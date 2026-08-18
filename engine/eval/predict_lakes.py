#!/usr/bin/env python3
"""Quick inference on specific lakes using a saved checkpoint.

Usage (on Sherlock, interactive GPU node):
    srun -p serc --gpus=1 --mem=32G --time=00:10:00 --pty bash
    ml system python/3.12.1 py-pytorch/2.2.1_py312 py-numpy/1.26.3_py312
    pip install --user netcdf4
    cd /oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision
    export PYTHONPATH=.
    python engine/eval/predict_lakes.py \
        --checkpoint models/essd/crossyear/lakevision_essd_crossyear.pth \
        --nc_dir /path/to/composites \
        --lake_ids CW2019_1681 CW2019_1653 CW2019_1578 CW2019_1794 CW2019_1577
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import netCDF4

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lakevision.models.classifier import LakeDrainageClassifier


CLASS_NAMES = ['ND', 'HF', 'MD', 'LD', 'CD']


def load_lake(nc_path, seq_len=153, channels=None):
    """Load a single lake composite and return (img_seq, area_seq, cloudy_seq)."""
    if channels is None:
        channels = ['red', 'green', 'blue', 'mask']

    with netCDF4.Dataset(str(nc_path)) as nc:
        nc.set_auto_mask(False)
        all_channels = list(nc.variables['imagery'].dimensions)
        # Channel names stored as coordinate
        if 'channel' in nc.variables:
            ch_names = [str(c) for c in nc.variables['channel'][:]]
        else:
            ch_names = ['red', 'green', 'blue', 'nir', 'swir16', 'cloudmask_scl', 'mask']

        ch_idxs = [ch_names.index(c) for c in channels]
        imagery = np.asarray(nc.variables['imagery'][:, ch_idxs, :, :], dtype=np.float32)

        water_area = np.asarray(nc.variables['water_area'][:], dtype=np.float32)

        if 'cloudy_seq_rgb' in nc.variables:
            cloudy_seq = np.asarray(nc.variables['cloudy_seq_rgb'][:], dtype=np.float32)
        else:
            # cloudy_seq polarity is 1=useful; the missing-data fallback must
            # be ones ("all frames useful"), matching LakeDataset. Zeros would
            # mean "all frames cloudy".
            cloudy_seq = np.ones(seq_len, dtype=np.float32)

        # Get label from attrs
        label = nc.getncattr('label') if 'label' in nc.ncattrs() else '?'

    # Mirror LakeDataset's training-time normalization: reflectance clipped to
    # [0,1] (mask passed through), area min-max normalized per sample.
    for i, ch in enumerate(channels):
        if ch != 'mask':
            imagery[:, i, :, :] = np.clip(imagery[:, i, :, :] / 10000.0, 0.0, 1.0)

    imagery = np.nan_to_num(imagery, nan=0.0)
    n_nan = int(np.isnan(water_area).sum())
    if n_nan:
        raise ValueError(
            f"{nc_path}: water_area contains {n_nan} NaN(s); composites "
            f"should be NaN-free — rebuild this file"
        )
    water_area = (water_area - water_area.min()) / (water_area.max() - water_area.min() + 1e-8)

    # Add batch dim (+ feature dim for the scalar sequences: [1, T, 1])
    img_seq = torch.from_numpy(imagery).unsqueeze(0)                       # [1, T, C, H, W]
    area_seq = torch.from_numpy(water_area).unsqueeze(0).unsqueeze(-1)     # [1, T, 1]
    cloudy_seq = torch.from_numpy(cloudy_seq).unsqueeze(0).unsqueeze(-1)   # [1, T, 1]

    return img_seq, area_seq, cloudy_seq, label


def main():
    parser = argparse.ArgumentParser(description='Predict drainage class for specific lakes')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model .pth checkpoint')
    parser.add_argument('--nc_dir', type=str, required=True,
                        help='Directory containing composite .nc files')
    parser.add_argument('--lake_ids', nargs='+', required=True,
                        help='Lake IDs to predict (e.g. CW2019_1681)')
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--seq_len', type=int, default=153)
    parser.add_argument('--frontcnn_base_channels', type=int, default=8)
    parser.add_argument('--frontcnn_num_layers', type=int, default=4)
    parser.add_argument('--clstm_hidden', type=int, default=32)
    parser.add_argument('--slstm_hidden', type=int, default=16)
    parser.add_argument('--classhead_hidden', type=int, default=64)
    parser.add_argument('--classhead_dropout', type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Build model with same architecture as training
    model = LakeDrainageClassifier(
        num_classes=args.num_classes,
        seq_len=args.seq_len,
        use_imgseq=True,
        use_areaseq=True,
        use_cloudyseq=False,
        use_nir=False,
        use_swir16=False,
        attention_type='none',
        frontcnn_base_channels=args.frontcnn_base_channels,
        frontcnn_num_layers=args.frontcnn_num_layers,
        clstm_hidden=args.clstm_hidden,
        slstm_hidden=args.slstm_hidden,
        classhead_hidden=args.classhead_hidden,
        classhead_dropout=args.classhead_dropout,
    )

    # Load checkpoint
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f'Loaded checkpoint: {args.checkpoint}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    nc_dir = Path(args.nc_dir)

    print(f'\n{"Lake ID":<20} {"True":>6} {"Pred":>6} {"Conf":>6}  Probabilities')
    print('-' * 75)

    for lake_id in args.lake_ids:
        nc_path = nc_dir / f'{lake_id}.nc'
        if not nc_path.exists():
            print(f'{lake_id:<20} FILE NOT FOUND: {nc_path}')
            continue

        img_seq, area_seq, cloudy_seq, true_label = load_lake(nc_path)
        img_seq = img_seq.to(device)
        area_seq = area_seq.to(device)
        cloudy_seq = cloudy_seq.to(device)

        with torch.no_grad():
            logits = model(img_seq, area_seq, cloudy_seq)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = probs.argmax()
            pred_label = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx]

        prob_str = '  '.join(f'{CLASS_NAMES[i]}={probs[i]:.3f}' for i in range(len(CLASS_NAMES)))
        match = '✓' if pred_label == true_label else '✗'
        print(f'{lake_id:<20} {true_label:>6} {pred_label:>6} {confidence:>5.1%}  {prob_str}  {match}')


if __name__ == '__main__':
    main()
