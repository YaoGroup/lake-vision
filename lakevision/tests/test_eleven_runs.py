"""
Tests for the eleven-runs pieces (docs/eleven_runs/ELEVEN_RUNS.html):

  dataset   validity channel, deposit masks as channels, mean-fill,
            the p_water observed flag, channel layout and counts
  model     GroupNorm, forget-gate bias, temporal readouts incl. attention with
            missing days masked, aux channels reaching the logits, exact channel
            check, per-chunk checkpointing being exact
  trainer   soft cross-entropy reducing to weighted CE on one-hot targets,
            warm-up + cosine schedule, provenance checkpoints loading both ways
"""
import json
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import torch
import torch.nn as nn

from lakevision.data.datasets import LakeDataset
from lakevision.models.blocks import FrontCNN, TemporalAttentionPool
from lakevision.models.classifier import LakeDrainageClassifier
from lakevision.models.clstm import CLSTM
from lakevision.models.checkpoint import save_checkpoint, load_checkpoint, describe_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine" / "training"))
import run_training as rt  # noqa: E402

T, H, W = 6, 8, 8
SEQ_LEN = 5
BANDS = ["B04", "B03", "B02", "B08", "B11", "B12"]
STATS = {c: {"mean": 5000.0, "std": 2000.0} for c in ["red", "green", "blue", "nir", "swir16", "swir22"]}


def write_deposit(fp, seed=0, blank_day=2, half_day=4):
    """Miniature stacks_v2 file: 6 bands, p_water with NaNs, both masks."""
    rng = np.random.default_rng(seed)
    refl = rng.uniform(1000, 9000, (T, 6, H, W)).astype(np.float32)
    refl[blank_day] = np.nan                 # no scene
    refl[half_day, :, :, : W // 2] = np.nan  # swath edge
    pw = np.array([np.nan, 0.2, np.nan, 0.9, np.nan, 0.4], np.float32)
    lb = np.zeros((H, W), np.uint8); lb[2:6, 2:6] = 1
    wm = np.zeros((T, H, W), np.uint8); wm[:, 3:5, 3:5] = 1; wm[blank_day] = 255
    with netCDF4.Dataset(fp, "w") as nc:
        for n, s in [("time", T), ("band", 6), ("y", H), ("x", W), ("string3", 3)]:
            nc.createDimension(n, s)
        v = nc.createVariable("reflectance", "f4", ("time", "band", "y", "x")); v[:] = refl
        v = nc.createVariable("band", "i4", ("band",)); v[:] = np.arange(6)
        v = nc.createVariable("band_name", "S1", ("band", "string3"))
        v.set_auto_chartostring(False)
        v[:] = np.array(BANDS, dtype="S3").view("S1").reshape(6, 3)
        v = nc.createVariable("p_water", "f4", ("time",)); v[:] = pw
        v = nc.createVariable("lake_boundary", "u1", ("y", "x")); v[:] = lb
        v = nc.createVariable("water_mask_ndwi", "u1", ("time", "y", "x"), fill_value=255); v[:] = wm
    return refl, pw, lb, wm


@pytest.fixture
def deposit(tmp_path):
    fp = tmp_path / "CW2019_0001.nc"
    return fp, write_deposit(fp)


# --------------------------------------------------------------------------- dataset
class TestDatasetAux:
    def test_default_path_unchanged(self, deposit):
        fp, _ = deposit
        ds = LakeDataset(fp, seq_len=SEQ_LEN, use_mask=False)
        img, area, cloudy, _, _ = ds[0]
        assert img.shape == (SEQ_LEN, 3, H, W) and ds.n_channels == 3 and ds.n_aux_channels == 0
        assert (cloudy == 1).all()

    def test_layout_and_counts(self, deposit):
        fp, _ = deposit
        ds = LakeDataset(fp, seq_len=SEQ_LEN, use_mask=False, use_nir=True, use_swir16=True,
                         band_stats=STATS, fill="mean", validity_channel=True, mask_source="both",
                         cloudy_seq_var="observed")
        img, area, cloudy, _, _ = ds[0]
        assert ds.aux_channel_names == ["validity", "mask_static", "mask_dynamic"]
        assert ds.n_channels == 8 and ds.n_aux_channels == 3 and ds.n_spectral_channels == 5
        assert img.shape == (SEQ_LEN, 8, H, W)

    def test_validity_and_mean_fill(self, deposit):
        fp, _ = deposit
        ds = LakeDataset(fp, seq_len=SEQ_LEN, use_mask=False, band_stats=STATS, fill="mean",
                         validity_channel=True)
        img, _, _, _, _ = ds[0]
        # window: peak at t=3, half=2 -> frames 1..5; blank day 2 -> index 1, half day 4 -> index 3
        assert img[1, 3].sum() == 0                      # validity all zero on the blank day
        assert img[3, 3].mean().item() == pytest.approx(0.5)
        assert (img[1, :3] == 0).all()                   # mean-filled then standardised -> 0
        assert (img[3, :3, :, : W // 2] == 0).all()
        assert (img[3, :3, :, W // 2:] != 0).any()

    def test_zero_fill_is_not_zero_after_standardising(self, deposit):
        fp, _ = deposit
        ds = LakeDataset(fp, seq_len=SEQ_LEN, use_mask=False, band_stats=STATS, fill="zero")
        img, _, _, _, _ = ds[0]
        assert torch.allclose(img[1, 0], torch.full((H, W), -2.5))  # (0-5000)/2000

    def test_masks(self, deposit):
        fp, (_, _, lb, wm) = deposit
        ds = LakeDataset(fp, seq_len=SEQ_LEN, use_mask=False, mask_source="both")
        img, _, _, _, _ = ds[0]
        assert torch.equal(img[0, 3], torch.tensor(lb, dtype=torch.float32))
        assert torch.equal(img[0, 4], torch.tensor(wm[1] == 1, dtype=torch.float32))
        assert img[1, 4].sum() == 0                      # 255 fill -> 0

    def test_observed_flag(self, deposit):
        fp, (_, pw, _, _) = deposit
        ds = LakeDataset(fp, seq_len=SEQ_LEN, use_mask=False, cloudy_seq_var="observed")
        _, _, cloudy, _, _ = ds[0]
        assert cloudy.squeeze().tolist() == np.isfinite(pw[1:6]).astype(float).tolist()

    def test_mean_fill_needs_stats(self, deposit):
        fp, _ = deposit
        with pytest.raises(ValueError, match="band_stats"):
            LakeDataset(fp, seq_len=SEQ_LEN, use_mask=False, fill="mean")

    def test_missing_mask_variable_raises(self, tmp_path):
        fp = tmp_path / "CW2019_0002.nc"
        write_deposit(fp)
        with netCDF4.Dataset(fp, "a") as nc:
            pass
        ds = LakeDataset(fp, seq_len=SEQ_LEN, use_mask=False, mask_source="static")
        ds[0]  # present -> fine
        fp2 = tmp_path / "CW2019_0003.nc"
        rng = np.random.default_rng(1)
        with netCDF4.Dataset(fp2, "w") as nc:
            for n, s in [("time", T), ("band", 6), ("y", H), ("x", W), ("string3", 3)]:
                nc.createDimension(n, s)
            v = nc.createVariable("reflectance", "f4", ("time", "band", "y", "x"))
            v[:] = rng.uniform(0, 1e4, (T, 6, H, W)).astype(np.float32)
            v = nc.createVariable("band", "i4", ("band",)); v[:] = np.arange(6)
            v = nc.createVariable("band_name", "S1", ("band", "string3"))
            v.set_auto_chartostring(False)
            v[:] = np.array(BANDS, dtype="S3").view("S1").reshape(6, 3)
            v = nc.createVariable("p_water", "f4", ("time",)); v[:] = np.linspace(0, 1, T)
        with pytest.raises(ValueError, match="lake_boundary"):
            LakeDataset(fp2, seq_len=SEQ_LEN, use_mask=False, mask_source="static")[0]


# --------------------------------------------------------------------------- model
def tiny(**kw):
    base = dict(use_imgseq=True, use_areaseq=True, use_cloudyseq=False, seq_len=4,
                input_H=32, input_W=32, frontcnn_base_channels=4, clstm_hidden=8,
                classhead_hidden=8, num_classes=5)
    base.update(kw)
    torch.manual_seed(0)
    m = LakeDrainageClassifier(**base)
    m.eval()
    return m


def inputs(n_ch, blank_day=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(2, 4, n_ch, 32, 32, generator=g)
    if blank_day is not None:
        x[0, blank_day] = 0
    return x, torch.rand(2, 4, 1, generator=g)


class TestModel:
    def test_groupnorm_present_and_clamped(self):
        f = FrontCNN(in_channels=3, base_channels=4, num_layers=2, norm="group", norm_groups=8)
        gns = [m for m in f.conv_block if isinstance(m, nn.GroupNorm)]
        assert [g.num_groups for g in gns] == [4, 8] and [g.num_channels for g in gns] == [4, 8]

    def test_forget_bias(self):
        c = CLSTM(input_channels=4, hidden_channels=8, forget_bias=1.0)
        b = c.cell.conv.bias.detach()
        assert torch.all(b[8:16] == 1.0)                       # forget gate only
        assert (b[:8].abs() < 1).all() and (b[16:].abs() < 1).all()  # i, o, g keep default init
        assert not torch.any(CLSTM(4, 8).cell.conv.bias.detach() == 1.0)

    @pytest.mark.parametrize("readout", ["last", "mean", "max", "attn"])
    @pytest.mark.parametrize("pool", ["avg", "both"])
    def test_readouts_run_and_backprop(self, readout, pool):
        m = tiny(temporal_readout=readout, pool_type=pool)
        m.train()
        x, a = inputs(3)
        out = m(x, a, None)
        assert out.shape == (2, 5)
        out.sum().backward()
        assert all(p.grad is not None for p in m.parameters() if p.requires_grad)

    def test_attention_masks_missing_days(self):
        m = tiny(temporal_readout="attn")
        x, a = inputs(3, blank_day=2)
        m(x, a, None)
        w = m.temporal_attn.last_weights
        assert w.shape == (2, 4)
        assert w[0, 2].item() == 0.0 and w[1, 2].item() > 0.0
        assert torch.allclose(w.sum(1), torch.ones(2))

    def test_attention_all_missing_stays_finite(self):
        pool = TemporalAttentionPool(6)
        h = torch.randn(1, 3, 6)
        out = pool(h, torch.zeros(1, 3, dtype=torch.bool))
        assert torch.isfinite(out).all() and pool.last_weights.sum().item() == pytest.approx(1.0)

    def test_aux_channels_reach_logits(self):
        m = tiny(n_aux_channels=2, expect_channels=5)
        x, a = inputs(5)
        x2 = x.clone(); x2[:, :, 3:] = 1 - x2[:, :, 3:]
        with torch.no_grad():
            assert (m(x, a, None) - m(x2, a, None)).abs().max() > 0

    def test_legacy_trailing_mask_still_discarded(self):
        m = tiny(n_aux_channels=1, expect_channels=5)   # 3 spectral + 1 aux + trailing mask
        x, a = inputs(5)
        x2 = x.clone(); x2[:, :, 4] = 1 - x2[:, :, 4]
        with torch.no_grad():
            assert (m(x, a, None) - m(x2, a, None)).abs().max() == 0

    def test_channel_count_enforced(self):
        m = tiny(n_aux_channels=1, expect_channels=4)
        x, a = inputs(6)
        with pytest.raises(RuntimeError, match="channels"):
            m(x, a, None)

    def test_chunked_checkpointing_is_exact(self):
        torch.manual_seed(0)
        f = FrontCNN(in_channels=3, base_channels=4, num_layers=2, chunk_size=3, checkpoint_chunks=True)
        f.train()
        x = torch.randn(2, 5, 3, 16, 16, requires_grad=True)
        y = f(x); y.sum().backward()
        g1 = x.grad.clone(); x.grad = None
        f.chunk_size = None; f.checkpoint_chunks = False
        y2 = f(x); y2.sum().backward()
        assert torch.allclose(y, y2, atol=1e-6) and torch.allclose(g1, x.grad, atol=1e-6)

    def test_whole_model_checkpoint_paths_agree(self):
        x, a = inputs(3)
        m1 = tiny(); m2 = tiny(frontcnn_chunk_size=2, gradient_checkpointing=True)
        m2.load_state_dict(m1.state_dict()); m1.train(); m2.train()
        assert torch.allclose(m1(x, a, None), m2(x, a, None), atol=1e-5)


# --------------------------------------------------------------------------- trainer
class TestTrainerPieces:
    def test_soft_ce_matches_weighted_ce_on_onehot(self):
        torch.manual_seed(0)
        logits = torch.randn(6, 5); y = torch.randint(0, 5, (6,))
        w = torch.tensor([1.0, 2.0, 0.5, 3.0, 1.5])
        hard = nn.CrossEntropyLoss(weight=w)(logits, y)
        soft = rt.soft_cross_entropy(logits, torch.nn.functional.one_hot(y, 5).float(), w)
        assert soft.item() == pytest.approx(hard.item(), rel=1e-5)

    def test_soft_ce_hedged_target_is_between(self):
        logits = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0]])
        w = torch.ones(5)
        p = torch.tensor([[0.5, 0.5, 0, 0, 0.0]])
        l0 = rt.soft_cross_entropy(logits, torch.eye(5)[[0]], w)
        l1 = rt.soft_cross_entropy(logits, torch.eye(5)[[1]], w)
        lp = rt.soft_cross_entropy(logits, p, w)
        assert l0 < lp < l1

    def test_soft_labels_loader(self, tmp_path):
        csv = tmp_path / "labels.csv"
        csv.write_text("lake_id,label,p_ND,p_HF,p_MD,p_LD,p_CD\n"
                       "A,HF,0,1.0,0,0,0\nB,MD,0,0.4,0.6,0,0\nC,LD,,,,,\n")
        soft = rt.load_soft_labels_essd_5class(csv, rt.CLASS_NAMES_ESSD_5CLASS)
        assert soft["B"].tolist() == pytest.approx([0, 0.4, 0.6, 0, 0])
        assert soft["C"].tolist() == [0, 0, 0, 1, 0]

    def test_warmup_cosine_schedule(self):
        opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1e-4)
        s = rt.build_lr_schedule(opt, "warmup_cosine", epochs=20, warmup_epochs=4, lr=1e-4, lr_min=1e-6)
        lrs = []
        for _ in range(20):
            lrs.append(opt.param_groups[0]["lr"]); s.step()
        assert lrs[0] == pytest.approx(0.25e-4) and lrs[3] == pytest.approx(1e-4)
        assert all(a >= b for a, b in zip(lrs[3:], lrs[4:]))
        assert lrs[-1] > 1e-6 and opt.param_groups[0]["lr"] == pytest.approx(1e-6, rel=1e-3)
        assert rt.build_lr_schedule(opt, "none", 5, 1, 1e-4, 1e-6) is None

    def test_checkpoint_both_formats(self, tmp_path):
        m = tiny()
        bare, prov = tmp_path / "bare.pth", tmp_path / "prov.pth"
        torch.save(m.state_dict(), bare)
        save_checkpoint(prov, m, {"config": {"lr": 1e-4}, "epoch": 3, "git_sha": "abc"})
        s1, meta1 = load_checkpoint(bare)
        s2, meta2 = load_checkpoint(prov)
        assert meta1 is None and meta2["epoch"] == 3 and meta2["git_sha"] == "abc"
        m2 = tiny(); m2.load_state_dict(s2)
        assert all(torch.equal(a, b) for a, b in zip(s1.values(), m2.state_dict().values()))
        assert "git_sha=abc" in describe_checkpoint(prov, meta2)
        s3, _ = load_checkpoint(prov, weights_only=True)   # falls back for our own file
        assert set(s3) == set(s1)

    def test_predictions_csv(self, tmp_path):
        metrics = {"lake_ids": ["A", "B"], "labels": np.array([0, 3]), "preds": np.array([1, 3]),
                   "probs": np.array([[.5, .3, .1, .05, .05], [.1, .1, .1, .6, .1]]),
                   "attn_weights": np.ones((2, 4)) / 4}
        out = tmp_path / "p.csv"
        rt.write_predictions_csv(out, metrics, rt.CLASS_NAMES_ESSD_5CLASS)
        rows = out.read_text().splitlines()
        assert rows[0] == "lake_id,true_label,pred_label,p_ND,p_HF,p_MD,p_LD,p_CD"
        assert rows[1].startswith("A,ND,HF,0.500000")
        assert np.load(out.with_suffix(".attn.npz"))["weights"].shape == (2, 4)


# --------------------------------------------------------------------------- end to end
def _deposit_dir(tmp_path, n=6, hw=32):
    """n tiny deposit files (T=6, hw x hw) plus a 5-class label CSV with soft columns."""
    d = tmp_path / "stacks"; d.mkdir()
    labels = ["ND", "HF", "MD", "LD", "CD", "LD"]
    rows = ["lake_id,label,p_ND,p_HF,p_MD,p_LD,p_CD"]
    for i in range(n):
        fp = d / f"CW2019_{i:04d}.nc"
        rng = np.random.default_rng(i)
        refl = rng.uniform(1000, 9000, (T, 6, hw, hw)).astype(np.float32)
        refl[2] = np.nan
        with netCDF4.Dataset(fp, "w") as nc:
            for nm, s in [("time", T), ("band", 6), ("y", hw), ("x", hw), ("string3", 3)]:
                nc.createDimension(nm, s)
            v = nc.createVariable("reflectance", "f4", ("time", "band", "y", "x")); v[:] = refl
            v = nc.createVariable("band", "i4", ("band",)); v[:] = np.arange(6)
            v = nc.createVariable("band_name", "S1", ("band", "string3"))
            v.set_auto_chartostring(False)
            v[:] = np.array(BANDS, dtype="S3").view("S1").reshape(6, 3)
            v = nc.createVariable("p_water", "f4", ("time",))
            v[:] = np.array([np.nan, 0.2, np.nan, 0.9, np.nan, 0.4], np.float32)
            v = nc.createVariable("lake_boundary", "u1", ("y", "x")); v[:] = (rng.random((hw, hw)) > 0.7)
            v = nc.createVariable("water_mask_ndwi", "u1", ("time", "y", "x"), fill_value=255)
            v[:] = (rng.random((T, hw, hw)) > 0.8).astype(np.uint8)
        p = np.zeros(5); p[["ND", "HF", "MD", "LD", "CD"].index(labels[i])] = 0.8; p[(i + 1) % 5] += 0.2
        rows.append(f"CW2019_{i:04d},{labels[i]}," + ",".join(f"{x:.2f}" for x in p))
    csv = tmp_path / "labels.csv"; csv.write_text("\n".join(rows) + "\n")
    ids = [f"CW2019_{i:04d}" for i in range(n)]
    for name, sub in [("train", ids[:4]), ("val", ids[4:5]), ("test", ids[5:6])]:
        json.dump(sub, open(tmp_path / f"{name}_ids.json", "w"))
    stats = {c: {"mean": 5000.0, "std": 2300.0} for c in ["red", "green", "blue", "nir", "swir16", "swir22"]}
    json.dump(stats, open(tmp_path / "band_stats.json", "w"))
    return d, csv


R10_FLAGS = dict(temporal_readout="attn", pool_type="both", frontcnn_norm="group", clstm_forget_bias=1.0,
                 lr_schedule="warmup_cosine", warmup_epochs=1, fill="mean", validity_channel=True,
                 use_cloudyseq=True, cloudy_seq_var="observed", mask_source="both", attention_type="spatial",
                 use_nir=True, use_swir16=True, soft_labels=True)


@pytest.mark.parametrize("flags", [dict(), R10_FLAGS, dict(frontcnn_base_channels=16, clstm_hidden=64)],
                         ids=["R0", "R10", "R9"])
def test_trainer_end_to_end(tmp_path, flags):
    d, csv = _deposit_dir(tmp_path)
    config = dict(
        labels_csv=[str(csv)], test_labels_csv=None, nc_dir=str(d), label_mode="essd_5class",
        id_col="lake_id", label_col="label", merge_classes=None,
        train_ids_file=str(tmp_path / "train_ids.json"), val_ids_file=str(tmp_path / "val_ids.json"),
        test_ids_file=str(tmp_path / "test_ids.json"),
        epochs=2, batch_size=2, lr=1e-3, weight_decay=0.0, amp=False, use_scheduler=False, lr_schedule="none",
        warmup_epochs=10, lr_min=1e-6, num_workers=0, seed=0, seq_len=153, stratify=True,
        band_stats=str(tmp_path / "band_stats.json") if flags.get("fill") == "mean" else None,
        no_mask=True, num_classes=5, frontcnn_num_layers=4, frontcnn_out_hw=None, classhead_dropout=0.0,
        gradient_checkpointing=False, accumulation_steps=1, preload_to_ram=False,
        save_path=str(tmp_path / "m.pth"), test_checkpoint="f1",
        test_predictions_csv=str(tmp_path / "pred.csv"), no_wandb=True,
    )
    config.update(flags)
    best_val_loss, test_metrics = rt.train(config)
    assert np.isfinite(best_val_loss) and np.isfinite(test_metrics["f1_macro"])
    assert (tmp_path / "m_bestf1.pth").exists()
    state, meta = load_checkpoint(tmp_path / "m_bestf1.pth")
    assert meta["config"].get("temporal_readout", "last") == flags.get("temporal_readout", "last")
    lines = (tmp_path / "pred.csv").read_text().splitlines()
    assert len(lines) == 2 and lines[1].startswith("CW2019_0005,LD,")
    if flags.get("temporal_readout") == "attn":
        assert (tmp_path / "pred.attn.npz").exists()
