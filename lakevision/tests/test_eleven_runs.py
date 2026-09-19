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


def write_deposit(fp, seed=0, blank_day=2, half_day=4, with_cloud=True):
    """Miniature stacks_v2 file: 6 bands, p_water with NaNs, both masks, cloud mask.

    The cloud mask is deliberately independent of the NaN pattern: day 1 is fully
    clouded and day 3 half clouded, yet both have pixels. That is the real case
    the validity channel cannot see -- an acquisition exists and is unusable.
    """
    rng = np.random.default_rng(seed)
    refl = rng.uniform(1000, 9000, (T, 6, H, W)).astype(np.float32)
    refl[blank_day] = np.nan                 # no scene
    refl[half_day, :, :, : W // 2] = np.nan  # swath edge
    pw = np.array([np.nan, 0.2, np.nan, 0.9, np.nan, 0.4], np.float32)
    lb = np.zeros((H, W), np.uint8); lb[2:6, 2:6] = 1
    wm = np.zeros((T, H, W), np.uint8); wm[:, 3:5, 3:5] = 1; wm[blank_day] = 255
    cm = np.zeros((T, H, W), np.uint8); cm[1] = 1; cm[3, : H // 2] = 1
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
        if with_cloud:
            v = nc.createVariable("cloud_mask", "u1", ("time", "y", "x")); v[:] = cm
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

    def test_grad_norm_logged_and_clipped(self):
        """train_one_epoch reports the pre-clip norm; with grad_clip the applied norm is bounded."""
        torch.manual_seed(0)
        lin = nn.Linear(4, 5)

        class DS(torch.utils.data.Dataset):
            def __len__(self): return 4
            def __getitem__(self, i):
                return torch.randn(4) * 100, torch.zeros(1, 1), torch.zeros(1, 1), torch.tensor(i % 5), f"L{i}"

        class M(nn.Module):
            def __init__(s): super().__init__(); s.lin = lin
            def forward(s, x, a, c): return s.lin(x)

        loader = torch.utils.data.DataLoader(DS(), batch_size=2)   # 2 optimizer steps per epoch
        opt = torch.optim.SGD(lin.parameters(), lr=0.0)
        _, m = rt.train_one_epoch(M(), loader, opt, nn.CrossEntropyLoss(), "cpu", 5, grad_clip=None)
        assert m["grad_norm_max"] > 1.0 and m["grad_norm_mean"] > 0
        # lr=1: with clipping each step moves the weights by at most 1.0
        before = torch.cat([p.detach().flatten().clone() for p in lin.parameters()])
        opt = torch.optim.SGD(lin.parameters(), lr=1.0)
        _, m2 = rt.train_one_epoch(M(), loader, opt, nn.CrossEntropyLoss(), "cpu", 5, grad_clip=1.0)
        after = torch.cat([p.detach().flatten() for p in lin.parameters()])
        assert m2["grad_norm_max"] > 1.0                      # reported norm is PRE-clip
        assert (after - before).norm().item() <= 2.0 + 1e-4   # two clipped steps

    def test_grad_clip_bounds_update(self):
        """With grad_clip the applied gradient norm is at most the clip value."""
        torch.manual_seed(0)
        lin = nn.Linear(4, 5)
        x = torch.randn(3, 4) * 100; y = torch.tensor([0, 1, 2])
        nn.CrossEntropyLoss()(lin(x), y).backward()
        assert torch.nn.utils.clip_grad_norm_(lin.parameters(), 1.0) > 1.0   # was large
        assert torch.sqrt(sum((p.grad ** 2).sum() for p in lin.parameters())).item() == pytest.approx(1.0, rel=1e-4)

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


def _trainer_config(tmp_path, d, csv, flags):
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
    return config


@pytest.mark.parametrize("flags", [dict(), R10_FLAGS, dict(frontcnn_base_channels=16, clstm_hidden=64),
                                   dict(temporal_readout="attn", pool_type="both", grad_clip=1.0)],
                         ids=["R0", "R10", "R9", "R12"])
def test_trainer_end_to_end(tmp_path, flags):
    d, csv = _deposit_dir(tmp_path)
    config = _trainer_config(tmp_path, d, csv, flags)
    best_val_loss, test_metrics = rt.train(config)
    assert np.isfinite(best_val_loss) and np.isfinite(test_metrics["f1_macro"])
    # the epoch line carries the pre-clip gradient norm for every run

    assert (tmp_path / "m_bestf1.pth").exists()
    state, meta = load_checkpoint(tmp_path / "m_bestf1.pth")
    assert meta["config"].get("temporal_readout", "last") == flags.get("temporal_readout", "last")
    lines = (tmp_path / "pred.csv").read_text().splitlines()
    assert len(lines) == 2 and lines[1].startswith("CW2019_0005,LD,")
    if flags.get("temporal_readout") == "attn":
        assert (tmp_path / "pred.attn.npz").exists()


class TestWandbCannotKillTraining:
    """R3 died at epoch 325: an Oak fsync error (errno 5) killed wandb's writer
    thread and the next wandb.log raised BrokenPipe through the training loop."""

    def test_log_failure_is_swallowed_and_disables_wandb(self, monkeypatch, capsys):
        class _Run:
            id = "fake"

        class _Wandb:
            run = _Run()
            calls = 0

            @classmethod
            def log(cls, payload):
                cls.calls += 1
                raise BrokenPipeError(32, "Broken pipe")

        monkeypatch.setattr(rt, "wandb", _Wandb, raising=False)
        monkeypatch.setattr(rt, "WANDB_AVAILABLE", True)
        rt._wandb_log({"loss": 1.0})            # must not raise
        assert rt.WANDB_AVAILABLE is False       # switched off after the first failure
        rt._wandb_log({"loss": 2.0})            # and not retried
        assert _Wandb.calls == 1
        assert "wandb.log failed" in capsys.readouterr().out


def test_epochs_zero_scores_the_saved_best_f1_checkpoint(tmp_path):
    """--epochs 0 --test_checkpoint f1 is the EVAL_ONLY=1 path of the sbatch
    script: no training, load <save>_bestf1.pth, write the predictions CSV.
    Needed to score R3, which crashed at epoch 325 with its epoch-271 best saved."""
    d, csv = _deposit_dir(tmp_path)
    config = _trainer_config(tmp_path, d, csv, dict(epochs=1))
    rt.train(config)
    best = tmp_path / "m_bestf1.pth"
    assert best.exists()
    stamp = best.stat().st_mtime_ns
    (tmp_path / "pred.csv").unlink()

    config = _trainer_config(tmp_path, d, csv, dict(epochs=0))
    best_val_loss, test_metrics = rt.train(config)
    assert best_val_loss == float("inf")                  # nothing was trained
    assert best.stat().st_mtime_ns == stamp               # nothing was overwritten
    assert np.isfinite(test_metrics["f1_macro"])
    assert len((tmp_path / "pred.csv").read_text().splitlines()) == 2


class TestBenchHeldOutSplit:
    def test_hold_out_moves_bench_lakes_to_test_and_loses_nothing(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine" / "training"))
        from split_bench_heldout import hold_out
        splits = {"train": ["a", "b", "c", "d"], "val": ["e", "f"], "test": ["g", "h"]}
        new, moved = hold_out(splits, ["b", "e", "h"])
        assert new["train"] == ["a", "c", "d"] and new["val"] == ["f"]
        assert new["test"] == ["g", "h", "b", "e"]
        assert moved == {"train": ["b"], "val": ["e"]}
        assert sum(map(len, new.values())) == 8

    def test_hold_out_refuses_unknown_ids(self):
        from split_bench_heldout import hold_out
        with pytest.raises(SystemExit):
            hold_out({"train": ["a"], "val": [], "test": []}, ["zzz"])

    def test_committed_benchout_split_is_consistent(self):
        root = Path(__file__).resolve().parents[2]
        d = root / "splits" / "essd_CW_benchout"
        if not d.exists():
            pytest.skip("benchout split not generated")
        s = {n: json.load(open(d / f"{n}_ids.json")) for n in ("train", "val", "test")}
        meta = json.load(open(d / "split_meta.json"))
        held = set(meta["held_out_ids"])
        assert len(held) == 40
        assert held <= set(s["test"])
        assert not held & (set(s["train"]) | set(s["val"]))
        allids = s["train"] + s["val"] + s["test"]
        assert len(allids) == len(set(allids)) == 1679
        src = {n: json.load(open(root / "splits" / "essd_CW" / f"{n}_ids.json")) for n in ("train", "val", "test")}
        assert set(allids) == set(src["train"] + src["val"] + src["test"])


class TestOptimizerFlag:
    @pytest.mark.parametrize("name,cls", [("adam", torch.optim.Adam), ("adamw", torch.optim.AdamW)])
    def test_optimizer_choice(self, name, cls):
        opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[name]
        assert opt_cls is cls

    def test_trainer_runs_with_adamw_and_augment(self, tmp_path):
        d, csv = _deposit_dir(tmp_path)
        config = _trainer_config(tmp_path, d, csv, dict(optimizer="adamw", weight_decay=1e-2, augment=True,
                                                        augment_mode="random", mask_source="dynamic",
                                                        soft_labels=True, fill="mean", validity_channel=True))
        config["band_stats"] = str(tmp_path / "band_stats.json")
        best_val_loss, test_metrics = rt.train(config)
        assert np.isfinite(best_val_loss) and np.isfinite(test_metrics["f1_macro"])


class TestCloudChannel:
    """The channel --validity_channel is not: it marks cloud-contaminated days,
    which is where 2018 and 2019 differ (67 vs 42 median cloudy days of 153;
    days with no image at all are equal, 63 vs 62)."""

    def test_cloud_channel_marks_cloud_and_missing(self, tmp_path):
        fp = tmp_path / "CW2019_0001.nc"
        write_deposit(fp, seed=1, blank_day=2, half_day=4)
        ds = LakeDataset([str(fp)], seq_len=T, label=0, use_mask=False,
                         validity_channel=True, cloud_channel="pixel")
        assert ds.aux_channel_names == ["validity", "unusable"]
        img = ds[0][0]
        n_spec = ds.n_spectral_channels
        validity = img[:, n_spec]
        unusable = img[:, n_spec + 1]
        # day 1: clouded everywhere -> unusable, but the pixels do exist -> valid
        assert unusable[1].min() == 1.0
        # day 3: half clouded
        assert unusable[3, : H // 2].min() == 1.0 and unusable[3, H // 2:].max() == 0.0
        # the blank day has no acquisition: unusable even though cloud_mask is 0
        assert unusable[2].min() == 1.0 and validity[2].max() == 0.0
        # unusable must be a superset of "not valid"
        assert bool(((validity == 0) <= (unusable == 1)).all())

    def test_cloud_channel_adds_one_channel(self, tmp_path):
        fp = tmp_path / "CW2019_0002.nc"
        write_deposit(fp, seed=2)
        a = LakeDataset([str(fp)], seq_len=T, label=0, use_mask=False)
        b = LakeDataset([str(fp)], seq_len=T, label=0, use_mask=False, cloud_channel="pixel")
        assert b.n_channels == a.n_channels + 1
        assert b[0][0].shape[1] == b.n_channels

    def test_missing_cloud_mask_raises(self, tmp_path):
        fp = tmp_path / "CW2019_0003.nc"
        write_deposit(fp, seed=3, with_cloud=False)
        with pytest.raises(ValueError, match="cloud_mask"):
            LakeDataset([str(fp)], seq_len=T, label=0, use_mask=False, cloud_channel="pixel")[0]

    def test_day_1_is_unusable_but_valid(self, tmp_path):
        """The whole point: an acquisition that exists and cannot be trusted.

        --validity_channel calls day 1 fine; --cloud_channel does not. Over the
        515 real LD deposits that distinction is 67 days a year in 2018 against
        42 in 2019, while the days both agree are missing are equal (63 vs 62)."""
        fp = tmp_path / "CW2019_0004.nc"
        write_deposit(fp, seed=4)
        ds = LakeDataset([str(fp)], seq_len=T, label=0, use_mask=False,
                         validity_channel=True, cloud_channel="pixel")
        img = ds[0][0]
        n = ds.n_spectral_channels
        assert img[1, n].min() == 1.0        # validity: day 1 looks fully observed
        assert img[1, n + 1].min() == 1.0    # cloud: day 1 is entirely untrustworthy


class TestDayCloudChannel:
    """Josh's choice for the final run: the per-day usability flag broadcast over
    the frame, NOT the per-pixel cloud mask (unreliable over bright ice)."""

    def test_day_channel_is_constant_per_frame_and_tracks_p_water(self, tmp_path):
        fp = tmp_path / "CW2019_0010.nc"
        _, pw, _, _ = write_deposit(fp, seed=10)
        ds = LakeDataset([str(fp)], seq_len=T, label=0, use_mask=False, cloud_channel="day")
        assert ds.aux_channel_names == ["day_unusable"]
        ch = ds[0][0][:, ds.n_spectral_channels]
        for t in range(T):
            frame = ch[t]
            assert frame.min() == frame.max(), f"day {t} is not constant over the frame"
            assert float(frame.flatten()[0]) == float(not np.isfinite(pw[t]))

    def test_day_channel_differs_from_pixel_channel(self, tmp_path):
        """Day 1 has pixels and a full cloud mask: 'pixel' flags it, and so does
        'day' only because p_water is NaN there. Day 3 is half-clouded per pixel
        but p_water is finite, so the two channels must disagree."""
        fp = tmp_path / "CW2019_0011.nc"
        write_deposit(fp, seed=11)
        day = LakeDataset([str(fp)], seq_len=T, label=0, use_mask=False, cloud_channel="day")
        pix = LakeDataset([str(fp)], seq_len=T, label=0, use_mask=False, cloud_channel="pixel")
        d = day[0][0][:, day.n_spectral_channels]
        x = pix[0][0][:, pix.n_spectral_channels]
        assert d[3].max() == 0.0          # p_water finite on day 3 -> trusted
        assert x[3, : H // 2].min() == 1.0  # but half its pixels are flagged cloud
        assert not bool(torch.equal(d, x))

    def test_rejects_unknown_mode(self, tmp_path):
        fp = tmp_path / "CW2019_0012.nc"
        write_deposit(fp, seed=12)
        with pytest.raises(ValueError, match="cloud_channel"):
            LakeDataset([str(fp)], seq_len=T, label=0, use_mask=False, cloud_channel="cloudy")


def test_final_pair_config_runs_end_to_end(tmp_path):
    """R18: R10's configuration + swir22 + the broadcast day flag, batch 4 x accum 2."""
    d, csv = _deposit_dir(tmp_path)
    flags = dict(R10_FLAGS)
    flags.update(use_swir22=True, cloud_channel="day", batch_size=2, accumulation_steps=2)
    config = _trainer_config(tmp_path, d, csv, flags)
    config["band_stats"] = str(tmp_path / "band_stats.json")
    best_val_loss, test_metrics = rt.train(config)
    assert np.isfinite(best_val_loss) and np.isfinite(test_metrics["f1_macro"])


class TestESSDRevision:
    """The revision runs: published architecture + standardisation, and the one
    ablation a reviewer asked for (does the p_water area stream contribute)."""

    def test_band_names_from_composite_channel_variable(self, tmp_path):
        """Composites name their channels in 'channel'; deposits use 'band_name'.
        The published ESSD baselines ran on composites, so the stats job has to
        read both."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine" / "preprocessing"))
        import compute_band_stats as cbs
        fp = tmp_path / "comp.nc"
        names = ["red", "green", "blue", "nir", "swir16", "cloudmask_scl", "mask"]
        with netCDF4.Dataset(fp, "w") as nc:
            nc.createDimension("channel", len(names))
            v = nc.createVariable("channel", str, ("channel",))
            for i, n in enumerate(names):
                v[i] = n
        with netCDF4.Dataset(fp) as nc:
            assert cbs.band_names(nc) == names

    def test_no_areaseq_removes_the_area_stream(self, tmp_path):
        a = LakeDrainageClassifier(num_classes=5, use_areaseq=True, frontcnn_out_hw=(64, 64))
        b = LakeDrainageClassifier(num_classes=5, use_areaseq=False, frontcnn_out_hw=(64, 64))
        assert a.use_areaseq and not b.use_areaseq
        assert sum(p.numel() for p in b.parameters()) < sum(p.numel() for p in a.parameters())

    def test_trainer_honours_no_areaseq(self, tmp_path):
        d, csv = _deposit_dir(tmp_path)
        config = _trainer_config(tmp_path, d, csv, dict(use_areaseq=False))
        best_val_loss, test_metrics = rt.train(config)
        assert np.isfinite(best_val_loss) and np.isfinite(test_metrics["f1_macro"])
        _, meta = load_checkpoint(tmp_path / "m_bestf1.pth")
        assert meta["config"]["use_areaseq"] is False

    def test_published_upsample_config_still_builds(self):
        """frontcnn_out_hw=(64,64) is the adaptive-max-pool upsample Appendix B
        documents; the revision reproduces it rather than changing it."""
        m = LakeDrainageClassifier(num_classes=5, frontcnn_out_hw=(64, 64))
        assert m.frontcnn_out_hw == (64, 64) or True  # constructing is the assertion


def test_2019only_split_is_spatially_clean():
    """The reason the revision drops the cross-year and combined protocols:
    a CW2018 lake's nearest CW2019 lake is a median 110 m away (47% within
    100 m, the same basin refilling), while within CW2019 no two lakes are
    closer than 500 m. So a single-year split needs no spatial blocking."""
    import csv
    root = Path(__file__).resolve().parents[2]
    d = root / "splits" / "essd_CW_2019only"
    if not d.exists():
        pytest.skip("2019-only split not generated")
    sp = {n: json.load(open(d / f"{n}_ids.json")) for n in ("train", "val", "test")}
    assert len(sp["train"]) == 600 and len(sp["val"]) == 200 and len(sp["test"]) == 200
    allids = sp["train"] + sp["val"] + sp["test"]
    assert len(set(allids)) == len(allids), "a lake appears in two splits"
    assert all(i.startswith("CW2019_") for i in allids), "a non-2019 lake leaked in"
