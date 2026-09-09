"""Activation memory per run at batch 8 x 153 frames, from saved-for-backward accounting at B=1, T=Tm, scaled linearly.

Run from the repo root:  ~/anaconda3/envs/lakevision/bin/python docs/eleven_runs/mem_bench.py docs/eleven_runs/mem_costs.json
"""
import sys, torch, torch.nn as nn, json
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from lakevision.models.classifier import LakeDrainageClassifier
from lakevision.models import blocks
torch.manual_seed(0)
Tm=6; H=512; B_FULL=8; T_FULL=153; scale = B_FULL*T_FULL/Tm

def build(in_ch=3, pool='avg', attn='none', base=8, hidden=32, gn=False, cloudy=False, attnpool=False):
    """Every run built through the real LakeDrainageClassifier flags (no stubs)."""
    m = LakeDrainageClassifier(num_classes=5, frontcnn_num_layers=4, frontcnn_base_channels=base, clstm_hidden=hidden,
            classhead_dropout=0.3, input_H=H, input_W=H, pool_type=pool, attention_type=attn, use_cloudyseq=cloudy,
            n_aux_channels=in_ch - 3, expect_channels=in_ch,
            frontcnn_norm='group' if gn else 'none', clstm_forget_bias=1.0 if gn else 0.0,
            temporal_readout='attn' if attnpool else 'last')
    m.train(); return m

def measure(m, in_ch, amp):
    seen = {}; phase = {'p': 'rest'}; tot = {'cnn': 0, 'rest': 0}
    def pack(t):
        st = t.untyped_storage(); k = st.data_ptr()
        if k not in seen: seen[k] = True; tot[phase['p']] += st.nbytes()
        return t
    m.frontcnn.register_forward_pre_hook(lambda *_: phase.__setitem__('p', 'cnn'))
    m.frontcnn.register_forward_hook(lambda *_: phase.__setitem__('p', 'rest'))
    x = torch.randn(1, Tm, in_ch, H, H); x[:, 1] = 0; area = torch.rand(1, Tm, 1); cl = torch.rand(1, Tm, 1)
    inp_ptr = x.untyped_storage().data_ptr()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        if amp:
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                out = m(x, area, cl)
        else:
            out = m(x, area, cl)
    # input storage counted if saved (conv1 saves its input); subtract so we can add the true fp32 batch separately
    if inp_ptr in seen: tot['cnn'] -= x.untyped_storage().nbytes()
    return tot

runs = [
 ('R0 reference',            dict()),
 ('R1 attn readout + both',  dict(pool='both', attnpool=True)),
 ('R2 GroupNorm',            dict(gn=True)),
 ('R3 +validity (4 ch)',     dict(in_ch=4)),
 ('R4 observed flag',        dict(cloudy=True)),
 ('R5 +2 masks (5 ch)',      dict(in_ch=5)),
 ('R6 spatial CBAM',         dict(attn='spatial')),
 ('R7 +NIR +SWIR16 (5 ch)',  dict(in_ch=5)),
 ('R8 soft targets',         dict()),
 ('R9 base16 hidden64',      dict(base=16, hidden=64)),
 ('R10 stack (8 ch)',        dict(in_ch=8, pool='both', attnpool=True, gn=True, cloudy=True, attn='spatial')),
]
GB=1e9
print(f"{'run':26s} {'in':>3s} {'batch':>6s} {'cnn':>6s} {'rest':>6s} | {'no-ckpt':>8s} {'ckpt':>6s}   (GB at B=8, T=153, bf16 autocast; +batch input fp32)")
rows=[]
for name, kw in runs:
    in_ch = kw.get('in_ch', 3)
    m = build(**kw)
    t = measure(m, in_ch, amp=True)
    batch = B_FULL*T_FULL*in_ch*H*H*4/GB
    cnn = t['cnn']*scale/GB; rest = t['rest']*scale/GB
    noc = batch + cnn + rest; ck = batch + max(cnn, rest)
    rows.append(dict(run=name, in_ch=in_ch, batch_gb=batch, cnn_gb=cnn, rest_gb=rest, peak_nockpt_gb=noc, peak_ckpt_gb=ck))
    print(f"{name:26s} {in_ch:3d} {batch:6.1f} {cnn:6.1f} {rest:6.1f} | {noc:8.1f} {ck:6.1f}")
json.dump(rows, open(sys.argv[1],'w'), indent=1)
