"""Params + forward FLOPs per lake for every run in docs/eleven_runs/ELEVEN_RUNS.html.

Run from the repo root:  ~/anaconda3/envs/lakevision/bin/python docs/eleven_runs/bench_variants.py docs/eleven_runs/bench_costs.json
"""
import sys, torch, torch.nn as nn, torch.nn.functional as F, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
from lakevision.models.classifier import LakeDrainageClassifier
from lakevision.models import blocks
from torch.utils.flop_counter import FlopCounterMode
torch.manual_seed(0)
T=3; H=512

def paper_frontcnn_forward(self, x):  # the ESSD-era forward: adaptive_max_pool UP to out_hw
    B,T_,C,H_,W_ = x.shape
    x = self.conv_block(x.view(B*T_,C,H_,W_))
    if self.out_hw is not None and tuple(x.shape[-2:]) != tuple(self.out_hw):
        x = F.adaptive_max_pool2d(x, self.out_hw)
    _,C2,H2,W2 = x.shape
    return x.view(B,T_,C2,H2,W2)

class AttnPool(nn.Module):  # temporal attention pooling over [B,T,D] with validity mask
    def __init__(self, d): super().__init__(); self.q = nn.Linear(d, 1)
    def forward(self, h, valid=None):
        s = self.q(h).squeeze(-1)
        if valid is not None: s = s.masked_fill(~valid, -1e4)
        a = torch.softmax(s, 1)
        return (a.unsqueeze(-1)*h).sum(1), a

def add_groupnorm(fcnn):
    layers=[]; 
    for m in fcnn.conv_block:
        layers.append(m)
        if isinstance(m, nn.Conv2d): layers.append(nn.GroupNorm(min(8, m.out_channels), m.out_channels))
    fcnn.conv_block = nn.Sequential(*layers)

def build(name, in_ch=3, out_hw=None, pool='avg', attn='none', base=8, layers=4, hidden=32, gn=False, cloudy=False, attnpool=False, essd=False):
    kw=dict(num_classes=5, frontcnn_num_layers=layers, frontcnn_base_channels=base, clstm_hidden=hidden,
            classhead_dropout=0.3, input_H=H, input_W=H, pool_type=pool, attention_type=attn, use_cloudyseq=cloudy)
    if essd:
        orig = blocks.FrontCNN.forward; blocks.FrontCNN.forward = paper_frontcnn_forward
        kw['frontcnn_out_hw']=(64,64)
        # the guard lives in __init__? no — in forward; also effective_hw check: construct then restore
        m = LakeDrainageClassifier(**kw); blocks.FrontCNN.forward = paper_frontcnn_forward
    else:
        m = LakeDrainageClassifier(**kw)
    if in_ch != 3:
        m.frontcnn = blocks.FrontCNN(in_channels=in_ch, base_channels=base, num_layers=layers, out_hw=kw.get('frontcnn_out_hw'), pool='max')
        m.n_imagery_channels = in_ch
        if attn == 'spatial':
            from lakevision.models.attention import SpatialCBAM
            m.attention = SpatialCBAM(m.frontcnn.output_channels, 7)
    if gn: add_groupnorm(m.frontcnn)
    if attnpool: m.attnpool = AttnPool(hidden*(2 if pool=='both' else 1))
    m.eval()
    x = torch.randn(1, T, in_ch, H, H); area = torch.rand(1, T, 1); cl = torch.rand(1, T, 1)
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad(): m(x, area, cl)
    flops = fc.get_total_flops()
    if attnpool:  # extra cost of pooling every hidden state instead of the last: GAP over T maps (negligible) — count it
        flops += T * hidden * (32*32)
    params = sum(p.numel() for p in m.parameters())
    per_frame = flops / T
    return dict(name=name, params=params, gflops_per_frame=per_frame/1e9, gflops_per_lake=per_frame*153/1e9)

rows = [
 build('ESSD as published (64×64 upsample)', essd=True),
 build('R0 reference (32×32, no upsample)'),
 build('R1 readout: temporal attention + avg+max', pool='both', attnpool=True),
 build('R2 conditioning: GroupNorm (+forget bias, cosine)', gn=True),
 build('R3 pixels: standardise + mean-fill + validity channel', in_ch=4),
 build('R4 observed flag: p_water flag via the cloudy_seq slot', cloudy=True),
 build('R5 masks: +static +dynamic mask channels', in_ch=5),
 build('R6 attention: spatial CBAM on the feature map', attn='spatial'),
 build('R7 spectral: +NIR +SWIR16', in_ch=5),
 build('R8 soft targets (no architecture change)'),
 build('R9 capacity: base 16 · hidden 64', base=16, hidden=64),
 build('R10 the stack (R1–R8 together)', in_ch=8, pool='both', attnpool=True, gn=True, cloudy=True, attn='spatial'),
]
for r in rows: print(f"{r['name']:58s} params {r['params']:>8,d}  GFLOP/frame {r['gflops_per_frame']:6.2f}  GFLOP/lake {r['gflops_per_lake']:7.1f}")
json.dump(rows, open(sys.argv[1],'w'), indent=1)
