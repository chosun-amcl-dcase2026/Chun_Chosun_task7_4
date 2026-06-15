"""
Chun_Chosun_task7_4_model.py — DCASE2026 Task 7 official submission model file.

Self-contained model definition + `load_model(task)` as required by
https://dcase.community/challenge2026/submission :

    load_model(task: int) -> torch.nn.Module
        task=2 : Step-2 system  (D2 5-fold soft-voting ensemble, adapters up to D2)
        task=3 : Step-3 system  (D3 K-Fold ensemble, val_macro-selected N models,
                                 adapters up to D3 — final system)

The returned module takes a raw mono waveform at 32 kHz
(shape [num_samples] or [batch, num_samples]) and returns log-probabilities
[batch, 10]; argmax over dim=-1 gives the predicted class index
(see CLASS_LABELS below for the index → name mapping).

Inference is domain-agnostic: a single forward pass with a fixed
`active_domain_count` — no domain label or routing is used.

Weights are loaded from the bundled dictionaries placed next to this file:
    Chun_Chosun_task7_4_D2_dictionary.pth
    Chun_Chosun_task7_4_D3_dictionary.pth
Each bundle stores the shared frozen CNN14 backbone (identical across all
ensemble members, D1-pretrained) once, plus per-member LoRA adapter weights.

Dependencies: torch, torchlibrosa (same as the official baseline).
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

# ── audio / model constants (frozen copies of the training config) ───────────
SAMPLE_RATE = 32000
WIN_SIZE    = 1024
HOP_SIZE    = 320
MEL_BINS    = 64
FMIN        = 50
FMAX        = 14000
NUM_CLASSES = 10

LORA_ALPHA      = 64.0
GATE_CFG        = {'use': True, 'init_logit': 4.0}
CHUNK_SECONDS     = 4.0
CHUNK_MIN_SECONDS = 1.0
TTA_GAINS         = [1.0, 0.85, 1.15]

# per-domain backbone CKA drift → LoRA ranks (identical to training config)
DOMAINS = [
    {   # D2
        'drift_conv_bn': [0.0230, 0.2463, 0.2756, 0.3021, 0.5021, 0.7848],
        'drift_bn':      [0.0171, 0.0215, 0.0000, 0.0744, 0.0455, 0.0244],
        'ceiling': 128,
    },
    {   # D3
        'drift_conv_bn': [0.2215, 0.1719, 0.1216, 0.1823, 0.2413, 0.2502],
        'drift_bn':      [0.1010, 0.0498, 0.1092, 0.1187, 0.2301, 0.2291],
        'ceiling': 80,
    },
]

CLASS_LABELS = {
    'alarm': 0, 'baby_cry': 1, 'dog_bark': 2, 'engine': 3, 'fire': 4,
    'footsteps': 5, 'knocking': 6, 'telephone_ringing': 7, 'piano': 8, 'speech': 9,
}

# CNN14 conv blocks (in_ch, out_ch)
BLOCK_CONFIGS = [(1, 64), (64, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048)]


def drift_to_block_configs(drift_conv_bn, drift_bn, ceiling):
    """Per-block CKA drift → per-block (r_A, r_B) LoRA ranks."""
    db  = [max(0.0, float(x)) for x in drift_bn]
    cb  = [float(x) for x in drift_conv_bn]
    tot = [a + b for a, b in zip(cb, db)]
    s   = float(ceiling) / max(tot)
    in_dims = [in_ch * 9 for (in_ch, _) in BLOCK_CONFIGS]
    r_A = [max(1, min(int(round(s * a)), in_dims[i])) for i, a in enumerate(cb)]
    r_B = [max(1, int(round(s * b))) for b in db]
    return list(zip(r_A, r_B))


# ── frozen CNN14 backbone (PANNs layout, D1-pretrained) ──────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, nb_tasks=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, (3, 3), (1, 1), (1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, (3, 3), (1, 1), (1, 1), bias=False)
        self.bnF = nn.ModuleList([nn.BatchNorm2d(out_ch) for _ in range(nb_tasks)])
        self.bnS = nn.ModuleList([nn.BatchNorm2d(out_ch) for _ in range(nb_tasks)])


class CNN14Backbone(nn.Module):
    def __init__(self, nb_tasks=3):
        super().__init__()
        self.spectrogram_extractor = Spectrogram(
            n_fft=WIN_SIZE, hop_length=HOP_SIZE, win_length=WIN_SIZE,
            window='hann', center=True, pad_mode='reflect', freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(
            sr=SAMPLE_RATE, n_fft=WIN_SIZE, n_mels=MEL_BINS,
            fmin=FMIN, fmax=FMAX, ref=1.0, amin=1e-10, top_db=None,
            freeze_parameters=True)
        self.bn0 = nn.ModuleList([nn.BatchNorm2d(64) for _ in range(nb_tasks)])
        self.conv_block1 = ConvBlock(1, 64, nb_tasks)
        self.conv_block2 = ConvBlock(64, 128, nb_tasks)
        self.conv_block3 = ConvBlock(128, 256, nb_tasks)
        self.conv_block4 = ConvBlock(256, 512, nb_tasks)
        self.conv_block5 = ConvBlock(512, 1024, nb_tasks)
        self.conv_block6 = ConvBlock(1024, 2048, nb_tasks)
        self.fc = nn.Linear(2048, NUM_CLASSES)


# ── per-domain additive LoRA adapter ─────────────────────────────────────────

class CorrAdditiveLoRABlock(nn.Module):
    """ΔW = gate · (s_A·(B@A) + s_B·(Bs@As)) injected additively per conv."""

    def __init__(self, in_ch, out_ch, r_A, r_B, alpha, gate_cfg=None):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.scale_A = alpha / r_A
        self.scale_B = alpha / r_B
        self.use_gate = bool(gate_cfg and gate_cfg.get('use', False))
        if self.use_gate:
            self.gate_theta = nn.Parameter(
                torch.tensor(float(gate_cfg.get('init_logit', 4.0))))
        self.A1 = nn.Parameter(torch.randn(r_A, in_ch * 9) * 0.02)
        self.A2 = nn.Parameter(torch.randn(r_A, out_ch * 9) * 0.02)
        self.B1 = nn.Parameter(torch.zeros(out_ch, r_A))
        self.B2 = nn.Parameter(torch.zeros(out_ch, r_A))
        self.As1 = nn.Parameter(torch.randn(r_B, in_ch * 9) * 0.02)
        self.As2 = nn.Parameter(torch.randn(r_B, out_ch * 9) * 0.02)
        self.Bs = nn.Parameter(torch.zeros(out_ch, r_B))

    def _gate(self):
        return torch.sigmoid(self.gate_theta) if self.use_gate else 1.0

    def get_delta_conv1(self):
        dW = (self.scale_A * (self.B1 @ self.A1)
              + self.scale_B * (self.Bs @ self.As1)).view(self.out_ch, self.in_ch, 3, 3)
        return self._gate() * dW if self.use_gate else dW

    def get_delta_conv2(self):
        dW = (self.scale_A * (self.B2 @ self.A2)
              + self.scale_B * (self.Bs @ self.As2)).view(self.out_ch, self.out_ch, 3, 3)
        return self._gate() * dW if self.use_gate else dW


class LoRACNN14(nn.Module):
    """Frozen CNN14 + per-domain additive LoRA. Domain-agnostic inference:
    adapters of the first `active_domain_count` domains are always summed."""

    def __init__(self, backbone, domain_block_cfgs, alpha=64.0, gate_cfg=None):
        super().__init__()
        self.backbone = backbone
        self.domain_loras = nn.ModuleList()
        for block_cfgs in domain_block_cfgs:
            lora_set = nn.ModuleList()
            for (in_ch, out_ch), (r_A, r_B) in zip(BLOCK_CONFIGS, block_cfgs):
                lora_set.append(CorrAdditiveLoRABlock(in_ch, out_ch, r_A, r_B,
                                                      alpha, gate_cfg=gate_cfg))
            self.domain_loras.append(lora_set)
        self.active_domain_count = 1

    @property
    def _conv_blocks(self):
        b = self.backbone
        return [b.conv_block1, b.conv_block2, b.conv_block3,
                b.conv_block4, b.conv_block5, b.conv_block6]

    def _apply_lora(self, x_in, base_out, block_idx, conv_in_block):
        out = base_out
        for d in range(self.active_domain_count):
            cb = self.domain_loras[d][block_idx]
            delta = cb.get_delta_conv1() if conv_in_block == 0 else cb.get_delta_conv2()
            out = out + F.conv2d(x_in, delta, padding=1)
        return out

    def forward(self, x):
        b = self.backbone
        x = b.spectrogram_extractor(x)
        x = b.logmel_extractor(x)
        x = x.transpose(1, 3); x = b.bn0[0](x); x = x.transpose(1, 3)
        for i, block in enumerate(self._conv_blocks):
            x1 = x
            x = self._apply_lora(x1, block.conv1(x1), i, 0)
            x = F.relu_(block.bnF[0](x))
            x2 = x
            x = self._apply_lora(x2, block.conv2(x2), i, 1)
            x = F.relu_(block.bnS[0](x))
            x = F.avg_pool2d(x, (2, 2))
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2); x2 = torch.mean(x, dim=2)
        return b.fc(x1 + x2)


# ── ensemble system (= the submitted model) ──────────────────────────────────

def _chunk_waveform(y):
    """Variable-length 1-D waveform tensor → [n_chunks, 4 s] (crop_4s policy)."""
    chunk_len = int(round(CHUNK_SECONDS * SAMPLE_RATE))
    min_len = int(round(CHUNK_MIN_SECONDS * SAMPLE_RATE))
    chunks = []
    for start in range(0, y.shape[0], chunk_len):
        seg = y[start:start + chunk_len]
        if seg.shape[0] < min_len and chunks:
            continue
        if seg.shape[0] < chunk_len:
            seg = F.pad(seg, (0, chunk_len - seg.shape[0]))
        chunks.append(seg)
    return torch.stack(chunks)


class EnsembleSystem(nn.Module):
    """Soft-voting ensemble: mean softmax over members × 4-s chunks × TTA gains.

    forward(waveform) -> log-probabilities [batch, NUM_CLASSES].
    waveform: float tensor [num_samples] or [batch, num_samples], mono 32 kHz.
    """

    def __init__(self, bundle):
        super().__init__()
        block_cfgs = [drift_to_block_configs(d['drift_conv_bn'], d['drift_bn'],
                                             d['ceiling']) for d in DOMAINS]
        backbone = CNN14Backbone(nb_tasks=3)
        backbone.load_state_dict(bundle['backbone'])
        self.members = nn.ModuleList()
        self.member_names = []
        for m in bundle['models']:
            net = LoRACNN14(backbone, block_cfgs, alpha=LORA_ALPHA, gate_cfg=GATE_CFG)
            net.domain_loras.load_state_dict(m['loras'])
            net.active_domain_count = int(bundle['active_domain_count'])
            self.members.append(net)
            self.member_names.append(m['name'])
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        self.eval()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        out = []
        for clip in x:
            chunks = _chunk_waveform(clip)
            p = None
            for net in self.members:
                for g in TTA_GAINS:
                    sp = F.softmax(net(chunks * g), dim=-1).mean(0)
                    p = sp if p is None else p + sp
            out.append(p / (len(self.members) * len(TTA_GAINS)))
        return torch.log(torch.stack(out).clamp_min(1e-12))


def load_model(task: int):
    """Official entry point. task=2 → Step-2 system, task=3 → Step-3 (final) system."""
    assert task in (2, 3), f'task must be 2 or 3, got {task}'
    here = os.path.dirname(os.path.abspath(__file__))
    fname = f'Chun_Chosun_task7_4_D{task}_dictionary.pth'
    # official package: next to this file / dev tree: ../result (bundle_official.py output)
    for d in (here, os.path.join(os.path.dirname(here), 'result')):
        path = os.path.join(d, fname)
        if os.path.exists(path):
            bundle = torch.load(path, map_location='cpu', weights_only=False)
            return EnsembleSystem(bundle)
    raise FileNotFoundError(f'{fname} not found — generate with bundle_official.py')


if __name__ == '__main__':
    # smoke test: load both systems and run a dummy clip
    for t in (2, 3):
        m = load_model(t)
        y = torch.randn(SAMPLE_RATE * 7) * 0.01
        lp = m(y)
        print(f'task={t}: {len(m.members)} members, out={tuple(lp.shape)}, '
              f'pred={int(lp.argmax(-1))}')
