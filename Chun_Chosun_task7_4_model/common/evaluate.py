"""
evaluate.py — 앙상블 추론·메트릭 공용 모듈 (exp16 eval_ensemble.py 에서 추출)

script/main.py(학습 후 앙상블)·inference.py(dictionary 추론)가 공유한다.
  - collect_probs       : 모델 1개 → 클립별 crop_4s + TTA softmax 확률행렬
  - macro_micro / per_class_metrics : macro(공식)·micro·클래스별 accuracy
  - load_model          : 체크포인트 → LoRACNN14 (count 지정, 도메인 라벨 미사용)
  - auto_n_by_val       : val_macro ≥ 평균 인 모델 수 = 제출 N (test 미참조, leakage-free)
  - write_predictions_csv : path,label 양식 (파일명만·소문자 클래스명)
"""

import os
import glob
import json

import numpy as np
import torch
import torch.nn.functional as F
import librosa

import config as c
from backbone import CNN14Backbone
from model_adaptor import LoRACNN14, drift_to_block_configs


# ── crop_4s 청킹 + 디코드 캐시 ────────────────────────────────────────────────

_WAV_CACHE = {}


def load_wav(path):
    y = _WAV_CACHE.get(path)
    if y is None:
        y, _ = librosa.load(path, sr=c.SAMPLE_RATE, mono=True)
        y = y.astype(np.float32); _WAV_CACHE[path] = y
    return y


def chunk_audio(y):
    chunk_len = int(round(c.CHUNK_SECONDS * c.SAMPLE_RATE))
    min_len   = int(round(c.CHUNK_MIN_SECONDS * c.SAMPLE_RATE))
    chunks = []
    for start in range(0, len(y), chunk_len):
        seg = y[start:start + chunk_len]
        if len(seg) < min_len: continue
        if len(seg) < chunk_len:
            seg = np.concatenate([seg, np.zeros(chunk_len - len(seg), dtype=np.float32)])
        chunks.append(seg.astype(np.float32))
    if not chunks:
        seg = np.concatenate([y, np.zeros(max(0, chunk_len - len(y)), dtype=np.float32)])[:chunk_len]
        chunks.append(seg)
    return chunks


# ── 모델 로드 (추론 전용 — count 로만 제어, 도메인 라벨 미사용) ────────────────

def block_cfgs_from_config():
    _dom = c.CFG.get('domains', []) or []
    _def = c.CFG.get('domain_defaults', {}) or {}
    out = []
    for d in _dom:
        cfgs, _, _ = drift_to_block_configs(d['drift_conv_bn'], d['drift_bn'],
                                            d.get('ceiling', _def.get('ceiling')))
        out.append(cfgs)
    return out


def load_model(ckpt_path, block_cfgs, count, device):
    backbone = CNN14Backbone(nb_tasks=3)
    backbone.load_d1_checkpoint(c.D1_CKPT); backbone.to(device)
    model = LoRACNN14(backbone, domain_block_cfgs=block_cfgs,
                      alpha=c.LORA_ALPHA, tune_bn=bool(c.CFG['lora'].get('tune_bn', True)),
                      gate_cfg=c.CFG.get('gate', {}) or {}).to(device)
    model.active_domain_count = count
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'] if isinstance(ck, dict) and 'model' in ck else ck)
    model.eval()
    return model


# ── 확률 산출 / 메트릭 ────────────────────────────────────────────────────────

@torch.no_grad()
def collect_probs(model, paths, device, tta_gains):
    """클립 경로 리스트 → 확률행렬 [n_clips, NUM_CLASSES] (crop_4s 청크 평균 × TTA gain 평균).
    paths 순서 고정 → 모델 간 행 정렬 일치(soft voting 가능)."""
    model.eval()
    probs = []
    for path in paths:
        chunks = chunk_audio(load_wav(path))
        xb = torch.from_numpy(np.stack(chunks)).float().to(device)
        p = None
        for g in tta_gains:
            logits, _ = model(xb * g)
            sp = F.softmax(logits, -1).mean(0)
            p = sp if p is None else p + sp
        probs.append((p / len(tta_gains)).cpu().numpy())
    return np.stack(probs)


def macro_micro(probs, labels):
    preds = probs.argmax(1)
    micro = float((preds == labels).mean() * 100)
    macro = float(np.mean([(preds[labels == cl] == cl).mean()
                           for cl in np.unique(labels)]) * 100)
    return micro, macro


def per_class_metrics(probs, labels):
    """클래스별 accuracy dict (macro accuracy 기준 보고용 — meta.yaml 채움에 사용)."""
    preds = probs.argmax(1)
    inv = {v: k for k, v in c.CLASS_LABELS.items()}
    out = {}
    for cl in sorted(np.unique(labels).tolist()):
        mask = labels == cl
        nc, nt = int((preds[mask] == cl).sum()), int(mask.sum())
        out[inv.get(cl, str(cl))] = {
            'correct': nc, 'total': nt,
            'acc': round(nc / nt * 100, 2) if nt else 0.0}
    return out


# ── 제출 N 선정 (test 미참조, leakage-free) ───────────────────────────────────

def auto_n_by_val(val_macros):
    """N = #{val_macro ≥ 평균}. test 결과를 전혀 보지 않고 N 결정."""
    if not val_macros:
        return 0
    mean_vm = float(np.mean(val_macros))
    return sum(1 for v in val_macros if v >= mean_vm), mean_vm


def discover_d3_entries(d3_root):
    """D3 체크포인트 → [{path, name, parent, val_macro}] (val_macro 내림차순).

    두 레이아웃 지원:
      flat (dictionary, _1/_2 컨벤션): d3_d2f{k}_d3f{j}.pth — 메타는 pth 내장 키 사용
      nested (학습 출력 checkpoint/) : d3_f{k}_{j}/d3.pth (+ meta.json)
    """
    entries = []
    flat = sorted(glob.glob(os.path.join(d3_root, 'd3_d2f*_d3f*.pth')))
    nested = sorted(glob.glob(os.path.join(d3_root, 'd3_f*_*', 'd3.pth')))
    for p in flat + nested:
        d = os.path.dirname(p)
        mp = os.path.join(d, 'meta.json')
        if p in nested and os.path.exists(mp):
            meta = json.load(open(mp))
            vm, parent = float(meta['val_macro']), int(meta['d2_fold'])
        else:
            ck = torch.load(p, map_location='cpu', weights_only=False)
            vm, parent = float(ck.get('val_macro', -1)), int(ck.get('d2_fold', -1))
            del ck
        name = (os.path.splitext(os.path.basename(p))[0] if p in flat
                else os.path.basename(d))
        entries.append({'path': p, 'name': name, 'parent': parent, 'val_macro': vm})
    entries.sort(key=lambda e: e['val_macro'], reverse=True)
    return entries


def discover_d2_ckpts(d2_root):
    """D2 체크포인트 → {fold: ckpt_path}. flat(d2_fold{k}.pth) / nested(d2_f{k}/d2.pth) 모두 지원."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d2_root, 'd2_fold*.pth'))):
        k = int(os.path.splitext(os.path.basename(p))[0].replace('d2_fold', ''))
        out[k] = p
    for p in sorted(glob.glob(os.path.join(d2_root, 'd2_f*', 'd2.pth'))):
        k = int(os.path.basename(os.path.dirname(p)).replace('d2_f', ''))
        out.setdefault(k, p)
    return out


# ── 제출 CSV ──────────────────────────────────────────────────────────────────

def write_predictions_csv(out_path, paths, probs):
    """path,label 양식. path=파일명만, label=소문자 클래스명."""
    inv = {v: k for k, v in c.CLASS_LABELS.items()}
    preds = probs.argmax(1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('path,label\n')
        for p, pr in zip(paths, preds):
            f.write(f'{os.path.basename(p)},{inv[int(pr)].lower()}\n')
