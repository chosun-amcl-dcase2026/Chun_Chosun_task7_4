"""
verify_official.py — 공식 패키지(model.py + 단일 dictionary pth) 동등성 검증

기존 파이프라인(evaluate.load_model + collect_probs, flat dictionary)과
공식 EnsembleSystem(load_model(task))이 같은 클립에서 같은 확률을 내는지 비교.

사용: CUDA_VISIBLE_DEVICES=2 python verify_official.py [--n-clips 10]
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_MODEL)
sys.path.insert(0, os.path.join(_MODEL, 'common'))
sys.path.insert(0, _HERE)   # 공식 model.py (script/ 에 보관)

os.environ['CONFIG'] = os.path.join(_MODEL, 'config.yaml')

import numpy as np
import torch

import config as c
from evaluate import (collect_probs, load_model, block_cfgs_from_config,
                      discover_d3_entries, auto_n_by_val, load_wav)

import importlib
official = importlib.import_module('Chun_Chosun_task7_4_model')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-clips', type=int, default=10)
    ap.add_argument('--task', type=int, default=3)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df = c.DF_TEST.reset_index(drop=True)
    paths = df['full_path'].tolist()[:args.n_clips]

    # ── 기존 파이프라인 ────────────────────────────────────────────────────────
    bc = block_cfgs_from_config()
    tta = [1.0, 0.85, 1.15]
    if args.task == 3:
        entries = discover_d3_entries(os.path.join(_ROOT, 'Chun_Chosun_task7_4_D3_dictionary'))
        n, _ = auto_n_by_val([e['val_macro'] for e in entries])
        ckpts, count = [e['path'] for e in entries[:n]], 2
    else:
        from evaluate import discover_d2_ckpts
        ckpts = list(discover_d2_ckpts(os.path.join(_ROOT, 'Chun_Chosun_task7_4_D2_dictionary')).values())
        count = 1
    probs_ref = None
    for p in ckpts:
        m = load_model(p, bc, count, device)
        pr = collect_probs(m, paths, device, tta)
        probs_ref = pr if probs_ref is None else probs_ref + pr
        del m; torch.cuda.empty_cache()
    probs_ref /= len(ckpts)
    print(f'[기존] {len(ckpts)}모델 앙상블, {len(paths)} clips')

    # ── 공식 패키지 ───────────────────────────────────────────────────────────
    sys_off = official.load_model(args.task).to(device)
    probs_off = []
    for p in paths:
        y = torch.from_numpy(load_wav(p)).to(device)
        lp = sys_off(y)
        probs_off.append(torch.exp(lp)[0].cpu().numpy())
    probs_off = np.stack(probs_off)
    print(f'[공식] {len(sys_off.members)}멤버 EnsembleSystem')

    diff = np.abs(probs_ref - probs_off).max()
    agree = (probs_ref.argmax(1) == probs_off.argmax(1)).mean()
    print(f'max|Δprob| = {diff:.2e},  argmax 일치율 = {agree*100:.1f}%')
    assert diff < 1e-4 and agree == 1.0, 'FAIL: 동등성 불일치'
    print('PASS: 공식 패키지 = 기존 파이프라인 동등')


if __name__ == '__main__':
    main()
