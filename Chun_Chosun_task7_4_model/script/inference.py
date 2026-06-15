"""
inference.py — dictionary 체크포인트 로드 → 추론 전용 (학습 없음)

소스: ../../D2_dictionary/ (step2, count=1), ../../D3_dictionary/ (step3, count=2)
출력: ../result/out.csv  (+ 루트 ../../Chun_Chosun_task7_4_output.csv 동기 기록. --single 일 때만 _output_single.csv)
로그: ../result/log.txt

앙상블: D3 dictionary pth 내장 val_macro 로 auto-N(≥평균) 선정 — test 미참조(leakage-free).
        --n 으로 수동 지정 가능. soft voting (softmax 확률 평균 → argmax) + crop_4s + TTA.

출력 포맷 (--format):
  csv (기본) : `path,label` 헤더 포함 — 작업지시서 양식
  tsv        : `filename\tpredicted_class` 헤더 없음 — DCASE 공식 평가 양식
               (https://dcase.community/challenge2026/ Task7: single TSV, no header)

사용법:
  python inference.py                          # step3, dev test 클립, auto-N
  python inference.py --step 2                 # step2 (D2 모델, D2 test)
  python inference.py --audio-dir /path/wavs   # 평가셋 등 임의 폴더 추론 (라벨 없음)
  python inference.py --n 10 --format tsv
"""

import argparse
import glob
import os
import sys
import json

_HERE   = os.path.dirname(os.path.abspath(__file__))
_MODEL  = os.path.dirname(_HERE)
_ROOT   = os.path.dirname(_MODEL)
_COMMON = os.path.join(_MODEL, 'common')
sys.path.insert(0, _COMMON)

RESULT_DIR = os.path.join(_MODEL, 'result')
D2_DICT    = os.path.join(_ROOT, 'Chun_Chosun_task7_4_D2_dictionary')
D3_DICT    = os.path.join(_ROOT, 'Chun_Chosun_task7_4_D3_dictionary')


def write_csv(out_path, paths, probs, inv, fmt):
    preds = probs.argmax(1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        if fmt == 'csv':
            f.write('path,label\n')
            for p, pr in zip(paths, preds):
                f.write(f'{os.path.basename(p)},{inv[int(pr)].lower()}\n')
        else:  # tsv — DCASE 공식 (헤더 없음, 탭 구분)
            for p, pr in zip(paths, preds):
                f.write(f'{os.path.basename(p)}\t{inv[int(pr)].lower()}\n')


def main():
    ap = argparse.ArgumentParser(description='exp17 dictionary 추론 전용')
    ap.add_argument('--config', default=os.path.join(_MODEL, 'config.yaml'))
    ap.add_argument('--step', type=int, choices=[2, 3], default=3,
                    help='2=D2_dictionary(count=1), 3=D3_dictionary(count=2, 최종 시스템)')
    ap.add_argument('--audio-dir', default=None,
                    help='추론할 wav 폴더 (미지정 시 dev test 클립 — 라벨 있으면 정확도도 로그)')
    ap.add_argument('--n', type=int, default=None,
                    help='앙상블 모델 수 (기본: config kfold.submission_n=20 고정; 그것이 null이면 val_macro≥평균 auto-N). test 미참조')
    ap.add_argument('--format', choices=['csv', 'tsv'], default='csv',
                    help='csv=path,label(지시서) | tsv=filename\\tclass 무헤더(DCASE 공식)')
    ap.add_argument('--out', default=None, help='출력 경로 (기본 ../result/out.csv)')
    ap.add_argument('--single', action='store_true',
                    help='앙상블 대신 단일 모델(checkpoint/single/d3.pth, count=2) 추론')
    args = ap.parse_args()
    os.environ['CONFIG'] = os.path.abspath(args.config)

    import numpy as np
    import torch
    import config as c
    from pipeline import require_cuda
    from evaluate import (collect_probs, macro_micro, per_class_metrics, load_model,
                          block_cfgs_from_config, auto_n_by_val,
                          discover_d3_entries, discover_d2_ckpts)
    from train_loop import make_log

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    require_cuda(device)
    os.makedirs(RESULT_DIR, exist_ok=True)
    log = make_log(os.path.join(RESULT_DIR, 'log.txt'))
    log(f'\n[inference] step={args.step} format={args.format} dict={"D2" if args.step==2 else "D3"}_dictionary')

    bc = block_cfgs_from_config()
    _tta = c.CFG.get('tta', {}) or {}
    tta_gains = list(_tta.get('gains', [1.0, 0.85, 1.15])) if _tta.get('use', False) else [1.0]
    inv = {v: k for k, v in c.CLASS_LABELS.items()}

    # ── 추론 대상 클립 ─────────────────────────────────────────────────────────
    labels = None
    if args.audio_dir:
        paths = sorted(glob.glob(os.path.join(args.audio_dir, '*.wav')))
        if not paths:
            raise FileNotFoundError(f'wav 없음: {args.audio_dir}')
        log(f'  대상: {args.audio_dir} ({len(paths)} clips, 라벨 없음)')
    else:
        if args.step == 2:
            df = c.DF_TEST[c.DF_TEST['domain'] == 'D2'].reset_index(drop=True)
        else:
            df = c.DF_TEST.reset_index(drop=True)          # 최종 시스템: D2+D3 test 전체
        paths = df['full_path'].tolist()
        labels = df['new_target'].values
        log(f'  대상: dev test ({len(paths)} clips, 라벨 있음 → 정확도 로그)')

    # ── dictionary 에서 모델 선택 (auto-N, test 미참조) ────────────────────────
    if args.single:
        sp = os.path.join(_MODEL, 'checkpoint', 'single', 'd3.pth')
        if not os.path.exists(sp):
            raise FileNotFoundError(f'단일 모델 없음: {sp} (single_main.py 가 생성)')
        ckpts, count, sel_names = [sp], 2, ['single_d3']
        log('  단일 모델 (checkpoint/single/d3.pth, 앙상블 없음)')
    elif args.step == 2:
        ckpts = list(discover_d2_ckpts(D2_DICT).values())
        if not ckpts:
            raise FileNotFoundError(f'D2_dictionary 비어있음: {D2_DICT} (main.py 가 복사로 채움)')
        count = 1
        sel_names = [os.path.splitext(os.path.basename(p))[0] for p in ckpts]
        log(f'  D2 모델 {len(ckpts)}개 전체 앙상블')
    else:
        entries = discover_d3_entries(D3_DICT)
        if not entries:
            raise FileNotFoundError(f'D3_dictionary 비어있음: {D3_DICT} (main.py 가 복사로 채움)')
        sub_n = (c.CFG.get('kfold', {}) or {}).get('submission_n', None)
        if args.n is not None:
            n = min(args.n, len(entries))
            log(f'  N={n} (수동 지정)')
        elif sub_n is not None:
            n = min(int(sub_n), len(entries))
            log(f'  N={n} (config kfold.submission_n 고정 — val_macro 상위 N, test 미참조)')
        else:
            n, mean_vm = auto_n_by_val([e['val_macro'] for e in entries])
            log(f'  N={n} (val_macro ≥ 평균 {mean_vm:.2f} auto — test 미참조)')
        sel = entries[:n]
        ckpts = [e['path'] for e in sel]
        sel_names = [e['name'] for e in sel]
        count = 2

    # ── soft voting ───────────────────────────────────────────────────────────
    probs = None
    for p in ckpts:
        m = load_model(p, bc, count, device)
        pr = collect_probs(m, paths, device, tta_gains)
        probs = pr if probs is None else probs + pr
        del m
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        log(f'    + {os.path.splitext(os.path.basename(p))[0]}')
    probs /= len(ckpts)

    # ── 출력 ──────────────────────────────────────────────────────────────────
    out_path = args.out or os.path.join(RESULT_DIR, 'out.csv')
    write_csv(out_path, paths, probs, inv, args.format)
    # 루트 제출본 동기 — 기본(앙상블)은 Chun_Chosun_task7_4_output.csv (제출 단일 파일, _1/_2 동일).
    #   --single 일 때만 _output_single.csv 로 분리(제출 미포함, 분석용).
    root_name = ('Chun_Chosun_task7_4_output_single.csv' if args.single
                 else 'Chun_Chosun_task7_4_output.csv')
    write_csv(os.path.join(_ROOT, root_name), paths, probs, inv, args.format)
    log(f'  out → {out_path}  (+ 루트 {root_name} 동기, {len(paths)} clips, {args.format})')

    if labels is not None:
        mi, ma = macro_micro(probs, labels)
        log(f'  정확도(참고): micro {mi:.2f} / macro {ma:.2f}')
        for cls, v in per_class_metrics(probs, labels).items():
            log(f'    {cls:22s} {v["acc"]:6.2f}%  ({v["correct"]}/{v["total"]})')
    log(f'[done] inference.py 종료 (모델: {", ".join(sel_names)})')


if __name__ == '__main__':
    main()
