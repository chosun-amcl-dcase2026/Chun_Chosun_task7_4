"""
main.py — 통합 파이프라인 (end-to-end)
  D1 체크포인트 로드 → D2 5-fold 학습 → D3 5×5 학습 → val_macro auto-N 앙상블 추론
  → metric.json / out.csv 기록 → D2_dictionary / D3_dictionary 에 체크포인트 복사.

경로 (script/ 기준 상대 — 하드코딩 절대경로 없음):
  체크포인트: ../checkpoint/ensemble/      로그: ../result/ensemble_log.txt
  결과      : ../result/{metric.json, out.csv}
  사전      : ../../D2_dictionary/, ../../D3_dictionary/  (복사로만 채움 — 원본 유지)

사용법:
  python main.py                                  # 전체 (D2 5 + D3 25 + 앙상블 + 복사)
  python main.py --d2-folds 0 1 2                 # 분할 실행 (멀티 GPU 레인)
  python main.py --d3-jobs 0:0 0:1 1:3            # D3 조합만
  python main.py --skip-train                     # 학습 생략, 앙상블+복사만 (완료분 재평가)
선정: 제출 N 은 val_macro ≥ 평균 (test 미참조, leakage-free). Top-N 비교표는 분석용으로만 기록.
"""

import argparse
import os
import shutil
import sys
import json
from collections import Counter

_HERE   = os.path.dirname(os.path.abspath(__file__))      # model/script
_MODEL  = os.path.dirname(_HERE)                           # model/
_ROOT   = os.path.dirname(_MODEL)                          # experiment root
_COMMON = os.path.join(_MODEL, 'common')
sys.path.insert(0, _COMMON)

CKPT_DIR   = os.path.join(_MODEL, 'checkpoint', 'ensemble')
RESULT_DIR = os.path.join(_MODEL, 'result')
D2_DICT    = os.path.join(_ROOT, 'Chun_Chosun_task7_4_D2_dictionary')
D3_DICT    = os.path.join(_ROOT, 'Chun_Chosun_task7_4_D3_dictionary')


def parse_args():
    ap = argparse.ArgumentParser(description='exp17 통합 파이프라인 (D2→D3 k-fold→앙상블)')
    ap.add_argument('--config', default=os.path.join(_MODEL, 'config.yaml'),
                    help='config.yaml 경로 (기본 ../config.yaml)')
    ap.add_argument('--d2-folds', type=int, nargs='+', default=None,
                    help='담당 D2 fold (기본 0..K-1 전체)')
    ap.add_argument('--d3-jobs', type=str, nargs='+', default=None,
                    help='담당 D3 조합 k:j (기본 전체 K×K)')
    ap.add_argument('--skip-train', action='store_true',
                    help='학습 생략 — 기존 체크포인트로 앙상블+사전복사만')
    ap.add_argument('--skip-ensemble', action='store_true',
                    help='앙상블 생략 — 학습만 (멀티 레인 분할 실행 시 마지막 레인만 앙상블)')
    return ap.parse_args()


def main():
    args = parse_args()
    os.environ['CONFIG'] = os.path.abspath(args.config)

    import torch
    import numpy as np
    import config as c
    import ddp_util as ddp
    from pipeline import (setup_determinism, require_cuda, build_domain_spec, make_prior_kd,
                          collect_kfg, kfold_cfg, run_phase_d2, run_phase_d3)
    from train_loop import make_log
    from evaluate import (collect_probs, macro_micro, per_class_metrics, load_model,
                          block_cfgs_from_config, auto_n_by_val, discover_d3_entries,
                          discover_d2_ckpts, write_predictions_csv)

    rank, local_rank, world_size, distributed, device = ddp.setup_distributed()
    if distributed:
        raise RuntimeError('exp17 은 비-DDP 설계입니다. torchrun 없이 python main.py 로 실행하세요.')
    require_cuda(device)
    det = setup_determinism()

    os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(RESULT_DIR, exist_ok=True)
    log = make_log(os.path.join(RESULT_DIR, 'ensemble_log.txt'))
    log(f'\n{"#"*70}\n# exp17 main.py  device={device}  deterministic={det}  '
        f'batch={c.BATCH_SIZE}\n# ckpt={CKPT_DIR}\n{"#"*70}')

    specs, block_cfgs, rank_meta = build_domain_spec(log)
    kfg = collect_kfg()
    prior_kd, kd_from_drift = make_prior_kd(specs)
    kf = kfold_cfg()
    log(f'  CoForge: cf_use={kfg["cf_use"]} beta={kfg["cf_beta"]} | KeepLoRA={kfg["kl_use"]} '
        f'| kd_from_drift={kd_from_drift} | K={kf["K"]}')

    d2_folds = args.d2_folds if args.d2_folds is not None else list(range(kf['K']))
    if args.d3_jobs is not None:
        d3_jobs = [tuple(int(x) for x in t.split(':')) for t in args.d3_jobs]
    else:
        d3_jobs = [(k, j) for k in range(kf['K']) for j in range(kf['K'])]

    # ── Phase d2 → Phase d3 (skip-if-exists → 재실행 안전) ──────────────────
    if not args.skip_train:
        log(f'\n[train] D2 folds={d2_folds}')
        run_phase_d2(d2_folds, CKPT_DIR, specs, block_cfgs, rank_meta, kfg, prior_kd, kf, device, log)
        log(f'\n[train] D3 jobs({len(d3_jobs)})={d3_jobs}')
        run_phase_d3(d3_jobs, CKPT_DIR, specs, block_cfgs, rank_meta, kfg, prior_kd, kf, device, log)
    else:
        log('\n[train] --skip-train: 학습 생략')

    if args.skip_ensemble:
        log('[ensemble] --skip-ensemble: 종료 (마지막 레인에서 앙상블 실행)')
        return

    # ══ 앙상블 추론 (val_macro auto-N — test 미참조 선정) ══════════════════════
    bc = block_cfgs_from_config()
    _tta = c.CFG.get('tta', {}) or {}
    tta_gains = list(_tta.get('gains', [1.0, 0.85, 1.15])) if _tta.get('use', False) else [1.0]
    entries = discover_d3_entries(CKPT_DIR)
    d2_ckpts = discover_d2_ckpts(CKPT_DIR)
    if not entries:
        raise FileNotFoundError(f'D3 체크포인트 없음: {CKPT_DIR}/d3_f*_*/d3.pth')
    log(f'\n[ensemble] D3 {len(entries)}개 / D2 {len(d2_ckpts)}개  TTA={tta_gains}')

    _auto_n, mean_vm = auto_n_by_val([e['val_macro'] for e in entries])
    sub_n_cfg = kf.get('submission_n', None)
    if sub_n_cfg is not None:
        n_sub = min(int(sub_n_cfg), len(entries))
        log(f'  제출 N = {n_sub} (config kfold.submission_n 고정 — val_macro 상위 N; test 미참조) '
            f'[참고: auto-N={_auto_n}, mean={mean_vm:.2f}]')
    else:
        n_sub = _auto_n
        log(f'  제출 N = {n_sub} (val_macro ≥ 평균 {mean_vm:.2f} auto; test 미참조)')

    df_d2_te = c.DF_TEST[c.DF_TEST['domain'] == 'D2'].reset_index(drop=True)
    df_d3_te = c.DF_TEST[c.DF_TEST['domain'] == 'D3'].reset_index(drop=True)
    paths_d2 = df_d2_te['full_path'].tolist(); lab_d2 = df_d2_te['new_target'].values
    paths_d3 = df_d3_te['full_path'].tolist(); lab_d3 = df_d3_te['new_target'].values

    # Step2: D2 모델(count=1) — fold별 확률 보관 (매칭 forgetting + Step2 보고)
    d2_probs_by_fold = {}
    for k, p in d2_ckpts.items():
        m = load_model(p, bc, 1, device)
        d2_probs_by_fold[k] = collect_probs(m, paths_d2, device, tta_gains)
        del m; torch.cuda.empty_cache()
    step2_probs = np.mean(list(d2_probs_by_fold.values()), axis=0)
    s2_mi, s2_ma = macro_micro(step2_probs, lab_d2)
    log(f'  Step2 D2@D2 (D2 {len(d2_probs_by_fold)}개 앙상블): micro {s2_mi:.2f} / macro {s2_ma:.2f}')

    # Step3: D3 모델(count=2) — 전 모델 확률 캐시 → Top-N 비교표 + 제출 N
    for e in entries:
        m = load_model(e['path'], bc, 2, device)
        e['p_d3'] = collect_probs(m, paths_d3, device, tta_gains)
        e['p_d2'] = collect_probs(m, paths_d2, device, tta_gains)
        del m; torch.cuda.empty_cache()
        log(f'    {e["name"]}  val_macro={e["val_macro"]:.2f}  (probs 캐시)')

    def topn_metrics(n):
        sel = entries[:n]
        p_d3 = np.mean([e['p_d3'] for e in sel], axis=0)
        p_d2 = np.mean([e['p_d2'] for e in sel], axis=0)
        par  = np.mean([d2_probs_by_fold[e['parent']] for e in sel
                        if e['parent'] in d2_probs_by_fold], axis=0)   # 매칭 D2@D2
        d3d3 = macro_micro(p_d3, lab_d3); d2d3 = macro_micro(p_d2, lab_d2)
        d2d2 = macro_micro(par, lab_d2)
        return {'N': n,
                'D2@D2': {'micro': d2d2[0], 'macro': d2d2[1]},
                'D2@D3': {'micro': d2d3[0], 'macro': d2d3[1]},
                'D3@D3': {'micro': d3d3[0], 'macro': d3d3[1]},
                'Acc': {'micro': (d2d3[0]+d3d3[0])/2, 'macro': (d2d3[1]+d3d3[1])/2},
                'Fr':  {'micro': d2d2[0]-d2d3[0],     'macro': d2d2[1]-d2d3[1]},
                'parents': dict(Counter(e['parent'] for e in sel)),
                'probs': (p_d3, p_d2)}

    log(f'\n  {"N":>3} | {"D2@D2":>7} {"D2@D3":>7} {"D3@D3":>7} | {"Acc":>7} {"Fr":>7}  (macro, D2@D2=매칭)')
    table = []
    for N in sorted(set(kf['topn'] + [n_sub])):
        if N > len(entries): continue
        r = topn_metrics(N)
        mark = ' ★제출N' if N == n_sub else ''
        log(f'  {N:>3} | {r["D2@D2"]["macro"]:>7.2f} {r["D2@D3"]["macro"]:>7.2f} '
            f'{r["D3@D3"]["macro"]:>7.2f} | {r["Acc"]["macro"]:>7.2f} {r["Fr"]["macro"]:>+7.2f}{mark}')
        table.append({k: v for k, v in r.items() if k != 'probs'})

    sub = topn_metrics(n_sub)
    p_d3_sub, p_d2_sub = sub['probs']

    # ── metric.json (macro 기준 + meta.yaml 채움용 per-class) ──────────────────
    metric = {
        'method': 'exp17 CoForge K-Fold (5x5) ensemble',
        'selection': f'val_macro auto-N = {n_sub} (mean={round(mean_vm,2)}, test 미참조)',
        'eval': 'crop_4s + TTA', 'tta_gains': tta_gains, 'deterministic': det,
        'n_models': {'d2': len(d2_ckpts), 'd3': len(entries)},
        'submission_N': n_sub,
        'Step2': {'Domain2': {'micro': round(s2_mi, 2), 'macro': round(s2_ma, 2),
                              'per_class': per_class_metrics(step2_probs, lab_d2)}},
        'Step3': {'Domain2': {'micro': round(sub['D2@D3']['micro'], 2),
                              'macro': round(sub['D2@D3']['macro'], 2),
                              'per_class': per_class_metrics(p_d2_sub, lab_d2)},
                  'Domain3': {'micro': round(sub['D3@D3']['micro'], 2),
                              'macro': round(sub['D3@D3']['macro'], 2),
                              'per_class': per_class_metrics(p_d3_sub, lab_d3)}},
        'Acc_macro': round(sub['Acc']['macro'], 2), 'Fr_macro': round(sub['Fr']['macro'], 2),
        'D2@D2_matched_macro': round(sub['D2@D2']['macro'], 2),
        'topn_table_analysis_only': table,
        'val_macro_ranking': [{'name': e['name'], 'val_macro': e['val_macro']} for e in entries],
    }
    with open(os.path.join(RESULT_DIR, 'metric.json'), 'w') as f:
        json.dump(metric, f, indent=2, ensure_ascii=False)
    log(f'\n  metric.json 저장 → {os.path.join(RESULT_DIR, "metric.json")}')
    log(f'  ★ 제출 N={n_sub}: Acc {sub["Acc"]["macro"]:.2f} / Fr {sub["Fr"]["macro"]:+.2f} '
        f'/ D3@D3 {sub["D3@D3"]["macro"]:.2f}')

    # ── out.csv (최종 시스템 = 제출 N 앙상블, dev test 전 클립) ────────────────
    all_paths = paths_d2 + paths_d3
    all_probs = np.concatenate([p_d2_sub, p_d3_sub], axis=0)
    write_predictions_csv(os.path.join(RESULT_DIR, 'out.csv'), all_paths, all_probs)
    log(f'  out.csv 저장 → {os.path.join(RESULT_DIR, "out.csv")} ({len(all_paths)} clips)')

    # ── dictionary 복사 (이동 금지 — 원본은 checkpoint/ 에 유지) ───────────────
    #   _1/_2 컨벤션과 동일한 flat 구조: d2_fold{k}.pth / d3_d2f{k}_d3f{j}.pth
    #   메타(val_macro 등)는 pth 내장 키로 충분 — meta.json 미복사.
    os.makedirs(D2_DICT, exist_ok=True); os.makedirs(D3_DICT, exist_ok=True)
    for k, p in d2_ckpts.items():
        shutil.copy2(p, os.path.join(D2_DICT, f'd2_fold{k}.pth'))
    for e in entries:
        ck = torch.load(e['path'], map_location='cpu', weights_only=False)
        d2f, d3f = int(ck.get('d2_fold', -1)), int(ck.get('d3_fold', -1))
        del ck
        shutil.copy2(e['path'], os.path.join(D3_DICT, f'd3_d2f{d2f}_d3f{d3f}.pth'))
    log(f'  dictionary 복사 완료 → {D2_DICT} ({len(d2_ckpts)}), {D3_DICT} ({len(entries)})')
    log('[done] main.py 종료')


if __name__ == '__main__':
    main()
