"""
make_classwise_dev.py — dev test 측정 → classwise_dev_{single|ensemble}.json

_2(Chun_Chosun_task7_2_model/results/classwise_dev.json)와 동일 형식:
  {"protocol": ..., "Step2": {"Domain2": {"average", "classwise"}},
   "Step3": {"Domain2": {...}, "Domain3": {...}}}
  - average = macro accuracy(%), classwise 키는 템플릿 축약명(dog/knock/baby/phone)
  - TTA 는 config tta.use 반영 (false → gain [1.0])

모드:
  --mode ensemble : Step2 = D2_dictionary 5-fold 전체 soft voting
                    Step3 = D3_dictionary val_macro auto-N soft voting
  --mode single   : Step2 = checkpoint/single/d2.pth (count=1)
                    Step3 = checkpoint/single/d3.pth (count=2)
  --mode ensemble_sweep : Step2 = D2_dictionary 5-fold 전체 (D2@D2 고정)
                    Step3 = D3_dictionary val_macro 내림차순 Top-N (config kfold.topn)
                    → classwise_dev_ensemble_sweep.json (N별 D2@D3/D3@D3/Acc/Fr 표)

사용: CUDA_VISIBLE_DEVICES=0 python make_classwise_dev.py --mode ensemble
"""

import argparse
import json
import os
import sys

_HERE   = os.path.dirname(os.path.abspath(__file__))
_MODEL  = os.path.dirname(_HERE)
_ROOT   = os.path.dirname(_MODEL)
_COMMON = os.path.join(_MODEL, 'common')
sys.path.insert(0, _COMMON)

RESULT_DIR  = os.path.join(_MODEL, 'result')
D2_DICT     = os.path.join(_ROOT, 'Chun_Chosun_task7_4_D2_dictionary')
D3_DICT     = os.path.join(_ROOT, 'Chun_Chosun_task7_4_D3_dictionary')
SINGLE_DIR  = os.path.join(_MODEL, 'checkpoint', 'single')

# 공식 meta.yaml 템플릿 축약명 (classwise 키)
SHORT = {'alarm': 'alarm', 'baby_cry': 'baby', 'dog_bark': 'dog', 'engine': 'engine',
         'fire': 'fire', 'footsteps': 'footsteps', 'knocking': 'knock',
         'telephone_ringing': 'phone', 'piano': 'piano', 'speech': 'speech'}


def soft_vote(ckpts, count, paths, device, tta_gains, log):
    import torch
    from evaluate import collect_probs, load_model, block_cfgs_from_config
    bc = block_cfgs_from_config()
    probs = None
    for p in ckpts:
        m = load_model(p, bc, count, device)
        pr = collect_probs(m, paths, device, tta_gains)
        probs = pr if probs is None else probs + pr
        del m
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        log(f'    + {os.path.splitext(os.path.basename(p))[0]}')
    return probs / len(ckpts)


def domain_block(probs, labels):
    """확률행렬 + 정답 → {'average': macro, 'classwise': {축약명: acc}}"""
    import numpy as np
    import config as c
    preds = probs.argmax(1)
    inv = {v: k for k, v in c.CLASS_LABELS.items()}
    cw = {}
    for cl in sorted(np.unique(labels).tolist()):
        mask = labels == cl
        cw[SHORT[inv[cl]]] = round(float((preds[mask] == cl).mean() * 100), 2)
    return {'average': round(sum(cw.values()) / len(cw), 2), 'classwise': cw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(_MODEL, 'config.yaml'))
    ap.add_argument('--mode', choices=['ensemble', 'single', 'ensemble_sweep'], required=True)
    args = ap.parse_args()
    os.environ['CONFIG'] = os.path.abspath(args.config)

    import torch
    import config as c
    from pipeline import require_cuda
    from evaluate import discover_d2_ckpts, discover_d3_entries, auto_n_by_val
    from train_loop import make_log

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    require_cuda(device)
    log = make_log(os.path.join(RESULT_DIR, 'log.txt'))

    _tta = c.CFG.get('tta', {}) or {}
    tta_gains = list(_tta.get('gains', [])) if _tta.get('use', False) else [1.0]
    tta_desc = f'TTA {tta_gains}' if len(tta_gains) > 1 else 'no TTA'

    df_d2 = c.DF_TEST[c.DF_TEST['domain'] == 'D2'].reset_index(drop=True)
    df_d3 = c.DF_TEST[c.DF_TEST['domain'] == 'D3'].reset_index(drop=True)
    paths_d2, lab_d2 = df_d2['full_path'].tolist(), df_d2['new_target'].values
    paths_d3, lab_d3 = df_d3['full_path'].tolist(), df_d3['new_target'].values

    # ── Top-N 스윕 (앙상블 크기 비교표) ────────────────────────────────────────
    if args.mode == 'ensemble_sweep':
        import numpy as np
        from evaluate import collect_probs, load_model, block_cfgs_from_config
        bc = block_cfgs_from_config()
        d2_ckpts = list(discover_d2_ckpts(D2_DICT).values())
        entries = discover_d3_entries(D3_DICT)             # val_macro 내림차순
        topns = [n for n in c.CFG['kfold']['topn'] if n <= len(entries)]
        log(f'\n[classwise_dev:ensemble_sweep] {tta_desc} | D2 {len(d2_ckpts)}모델, D3 {len(entries)}모델 '
            f'| Top-N={topns}')

        # D2@D2 = D2 5-fold 전체 앙상블 (모든 N 공통, 고정)
        log('  Step2 Domain2 (D2 전체 앙상블 — 모든 N 공통):')
        p_s2 = soft_vote(d2_ckpts, 1, paths_d2, device, tta_gains, log)
        d2d2 = domain_block(p_s2, lab_d2)

        # D3 모델별 확률 1회 캐시 (D2+D3 test) → N별 평균 재사용
        log('  Step3 (D3 모델별 확률 캐시, D2+D3 test):')
        per_model = []
        for e in entries:
            pr = soft_vote([e['path']], 2, paths_d2 + paths_d3, device, tta_gains, log)
            per_model.append(pr)

        rows = []
        n2 = len(paths_d2)
        for N in topns:
            p = np.mean(per_model[:N], axis=0)
            b_d2 = domain_block(p[:n2], lab_d2)            # D2@D3
            b_d3 = domain_block(p[n2:], lab_d3)            # D3@D3
            acc = round((b_d2['average'] + b_d3['average']) / 2, 2)
            fr  = round(d2d2['average'] - b_d2['average'], 2)
            rows.append({'N': N, 'D2@D2': d2d2['average'], 'D2@D3': b_d2['average'],
                         'D3@D3': b_d3['average'], 'Acc': acc, 'Fr': fr,
                         'classwise': {'D2@D3': b_d2['classwise'], 'D3@D3': b_d3['classwise']}})
            log(f'    Top-{N:>2}: D2@D2 {d2d2["average"]:.2f} | D2@D3 {b_d2["average"]:.2f} '
                f'| D3@D3 {b_d3["average"]:.2f} | Acc {acc:.2f} | Fr {fr:+.2f}')

        out = {'protocol': f'crop_4s, {tta_desc}, soft voting Top-N sweep '
                           f'(Step2: D2 5-fold 전체 / Step3: val_macro Top-N)',
               'D2@D2_fixed': d2d2['average'], 'D2@D2_classwise': d2d2['classwise'],
               'topn': rows,
               'val_macro_ranking': [{'name': e['name'], 'val_macro': round(e['val_macro'], 2)}
                                     for e in entries]}
        out_path = os.path.join(RESULT_DIR, 'classwise_dev_ensemble_sweep.json')
        with open(out_path, 'w') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        log(f'[done] → {out_path}')
        print(json.dumps({'D2@D2_fixed': out['D2@D2_fixed'], 'topn': rows}, ensure_ascii=False, indent=2))
        return

    if args.mode == 'ensemble':
        d2_ckpts = list(discover_d2_ckpts(D2_DICT).values())
        entries = discover_d3_entries(D3_DICT)
        n, mean_vm = auto_n_by_val([e['val_macro'] for e in entries])
        d3_ckpts = [e['path'] for e in entries[:n]]
        protocol = f'crop_4s, {tta_desc}, soft voting (Step2: D2 5-fold / Step3: auto-N={n})'
        log(f'\n[classwise_dev:ensemble] {tta_desc} | Step2 {len(d2_ckpts)}모델, Step3 {n}모델')
    else:
        d2_ckpts = [os.path.join(SINGLE_DIR, 'd2.pth')]
        d3_ckpts = [os.path.join(SINGLE_DIR, 'd3.pth')]
        for p in d2_ckpts + d3_ckpts:
            if not os.path.exists(p):
                raise FileNotFoundError(f'단일 모델 없음: {p} (single_main.py 가 생성)')
        protocol = f'crop_4s, {tta_desc}, single model (no ensemble)'
        log(f'\n[classwise_dev:single] {tta_desc}')

    # Step2: D2 모델(count=1) on D2 test
    log('  Step2 Domain2:')
    p_s2 = soft_vote(d2_ckpts, 1, paths_d2, device, tta_gains, log)
    # Step3: D3 모델(count=2) on D2 + D3 test
    log('  Step3 (D2+D3 test):')
    p_s3 = soft_vote(d3_ckpts, 2, paths_d2 + paths_d3, device, tta_gains, log)
    p_s3_d2, p_s3_d3 = p_s3[:len(paths_d2)], p_s3[len(paths_d2):]

    out = {
        'protocol': protocol,
        'Step2': {'Domain2': domain_block(p_s2, lab_d2)},
        'Step3': {'Domain2': domain_block(p_s3_d2, lab_d2),
                  'Domain3': domain_block(p_s3_d3, lab_d3)},
    }
    out_path = os.path.join(RESULT_DIR, f'classwise_dev_{args.mode}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f'  Step2 D2 {out["Step2"]["Domain2"]["average"]} | '
        f'Step3 D2 {out["Step3"]["Domain2"]["average"]} / D3 {out["Step3"]["Domain3"]["average"]}')
    log(f'[done] → {out_path}')
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
