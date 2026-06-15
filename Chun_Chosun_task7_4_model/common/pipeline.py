"""
pipeline.py — CoForge K-Fold 파이프라인 절차 (exp16 train_kfold.py 에서 추출한 공용 모듈)

script/main.py(앙상블)·single_main.py(단일)가 공유하는 도메인 학습 1회분 + K-Fold phase 러너.
  - setup_determinism / set_all_seeds : 결정성·시드
  - build_domain_spec / make_prior_kd : config 구동 도메인 spec + drift-비율 KD
  - build_model_adaptor               : D1 동결 backbone + LoRACNN14 + DeltaGPAdaptor
  - train_one_domain                  : teacher snapshot → A init → hooks → KeepLoRA → train_phase
  - run_phase_d2 / run_phase_d3       : K-Fold 5 + 25 런 (skip-if-exists, 디스크 독립)
"""

import copy
import json
import os
import random
import time

import numpy as np
import torch

import config as c
from backbone import CNN14Backbone
from dataset import make_kfold_split
from model_adaptor import LoRACNN14, drift_to_block_configs
from train_adaptor import DeltaGPAdaptor
from train_loop import train_phase, freeze, save_confmat_md


# ── 결정성 / 시드 ─────────────────────────────────────────────────────────────

def setup_determinism():
    """config `deterministic` 토글. false=cudnn.benchmark(빠름, 런간 ~0.8%p 노이즈)."""
    det = bool(c.CFG.get('deterministic', True))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = (not det)
        torch.backends.cudnn.deterministic = det
        torch.use_deterministic_algorithms(det, warn_only=True)
        if det:
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.set_float32_matmul_precision('high')
    return det


def set_all_seeds(seed):
    """random/np/torch/cuda + c.SEED 동시 설정 (조합별 결정성)."""
    seed = int(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    c.set_seed(seed)


def require_cuda(device):
    """CPU 폴백 차단: CUDA_VISIBLE_DEVICES 가 잘못되면 조용히 CPU 학습으로 떨어진다 → 즉시 에러."""
    if device == 'cpu' or not torch.cuda.is_available():
        raise RuntimeError(
            f'CUDA GPU 를 못 찾았습니다 → CPU 폴백 차단. '
            f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")!r} (GPU 정수 id 여야 함).')


# ── adaptor 보호상태 디스크 직렬화 (post-D2 → D3 가 로드) ─────────────────────

def _to_cpu_list(lst):
    return [None if t is None else t.detach().cpu() for t in lst]


def save_protect(adaptor, path):
    """D2 학습 후 보호상태 저장: _protect_v + Σ-cov 캐시. D3 가 hook·A초기화·R 에 사용."""
    torch.save({'_protect_v': _to_cpu_list(adaptor._protect_v),
                '_cov_U':     _to_cpu_list(adaptor._cov_U),
                '_cov_lam':   _to_cpu_list(adaptor._cov_lam),
                '_cov_bnvar': _to_cpu_list(adaptor._cov_bnvar)}, path)


def load_protect(adaptor, path, device):
    d = torch.load(path, map_location='cpu', weights_only=False)
    adaptor._protect_v = list(d['_protect_v'])
    adaptor._cov_U     = [None if t is None else t.to(device) for t in d['_cov_U']]
    adaptor._cov_lam   = [None if t is None else t.to(device) for t in d['_cov_lam']]
    adaptor._cov_bnvar = [None if t is None else t.to(device) for t in d['_cov_bnvar']]
    adaptor.clear_orth_hooks()


# ── config 구동 도메인 spec ───────────────────────────────────────────────────

def build_domain_spec(log):
    _dom_cfg = c.CFG.get('domains', []) or []
    _dom_def = c.CFG.get('domain_defaults', {}) or {}
    if len(_dom_cfg) != 2:
        raise RuntimeError(f'2-phase 파이프라인은 domains=[D2, D3] 2개 필요. 현재 {len(_dom_cfg)}개.')

    def _dget(d, key, default=None):
        return d.get(key, _dom_def.get(key, default))

    specs, block_cfgs, rank_meta = [], [], {}
    for d in _dom_cfg:
        name = d['name']
        cfgs, r_A, r_B = drift_to_block_configs(d['drift_conv_bn'], d['drift_bn'], _dget(d, 'ceiling'))
        block_cfgs.append(cfgs)
        rank_meta[f'{name.lower()}_rank_A'] = r_A
        rank_meta[f'{name.lower()}_rank_B'] = r_B
        drift_j = sum(max(0.0, float(v)) for v in d['drift_conv_bn']) / len(d['drift_conv_bn'])
        specs.append(dict(
            name=name, lr=float(_dget(d, 'lr')), epochs=int(_dget(d, 'epochs')),
            patience=int(_dget(d, 'patience')), min_epochs=int(_dget(d, 'min_epochs')),
            warm=bool(_dget(d, 'warm', False)), kd=dict(_dget(d, 'kd')), drift=drift_j))
        log(f'  [{name}] rank_A={r_A} rank_B={r_B} (auto from drift, ceil={_dget(d,"ceiling")}) | '
            f'lr={specs[-1]["lr"]} epochs={specs[-1]["epochs"]} warm={specs[-1]["warm"]}')
    return specs, block_cfgs, rank_meta


def make_prior_kd(specs):
    """drift 비율 KD α: t=0(D2)→α_lo 바닥, t=1(D3)→prior/(prior+own). kd_from_drift=false→None."""
    _kd = c.CFG.get('kd_prior', {}) or {}
    _cf = c.CFG.get('cov_forget', {}) or {}
    kd_a_lo = float(_kd.get('alpha_lo', 0.1)); kd_a_hi = float(_kd.get('alpha_hi', 1.0))
    kd_from_drift = bool(_cf.get('kd_from_drift', _kd.get('use', False)))

    def _prior_kd(t):
        if not kd_from_drift:
            return None
        prior = sum(specs[j]['drift'] for j in range(t))
        own   = float(specs[t]['drift'])
        ratio = prior / (prior + own) if (prior + own) > 0 else 0.0
        alpha = min(kd_a_hi, max(kd_a_lo, ratio))
        cfg = dict(specs[t]['kd']); cfg['kd_alpha'] = alpha
        return cfg, prior, own, ratio
    return _prior_kd, kd_from_drift


def build_model_adaptor(block_cfgs, device):
    """fresh backbone(D1 동결) + LoRACNN14 + DeltaGPAdaptor."""
    gate = c.CFG.get('gate', {}) or {}
    backbone = CNN14Backbone(nb_tasks=3)
    backbone.load_d1_checkpoint(c.D1_CKPT); backbone.freeze_backbone(); backbone.to(device)
    model = LoRACNN14(backbone, domain_block_cfgs=block_cfgs,
                      alpha=c.LORA_ALPHA, tune_bn=bool(c.CFG['lora'].get('tune_bn', True)),
                      gate_cfg=gate).to(device)
    return model, DeltaGPAdaptor(model)


def collect_kfg():
    """학습기법 설정 (KeepLoRA / CovForget)."""
    _kl = c.CFG.get('keeplora_binit', {}) or {}
    _cf = c.CFG.get('cov_forget', {}) or {}
    return dict(
        kl_use=bool(_kl.get('use', False)), kl_eta=float(_kl.get('eta', 0.1)),
        kl_nb=int(_kl.get('n_batches', 4)), kl_align=bool(_kl.get('align_A', False)),
        cf_use=bool(_cf.get('use', False)), cf_every=int(_cf.get('cov_every', 8)),
        cf_energy=float(_cf.get('energy', 0.95)), cf_kmax=int(_cf.get('k_max', 256)),
        cf_ep=float(_cf.get('energy_protect', 0.9)), cf_beta=float(_cf.get('beta', 1.0e-3)))


def p_svd_d1():
    _pace = c.CFG['pace']
    return _pace.get('p_svd_d1', _pace.get('p_svd_d2'))


# ── 한 도메인 phase 학습 (D2 또는 D3) ─────────────────────────────────────────

def train_one_domain(model, adaptor, t, spec, df_tr, df_val, device, log,
                     kfg, prior_kd, save_cm_path=None, cm_title=''):
    """teacher snapshot → A init(⊥_protect_v) → hooks → KeepLoRA → train_phase. best restore 후 반환."""
    name = spec['name']
    teacher = freeze(copy.deepcopy(model)); teacher.active_domain_count = t; teacher.to(device)

    model.active_domain_count = t + 1
    adaptor.init_A_in_nullspace(domain_idx=t)
    model.set_domain_trainable(t)
    adaptor.install_orth_hooks(domain_idx=t)
    if kfg['kl_use']:
        adaptor.keeplora_init_AB(domain_idx=t, df=df_tr, device=device,
                                 eta=kfg['kl_eta'], n_batches=kfg['kl_nb'],
                                 align_A=kfg['kl_align'], log_fn=log)

    reg_fn   = (adaptor.forgetting_penalty if (kfg['cf_use'] and adaptor._cov_U) else None)
    reg_beta = (kfg['cf_beta'] if reg_fn is not None else 0.0)

    _kd = prior_kd(t)
    if _kd is not None:
        mode_cfg, _p, _o, _r = _kd
        log(f'  drift-비율 α: prior={_p:.4f} own={_o:.4f} → ratio={_r:.4f} → clamp = {mode_cfg["kd_alpha"]:.3f}')
    else:
        mode_cfg = spec['kd']
    log(f'  KD mode_cfg = {{alpha:{mode_cfg["kd_alpha"]:.3f}, temp:{mode_cfg["kd_temp"]}, '
        f'feat_w:{mode_cfg["kd_feat_weight"]:.3f}}}')

    on_best = None
    if save_cm_path is not None:
        def on_best(m, val_micro, val_loss, epoch, detail):
            save_confmat_md(detail.get('confusion_matrix', {}), save_cm_path,
                            f'{cm_title} — epoch={epoch} vacc={val_micro:.2f}%')

    stopper = train_phase(
        model, df_tr, device, log,
        phase_name=f'{name} (CoForge{" + R-penalty" if reg_fn else ""}'
                   f'{" + WarmRestarts" if spec["warm"] else ""})',
        lr=spec['lr'], teacher_model=teacher, mode_cfg=mode_cfg,
        epochs=spec['epochs'], val_df=df_val,
        patience=spec['patience'], min_epochs=spec['min_epochs'],
        on_best=on_best,
        rank=0, world_size=1, distributed=False,
        use_warm_restarts=spec['warm'],
        reg_fn=reg_fn, reg_beta=reg_beta,
        cov_collect=kfg['cf_use'], cov_every=kfg['cf_every'])
    adaptor.clear_orth_hooks()
    return stopper


# ── K-Fold phase 러너 (skip-if-exists, 디스크 독립) ───────────────────────────

def run_phase_d2(folds, out_dir, specs, block_cfgs, rank_meta, kfg, prior_kd, kf, device, log):
    """D2 fold 학습 → {out_dir}/d2_f{k}/{d2.pth, protect.pt}. 완료분 skip."""
    df_d2 = c.DF_TRAIN[c.DF_TRAIN['domain'] == specs[0]['name']]
    for k in folds:
        t0 = time.time()
        ddir = os.path.join(out_dir, f'd2_f{k}')
        if (os.path.exists(os.path.join(ddir, 'd2.pth'))
                and os.path.exists(os.path.join(ddir, 'protect.pt'))):
            log(f'[Phase d2] D2 fold {k}: 이미 완료 → skip')
            continue
        log(f'\n{"="*70}\n[Phase d2] D2 fold {k}/{kf["K"]}\n{"="*70}')
        model, adaptor = build_model_adaptor(block_cfgs, device)
        adaptor.build_base_protect(p_svd=p_svd_d1(), eps=c.SVD_EPS)

        set_all_seeds(kf['d2_sbase'] + k)
        d2_tr, d2_val = make_kfold_split(df_d2, fold_k=k, k_total=kf['K'], seed=kf['kf_seed'])
        log(f'  D2 split: train={len(d2_tr)} val={len(d2_val)} (fold {k})')
        os.makedirs(ddir, exist_ok=True)
        st = train_one_domain(model, adaptor, 0, specs[0], d2_tr, d2_val, device, log,
                              kfg, prior_kd, os.path.join(ddir, 'confmat_d2_val.md'),
                              f'D2 fold{k} Val CM')
        model.active_domain_count = 1
        torch.save({'model': model.state_dict(), 'phase': 'after_D2', 'd2_fold': k,
                    'val_macro': float(st.best_macro), 'active_domain_count': 1, **rank_meta},
                   os.path.join(ddir, 'd2.pth'))
        if kfg['cf_use']:
            adaptor.finalize_cov_basis(0, energy=kfg['cf_energy'], k_max=kfg['cf_kmax'], log_fn=log)
            adaptor.accumulate_cov_protect(energy_protect=kfg['cf_ep'], log_fn=log)
        save_protect(adaptor, os.path.join(ddir, 'protect.pt'))
        log(f'  ✓ saved {ddir}/d2.pth + protect.pt (val_macro={st.best_macro:.2f}, '
            f'{(time.time()-t0)/60:.1f} min)')
        del model, adaptor
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def run_phase_d3(jobs, out_dir, specs, block_cfgs, rank_meta, kfg, prior_kd, kf, device, log):
    """(k,j) 조합 학습: 부모 {out_dir}/d2_f{k} 디스크 로드 → 완전 독립 런. 완료분 skip."""
    df_d3 = c.DF_TRAIN[c.DF_TRAIN['domain'] == specs[1]['name']]
    for (k, j) in jobs:
        t0 = time.time()
        combo_seed = kf['combo_sbase'] + k * 10 + j
        ddir = os.path.join(out_dir, f'd3_f{k}_{j}')
        if os.path.exists(os.path.join(ddir, 'd3.pth')):
            log(f'[Phase d3] D2f{k} × D3f{j}: 이미 완료 → skip')
            continue
        log(f'\n{"="*70}\n[Phase d3] D2f{k} × D3f{j}  (seed={combo_seed})\n{"="*70}')
        d2dir = os.path.join(out_dir, f'd2_f{k}')
        d2pth, protpth = os.path.join(d2dir, 'd2.pth'), os.path.join(d2dir, 'protect.pt')
        for p in (d2pth, protpth):
            if not os.path.exists(p):
                raise FileNotFoundError(f'부모 D2 산출물 없음: {p} (Phase d2 를 먼저 완료하세요)')

        model, adaptor = build_model_adaptor(block_cfgs, device)
        ck = torch.load(d2pth, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        load_protect(adaptor, protpth, device)
        model.active_domain_count = 1

        set_all_seeds(combo_seed)
        d3_tr, d3_val = make_kfold_split(df_d3, fold_k=j, k_total=kf['K'], seed=kf['kf_seed'])
        log(f'  D3 split: train={len(d3_tr)} val={len(d3_val)} (fold {j}) | parent d2_f{k} '
            f'(val_macro={ck.get("val_macro")})')
        os.makedirs(ddir, exist_ok=True)
        st = train_one_domain(model, adaptor, 1, specs[1], d3_tr, d3_val, device, log,
                              kfg, prior_kd, os.path.join(ddir, 'confmat_d3_val.md'),
                              f'D3 d2f{k}_d3f{j} Val CM')
        model.active_domain_count = 2
        torch.save({'model': model.state_dict(), 'phase': 'after_D3', 'd2_fold': k, 'd3_fold': j,
                    'combo_seed': combo_seed, 'val_macro': float(st.best_macro),
                    'active_domain_count': 2, **rank_meta},
                   os.path.join(ddir, 'd3.pth'))
        with open(os.path.join(ddir, 'meta.json'), 'w') as f:
            json.dump({'d2_fold': k, 'd3_fold': j, 'combo_seed': combo_seed,
                       'val_macro': float(st.best_macro), 'val_micro': float(st.best_micro)},
                      f, indent=2)
        log(f'  ✓ saved {ddir}/d3.pth (val_macro={st.best_macro:.2f}, {(time.time()-t0)/60:.1f} min)')
        del model, adaptor
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def kfold_cfg():
    """config kfold 섹션 → dict (out_dir 은 호출자가 결정 — 스크립트 기준 상대경로)."""
    _kf = c.CFG.get('kfold', {}) or {}
    return dict(K=int(_kf.get('k_total', 5)), kf_seed=int(_kf.get('seed', 42)),
                d2_sbase=int(_kf.get('d2_seed_base', 42)),
                combo_sbase=int(_kf.get('combo_seed_base', 42)),
                topn=list(_kf.get('topn', [5, 10, 15, 20, 25])),
                submission_n=_kf.get('submission_n', None))
