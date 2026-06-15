"""
train_loop.py — 순수 학습 루프 + 얼리스탑 + 로그 헬퍼 (exp16 train.py 에서 추출한 공용 모듈)

script/ 의 main.py·single_main.py 가 공유한다. 중복 코드 금지 원칙에 따라
train_phase(CE + KD + R penalty + Σ활성수집)와 MultiCriterionStopper 는 여기에만 존재한다.
method 는 exp15 CoForge 와 동일 (early-stop best-val restore).
"""

import os
import time

import torch
import torch.nn.functional as F

import config as c
from dataset import build_train_loader, mixup_batch, make_criterion, evaluate
import ddp_util as ddp


def _fmt_dur(sec):
    """초 → 사람이 읽는 길이 (3s / 9m58s / 1h02m). 로그 전용."""
    sec = int(sec)
    if sec < 60:   return f'{sec}s'
    if sec < 3600: return f'{sec//60}m{sec%60:02d}s'
    return f'{sec//3600}h{(sec%3600)//60:02d}m'


def make_log(log_path):
    """stdout + 파일 동시 기록 로거. 모든 학습 로그는 model/result/ 안에만 남긴다."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as f: f.write(msg + '\n')
    return log


def freeze(model):
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model


def save_confmat_md(confmat, path, title):
    if not confmat: return
    labels = list(confmat.keys())
    lines  = [f'# {title}', '',
              '| true \\ pred | ' + ' | '.join(labels) + ' | acc% |',
              '|' + '---|' * (len(labels) + 2)]
    for tl in labels:
        row = confmat.get(tl, {}); total = sum(row.values())
        acc   = f'{100*row.get(tl,0)/total:.1f}' if total > 0 else '—'
        cells = ' | '.join(str(row.get(pl, 0)) for pl in labels)
        lines.append(f'| {tl} | {cells} | {acc} |')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f: f.write('\n'.join(lines) + '\n')


# ── Early stopping ───────────────────────────────────────────────────────────

class MultiCriterionStopper:
    """patience(얼리스탑)와 best(복원/저장)를 **분리** 모니터.
      patience_on: 무엇이 개선되면 patience 리셋 (loss | micro | macro | any[3중-OR]) → stop 신호.
      best_on    : 복원/저장 모델 선택 기준 (macro=공식 | loss | micro)."""
    def __init__(self, patience, min_epochs=0, min_delta=0.0, patience_on='loss', best_on='macro'):
        self.patience = patience; self.min_epochs = min_epochs; self.min_delta = min_delta
        self.patience_on = patience_on; self.best_on = best_on
        self.best_loss = float('inf'); self.best_micro = -1.0; self.best_macro = -1.0
        self.best_epoch = 0; self.best_state = None; self.best_meta = None
        self.counter = 0

    def step(self, loss, micro, macro, model, epoch, meta=None):
        imp = {'loss':  loss  < self.best_loss  - self.min_delta,
               'micro': micro > self.best_micro + self.min_delta,
               'macro': macro > self.best_macro + self.min_delta}
        if imp['loss']:  self.best_loss  = loss
        if imp['micro']: self.best_micro = micro
        if imp['macro']: self.best_macro = macro
        improved_best = imp[self.best_on]
        if improved_best:
            self.best_epoch = epoch
            self.best_state = {k: v.detach().to('cpu', copy=True)
                               for k, v in model.state_dict().items()}; self.best_meta = meta
        reset = (imp['loss'] or imp['micro'] or imp['macro']) if self.patience_on == 'any' \
                else imp[self.patience_on]
        self.counter = 0 if reset else self.counter + 1
        stop = (epoch >= self.min_epochs) and (self.counter >= self.patience)
        return improved_best, stop

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

    @property
    def best_str(self):
        v = {'loss': f'loss={self.best_loss:.4f}', 'micro': f'micro={self.best_micro:.2f}',
             'macro': f'macro={self.best_macro:.2f}'}[self.best_on]
        return f'{v}@{self.best_epoch}'


# ── 훈련 phase (exp15/16 과 동일 로직) ───────────────────────────────────────

def train_phase(
    model, train_df, device, log_fn, phase_name,
    lr, teacher_model=None, mode_cfg=None,
    epochs=None, val_df=None, patience=None, min_epochs=0,
    on_best=None,
    rank=0, world_size=1, distributed=False,
    use_warm_restarts=False,
    reg_fn=None, reg_beta=0.0,   # 망각 penalty R=β·tr(D⁻¹ ΔW Σ_prior ΔWᵀ)
    cov_collect=False, cov_every=8,   # 학습 forward 도중 활성 통계 수집(별도 패스 없음)
):
    assert epochs is not None
    is_main = (rank == 0)
    log_fn(f'\n========== Phase: {phase_name} ==========')

    ds, loader = build_train_loader(train_df, rank, world_size, distributed)
    if is_main:
        log_fn(f'  Train chunks: {len(ds)}')
        if val_df is not None:
            log_fn(f'  Val clips: {len(val_df)} | patience={patience} min_epochs={min_epochs}')

    raw_model = ddp.raw(model)
    trainable = [p for p in raw_model.parameters() if p.requires_grad]
    if is_main:
        log_fn(f'  Trainable params: {sum(p.numel() for p in trainable):,}  LR: {lr}')
        log_fn(f'  Scheduler: {"WarmRestarts" if use_warm_restarts else "CosineAnnealing"}')

    optim = torch.optim.AdamW(trainable, lr=lr, weight_decay=c.WEIGHT_DECAY)
    if use_warm_restarts:
        T0 = max(1, epochs // 4)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim, T_0=T0, T_mult=2, eta_min=c.LR_ETA_MIN)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=epochs, eta_min=c.LR_ETA_MIN)

    criterion = make_criterion()
    _es = c.CFG.get('early_stop', {}) or {}
    _pat_on = str(_es.get('patience_on', 'loss')).lower()
    _best_on = str(_es.get('best_on', 'macro')).lower()
    stopper = (MultiCriterionStopper(patience, min_epochs, c.MIN_DELTA,
                                     patience_on=_pat_on, best_on=_best_on)
               if (val_df is not None and patience is not None and is_main)
               else None)
    if is_main and stopper is not None:
        log_fn(f'  EarlyStop: patience_on={_pat_on}(정지신호) | best_on={_best_on}(복원/저장)')

    if cov_collect and is_main:
        raw_model._captured_acts = {}     # 학습 중 활성 reservoir 초기화 (별도 패스 없음)

    _t_sum = 0.0
    for epoch in range(1, epochs + 1):
        _ep_start = time.time()
        if distributed and hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(epoch)

        model.train()
        if not raw_model.tune_bn:
            raw_model.backbone.set_bn_eval()
        raw_model.set_bn_train_for_tuned()

        sum_cls = sum_kd = sum_feat = sum_reg = n = 0

        for batch_i, (audio, target, _, _) in enumerate(loader):
            audio      = audio.float().to(device)
            target_idx = target.float().to(device).argmax(-1)

            audio_m, ya, yb, lam = mixup_batch(audio, target_idx, device)

            if cov_collect:   # rank0 only, mixup 없는 배치만 → 깨끗한 Σ
                raw_model._capture_acts = (is_main and batch_i % cov_every == 0 and lam == 1.0)

            logits, feat = model(audio_m, use_spec_aug=c.USE_SPEC_AUG)
            loss_cls = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            loss = loss_cls

            if teacher_model is not None:
                kd_alpha = mode_cfg['kd_alpha']; kd_temp = mode_cfg['kd_temp']
                kd_fw    = mode_cfg['kd_feat_weight']
                teacher_model.eval()
                with torch.no_grad():
                    t_logits, t_feat = teacher_model(audio_m)
                kd_loss = F.kl_div(
                    F.log_softmax(logits / kd_temp, dim=-1),
                    F.softmax(t_logits / kd_temp, dim=-1),
                    reduction='batchmean') * (kd_temp ** 2)
                feat_loss = 1.0 - F.cosine_similarity(feat, t_feat, dim=1, eps=1e-8).mean()
                loss = loss + kd_alpha * kd_loss + kd_fw * feat_loss
                sum_kd += float(kd_loss.detach()); sum_feat += float(feat_loss.detach())

            if reg_fn is not None and reg_beta > 0.0:    # 망각 penalty (캐시 Σ_prior 통계만 사용)
                R = reg_fn()
                if R is not None:
                    loss = loss + reg_beta * R
                    sum_reg += float(R.detach())

            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, c.GRAD_CLIP)
            optim.step(); sum_cls += float(loss_cls.detach()); n += 1

        if cov_collect:
            raw_model._capture_acts = False   # val/eval 중 캡처 방지

        avg_cls = ddp.all_reduce_mean(sum_cls / max(n, 1), device, distributed)
        sched.step()

        stop = False
        if is_main and stopper is not None:
            _vc = bool(c.CFG.get('val_eval', {}).get('chunk', True))
            val_micro, val_detail, val_loss = evaluate(
                raw_model, val_df, device, return_detail=True, criterion=criterion,
                chunk=_vc, bucket=not _vc)
            val_macro = val_detail['macro']
            improved_best, stop = stopper.step(val_loss, val_micro, val_macro, raw_model, epoch,
                                               meta={'epoch': epoch, 'val_loss': val_loss,
                                                     'val_micro': val_micro, 'val_macro': val_macro})
            if improved_best and on_best is not None:
                on_best(raw_model, val_micro, val_loss, epoch, val_detail)

        stop = ddp.broadcast_stop(stop, device, distributed)

        _ep_dt = time.time() - _ep_start
        _t_sum += _ep_dt
        _eta = (_t_sum / epoch) * (epochs - epoch)

        if is_main and (epoch % c.LOG_EVERY == 0 or epoch == 1 or epoch == epochs or stop):
            parts = [f'Epoch {epoch:3d}/{epochs}', f'loss {avg_cls:.4f}']
            if stopper is not None:
                parts.append(f'val acc {val_micro:.1f} macro {val_detail["macro"]:.1f}')
                parts.append(f'best macro {stopper.best_macro:.1f}@{stopper.best_epoch} '
                             f'(stall {stopper.counter}/{patience})')
            if teacher_model is not None:
                parts.append(f'KD {sum_kd/max(n,1):.3f} feat {sum_feat/max(n,1):.3f}')
            if reg_fn is not None and reg_beta > 0.0:
                parts.append(f'R {sum_reg/max(n,1):.4f}')
            parts.append(f'lr {optim.param_groups[0]["lr"]:.2e}')
            parts.append(f'{_ep_dt:.1f}s eta {_fmt_dur(_eta)}')
            log_fn('  ' + ' | '.join(parts))

        if stop:
            if is_main:
                log_fn(f'  Early stop @ epoch {epoch} (best {stopper.best_str}, {patience}ep 정체)')
            break

    if is_main and stopper is not None:
        stopper.restore(raw_model)
        log_fn(f'  Restored best model ({stopper.best_str}, best_on={stopper.best_on}, '
               f'patience_on={stopper.patience_on})')
    ddp.sync_model(raw_model, distributed)
    return stopper
