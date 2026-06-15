"""
dataset.py — exp07_AdaptorSplit 데이터 + 증강 레이어 (train.py 에서 분리)

데이터셋·청킹·증강·로더 구성·손실·val split·평가 등 "데이터 관련 로직 전부" 를 모은다.
train.py 는 이 모듈의 함수만 호출하고 데이터 세부는 알지 못한다 (순수 학습 루프만 보유).

증강 (baseline 대비 추가/정리된 것):
  - gain aug   : AudioDataset.__getitem__ (chunk 단위 random gain)
  - chunking   : chunk_audio (4s non-overlap, <min_sec 폐기)
  - mixup      : mixup_batch (batch 단위, prob 적용)  ← train_phase 인라인에서 추출
  - spec_aug   : 모델 forward(backbone.spec_augmenter) 쪽이라 여기 없음
  - balanced sampling : AudioDataset.get_balanced_sampler

저수준 디코드 캐시·청킹은 공유 common/trainer.py 와 동일 구현을 exp 로컬로 둔다
(실험 스냅샷 재현성 — common 변경에 영향받지 않도록).
"""
import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, DistributedSampler, Sampler
import librosa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as c

# ── 조건부 gain 예외 (프로토콜 §1-2, exp04 설정) ─────────────────────────────
#   COND_CLASSES 에 속한 클래스는 gain aug 미적용 (Mixup·SpecAug 는 전 클래스 그대로).
#   D3 극소 클래스(baby_cry=1 24샘플, telephone_ringing=7 31샘플) 진폭 보존.
#   config.yaml 의 cond_classes 에서 읽음 (없으면 빈 set → exp15 동작: 전 클래스 gain).
COND_CLASSES = set(int(x) for x in (c.CFG.get('cond_classes', []) or []))


# ============================================================
# 디코딩 캐시 (path → waveform)
#   librosa.load(~57ms/clip)가 지배적 병목. 클립당 1회만 디코딩한다.
#   - __init__ 이 main 프로세스에서 캐시를 채우면 fork 된 DataLoader 워커가
#     COW 로 공유 → 워커 재디코딩 0.
#   - evaluate() 가 매 epoch 새 Dataset 을 만들어도 전역 캐시라 재디코딩 안 함.
# ============================================================
_AUDIO_CACHE = {}


def _load_audio(path):
    y = _AUDIO_CACHE.get(path)
    if y is None:
        y, _ = librosa.load(path, sr=c.SAMPLE_RATE, mono=True)
        y = y.astype(np.float32)
        _AUDIO_CACHE[path] = y
    return y


# ============================================================
# Audio chunk 분할 (§8 — in-memory)
# ============================================================
def pad_trunc(x, tlen):
    return np.concatenate([x, np.zeros(tlen - len(x))]) if len(x) < tlen else x[:tlen]


def chunk_audio(y, sr=None, chunk_seconds=None, min_seconds=None):
    """
    공식 baseline 오프라인 utils/chunking.py main 루프의 in-memory 버전.
      - non-overlap chunk_seconds(기본 4s) 분할.
      - 마지막 chunk 가 min_seconds(기본 2s) 미만이면 폐기, 이상이면 zero-pad.
      - 각 chunk 가 독립 training sample.
    1차원 waveform 1개를 받아 [chunk, samples] 리스트로 반환.
    """
    sr = sr or c.SAMPLE_RATE
    chunk_seconds = c.CHUNK_SECONDS if chunk_seconds is None else chunk_seconds
    min_seconds = c.CHUNK_MIN_SECONDS if min_seconds is None else min_seconds
    chunk_len = int(round(chunk_seconds * sr))
    min_len = int(round(min_seconds * sr))

    chunks = []
    for start in range(0, len(y), chunk_len):
        seg = y[start:start + chunk_len]
        if len(seg) < min_len:
            continue                                  # 2초 미만 → 폐기
        if len(seg) < chunk_len:
            seg = pad_trunc(seg, chunk_len)           # 2~4초 → zero-pad
        chunks.append(seg.astype(np.float32))
    if not chunks:                                    # 전체가 min_len 미만이면 1개는 보존
        chunks.append(pad_trunc(y, chunk_len).astype(np.float32))
    return chunks


# ============================================================
# Dataset
# ============================================================
class AudioDataset(Dataset):
    """
    is_train=True  : chunk_audio 로 4s non-overlap 분할 → 각 chunk 독립 sample.
    is_train=False : 가변 길이(truncate 없음) → eval loader batch_size=1 필수.
    """
    def __init__(self, df, is_train=True):
        self.is_train = is_train
        self.labels, self.fnames, self.domains = [], [], []
        if is_train:
            # chunk 분할: 디코드 캐시에서 로드(1회) → chunk 단위로 sample 펼침.
            #   모든 chunk 는 CLIP_SAMPLES 길이라 연속 배열로 적재 → 워커 COW 안전.
            chunks_all = []
            for _, row in df.iterrows():
                y = _load_audio(row['full_path'])
                for seg in chunk_audio(y):
                    chunks_all.append(seg)
                    self.labels.append(int(row['new_target']))
                    self.fnames.append(row['filename'])
                    self.domains.append(c.DOMAIN_TO_IDX[row['domain']])
            self._samples = (np.stack(chunks_all).astype(np.float32)
                             if chunks_all else np.zeros((0, c.CLIP_SAMPLES), np.float32))
        else:
            self.paths = []
            for _, row in df.iterrows():
                _load_audio(row['full_path'])         # 전역 캐시 워밍(main) → 워커 재디코딩 0
                self.paths.append(row['full_path'])
                self.labels.append(int(row['new_target']))
                self.fnames.append(row['filename'])
                self.domains.append(c.DOMAIN_TO_IDX[row['domain']])

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        if self.is_train:
            x = self._samples[idx].copy()                                # 캐시 원본 보호
            if self.labels[idx] not in COND_CLASSES:                     # gain: COND 클래스 제외(§1-2)
                x = x * np.random.uniform(c.GAIN_AUG_LOW, c.GAIN_AUG_HIGH)
        else:
            y = _load_audio(self.paths[idx])          # 가변 길이, truncate 없음
            # CLIP_SAMPLES(4s) 미만 클립은 zero-pad: backbone 6× avg_pool2d 가
            #   짧은 클립에서 공간 차원을 0 으로 붕괴시키는 것 방지(학습 chunk pad 와 동일).
            #   긴 클립은 그대로 → "가변 길이 truncate 없음"(§8) 유지.
            x = y if len(y) >= c.CLIP_SAMPLES else pad_trunc(y, c.CLIP_SAMPLES)
            x = np.asarray(x, dtype=np.float32).copy()  # 캐시 원본 보호
        onehot = np.zeros(c.NUM_CLASSES, dtype=np.float32)
        onehot[self.labels[idx]] = 1.0
        return x, onehot, self.fnames[idx], self.domains[idx]

    def get_sample_weights(self):
        # §9: 클래스별 가중치 = 1/count → 각 클래스 균등 비율. (단일/DDP 공용)
        labels = np.array(self.labels)
        counts = np.bincount(labels, minlength=c.NUM_CLASSES).astype(np.float64)
        w = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
        if c.USE_BALANCED_SAMPLING:
            for hc in c.HARD_CLASSES:
                w[hc] *= c.HARD_CLASS_MULTIPLIER      # multiplier=1 → 추가 가중 없음
        return w[labels]

    def get_balanced_sampler(self):
        sample_w = self.get_sample_weights()
        return WeightedRandomSampler(sample_w, len(sample_w), replacement=True)


# ============================================================
# Loss / 증강
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.ls = label_smoothing
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def make_criterion():
    """config 에 따라 FocalLoss / CrossEntropy 선택 (train_phase·KeepLoRA 공통)."""
    return (FocalLoss(gamma=c.FOCAL_GAMMA, label_smoothing=c.LABEL_SMOOTHING)
            if c.USE_FOCAL_LOSS
            else nn.CrossEntropyLoss(label_smoothing=c.LABEL_SMOOTHING))


def mixup_batch(audio, target_idx, device):
    """batch 단위 mixup 증강 (train_phase 인라인에서 추출).
    확률 c.MIXUP_PROB 로 적용, 아니면 항등. 반환 (audio_m, ya, yb, lam).
    RNG 호출 순서는 원본과 동일: np.random.random → (적용 시) np.random.beta → torch.randperm."""
    if c.USE_MIXUP and np.random.random() < c.MIXUP_PROB:
        lam = np.random.beta(c.MIXUP_ALPHA, c.MIXUP_ALPHA)
        idx = torch.randperm(audio.size(0), device=device)
        return lam * audio + (1 - lam) * audio[idx], target_idx, target_idx[idx], lam
    return audio, target_idx, target_idx, 1.0


# ============================================================
# DataLoader 구성 (train 전용)
# ============================================================
class DistributedWeightedSampler(Sampler):
    """클래스 균형(가중) 샘플링 + rank 분할 — DDP 에서도 use_balanced 적용.
    매 epoch 전 rank 동일 seed 로 전역 가중 추출 후 rank 별로 strided 분배 → 중복 없는 균형 샤딩.
    set_epoch(epoch) 로 epoch 마다 재추출(DistributedSampler 와 동일 인터페이스)."""
    def __init__(self, weights, num_replicas, rank, seed=0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.per_rank = len(self.weights) // num_replicas   # rank 당 동일 개수(나머지 버림 → DDP 균형)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.per_rank

    def __iter__(self):
        g = torch.Generator(); g.manual_seed(self.seed + self.epoch)   # 전 rank 동일 → 분할이 partition
        n = self.per_rank * self.num_replicas
        idx = torch.multinomial(self.weights, n, replacement=True, generator=g)
        return iter(idx[self.rank::self.num_replicas].tolist())


def build_train_loader(train_df, rank=0, world_size=1, distributed=False):
    """학습용 (dataset, loader). use_balanced 를 DDP/단일 동일하게 적용:
      use_balanced=false → 균등 (DDP: DistributedSampler, 단일: shuffle)  ← 기존/backup 과 동일.
      use_balanced=true  → 균형 (DDP: DistributedWeightedSampler, 단일: WeightedRandomSampler).
    워커 augmentation RNG 결정성: 워커마다 시드 고정.  반환: (AudioDataset, DataLoader)."""
    ds = AudioDataset(train_df, is_train=True)

    def _seed_worker(worker_id):
        ws = torch.initial_seed() % 2**32
        np.random.seed(ws); random.seed(ws)
    gen = torch.Generator(); gen.manual_seed(c.SEED + rank)
    bal = c.USE_BALANCED_SAMPLING

    if distributed:
        if bal:
            sampler = DistributedWeightedSampler(ds.get_sample_weights(),
                                                 world_size, rank, seed=c.SEED)
        else:
            sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                         shuffle=True, drop_last=True)   # 기존(backup) 그대로
        loader = DataLoader(ds, batch_size=c.BATCH_SIZE, sampler=sampler,
                            num_workers=c.NUM_WORKERS, pin_memory=True,
                            worker_init_fn=_seed_worker, generator=gen)
    else:
        loader = DataLoader(
            ds, batch_size=c.BATCH_SIZE,
            sampler=ds.get_balanced_sampler() if bal else None,
            shuffle=not bal,
            num_workers=c.NUM_WORKERS, pin_memory=True, drop_last=True,
            worker_init_fn=_seed_worker, generator=gen)
    return ds, loader


# ============================================================
# Stratified val split (§10)
# ============================================================
def stratified_val_split(df, val_ratio=None, seed=None):
    """
    클래스 stratified split → (train_df, val_df).
      - 희소 클래스(<2 샘플)는 전부 train 에 유지 (val 로 떼지 않음).
      - seed(VAL_SEED) 고정 → 모든 ablation 동일 split → 페어 비교 보장.
    """
    val_ratio = c.VAL_RATIO if val_ratio is None else val_ratio
    seed = c.VAL_SEED if seed is None else seed
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for cls, grp in df.groupby('new_target'):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_ratio))
        if len(idx) < 2:
            n_val = 0                                 # 희소 클래스 전부 train
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return df.loc[train_idx], df.loc[val_idx]


# ============================================================
# K-Fold split (프로토콜 §2-1) — exp16
# ============================================================
def make_kfold_split(df, fold_k, k_total=5, seed=42):
    """클래스별 stratified K-Fold → (train_df, val_df). fold_k 가 val, 나머지가 train.
      - 클래스별 독립 분할: 각 클래스 샘플을 seed 로 셔플 후 i%k_total 로 fold 배정
        → 모든 fold 가 동일 클래스 분포(stratified). seed 고정 → 재현·페어비교.
      - 클래스 샘플 수 < k_total 이면 일부 fold 는 그 클래스 val 0개(best-effort stratified).
      - 프로토콜 §2-1 make_kfold_split 과 동일 로직 ('new_target' 라벨 기준).
    """
    df = df.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    fold_assign = np.full(len(df), -1, dtype=int)
    tgt = df['new_target'].values
    for cls in range(c.NUM_CLASSES):                       # 클래스별 독립 분할
        idx = np.where(tgt == cls)[0]
        idx = idx[rng.permutation(len(idx))]
        for i, pos in enumerate(idx):
            fold_assign[pos] = i % k_total
    val_mask = fold_assign == fold_k
    return (df[~val_mask].reset_index(drop=True),           # train (≈80%)
            df[val_mask].reset_index(drop=True))            # val   (≈20%, fold_k)


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate(model, test_df, device, return_detail=False, criterion=None,
             chunk=False, bucket=False, eval_batch=64):
    """평가 방식 3가지:
      chunk=True       : 각 클립을 4s 청크로 분할(train 로딩과 동일)해 청크 단위 평가 — 빠름.
      bucket=True      : 원본 full-clip 을 같은 길이끼리 묶어 배치 — batch=1 과 per-sample 비트 동일(재현),
                         같은 길이 클립이 많으면 빠름(예: D2). DataLoader 유지로 RNG 도 batch=1 과 동일.
      (둘 다 False)    : 원본 full-clip batch=1 — backup 그대로(test 용).
    micro accuracy, criterion 주면 cls val loss(gain=1.0 logit 기준)도 반환."""
    model.eval()
    tta_gains = c.TTA_GAINS if c.USE_TTA else [1.0]
    loss_sum, loss_n = 0.0, 0
    if chunk:
        chunks, labels = [], []
        for _, row in test_df.iterrows():
            lab = int(row['new_target'])
            for seg in chunk_audio(_load_audio(row['full_path'])):   # 전부 4s (CLIP_SAMPLES)
                chunks.append(seg); labels.append(lab)
        labels = np.array(labels)
        preds = np.empty(len(chunks), dtype=np.int64)
        for s in range(0, len(chunks), eval_batch):
            xb = torch.from_numpy(np.stack(chunks[s:s + eval_batch])).float().to(device)
            logits_base, probs_list = None, []
            for g in tta_gains:
                logits, _ = model(xb * g)
                if logits_base is None:
                    logits_base = logits
                probs_list.append(F.softmax(logits, -1))
            preds[s:s + xb.size(0)] = torch.stack(probs_list).mean(0).argmax(-1).cpu().numpy()
            if criterion is not None:
                tgt = torch.from_numpy(labels[s:s + xb.size(0)]).to(device)
                loss_sum += float(criterion(logits_base, tgt)) * xb.size(0)
                loss_n += xb.size(0)
    elif bucket:
        ds = AudioDataset(test_df, is_train=False)
        # 같은 길이끼리 묶는 batch_sampler. 사전 패스는 is_train=False 라 증강 난수 0 → RNG 불변.
        _bylen = {}
        for i in range(len(ds)):
            _bylen.setdefault(ds[i][0].shape[-1], []).append(i)
        _bat = [g[s:s + eval_batch] for g in _bylen.values() for s in range(0, len(g), eval_batch)]
        loader = DataLoader(ds, batch_sampler=_bat, num_workers=c.NUM_WORKERS)
        preds_l, labels_l = [], []
        for audio, target, _, _ in loader:
            audio = audio.float().to(device)
            logits_base, probs_list = None, []
            for g in tta_gains:
                logits, _ = model(audio * g)
                if logits_base is None:
                    logits_base = logits          # cls loss 는 gain=1.0(첫 gain) logit 기준
                probs_list.append(F.softmax(logits, -1))
            preds_l.extend(torch.stack(probs_list).mean(0).argmax(-1).tolist())
            labels_l.extend(target.argmax(-1).tolist())
            if criterion is not None:
                loss_sum += float(criterion(logits_base, target.to(device).argmax(-1))) * audio.size(0)
                loss_n += audio.size(0)
        preds, labels = np.array(preds_l), np.array(labels_l)
    else:
        ds = AudioDataset(test_df, is_train=False)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=c.NUM_WORKERS)
        preds_l, labels_l = [], []
        for audio, target, _, _ in loader:
            audio = audio.float().to(device)
            logits_base, probs_list = None, []
            for g in tta_gains:
                logits, _ = model(audio * g)
                if logits_base is None:
                    logits_base = logits          # cls loss 는 gain=1.0(첫 gain) logit 기준
                probs_list.append(F.softmax(logits, -1))
            preds_l.append(torch.stack(probs_list).mean(0).argmax(-1).item())
            labels_l.append(target.argmax(-1).item())
            if criterion is not None:
                loss_sum += float(criterion(logits_base, target.to(device).argmax(-1)))
                loss_n += 1
        preds, labels = np.array(preds_l), np.array(labels_l)
    micro = float(np.sum(preds == labels) / len(labels) * 100)
    val_loss = (loss_sum / loss_n) if (criterion is not None and loss_n > 0) else None
    if not return_detail:
        return (micro, val_loss) if criterion is not None else micro
    idx_to_label = {v: k for k, v in c.CLASS_LABELS.items()}
    per_class, accs = {}, []
    for cls in np.unique(labels):
        mask = labels == cls
        acc = float(np.sum(preds[mask] == labels[mask]) / mask.sum() * 100)
        per_class[idx_to_label.get(int(cls), str(int(cls)))] = acc
        accs.append(acc)
    detail = {'macro': float(np.mean(accs)) if accs else 0.0, 'per_class': per_class}
    return micro, detail, val_loss
