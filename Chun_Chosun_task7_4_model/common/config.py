"""
LoRA 공통 설정 — 여러 ablation 실험이 공유하는 코드.

  하이퍼파라미터는 CONFIG 환경변수가 가리키는 config.yaml 에서 읽는다.
    - common 은 공유 코드라 특정 실험 폴더를 모른다.
      실행 주체(각 실험의 train.py)가 자기 폴더 config.yaml 경로를 CONFIG 로 넘긴다.
    - seed 만 train.py --seed 로 런타임 오버라이드 (multi-seed ablation)
  데이터셋 경로(TASK7_*)만 인프라 환경의존이라 env 로 남긴다.

  값은 c.XXX 상수 인터페이스로 노출 → trainer.py/train.py 가 그대로 사용.
"""
import os
import time
import yaml
import pandas as pd


# ============================================================
# KST 시각 유틸 (Asia/Seoul tzdata 없는 컨테이너 대응: POSIX TZ='UTC-9')
# ============================================================
def kst_timestamp(fmt='%Y%m%d-%H%M%S'):
    """현재 시각을 KST(+9) 기준 문자열로 반환. 환경 tzdata 의존성 제거."""
    prev_tz = os.environ.get('TZ')
    os.environ['TZ'] = 'UTC-9'
    time.tzset()
    try:
        return time.strftime(fmt)
    finally:
        if prev_tz is None:
            os.environ.pop('TZ', None)
        else:
            os.environ['TZ'] = prev_tz
        time.tzset()


# ============================================================
# config.yaml 로드 → 모듈 상수
#   common 은 여러 ablation 실험이 공유하는 코드 → 특정 실험 폴더를 모른다.
#   실행 주체(각 실험의 train.py)가 자기 폴더 config.yaml 경로를 CONFIG env 로 넘긴다.
# ============================================================
CONFIG_PATH = os.environ.get('CONFIG', '').strip()
if not CONFIG_PATH:
    raise RuntimeError(
        'CONFIG 환경변수가 설정되지 않았습니다. common/config.py 는 공유 코드라 '
        '실험별 config.yaml 경로를 직접 알 수 없습니다. 실험 폴더의 train.py 를 통해 '
        '실행하거나 CONFIG=/path/to/config.yaml 을 지정하세요.')
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f'config.yaml not found: {CONFIG_PATH}')

with open(CONFIG_PATH) as _f:
    CFG = yaml.safe_load(_f)


def set_seed(seed):
    """train.py --seed 런타임 오버라이드용. SEED 상수를 갱신한다."""
    global SEED
    SEED = int(seed)
    CFG['seed'] = SEED


# ── 식별자 ────────────────────────────────────────────────
MODE = CFG['mode']

# ── 실험 메타데이터 (results/metrics.json 대시보드용) ──────
_exp = CFG.get('exp', {}) or {}
EXP_ID = str(_exp.get('id', '') or '').strip()
EXP_METHOD = str(_exp.get('method', '') or '').strip()
EXP_DATE = str(_exp.get('date', '') or '').strip()
EXP_NOTES = str(_exp.get('notes', '') or '').strip()

# ── 오디오 / mel ──────────────────────────────────────────
_audio = CFG['audio']
SAMPLE_RATE = _audio['sample_rate']
MEL_BINS = _audio['mel_bins']
FMIN = _audio['fmin']
FMAX = _audio['fmax']
WIN_SIZE = _audio['win_size']
HOP_SIZE = _audio['hop_size']
NUM_CLASSES = _audio['num_classes']
CLIP_SAMPLES = SAMPLE_RATE * int(CFG['chunk']['seconds'])

# 인덱스 순서는 baseline(dict_class_labels) / D1 체크포인트 fc 와 정합해야 한다.
#   baseline 단축명: alarm,baby,dog,engine,fire,footsteps,knock,phone,piano,speech
#   CSV/코드 장음명: alarm,baby_cry,dog_bark,...,telephone_ringing(=phone),piano,speech
#   → 인덱스 동일. CSV 의 'class' 가 장음명이므로 키는 장음명, 인덱스는 baseline 순서.
#     (telephone_ringing=7, piano=8, speech=9)
CLASS_LABELS = {
    'alarm': 0, 'baby_cry': 1, 'dog_bark': 2, 'engine': 3, 'fire': 4,
    'footsteps': 5, 'knocking': 6, 'telephone_ringing': 7, 'piano': 8, 'speech': 9,
}

# ── 재현성 / val split ────────────────────────────────────
SEED = CFG['seed']
VAL_SEED = CFG['val_seed']
VAL_RATIO = CFG['val_ratio']

# ── LoRA ──────────────────────────────────────────────────
_lora = CFG['lora']
D2_LORA_RANK = _lora['d2_rank']
D3_LORA_RANK = _lora['d3_rank']
LORA_ALPHA = _lora['alpha']

# ── 학습 공통 ─────────────────────────────────────────────
_train = CFG['train']
BATCH_SIZE = _train['batch_size']
WEIGHT_DECAY = _train['weight_decay']
LR_D2 = _train['lr_d2']
LR_D3 = _train['lr_d3']
EPOCHS_OVERRIDE = _train['epochs_override']     # None=미사용 (smoke test 단일 오버라이드)
D2_EPOCHS = EPOCHS_OVERRIDE if EPOCHS_OVERRIDE else _train['d2_epochs']
D3_EPOCHS = EPOCHS_OVERRIDE if EPOCHS_OVERRIDE else _train['d3_epochs']
GRAD_CLIP = _train['grad_clip']
NUM_WORKERS = _train['num_workers']
LR_ETA_MIN = _train['lr_eta_min']
LOG_EVERY = _train['log_every']

# ── Early stopping ────────────────────────────────────────
_es = CFG['early_stop']
D2_PATIENCE = _es['d2_patience']
D3_PATIENCE = _es['d3_patience']
D2_MIN_EPOCHS = _es['d2_min_epochs']
D3_MIN_EPOCHS = _es['d3_min_epochs']
MIN_DELTA = _es['min_delta']

# ── KD (phase별 teacher 설정 dict) ────────────────────────
KD_D1 = CFG['kd']['d1']     # D2 phase: teacher=D1 (logit-only)
KD_D2 = CFG['kd']['d2']     # D3 phase: teacher=D2 (cosine feature KD)

# ── PaCE ──────────────────────────────────────────────────
_pace = CFG['pace']
P_SVD_D2 = _pace['p_svd_d2']
P_SVD_D3 = _pace['p_svd_d3']
SVD_EPS = _pace['svd_eps']

# ── 학습 트릭 ─────────────────────────────────────────────
_focal = CFG['focal']
USE_FOCAL_LOSS = _focal['use']
FOCAL_GAMMA = _focal['gamma']
LABEL_SMOOTHING = _focal['label_smoothing']

_mixup = CFG['mixup']
USE_MIXUP = _mixup['use']
MIXUP_ALPHA = _mixup['alpha']
MIXUP_PROB = _mixup['prob']

USE_SPEC_AUG = CFG['spec_aug']['use']

_tta = CFG['tta']
USE_TTA = _tta['use']
TTA_GAINS = _tta['gains']

_gain = CFG['gain_aug']
GAIN_AUG_LOW = _gain['low']
GAIN_AUG_HIGH = _gain['high']

DROPOUT = CFG['dropout']

# ── Sampling ──────────────────────────────────────────────
_samp = CFG['sampling']
USE_BALANCED_SAMPLING = _samp['use_balanced']
HARD_CLASSES = _samp['hard_classes']
HARD_CLASS_MULTIPLIER = _samp['hard_class_multiplier']

# ── Audio chunk 분할 ──────────────────────────────────────
_chunk = CFG['chunk']
CHUNK_SECONDS = _chunk['seconds']
CHUNK_MIN_SECONDS = _chunk['min_seconds']

# ── D1 체크포인트 경로 — env(TASK7_D1_CKPT) > config(d1_ckpt).
#   상대경로면 실험 루트(exp17_KFoldCoForge/) 기준 해석 → 패키지에 번들된 D1_dictionary/ 사용 가능 (이식성)
_EXP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
D1_CKPT = (os.environ.get('TASK7_D1_CKPT') or str(CFG.get('d1_ckpt', '') or '').strip())
if D1_CKPT and not os.path.isabs(D1_CKPT):
    D1_CKPT = os.path.join(_EXP_ROOT, D1_CKPT)


# ============================================================
# 데이터셋 메타 (경로만 인프라 env 의존)
# ============================================================
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# 데이터 루트 해석 우선순위: env(TASK7_LORA_DATASET_DIR) > config.yaml(data_dir) > 폴백.
#   팀 SETUP 컨벤션은 config.yaml 에 경로를 둔다 → data_dir 키를 우선 읽고, env 로 오버라이드 허용.
DATASET_DIR = (os.environ.get('TASK7_LORA_DATASET_DIR')
               or str(CFG.get('data_dir', '') or '').strip()
               or '/data2/ejkim/DCASE2026/datasets')
LOCAL_TASK7_DATA_DIR = os.environ.get('TASK7_DATA_DIR', os.path.join(REPO_ROOT, 'task7_data'))


def _build_meta():
    metadata_dir = os.path.join(DATASET_DIR, 'metadata')
    if not os.path.exists(os.path.join(metadata_dir, 'd2-dev-train.csv')):
        train_path = os.path.join(LOCAL_TASK7_DATA_DIR, 'evaluation_setup', 'development_train.txt')
        test_path = os.path.join(LOCAL_TASK7_DATA_DIR, 'evaluation_setup', 'development_test.txt')
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(
                'Dataset metadata not found. Set TASK7_LORA_DATASET_DIR for metadata/*.csv '
                'or TASK7_DATA_DIR for evaluation_setup/development_*.txt.'
            )
        cols = ['filename', 'target', 'domain', 'new_target']
        df_train = pd.read_csv(train_path, sep='\t', names=cols)
        df_test = pd.read_csv(test_path, sep='\t', names=cols)
        for df in (df_train, df_test):
            df['full_path'] = df['filename'].apply(lambda f: os.path.join(LOCAL_TASK7_DATA_DIR, f))
        return df_train, df_test

    splits = {
        'train': {'D2': 'd2-dev-train.csv', 'D3': 'd3-dev-train.csv'},
        'test':  {'D2': 'd2-dev-test.csv',  'D3': 'd3-dev-test.csv'},
    }
    audio_dirs = {
        'train': {'D2': 'd2-dev-train', 'D3': 'd3-dev-train'},
        'test':  {'D2': 'd2-dev-test',  'D3': 'd3-dev-test'},
    }
    frames_tr, frames_te = [], []
    for split, domains in splits.items():
        for dom, fn in domains.items():
            df = pd.read_csv(os.path.join(DATASET_DIR, 'metadata', fn))
            df['domain'] = dom
            df['new_target'] = df['class'].map(CLASS_LABELS)
            adir = os.path.join(DATASET_DIR, audio_dirs[split][dom])
            df['full_path'] = df['filename'].apply(lambda f: os.path.join(adir, f))
            df = df.rename(columns={'class': 'target'})
            (frames_tr if split == 'train' else frames_te).append(df)
    return pd.concat(frames_tr, ignore_index=True), pd.concat(frames_te, ignore_index=True)

DF_TRAIN, DF_TEST = _build_meta()
DOMAIN_TO_IDX = {'D1': 0, 'D2': 1, 'D3': 2}
