# Chun_Chosun_task7_4 — OR-KDL CoLoRA Selective Augment (OR-KDL-CLS)

DCASE 2026 Challenge **Task 7**, System 4 (Chosun University).

Selective-augmentation variant of System 3: covariance protection (**CovForget**)
with forgetting regularization, orthogonal gradient projection, null-space LoRA,
distillation, and k-fold ensembling. Gain augmentation is disabled for the
`baby_cry` and `telephone_ringing` classes.

## Model checkpoints

Weights are not stored in this repo due to size. Download from Google Drive:

**https://drive.google.com/drive/folders/1RZOGw4nUFIsj8NDBRuKDV8oiI3SBQH4D**

Use the archive **`Chun_Chosun_task7_4_checkpoints.tar.gz`** (30 checkpoints:
D2 dictionary 5-fold + D3 dictionary 25). Extract into the repo root:

```bash
tar -xzf Chun_Chosun_task7_4_checkpoints.tar.gz -C .
```

This restores `Chun_Chosun_task7_4_D2_dictionary/` and `Chun_Chosun_task7_4_D3_dictionary/`.
