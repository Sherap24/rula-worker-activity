# Week 4 Implementation Plan

**Project:** Vision-Based Worker Activity and Ergonomic Risk Recognition Using Construction Videos  
**Last updated:** 2026-07-14  
**Status:** Week 4 camera_1 baseline complete (features + classifiers). Multi-camera expansion optional.

---

## Week 4 — Items

### CWPV data completion

- [x] **Re-download CWPV** — archive size matches Figshare; `7z t` passed
- [x] **Extract full archive** — 2773 videos under `cwpv/extracted/`
- [x] **Rebuild inventory** — 23 701 rows total; **2773** CWPV videos
- [x] **Inspect CWPV labels** — 21 subjects, 8 motions; provisional map in `configs/label_map_cwpv.yaml`

### CWPV baseline splits

- [x] `build-cwpv-baseline` CLI — subject-disjoint splits, manifests, leakage audit
- [x] `configs/label_map_cwpv.yaml` applied (provisional, not frozen)
- [x] Full-dataset splits — **693** logical samples; train/val/test **1848 / 396 / 529** videos
- [x] Filename parser supports 4-digit README and 5-digit `PP+M+block+T` extracted names

### Video pose features

- [x] `pose_skeleton_features.py` — joint angles, hip kinematics, sliding windows
- [x] `extract-cwpv-features` CLI — train/validation manifests only (`--view`, progress logs)
- [x] `configs/cwpv_pose_features.yaml` — `frame_stride: 2`
- [x] Feature extraction on train/val **`camera_1`** — 561 rows (462 train + 99 val), 0 errors
- [ ] Optional: extract remaining cameras (`camera_2`–`camera_4`)

### Baseline classifiers

- [x] `train-cwpv-baseline` CLI — LogisticRegression + RandomForest
- [x] Train on `train` only; evaluate on `validation` only; **test held out**
- [x] End-to-end run on `camera_1` CWPV features
  - Logistic Regression: **acc 0.929 / macro-F1 0.906**
  - Random Forest: **acc 0.980 / macro-F1 0.976**

---

## Week 4 — Acceptance criteria

- [x] Full CWPV inventory (~2000+ videos; actual **2773**)
- [x] CWPV train/val/test manifests with leakage audit passed
- [x] Pose features extracted for train + validation (`camera_1`)
- [x] Baseline classifier report on validation split
- [x] Test split never used for training or hyperparameter tuning

---

## Commands

```powershell
$env:RULA_DATA_ROOT = "C:\Users\asahi\Datasets\RULA"
python -m worker_activity extract-cwpv-features --view camera_1
python -m worker_activity train-cwpv-baseline
# Optional: all cameras
# python -m worker_activity extract-cwpv-features
```
