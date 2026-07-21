# Week 5 Implementation Plan

**Project:** Vision-Based Worker Activity and Ergonomic Risk Recognition Using Construction Videos  
**Last updated:** 2026-07-21  
**Status:** Week 5 domain-transfer pipeline complete (10 phone clips evaluated).

---

## Week 5 — Goal

Evaluate how well **CWPV-trained** pose-feature baselines transfer to **local smartphone video** of the same five activity classes. Phone clips are **evaluation-only** (no retraining). CWPV **test** split stays held out.

---

## Week 5 — Items

### Capture / data

- [x] Self-filmed clips under `RULA_DATA_ROOT/local_smartphone/` (10 videos)
- [x] Filename convention: `{canonical_activity}_{nn}.mp4`
- [x] Enable `local_smartphone` in `configs/data_sources.yaml`

### Pipeline

- [x] Parse smartphone filenames → inventory labels
- [x] Persist CWPV baseline models (`outputs/models/`)
- [x] `extract-smartphone-features` CLI
- [x] `evaluate-domain-transfer` CLI + report

### Acceptance criteria

- [x] Inventory includes smartphone videos with canonical labels
- [x] Pose features extracted for phone clips (10/10)
- [x] Domain-transfer metrics vs CWPV validation baseline
- [x] CWPV test never used

### Results (2026-07-21)

| Model | CWPV val acc | Phone acc | CWPV val F1 | Phone F1 |
| --- | ---: | ---: | ---: | ---: |
| logistic_regression | 0.929 | 0.100 | 0.906 | 0.067 |
| random_forest | 0.980 | 0.200 | 0.976 | 0.114 |

Large drop confirms domain shift (phone vs CWPV multi-view). Report: `reports/domain_transfer_smartphone.md`.

---

## Commands

```powershell
$env:RULA_DATA_ROOT = "C:\Users\asahi\Datasets\RULA"
python -m worker_activity build-inventory
python -m worker_activity train-cwpv-baseline
python -m worker_activity extract-smartphone-features
python -m worker_activity evaluate-domain-transfer
```
