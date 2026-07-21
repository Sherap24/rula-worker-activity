# Week 2 & Week 3 Implementation Plan

**Project:** Vision-Based Worker Activity and Ergonomic Risk Recognition Using Construction Videos  
**Last updated:** 2026-07-06  
**Status:** Week 2 complete; CML skeleton features done; Week 3 CWPV pose pipeline complete; Week 4 baseline started.

Workstreams (keep separate in design and outputs):

| Stream | Scope |
|--------|--------|
| **A. Activity recognition** | walking, bending, lifting/lowering, carrying, kneeling, squatting, overhead work/reaching, standing/idle, unknown |
| **B. Ergonomic screening** | joint geometry, repeated bending, overhead duration, kneeling/squatting duration, awkward-posture proxies |
| **C. Zone events** | body-center or foot-point crossing a configured polygon; restricted-zone entry is an **event**, not an activity class |

---

## Week 2 — Mandatory items

### Repository & environment

- [x] Initialize git repository with `.gitignore`
- [x] Add `README.md` with setup, data placement, and ethical/safety scope statement
- [x] Create `pyproject.toml` pinned for reproducible install
- [x] Create project-local `.venv` and document activation for Windows
- [x] Decide pose baseline for Week 3: **MediaPipe Pose** (venv 0.10.35)
- [x] Document OpenCV-only metadata fallback in README
- [x] Add `configs/paths.yaml` and `.env` template for `RULA_DATA_ROOT`

### Data acquisition & inventory

- [x] Document official CWPV/CML URLs, DOIs, licenses in `configs/data_sources.yaml`
- [x] Document external data layout via `RULA_DATA_ROOT` and `docs/DATA_ACQUISITION.md`
- [x] **CML downloaded and extracted** to `RULA_DATA_ROOT\cml\extracted\`
- [ ] **CWPV download and extract** — partial extract only (74/~2000 videos); RAR corrupt, re-download needed
- [x] Run integrity checks on CML files via `build-inventory` (20,928 JSON, 0 failures)
- [x] Produce CML inventory: `data/manifests/clip_inventory.csv` + `reports/dataset_inventory.md`

### Schema & label mapping

- [x] Define clip inventory schema in `docs/DATA_SCHEMA.md` and `src/worker_activity/data/schema.py`
- [x] Create **candidate** taxonomy in `configs/taxonomy_candidate.yaml` (not frozen)
- [x] Inspect real CML labels — `reports/cml_label_inspection.md`
- [x] Draft CML label map: `configs/label_map_cml.yaml` (user-confirmed for CML work)
- [x] Draft CWPV label map: `configs/label_map_cwpv.yaml` (provisional, from README)
- [ ] Inspect CWPV inventory labels after full extract (74 partial; 5-digit filename parser fixed — re-run `build-inventory`)

### Preprocessing & manifests

- [x] Implement video metadata extraction (`src/worker_activity/video/metadata.py`)
- [x] Implement clip/file indexing via `build-inventory`
- [x] Implement subject-disjoint split logic (`src/worker_activity/data/splits.py`) with tests
- [x] Emit CML train/val/test manifests via `build-cml-baseline`
- [x] Class-balance report for CML (`reports/class_balance_cml.md`)
- [x] Data-quality report for CML (`reports/data_quality.md`)

### CLI & tests (Week 2 foundation)

- [x] `show-sources`, `audit-environment`, `audit-data`, `validate-config`, `build-inventory`
- [x] `build-cml-baseline`, `extract-cml-features`, `inspect-cwpv`
- [x] Unit tests: config, registry, schema, metadata, inventory, splits, CML baseline, CWPV metadata, pose (53+ tests)

---

## Week 2 — Acceptance criteria

- [ ] First git commit (optional — repo initialized)
- [x] Fresh venv + `pip install -e ".[dev]"` runs without errors
- [x] CLI commands run without stack traces when data absent
- [x] Feasible data sources documented with verified URLs/DOIs
- [x] Standard schema documented and reflected in manifest CSV headers
- [x] Subject-aware split logic implemented and tested
- [ ] Baseline class set **frozen** (blocked on CWPV label inspection)
- [x] CML dataset inventory, class-balance, and data-quality reports from real data
- [x] No model training code
- [x] No basketball-specific logic in `src/`

---

## Week 3 — Starter items

- [x] Create `src/worker_activity/pose/` package (MediaPipe estimator + pipeline)
- [x] Create `src/worker_activity/features/` package (CML skeleton features)
- [x] Create `src/worker_activity/viz/` package (annotated video rendering)
- [x] Config: `configs/pose_mediapipe.yaml`
- [x] Document joint topology in `docs/joint_mapping.md`
- [x] `extract-pose` and `render-annotated` CLI commands
- [x] Run pose extraction on CWPV subset (4 real videos, subject 01; sample synthetic video)

---

## Week 3 — Acceptance criteria

- [x] CWPV videos inventoried with parsed subject/motion/camera fields (74 partial; re-run `build-inventory` after parser fix)
- [x] Pose landmarks extracted for a small CWPV subset (`data/processed/cwpv/pose/`)
- [x] At least one annotated sample video (`outputs/cwpv/annotated/0111Camera_1_annotated.mp4`)
- [x] CWPV label inspection and class-balance reports generated
- [ ] Full CWPV inventory (~2000 videos) — **re-download in progress** (corrupt RAR confirmed)

---

## Week 4 — Started (see TASKS_WEEK4.md)

- [x] `build-cwpv-baseline` — subject-disjoint splits, manifests, leakage audit
- [x] `extract-cwpv-features` — pose + sliding-window kinematic features (train/val only)
- [x] `train-cwpv-baseline` — LogisticRegression + RandomForest (no test training)
- [ ] Full CWPV extract + end-to-end Week 4 run

---

## Intentionally postponed

| Item | Target |
|------|--------|
| Sliding-window classifiers | Week 4 |
| Ergonomic rules + zone events | Week 6 |
| CMA / Kaggle / ICON-Pose | After core inspection |
| Local smartphone capture | Week 5 |
| Model training | Week 4+ |
| Auto multi-GB download without user approval | By design |

---

## Data or decisions requiring user intervention

| # | Action | Status |
|---|--------|--------|
| 1 | CML at `RULA_DATA_ROOT\cml\` | **Done** (`C:\Users\asahi\Datasets\RULA`) |
| 2 | CWPV download + extract | **In progress** |
| 3 | Set `RULA_DATA_ROOT` persistently | **`.env` created** — set Windows user env for new shells |
| 4 | Review CML label map → freeze taxonomy | **CML confirmed; CWPV pending** |
| 5 | Install FFmpeg (optional) | Optional — OpenCV works |
