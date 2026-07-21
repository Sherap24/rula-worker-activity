# Dataset Decisions

**Status:** Planning — taxonomy and label mapping are **not frozen**.

## Priority order

| Priority | Source | Role | Status |
|----------|--------|------|--------|
| 1 | CML | Skeleton + activity labels; taxonomy reference | `present_unvalidated` |
| 2 | CWPV | Primary Week 3 video + posture labels | `not_downloaded` (acquisition in progress) |
| 3 | Kaggle construction | Optional; labels unverified | Deferred |
| 4 | CMA | Optional RGB comparison | Deferred |
| 5 | ICON-Pose | Egocentric extension | Deferred |
| 6 | Local smartphone | Domain transfer | Week 5 |

## CML decisions

- Use for **3D skeleton** and activity-label alignment — not for 2D pose estimation.
- Public release: ~6,131 samples; ~4,333 construction-related.
- 15-joint and 20-joint structures documented in dataset and supporting GitHub repo.
- Larger experimental library from the paper is **not fully redistributed** — do not assume completeness.

## CWPV decisions

- Primary **video** source for pose pipeline (Week 3).
- Version 3 (2026-02-18); 21 subjects; 8 tasks/postures; 3 repetitions; 504 video sets; 4 camera views.
- Vision baseline uses **video only**; IMU deferred.
- Requires ~11.6 GB — must live under `RULA_DATA_ROOT`, not OneDrive project folder.

## Deferred sources

### CMA

- GitHub: https://github.com/S1mpleyang/ConstructionActionRecognition
- Baidu Netdisk with extraction code `z7a2`
- Non-commercial research restriction
- Legacy Python 3.6 — not compatible with project 3.12 env without isolation

### ICON-Pose

- Paper-linked egocentric dataset
- Not in core download path for Weeks 2–3

### Kaggle

- Page shows CC0 — still requires local label/subject/split audit
- Must **not** drive taxonomy until inspected

## Taxonomy freeze criteria

`configs/taxonomy_candidate.yaml` remains **candidate** until:

1. CML raw activity labels are inspected locally
2. CWPV posture/task labels are inspected locally
3. Mapping table documents ambiguous or collapsed classes
4. Class imbalance report is generated from real manifests

## Workstream separation (non-negotiable)

| Concept | Type | Example |
|---------|------|---------|
| Activity class | Classification label | `bending_stooping` |
| Ergonomic indicator | Derived metric | repeated bending duration |
| Zone event | Spatial crossing event | `restricted_zone_entry` |

Restricted-zone entry is an **event**, not an activity class.

## Basketball All Net

A separate basketball project exists elsewhere. **This repository does not depend on it.** Generic patterns (metadata, CSV exports) were reimplemented here without importing basketball code.
