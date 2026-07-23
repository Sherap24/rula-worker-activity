# Week 6 Implementation Plan

**Project:** Vision-Based Worker Activity and Ergonomic Risk Recognition Using Construction Videos  
**Last updated:** 2026-07-22  
**Status:** Week 6 complete — ergonomic screening + zone events + end-to-end report.

---

## Week 6 — Goal

Produce **screening-level** ergonomic indicators and **restricted-zone entry/exit events** from MediaPipe pose on phone and CWPV video, then an end-to-end report that keeps workstreams separate:

| Stream | Output |
|--------|--------|
| **A Activity** | Existing CWPV/phone classifier predictions (no retrain) |
| **B Ergonomics** | Duration/frequency indicators from joint angles |
| **C Zones** | Polygon entry/exit events (demo YAML polygons) |

CWPV **test** split stays held out. Zone events and ergonomic indicators are **not** activity classes.

---

## Week 6 — Items

### Task tracker / docs

- [x] Create this file (`TASKS_WEEK6.md`)
- [x] Configs: `configs/ergonomics_screening.yaml`, `configs/zones_demo.yaml`

### Ergonomic screening (B)

- [x] `src/worker_activity/ergonomics/` — rules + duration/frequency aggregation
- [x] CLI: `screen-ergonomics`
- [x] Run on all 10 smartphone clips + small CWPV `camera_1` sample

### Zone events (C)

- [x] `src/worker_activity/zones/` — point-in-polygon + entry/exit
- [x] Demo polygons in YAML (normalized image coords; no real site survey required)
- [x] CLI: `detect-zone-events`

### End-to-end report

- [x] CLI: `build-week6-report`
- [x] `reports/week6_end_to_end.md` with separate A / B / C sections

### Tests

- [x] Unit tests for screening rules and polygon entry/exit
- [x] `pytest tests/ -q` (75 passed)

### Acceptance criteria

- [x] Ergonomic indicators (bending frequency, overhead/kneel/squat duration, awkward proxy) from pose
- [x] Zone entry/exit events from body-center vs config polygon
- [x] Phone + CWPV sample covered in report
- [x] Streams not conflated; CWPV test unused; no hard-coded absolute paths

### Results (2026-07-22)

| Pipeline | Videos | Notes |
|----------|-------:|-------|
| Ergonomic screening | 15 | 10 phone + 5 CWPV `camera_1` sample |
| Zone events | 15 | 49 entry/exit events vs demo polygon |
| Tests | 75 | Including 7 Week 6 unit tests |

Artifacts: `reports/week6_end_to_end.md`, `reports/ergonomic_screening.md`, `reports/zone_events.md`, `data/processed/week6/*.csv`.

---

## Commands

```powershell
$env:RULA_DATA_ROOT = "C:\Users\asahi\Datasets\RULA"
.\.venv\Scripts\Activate.ps1

python -m worker_activity screen-ergonomics
python -m worker_activity detect-zone-events
python -m worker_activity build-week6-report
pytest tests/ -q
```

Optional limits:

```powershell
python -m worker_activity screen-ergonomics --source phone --max-videos 2
python -m worker_activity detect-zone-events --source cwpv --max-videos 4
python -m worker_activity build-week6-report --skip-pipelines
```
