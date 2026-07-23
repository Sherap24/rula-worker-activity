# Week 6 Handoff — Continue Here

**Project:** Vision-Based Worker Activity and Ergonomic Risk Recognition Using Construction Videos  
**Repo:** `C:\Users\asahi\OneDrive\Desktop\RULA\Project`  
**Data root:** `RULA_DATA_ROOT=C:\Users\asahi\Datasets\RULA`  
**Last updated:** 2026-07-22  
**Status entering Week 6:** Weeks 2–5 complete. Next: ergonomic screening + zone events + end-to-end report.

---

## Prompt for new chat (copy/paste)

```
Continue RULA construction video project at C:\Users\asahi\OneDrive\Desktop\RULA\Project.

Read first:
- WEEK6_HANDOFF.md (this file)
- docs/RESEARCH_SCOPE.md
- TASKS_WEEK4.md, TASKS_WEEK5.md
- docs/joint_mapping.md (Week 6 ergonomic note)

Context:
- Package: src/worker_activity/ (CLI: python -m worker_activity)
- Editable install: .venv + pip install -e ".[dev]"
- Three workstreams MUST stay separate: (A) activity classes, (B) ergonomic screening indicators, (C) zone events
- Test CWPV split is held out — never train or tune on it
- Taxonomy / label_map_cwpv.yaml is provisional, not frozen

Done:
- Week 2–3: CML baseline, MediaPipe pose, inventory
- Week 4: Full CWPV (~2773 videos), subject-disjoint splits, pose features (camera_1), LR/RF baselines (RF ~98% val)
- Week 5: 10 self-filmed phone clips; domain transfer eval — phone acc ~10–20% (domain shift; clips may also be a factor)

Start Week 6:
1. Create TASKS_WEEK6.md
2. Implement ergonomic screening from MediaPipe joint angles (duration/frequency indicators — NOT activity classes)
3. Implement zone-event detection (polygon entry/exit — NOT activity classes)
4. Produce an end-to-end report tying activity + screening + zone outputs
5. Keep scope minimal; match existing module layout; run pytest

Do not merge workstreams. Do not retrain on phone video or CWPV test.
```

---

## Environment

```powershell
cd C:\Users\asahi\OneDrive\Desktop\RULA\Project
$env:RULA_DATA_ROOT = "C:\Users\asahi\Datasets\RULA"
.\.venv\Scripts\Activate.ps1
python -m worker_activity --help
pytest tests/ -q
```

Key paths:

| What | Where |
|------|--------|
| Code | `src/worker_activity/` |
| Configs | `configs/` |
| Reports | `reports/` |
| CWPV features (camera_1) | `data/processed/cwpv/features_train_val.parquet` |
| CWPV models | `outputs/models/` |
| Phone features | `data/processed/local_smartphone/features.parquet` |
| Phone videos | `C:\Users\asahi\Datasets\RULA\local_smartphone\` |
| Domain-transfer report | `reports/domain_transfer_smartphone.md` |

---

## Weeks 4–5 snapshot (for continuity)

### Week 4
- Full CWPV inventory + subject-disjoint train/val/test; leakage audit passed
- Pose → kinematic features; baselines on train, eval on val only
- RF ~0.98 acc / ~0.98 macro-F1 on CWPV validation (`camera_1`)

### Week 5
- Smartphone domain transfer (eval-only)
- Phone accuracy ~0.10–0.20 vs CWPV val ~0.93–0.98
- **Possible clip issues:** short clips, framing, lighting, atypical self-demo motion, single view — investigate later if needed; do not block Week 6

---

## Week 6 goals (from research scope)

Keep these **separate** from activity classification:

### B. Ergonomic / posture screening
Rule/geometry indicators from MediaPipe angles on CWPV (and optionally phone) video:

- repeated bending frequency
- overhead posture duration
- kneeling / squatting duration
- awkward-posture proxies from joint angles

**Not** final ergonomic/safety judgments — screening-level only.

### C. Zone-event detection
- Body-center or foot-point vs configured polygon
- Events: `restricted_zone_entry` / `restricted_zone_exit`
- **Not** an activity class

### Deliverable
End-to-end report showing how activity outputs + screening indicators + zone events fit together without conflating them.

Suggested reuse:

- Pose landmarks: `src/worker_activity/pose/`
- Angles already in: `src/worker_activity/features/pose_skeleton_features.py`
- Joint notes: `docs/joint_mapping.md`
- Viz: `src/worker_activity/viz/`

Suggested new layout (match existing style):

- `src/worker_activity/ergonomics/` — rules + duration/frequency aggregation
- `src/worker_activity/zones/` — polygon config + crossing events
- CLI commands e.g. `screen-ergonomics`, `detect-zone-events`
- Configs under `configs/` (thresholds, polygon vertices)
- Reports under `reports/`

---

## Hard rules for the next agent

1. Do not hard-code absolute user paths in committed code (use `RULA_DATA_ROOT` / configs).
2. Do not use CWPV **test** for training or hyperparameter tuning.
3. Do not treat zone events or ergonomic indicators as activity class labels.
4. Minimize scope; prefer small CLIs + tests + markdown reports.
5. Large data stays outside OneDrive under `RULA_DATA_ROOT`.

---

## Optional follow-ups (after Week 6 core)

- Revisit phone clips / capture protocol if domain transfer needs a second pass
- Multi-camera CWPV feature extraction (Week 4 leftover)
- Week 7 ablations / optional CMA
