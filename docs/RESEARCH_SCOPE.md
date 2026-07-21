# Research Scope

**Project:** Vision-Based Worker Activity and Ergonomic Risk Recognition Using Construction Videos  
**Duration:** 8 weeks (~20 hours/week)

## Research question

Can an interpretable pose-and-movement pipeline classify construction worker activities and produce screening-level ergonomic indicators from multi-view site video, with documented domain-transfer limitations?

## Three workstreams (keep separate)

### A. Activity recognition

Frame- or window-level classification among candidate classes:

- standing / idle
- walking
- bending / stooping
- lifting / lowering
- carrying
- kneeling
- squatting
- overhead work / reaching
- other
- unknown (low confidence)

### B. Ergonomic / posture screening

Rule-based or geometry-based **indicators**, not activity classes:

- repeated bending frequency
- overhead posture duration
- kneeling / squatting duration
- awkward-posture proxies from joint angles

**Scope limit:** Screening indicators are not final ergonomic or safety judgments.

### C. Zone-event detection

- Body-center or foot-point crossing a configured polygon
- `restricted_zone_entry` / `restricted_zone_exit` events
- **Not** an activity class

## Timeline alignment

| Week | Focus |
|------|-------|
| 1 | Scoping, literature, feasibility |
| 2 | Data schema, inventory, environment (**current foundation**) |
| 3 | Pose extraction, movement features, annotated samples |
| 4 | Sliding-window baseline classifiers |
| 5 | Local smartphone domain transfer |
| 6 | Ergonomic rules + zone events + end-to-end report |
| 7 | Refinement, ablations, optional CMA comparison |
| 8 | Final validation, documentation, manuscript |

## Out of scope (this phase)

- Model training (Week 4+)
- Automatic multi-GB dataset downloads
- IMU fusion (CWPV IMU deferred)
- 2D pose estimation on CML skeleton data
- Basketball-specific logic
- Transformer / large RGB video models (stretch)
- Multi-worker tracking (stretch)

## Ethical and safety limitations

- Outputs are research prototypes for activity recognition and **screening-level** posture indicators.
- Do not present model outputs as certified safety compliance or medical ergonomic assessment.
- Local smartphone capture requires consent protocol (Week 5).
- Respect dataset licenses (CC BY 4.0 for CML/CWPV; verify others before use).

## Reproducibility commitments

- Pinned dependencies in `pyproject.toml`
- External data via `RULA_DATA_ROOT`
- Manifest-driven processing
- Subject-disjoint splits when `subject_id` is available
- Experiment configs and reports under `reports/` and `outputs/`
