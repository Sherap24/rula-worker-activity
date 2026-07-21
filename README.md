# Vision-Based Worker Activity and Ergonomic Risk Recognition

Research software for classifying construction worker activities and producing screening-level ergonomic indicators from site video. **Week 2 data foundation is complete**; Week 3 adds CWPV video pose extraction via MediaPipe.

## Scope

Three **separate** workstreams:

| Stream | Description |
|--------|-------------|
| **Activity recognition** | walking, bending, lifting, carrying, kneeling, squatting, overhead work, standing/idle, unknown |
| **Ergonomic screening** | posture duration/frequency indicators — not activity classes |
| **Zone events** | restricted-area polygon crossings — events, not classes |

## Important: data location

The project folder is in **OneDrive**. Large datasets must **not** be stored here.

Set an external data root:

```powershell
$env:RULA_DATA_ROOT = "C:\Users\asahi\Datasets\RULA"
```

Or copy `.env.example` to `.env` (see [docs/DATA_ACQUISITION.md](docs/DATA_ACQUISITION.md)).

| Dataset | Size (approx.) | Role |
|---------|----------------|------|
| [CML](https://doi.org/10.6084/m9.figshare.20480787) | 1.2 GB | 3D skeleton + activity labels |
| [CWPV](https://doi.org/10.6084/m9.figshare.27907818) | 11.6 GB | Primary Week 3 video source |

## Setup

Requires **Python 3.10–3.12**.

```powershell
cd C:\Users\asahi\OneDrive\Desktop\RULA\Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item configs\paths.example.yaml configs\paths.yaml
Copy-Item .env.example .env
```

## CLI commands

```powershell
python -m worker_activity show-sources
python -m worker_activity audit-environment
python -m worker_activity validate-config
python -m worker_activity audit-data
python -m worker_activity build-inventory
python -m worker_activity build-cml-baseline
python -m worker_activity extract-cml-features
python -m worker_activity inspect-cwpv
python -m worker_activity extract-pose --subject 09 --motion 3 --max-videos 2
python -m worker_activity render-annotated --video "PATH\to\video.avi"
python -m worker_activity build-cwpv-baseline
python -m worker_activity extract-cwpv-features
python -m worker_activity train-cwpv-baseline
```

| Command | Purpose |
|---------|---------|
| `show-sources` | List registered datasets, licenses, priorities |
| `audit-environment` | Verify Python packages and tooling |
| `validate-config` | Check YAML configs |
| `audit-data` | Inspect external data directories |
| `build-inventory` | Build clip inventory CSV/Parquet |
| `build-cml-baseline` | CML mapping, splits, leakage audit, manifests |
| `extract-cml-features` | CML skeleton kinematic features (train only, 15/20-node) |
| `inspect-cwpv` | CWPV label inspection and class-balance reports |
| `extract-pose` | MediaPipe pose on CWPV video subset |
| `render-annotated` | Draw pose landmarks on a video |
| `build-cwpv-baseline` | CWPV mapping, subject-disjoint splits, manifests |
| `extract-cwpv-features` | Pose + kinematic features (train/val only) |
| `train-cwpv-baseline` | Baseline classifiers (train on train, eval on val) |

### CWPV download (manual / script)

```powershell
python scripts/download_cwpv.py --data-root C:\Users\asahi\Datasets\RULA
# If RAR is corrupt, force a clean re-download:
python scripts/download_cwpv.py --data-root C:\Users\asahi\Datasets\RULA --force-download
python -m worker_activity build-inventory
python -m worker_activity build-cwpv-baseline
```

Requires [7-Zip](https://www.7-zip.org/) for RAR extraction.

## Testing

```powershell
pytest tests/ -v
```

## Project layout

```
Project/
├── configs/                # data_sources, taxonomy, splits, paths, pose
├── docs/                   # acquisition, schema, scope, joint_mapping
├── scripts/                # download_cwpv.py
├── src/worker_activity/    # Python package (data, pose, features, viz)
├── tests/
├── data/manifests/         # Generated inventories (gitignored)
├── data/processed/         # Features and pose outputs (gitignored)
└── reports/                # Audit and validation reports
```

## Documentation

- [Data acquisition](docs/DATA_ACQUISITION.md)
- [Data schema](docs/DATA_SCHEMA.md)
- [Dataset decisions](docs/DATASET_DECISIONS.md)
- [Research scope](docs/RESEARCH_SCOPE.md)
- [Joint mapping (CML ↔ MediaPipe)](docs/joint_mapping.md)
- [Week 2/3 tasks](TASKS_WEEK2_WEEK3.md)
- [Week 4 tasks](TASKS_WEEK4.md)

## Limitations

- Outputs are **research prototypes**, not certified safety or ergonomic assessments.
- Taxonomy is **not frozen** until CWPV labels are inspected locally.
- FFmpeg is optional; video metadata uses **OpenCV** by default.
- This project does **not** depend on any external basketball codebase.

## License

Project code: MIT. Dataset licenses vary — see `configs/data_sources.yaml`.