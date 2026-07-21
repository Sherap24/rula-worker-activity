# Clip Inventory Schema (Week 2)

Canonical schema for the clip/file inventory produced by `build-inventory`. This describes **files and clips**, not per-frame pose rows (Week 3).

## Design principles

- All paths in manifests are **relative to `RULA_DATA_ROOT`**, never absolute machine paths.
- Nullable fields are expected before datasets are downloaded or labels are mapped.
- Three workstreams remain separate: activity classification, ergonomic screening, zone events.

## Columns

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `source` | string | yes | Canonical source ID (`cml`, `cwpv`, …) from `configs/data_sources.yaml` |
| `source_version` | string | no | Dataset version string when known (e.g. CWPV v3) |
| `source_type` | string | yes | Registry source type (`skeleton`, `video_and_imu`, …) |
| `relative_path` | string | yes | Path relative to `RULA_DATA_ROOT` using forward slashes |
| `file_name` | string | yes | Base file name |
| `extension` | string | yes | Lowercase extension including dot (`.mp4`, `.csv`) |
| `size_bytes` | int | no | File size from filesystem |
| `checksum` | string | no | SHA-256 hex when integrity check succeeds |
| `video_id` | string | no | Dataset-specific video identifier (after inspection) |
| `clip_id` | string | no | Unique clip identifier within project |
| `subject_id` | string | no | Subject identifier for subject-disjoint splits |
| `view_id` | string | no | Camera view (CWPV has four positions) |
| `repetition_id` | string | no | Repetition index (CWPV: 3 repetitions) |
| `raw_activity_label` | string | no | Label string from dataset metadata |
| `canonical_activity` | string | no | Mapped label from `taxonomy_candidate.yaml` |
| `label_mapping_status` | enum | no | `pending_inspection`, `mapped`, `unmapped`, `not_applicable` |
| `fps` | float | no | Frames per second (video only) |
| `frame_count` | int | no | Total frames (video only) |
| `duration_seconds` | float | no | Duration in seconds |
| `width` | int | no | Frame width in pixels |
| `height` | int | no | Frame height in pixels |
| `codec` | string | no | FourCC or codec name when available |
| `metadata_status` | enum | no | `pending`, `extracted`, `failed`, `not_applicable` |
| `integrity_status` | enum | no | `pending`, `ok`, `failed`, `skipped` |
| `notes` | string | no | Free-text diagnostics |

## Enumerations

### `metadata_status`

| Value | Meaning |
|-------|---------|
| `pending` | Video not yet probed |
| `extracted` | Metadata successfully read |
| `failed` | OpenCV/ffprobe could not read file |
| `not_applicable` | Non-video files (e.g. CML skeleton) |

### `integrity_status`

| Value | Meaning |
|-------|---------|
| `pending` | Not checked |
| `ok` | File readable; checksum stored |
| `failed` | Missing, zero-byte, or unreadable |
| `skipped` | Check intentionally skipped |

### `label_mapping_status`

| Value | Meaning |
|-------|---------|
| `pending_inspection` | Default before CML/CWPV labels inspected |
| `mapped` | Raw label mapped to `canonical_activity` |
| `unmapped` | Raw label has no mapping |
| `not_applicable` | Skeleton-only or unlabeled file |

## Output formats

| Format | Path |
|--------|------|
| CSV | `data/manifests/clip_inventory.csv` |
| Parquet | `data/manifests/clip_inventory.parquet` |
| Markdown summary | `reports/dataset_inventory.md` |

## Validation

The `worker_activity.data.schema` module validates column presence and enum values. Run:

```powershell
python -m worker_activity build-inventory
```

## Future schemas (not Week 2)

| Schema | Week | Description |
|--------|------|-------------|
| Per-frame pose rows | 3 | `video_id`, `frame_idx`, keypoints, visibility |
| Movement trajectory | 3 | body-center, speed, heading |
| Activity windows | 4 | sliding-window features and labels |
| Zone events | 6 | polygon ID, crossing timestamp — **not an activity class** |
| Ergonomic indicators | 6 | duration/frequency metrics — **not activity classes** |
