"""Smartphone pose feature extraction pipeline (Week 5 domain transfer)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from worker_activity.config import find_repo_root, load_yaml, resolve_paths_config
from worker_activity.features.pose_skeleton_features import extract_video_pose_features
from worker_activity.pose.mediapipe_estimator import MediaPipePoseEstimator
from worker_activity.pose.pipeline import load_pose_config


@dataclass
class SmartphoneFeatureExtractionResult:
    features: pd.DataFrame
    output_path: Path
    report_path: Path
    pose_outputs: list[Path] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)


METADATA_COLUMNS = [
    "relative_path",
    "file_name",
    "canonical_activity",
    "raw_activity_label",
    "subject_id",
    "clip_id",
    "video_id",
    "frame_count",
    "fps",
]


def load_smartphone_feature_config(path: Path | None = None) -> dict[str, Any]:
    root = find_repo_root()
    cfg_path = path or root / "configs" / "smartphone_pose_features.yaml"
    if not cfg_path.is_file():
        return {
            "sliding_window": {"window_size": 30, "stride": 15},
            "extraction": {"frame_stride": 2},
        }
    return load_yaml(cfg_path)


def _pose_output_path(pose_dir: Path, video_path: Path) -> Path:
    return pose_dir / f"{video_path.stem}_pose.parquet"


def _load_smartphone_inventory(paths) -> pd.DataFrame:
    inv_path = paths.manifests_dir / "clip_inventory.csv"
    if not inv_path.is_file():
        raise FileNotFoundError(
            f"Inventory not found: {inv_path}. Run build-inventory first."
        )
    df = pd.read_csv(inv_path, low_memory=False)
    phone = df[df["source"] == "local_smartphone"].copy()
    if phone.empty:
        raise FileNotFoundError(
            "No local_smartphone rows in inventory. "
            "Place videos under RULA_DATA_ROOT/local_smartphone and re-run build-inventory."
        )
    return phone


def extract_smartphone_features(
    *,
    max_videos: int | None = None,
    skip_pose_if_exists: bool = True,
    frame_stride: int | None = None,
) -> SmartphoneFeatureExtractionResult:
    """Extract MediaPipe pose + kinematic features for smartphone clips."""
    root = find_repo_root()
    paths = resolve_paths_config(repo_root=root, require_data_root=True)
    assert paths.data_root is not None

    feature_cfg = load_smartphone_feature_config()
    sw_cfg = feature_cfg.get("sliding_window", {})
    window_size = int(sw_cfg.get("window_size", 30))
    stride = int(sw_cfg.get("stride", 15))
    if frame_stride is None:
        frame_stride = int(feature_cfg.get("extraction", {}).get("frame_stride", 2))

    pose_dir = paths.processed_dir / "local_smartphone" / "pose"
    pose_dir.mkdir(parents=True, exist_ok=True)

    inventory = _load_smartphone_inventory(paths)
    if max_videos is not None:
        inventory = inventory.head(max_videos)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    pose_outputs: list[Path] = []
    pose_cfg = load_pose_config()
    total = len(inventory)
    print(f"Extracting smartphone features: {total} videos (frame_stride={frame_stride})", flush=True)

    with MediaPipePoseEstimator(pose_cfg) as estimator:
        for i, (_, row) in enumerate(inventory.iterrows(), start=1):
            rel = str(row["relative_path"])
            video_path = paths.data_root / rel.replace("/", "\\")
            if not video_path.is_file():
                errors.append({"relative_path": rel, "error": "file_not_found"})
                continue

            pose_path = _pose_output_path(pose_dir, video_path)
            try:
                if skip_pose_if_exists and pose_path.is_file():
                    pose_df = pd.read_parquet(pose_path)
                    reused = True
                else:
                    sequence = estimator.extract_from_video(
                        video_path,
                        relative_path=rel,
                        fps=row.get("fps") if pd.notna(row.get("fps")) else None,
                        frame_stride=frame_stride,
                    )
                    pose_df = pd.DataFrame(sequence.to_records())
                    pose_df.to_parquet(pose_path, index=False)
                    reused = False
                pose_outputs.append(pose_path)

                feats = extract_video_pose_features(
                    pose_df,
                    window_size=window_size,
                    stride=stride,
                )
                feat_row: dict[str, Any] = {col: row.get(col) for col in METADATA_COLUMNS}
                feat_row["source"] = "local_smartphone"
                feat_row["domain"] = "smartphone"
                feat_row["split"] = "domain_transfer"
                feat_row.update(feats)
                feat_row["extraction_status"] = "ok"
                feat_row["pose_parquet"] = str(pose_path)
                rows.append(feat_row)
                print(
                    f"  [{i}/{total}] {'reuse' if reused else 'extract'} {video_path.name}",
                    flush=True,
                )
            except OSError as exc:
                errors.append({"relative_path": rel, "error": str(exc)})
                print(f"  ERROR {rel}: {exc}", flush=True)

    features_df = pd.DataFrame(rows)
    out_cfg = feature_cfg.get("output", {})
    output_path = root / out_cfg.get(
        "features",
        "data/processed/local_smartphone/features.parquet",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False)

    report_path = root / out_cfg.get(
        "report",
        "reports/smartphone_pose_feature_extraction.md",
    )
    _write_report(features_df, errors, report_path, window_size, stride, frame_stride)

    return SmartphoneFeatureExtractionResult(
        features=features_df,
        output_path=output_path,
        report_path=report_path,
        pose_outputs=pose_outputs,
        errors=errors,
    )


def _write_report(
    df: pd.DataFrame,
    errors: list[dict[str, str]],
    path: Path,
    window_size: int,
    stride: int,
    frame_stride: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Smartphone Pose Feature Extraction",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_",
        "",
        "**Scope:** Local smartphone clips for Week 5 domain-transfer evaluation only.",
        "",
        f"- Feature rows: {len(df)}",
        f"- Errors: {len(errors)}",
        f"- Sliding window: size={window_size}, stride={stride}",
        f"- Frame stride: {frame_stride}",
        "",
    ]
    if not df.empty and "canonical_activity" in df.columns:
        lines.append("## Rows per activity")
        lines.append("")
        for label, count in df["canonical_activity"].value_counts().items():
            lines.append(f"- {label}: {count}")
        lines.append("")
    if errors:
        lines.append("## Errors")
        for err in errors[:20]:
            lines.append(f"- `{err['relative_path']}`: {err['error']}")
    path.write_text("\n".join(lines), encoding="utf-8")
