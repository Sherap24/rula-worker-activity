"""CWPV pose feature extraction pipeline (train/validation manifests only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from worker_activity.config import find_repo_root, load_yaml, resolve_paths_config
from worker_activity.features.pose_skeleton_features import extract_video_pose_features
from worker_activity.pose.mediapipe_estimator import MediaPipePoseConfig, MediaPipePoseEstimator
from worker_activity.pose.pipeline import load_pose_config


@dataclass
class CwpvFeatureExtractionResult:
    features: pd.DataFrame
    output_path: Path
    report_path: Path
    pose_outputs: list[Path] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)


METADATA_COLUMNS = [
    "relative_path",
    "file_name",
    "logical_sample_id",
    "representation_group_id",
    "canonical_activity",
    "raw_activity_label",
    "subject_id",
    "motion_id",
    "view_id",
    "split",
    "frame_count",
    "fps",
]


def load_cwpv_feature_config(path: Path | None = None) -> dict[str, Any]:
    root = find_repo_root()
    cfg_path = path or root / "configs" / "cwpv_pose_features.yaml"
    if not cfg_path.is_file():
        return {"sliding_window": {"window_size": 30, "stride": 15}}
    return load_yaml(cfg_path)


def _manifest_paths(paths, splits: list[str]) -> list[tuple[str, Path]]:
    mapping = {
        "train": paths.manifests_dir / "cwpv_train.csv",
        "validation": paths.manifests_dir / "cwpv_validation.csv",
    }
    return [(name, mapping[name]) for name in splits if name in mapping]


def _pose_output_path(pose_dir: Path, video_path: Path) -> Path:
    return pose_dir / f"{video_path.stem}_pose.parquet"


def extract_cwpv_features(
    *,
    splits: list[str] | None = None,
    max_videos: int | None = None,
    skip_pose_if_exists: bool = True,
    view_id: str | None = None,
    frame_stride: int | None = None,
) -> CwpvFeatureExtractionResult:
    """Extract pose landmarks and kinematic features for train/validation splits."""
    root = find_repo_root()
    paths = resolve_paths_config(repo_root=root, require_data_root=True)
    assert paths.data_root is not None

    splits = splits or ["train", "validation"]
    feature_cfg = load_cwpv_feature_config()
    sw_cfg = feature_cfg.get("sliding_window", {})
    window_size = int(sw_cfg.get("window_size", 30))
    stride = int(sw_cfg.get("stride", 15))
    if frame_stride is None:
        frame_stride = int(feature_cfg.get("extraction", {}).get("frame_stride", 1))

    pose_dir = paths.processed_dir / "cwpv" / "pose"
    feat_dir = paths.processed_dir / "cwpv" / "features"
    pose_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    manifest_list = _manifest_paths(paths, splits)
    if not manifest_list:
        raise FileNotFoundError(
            "CWPV train/validation manifests not found. Run build-cwpv-baseline first."
        )

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    pose_outputs: list[Path] = []
    pose_cfg = load_pose_config()

    with MediaPipePoseEstimator(pose_cfg) as estimator:
        for split_name, manifest_path in manifest_list:
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Manifest not found: {manifest_path}")
            manifest = pd.read_csv(manifest_path)
            if view_id:
                view_series = manifest["view_id"].astype(str)
                file_series = manifest["file_name"].astype(str)
                # Prefer view_id; fall back to filename Camera_N pattern
                cam_suffix = view_id.replace("camera_", "")
                mask = view_series.eq(view_id) | file_series.str.contains(
                    rf"Camera_{cam_suffix}\.", case=False, regex=True, na=False
                )
                manifest = manifest[mask]
            if max_videos is not None:
                manifest = manifest.head(max_videos)

            total = len(manifest)
            print(
                f"Extracting {split_name}: {total} videos "
                f"(frame_stride={frame_stride}"
                f"{f', view={view_id}' if view_id else ''})",
                flush=True,
            )

            for i, (_, row) in enumerate(manifest.iterrows(), start=1):
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
                    feat_row["split"] = split_name
                    feat_row.update(feats)
                    feat_row["extraction_status"] = "ok"
                    feat_row["pose_parquet"] = str(pose_path)
                    rows.append(feat_row)
                    if i % 10 == 0 or i == total or not reused:
                        print(
                            f"  [{split_name}] {i}/{total} "
                            f"{'reuse' if reused else 'extract'} {video_path.name}",
                            flush=True,
                        )
                except OSError as exc:
                    errors.append({"relative_path": rel, "error": str(exc)})
                    print(f"  ERROR {rel}: {exc}", flush=True)
    features_df = pd.DataFrame(rows)
    out_cfg = feature_cfg.get("output", {})
    output_path = root / out_cfg.get(
        "train_val_features",
        "data/processed/cwpv/features_train_val.parquet",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False)

    report_path = root / out_cfg.get(
        "report",
        "reports/cwpv_pose_feature_extraction.md",
    )
    _write_report(features_df, errors, splits, report_path, window_size, stride)

    return CwpvFeatureExtractionResult(
        features=features_df,
        output_path=output_path,
        report_path=report_path,
        pose_outputs=pose_outputs,
        errors=errors,
    )


def _write_report(
    df: pd.DataFrame,
    errors: list[dict[str, str]],
    splits: list[str],
    path: Path,
    window_size: int,
    stride: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CWPV Pose Feature Extraction",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_",
        "",
        "**Scope:** Train and validation manifests only. Test split is held out.",
        "",
        f"- Splits processed: {', '.join(splits)}",
        f"- Feature rows: {len(df)}",
        f"- Errors: {len(errors)}",
        f"- Sliding window: size={window_size}, stride={stride}",
        "",
    ]
    if not df.empty and "split" in df.columns:
        lines.append("## Rows per split")
        lines.append("")
        for split_name, count in df["split"].value_counts().items():
            lines.append(f"- {split_name}: {count}")
        lines.append("")
    if errors:
        lines.append("## Errors")
        for err in errors[:20]:
            lines.append(f"- `{err['relative_path']}`: {err['error']}")
    path.write_text("\n".join(lines), encoding="utf-8")
