"""Pose extraction pipeline for CWPV (and other) videos."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from worker_activity.config import find_repo_root, load_yaml, resolve_paths_config
from worker_activity.data.cwpv_metadata import parse_cwpv_filename
from worker_activity.pose.mediapipe_estimator import MediaPipePoseConfig, MediaPipePoseEstimator
from worker_activity.reporting.markdown import bullet_list, write_markdown_report
from worker_activity.video.metadata import VIDEO_EXTENSIONS


@dataclass
class PoseExtractionResult:
    outputs: list[Path] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    report_path: Path | None = None


def load_pose_config(path: Path | None = None) -> MediaPipePoseConfig:
    root = find_repo_root()
    cfg_path = path or root / "configs" / "pose_mediapipe.yaml"
    raw = load_yaml(cfg_path)
    mp_cfg = raw.get("mediapipe", {})
    model_path = mp_cfg.get("model_path")
    return MediaPipePoseConfig(
        model_path=root / model_path if model_path else None,
        model_url=str(
            mp_cfg.get(
                "model_url",
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            )
        ),
        num_poses=int(mp_cfg.get("num_poses", 1)),
        min_pose_detection_confidence=float(mp_cfg.get("min_pose_detection_confidence", 0.5)),
        min_pose_presence_confidence=float(mp_cfg.get("min_pose_presence_confidence", 0.5)),
        min_tracking_confidence=float(mp_cfg.get("min_tracking_confidence", 0.5)),
    )


def _select_cwpv_videos(
    inventory: pd.DataFrame,
    *,
    subject_id: str | None,
    motion_id: str | None,
    camera_id: str | None,
    max_videos: int,
) -> pd.DataFrame:
    df = inventory[inventory["source"] == "cwpv"].copy()
    if df.empty:
        return df
    keep_indices: list[Any] = []
    for idx, row in df.iterrows():
        parsed = parse_cwpv_filename(str(row["file_name"]))
        if parsed is None:
            continue
        if subject_id and parsed.participant_id != subject_id.zfill(2):
            continue
        if motion_id and parsed.motion_id != str(motion_id):
            continue
        if camera_id and parsed.camera_id != str(camera_id):
            continue
        keep_indices.append(idx)
    if not keep_indices:
        return df.iloc[0:0]
    return df.loc[keep_indices].head(max_videos)


def extract_pose_from_inventory(
    *,
    subject_id: str | None = None,
    motion_id: str | None = None,
    camera_id: str | None = None,
    max_videos: int = 2,
    max_frames: int | None = None,
    inventory_path: Path | None = None,
) -> PoseExtractionResult:
    root = find_repo_root()
    paths = resolve_paths_config(repo_root=root, require_data_root=True)
    assert paths.data_root is not None

    inv_path = inventory_path or paths.manifests_dir / "clip_inventory.csv"
    if not inv_path.is_file():
        raise FileNotFoundError(f"Inventory not found: {inv_path}")

    inventory = pd.read_csv(inv_path, low_memory=False)
    selected = _select_cwpv_videos(
        inventory,
        subject_id=subject_id,
        motion_id=motion_id,
        camera_id=camera_id,
        max_videos=max_videos,
    )

    out_dir = paths.processed_dir / "cwpv" / "pose"
    out_dir.mkdir(parents=True, exist_ok=True)

    pose_cfg = load_pose_config()
    result = PoseExtractionResult()
    with MediaPipePoseEstimator(pose_cfg) as estimator:
        for _, row in selected.iterrows():
            rel = str(row["relative_path"])
            video_path = paths.data_root / rel.replace("/", "\\")
            if not video_path.is_file():
                result.errors.append({"relative_path": rel, "error": "file_not_found"})
                continue
            try:
                sequence = estimator.extract_from_video(
                    video_path,
                    relative_path=rel,
                    max_frames=max_frames,
                    fps=row.get("fps") if pd.notna(row.get("fps")) else None,
                )
                out_path = out_dir / f"{video_path.stem}_pose.parquet"
                pd.DataFrame(sequence.to_records()).to_parquet(out_path, index=False)
                result.outputs.append(out_path)
            except OSError as exc:
                result.errors.append({"relative_path": rel, "error": str(exc)})

    report_path = paths.reports_dir / "cwpv_pose_extraction.md"
    write_markdown_report(
        report_path,
        "CWPV Pose Extraction",
        {
            "Generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Filters": bullet_list(
                [
                    f"subject_id={subject_id or 'any'}",
                    f"motion_id={motion_id or 'any'}",
                    f"camera_id={camera_id or 'any'}",
                    f"max_videos={max_videos}",
                    f"max_frames={max_frames or 'all'}",
                ]
            ),
            "Outputs": bullet_list([str(p) for p in result.outputs]) or ["(none)"],
            "Errors": bullet_list(
                [f"{e['relative_path']}: {e['error']}" for e in result.errors]
            )
            or ["(none)"],
        },
    )
    result.report_path = report_path
    return result
