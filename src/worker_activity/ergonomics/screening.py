"""Ergonomic screening pipeline over pose parquet sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from worker_activity.config import find_repo_root, load_yaml, resolve_paths_config
from worker_activity.ergonomics.rules import (
    count_bouts,
    duration_seconds,
    frame_indicators,
)
from worker_activity.features.pose_skeleton_features import pose_frames_from_parquet
from worker_activity.reporting.markdown import (
    bullet_list,
    dataframe_to_markdown,
    write_markdown_report,
)
from worker_activity.week6_sources import (
    PoseVideoRef,
    select_cwpv_sample_videos,
    select_phone_videos,
)


@dataclass
class ErgonomicScreeningResult:
    metrics: pd.DataFrame
    output_path: Path
    report_path: Path
    errors: list[dict[str, str]] = field(default_factory=list)


def load_ergonomics_config(path: Path | None = None) -> dict[str, Any]:
    root = find_repo_root()
    cfg_path = path or root / "configs" / "ergonomics_screening.yaml"
    return load_yaml(cfg_path)


def screen_pose_frames(
    frames: list[dict[str, dict[str, float]]],
    cfg: dict[str, Any],
    *,
    fps: float | None = None,
) -> dict[str, float]:
    """Aggregate screening indicators for one pose sequence."""
    min_consecutive = int(cfg.get("min_consecutive_frames", 3))
    use_fps = float(fps) if fps and fps > 0 else float(cfg.get("default_fps", 30.0))

    bending_flags: list[bool] = []
    overhead_flags: list[bool] = []
    kneel_flags: list[bool] = []
    squat_flags: list[bool] = []
    awkward_flags: list[bool] = []

    for landmarks in frames:
        ind = frame_indicators(landmarks, cfg)
        bending_flags.append(bool(ind["bending"]))
        overhead_flags.append(bool(ind["overhead"]))
        kneel_flags.append(bool(ind["kneeling"]))
        squat_flags.append(bool(ind["squatting"]))
        awkward_flags.append(bool(ind["awkward"]))

    return {
        "frame_count": float(len(frames)),
        "fps_used": use_fps,
        "repeated_bending_count": float(count_bouts(bending_flags, min_consecutive=min_consecutive)),
        "bending_duration_s": duration_seconds(
            bending_flags, fps=use_fps, min_consecutive=min_consecutive
        ),
        "overhead_duration_s": duration_seconds(
            overhead_flags, fps=use_fps, min_consecutive=min_consecutive
        ),
        "kneeling_duration_s": duration_seconds(
            kneel_flags, fps=use_fps, min_consecutive=min_consecutive
        ),
        "squatting_duration_s": duration_seconds(
            squat_flags, fps=use_fps, min_consecutive=min_consecutive
        ),
        "awkward_posture_frame_count": float(
            sum(1 for f in awkward_flags if f)
        ),
        "awkward_posture_duration_s": duration_seconds(
            awkward_flags, fps=use_fps, min_consecutive=min_consecutive
        ),
    }


def _screen_video(ref: PoseVideoRef, cfg: dict[str, Any]) -> dict[str, Any]:
    if not ref.pose_parquet.is_file():
        raise FileNotFoundError(f"Pose parquet missing: {ref.pose_parquet}")
    pose_df = pd.read_parquet(ref.pose_parquet)
    frames = pose_frames_from_parquet(pose_df)
    metrics = screen_pose_frames(frames, cfg, fps=ref.fps)
    row: dict[str, Any] = {
        "source": ref.source,
        "relative_path": ref.relative_path,
        "file_name": ref.file_name,
        "canonical_activity": ref.canonical_activity,
        "subject_id": ref.subject_id,
        "motion_id": ref.motion_id,
        "view_id": ref.view_id,
        "split": ref.split,
        "pose_parquet": str(ref.pose_parquet),
        **metrics,
    }
    return row


def screen_ergonomics(
    *,
    source: str = "both",
    max_videos: int | None = None,
    config_path: Path | None = None,
) -> ErgonomicScreeningResult:
    """Run ergonomic screening on phone and/or CWPV sample pose sequences."""
    root = find_repo_root()
    paths = resolve_paths_config(repo_root=root)
    cfg = load_ergonomics_config(config_path)

    refs: list[PoseVideoRef] = []
    if source in ("both", "phone"):
        refs.extend(select_phone_videos(paths, max_videos=max_videos))
    if source in ("both", "cwpv"):
        sample_cfg = cfg.get("cwpv_sample", {})
        cwpv_max = max_videos if max_videos is not None else int(sample_cfg.get("max_videos", 5))
        refs.extend(
            select_cwpv_sample_videos(
                paths,
                view_id=str(sample_cfg.get("view_id", "camera_1")),
                motion_ids=[str(m) for m in sample_cfg.get("motion_ids", ["1", "3", "4", "6", "8"])],
                max_videos=cwpv_max,
            )
        )

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for ref in refs:
        try:
            rows.append(_screen_video(ref, cfg))
        except Exception as exc:  # noqa: BLE001 — collect per-video failures
            errors.append({"relative_path": ref.relative_path, "error": str(exc)})

    metrics_df = pd.DataFrame(rows)
    out_dir = paths.processed_dir / "week6"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "ergonomic_screening.csv"
    metrics_df.to_csv(output_path, index=False)

    report_path = paths.reports_dir / "ergonomic_screening.md"
    _write_screening_report(report_path, metrics_df, errors)
    return ErgonomicScreeningResult(
        metrics=metrics_df,
        output_path=output_path,
        report_path=report_path,
        errors=errors,
    )


def _write_screening_report(
    path: Path,
    metrics: pd.DataFrame,
    errors: list[dict[str, str]],
) -> None:
    if metrics.empty:
        summary = "No videos screened."
        table = "_No rows._"
    else:
        summary = (
            f"Videos screened: **{len(metrics)}**  \n"
            f"Sources: {', '.join(sorted(metrics['source'].astype(str).unique()))}  \n"
            f"_Screening-level indicators only — not ergonomic certification._"
        )
        cols = [
            "source",
            "file_name",
            "canonical_activity",
            "repeated_bending_count",
            "overhead_duration_s",
            "kneeling_duration_s",
            "squatting_duration_s",
            "awkward_posture_duration_s",
        ]
        present = [c for c in cols if c in metrics.columns]
        table = dataframe_to_markdown(metrics[present])
    write_markdown_report(
        path,
        "Ergonomic Screening Indicators (Week 6)",
        {
            "Summary": summary,
            "Per-video indicators": table,
            "Errors": bullet_list(
                [f"{e['relative_path']}: {e['error']}" for e in errors]
            ),
            "Notes": bullet_list(
                [
                    "Indicators are rule/geometry proxies from MediaPipe joint angles.",
                    "Not activity class labels and not final safety judgments.",
                    f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
                ]
            ),
        },
    )
