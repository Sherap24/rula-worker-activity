"""Zone entry/exit event detection from pose sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from worker_activity.config import find_repo_root, load_yaml, resolve_paths_config
from worker_activity.features.pose_skeleton_features import (
    _landmark_point,
    hip_center,
    pose_frames_from_parquet,
)
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
from worker_activity.zones.geometry import normalize_vertices, point_in_polygon


@dataclass
class ZoneEventDetectionResult:
    events: pd.DataFrame
    summary: pd.DataFrame
    events_path: Path
    summary_path: Path
    report_path: Path
    errors: list[dict[str, str]] = field(default_factory=list)


def load_zones_config(path: Path | None = None) -> dict[str, Any]:
    root = find_repo_root()
    cfg_path = path or root / "configs" / "zones_demo.yaml"
    return load_yaml(cfg_path)


def _track_point(
    landmarks: dict[str, dict[str, float]],
    *,
    mode: str,
) -> tuple[float, float] | None:
    if mode == "foot_point":
        left = _landmark_point(landmarks, "LEFT_ANKLE")
        right = _landmark_point(landmarks, "RIGHT_ANKLE")
        if left is None and right is None:
            left = _landmark_point(landmarks, "LEFT_HEEL")
            right = _landmark_point(landmarks, "RIGHT_HEEL")
        if left is None and right is None:
            return None
        if left is None:
            return float(right[0]), float(right[1])
        if right is None:
            return float(left[0]), float(left[1])
        return float((left[0] + right[0]) / 2.0), float((left[1] + right[1]) / 2.0)

    center = hip_center(landmarks)
    if center is None:
        return None
    return float(center[0]), float(center[1])


def polygons_for_video(cfg: dict[str, Any], stem: str) -> list[dict[str, Any]]:
    overrides = cfg.get("video_polygons") or {}
    if stem in overrides and overrides[stem]:
        raw_list = overrides[stem]
    else:
        raw_list = cfg.get("default_polygons") or []
    polygons: list[dict[str, Any]] = []
    for item in raw_list:
        verts = normalize_vertices(item.get("vertices") or [])
        if len(verts) < 3:
            continue
        polygons.append(
            {
                "id": str(item.get("id", "zone")),
                "label": str(item.get("label", item.get("id", "zone"))),
                "vertices": verts,
            }
        )
    return polygons


def detect_events_for_frames(
    frames: list[dict[str, dict[str, float]]],
    polygons: list[dict[str, Any]],
    *,
    point_mode: str,
    fps: float,
    min_consecutive: int,
) -> list[dict[str, Any]]:
    """Emit restricted_zone_entry / restricted_zone_exit events."""
    if min_consecutive < 1:
        min_consecutive = 1
    events: list[dict[str, Any]] = []

    for poly in polygons:
        inside_flags: list[bool | None] = []
        for landmarks in frames:
            pt = _track_point(landmarks, mode=point_mode)
            if pt is None:
                inside_flags.append(None)
            else:
                inside_flags.append(point_in_polygon(pt[0], pt[1], poly["vertices"]))

        # Fill short missing gaps with previous known state for stability
        cleaned: list[bool] = []
        last = False
        for flag in inside_flags:
            if flag is None:
                cleaned.append(last)
            else:
                cleaned.append(flag)
                last = flag

        confirmed: list[bool] = []
        # Confirm state only after min_consecutive identical frames
        pending_val = cleaned[0] if cleaned else False
        pending_len = 0
        current = False
        for flag in cleaned:
            if flag == pending_val:
                pending_len += 1
            else:
                pending_val = flag
                pending_len = 1
            if pending_len >= min_consecutive:
                current = pending_val
            confirmed.append(current)

        prev = False
        for idx, cur in enumerate(confirmed):
            if cur and not prev:
                events.append(
                    {
                        "event_type": "restricted_zone_entry",
                        "zone_id": poly["id"],
                        "zone_label": poly["label"],
                        "frame_index": idx,
                        "timestamp_s": float(idx) / fps if fps > 0 else float(idx),
                        "point_mode": point_mode,
                    }
                )
            elif prev and not cur:
                events.append(
                    {
                        "event_type": "restricted_zone_exit",
                        "zone_id": poly["id"],
                        "zone_label": poly["label"],
                        "frame_index": idx,
                        "timestamp_s": float(idx) / fps if fps > 0 else float(idx),
                        "point_mode": point_mode,
                    }
                )
            prev = cur
    return events


def _detect_video(ref: PoseVideoRef, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not ref.pose_parquet.is_file():
        raise FileNotFoundError(f"Pose parquet missing: {ref.pose_parquet}")
    pose_df = pd.read_parquet(ref.pose_parquet)
    frames = pose_frames_from_parquet(pose_df)
    fps = float(ref.fps) if ref.fps and ref.fps > 0 else float(cfg.get("default_fps", 30.0))
    point_mode = str(cfg.get("point_mode", "body_center"))
    min_consecutive = int(cfg.get("min_consecutive_frames", 2))
    polygons = polygons_for_video(cfg, ref.stem)
    events = detect_events_for_frames(
        frames,
        polygons,
        point_mode=point_mode,
        fps=fps,
        min_consecutive=min_consecutive,
    )
    enriched = []
    for ev in events:
        enriched.append(
            {
                "source": ref.source,
                "relative_path": ref.relative_path,
                "file_name": ref.file_name,
                "canonical_activity": ref.canonical_activity,
                "subject_id": ref.subject_id,
                "motion_id": ref.motion_id,
                "view_id": ref.view_id,
                "split": ref.split,
                **ev,
            }
        )
    summary = {
        "source": ref.source,
        "relative_path": ref.relative_path,
        "file_name": ref.file_name,
        "canonical_activity": ref.canonical_activity,
        "n_polygons": len(polygons),
        "n_entry": sum(1 for e in enriched if e["event_type"] == "restricted_zone_entry"),
        "n_exit": sum(1 for e in enriched if e["event_type"] == "restricted_zone_exit"),
        "frame_count": len(frames),
        "point_mode": point_mode,
    }
    return enriched, summary


def detect_zone_events(
    *,
    source: str = "both",
    max_videos: int | None = None,
    config_path: Path | None = None,
) -> ZoneEventDetectionResult:
    root = find_repo_root()
    paths = resolve_paths_config(repo_root=root)
    cfg = load_zones_config(config_path)

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

    all_events: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for ref in refs:
        try:
            events, summary = _detect_video(ref, cfg)
            all_events.extend(events)
            summaries.append(summary)
        except Exception as exc:  # noqa: BLE001
            errors.append({"relative_path": ref.relative_path, "error": str(exc)})

    events_df = pd.DataFrame(all_events)
    summary_df = pd.DataFrame(summaries)
    out_dir = paths.processed_dir / "week6"
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "zone_events.csv"
    summary_path = out_dir / "zone_events_summary.csv"
    events_df.to_csv(events_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    report_path = paths.reports_dir / "zone_events.md"
    _write_zone_report(report_path, events_df, summary_df, errors)
    return ZoneEventDetectionResult(
        events=events_df,
        summary=summary_df,
        events_path=events_path,
        summary_path=summary_path,
        report_path=report_path,
        errors=errors,
    )


def _write_zone_report(
    path: Path,
    events: pd.DataFrame,
    summary: pd.DataFrame,
    errors: list[dict[str, str]],
) -> None:
    n_entry = int((events["event_type"] == "restricted_zone_entry").sum()) if not events.empty else 0
    n_exit = int((events["event_type"] == "restricted_zone_exit").sum()) if not events.empty else 0
    summary_text = (
        f"Videos processed: **{len(summary)}**  \n"
        f"Entry events: **{n_entry}**  \n"
        f"Exit events: **{n_exit}**  \n"
        f"_Demo polygons from configs/zones_demo.yaml — not real jobsite annotations._"
    )
    summary_table = dataframe_to_markdown(summary) if not summary.empty else "_No rows._"
    events_preview = (
        dataframe_to_markdown(events.head(40))
        if not events.empty
        else "_No zone crossings detected (body-center never entered the demo polygon)._"
    )
    write_markdown_report(
        path,
        "Zone Events (Week 6)",
        {
            "Summary": summary_text,
            "Per-video summary": summary_table,
            "Events (preview)": events_preview,
            "Errors": bullet_list(
                [f"{e['relative_path']}: {e['error']}" for e in errors]
            ),
            "Notes": bullet_list(
                [
                    "Zone events are spatial crossings, not activity classes.",
                    "Default point is hip midpoint (body_center).",
                    f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
                ]
            ),
        },
    )
