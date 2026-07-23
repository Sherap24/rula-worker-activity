"""Shared Week 6 video / pose selection helpers (phone + CWPV sample)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from worker_activity.config import PathsConfig
from worker_activity.data.cwpv_metadata import parse_cwpv_filename


@dataclass
class PoseVideoRef:
    source: str
    relative_path: str
    file_name: str
    pose_parquet: Path
    fps: float | None = None
    canonical_activity: str | None = None
    subject_id: str | None = None
    motion_id: str | None = None
    view_id: str | None = None
    split: str | None = None
    stem: str = ""


def _pose_path_for(processed_dir: Path, source: str, file_name: str) -> Path:
    stem = Path(file_name).stem
    if source == "local_smartphone":
        return processed_dir / "local_smartphone" / "pose" / f"{stem}_pose.parquet"
    return processed_dir / "cwpv" / "pose" / f"{stem}_pose.parquet"


def select_phone_videos(
    paths: PathsConfig,
    *,
    max_videos: int | None = None,
    require_pose: bool = True,
) -> list[PoseVideoRef]:
    inv_path = paths.manifests_dir / "clip_inventory.csv"
    if not inv_path.is_file():
        return []
    df = pd.read_csv(inv_path, low_memory=False)
    phone = df[df["source"] == "local_smartphone"].copy()
    if phone.empty:
        return []
    if max_videos is not None:
        phone = phone.head(max_videos)

    refs: list[PoseVideoRef] = []
    for _, row in phone.iterrows():
        file_name = str(row["file_name"])
        pose_path = _pose_path_for(paths.processed_dir, "local_smartphone", file_name)
        if require_pose and not pose_path.is_file():
            continue
        fps = float(row["fps"]) if pd.notna(row.get("fps")) else None
        refs.append(
            PoseVideoRef(
                source="local_smartphone",
                relative_path=str(row["relative_path"]),
                file_name=file_name,
                pose_parquet=pose_path,
                fps=fps,
                canonical_activity=(
                    str(row["canonical_activity"])
                    if pd.notna(row.get("canonical_activity"))
                    else None
                ),
                subject_id=str(row["subject_id"]) if pd.notna(row.get("subject_id")) else None,
                stem=Path(file_name).stem,
            )
        )
    return refs


def select_cwpv_sample_videos(
    paths: PathsConfig,
    *,
    view_id: str = "camera_1",
    motion_ids: list[str] | None = None,
    max_videos: int = 5,
    require_pose: bool = True,
    splits: list[str] | None = None,
) -> list[PoseVideoRef]:
    """Pick a small CWPV train/validation sample (never test)."""
    splits = splits or ["train", "validation"]
    motion_ids = motion_ids or ["1", "3", "4", "6", "8"]
    frames: list[pd.DataFrame] = []
    for split in splits:
        path = paths.manifests_dir / f"cwpv_{split}.csv"
        if not path.is_file():
            # fallback naming used by baseline pipeline
            alt = paths.manifests_dir / f"cwpv_{'validation' if split == 'validation' else split}.csv"
            path = alt if alt.is_file() else path
        if not path.is_file():
            continue
        part = pd.read_csv(path, low_memory=False)
        part["split"] = split if "split" not in part.columns else part["split"]
        frames.append(part)
    if not frames:
        return []

    df = pd.concat(frames, ignore_index=True)
    # Never include test if somehow present
    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() != "test"]

    preferred: list[PoseVideoRef] = []
    used_motions: set[str] = set()
    leftovers: list[PoseVideoRef] = []

    for _, row in df.iterrows():
        file_name = str(row.get("file_name") or "")
        parsed = parse_cwpv_filename(file_name)
        row_view = str(row["view_id"]) if pd.notna(row.get("view_id")) else (
            f"camera_{parsed.camera_id}" if parsed else None
        )
        row_motion = str(row["motion_id"]) if pd.notna(row.get("motion_id")) else (
            parsed.motion_id if parsed else None
        )
        if view_id and row_view != view_id:
            continue
        pose_path = _pose_path_for(paths.processed_dir, "cwpv", file_name)
        if require_pose and not pose_path.is_file():
            continue
        fps = float(row["fps"]) if pd.notna(row.get("fps")) else None
        ref = PoseVideoRef(
            source="cwpv",
            relative_path=str(row["relative_path"]),
            file_name=file_name,
            pose_parquet=pose_path,
            fps=fps,
            canonical_activity=(
                str(row["canonical_activity"])
                if pd.notna(row.get("canonical_activity"))
                else None
            ),
            subject_id=str(row["subject_id"]) if pd.notna(row.get("subject_id")) else None,
            motion_id=row_motion,
            view_id=row_view,
            split=str(row["split"]) if pd.notna(row.get("split")) else None,
            stem=Path(file_name).stem,
        )
        if row_motion in motion_ids and row_motion not in used_motions:
            preferred.append(ref)
            used_motions.add(row_motion)
        else:
            leftovers.append(ref)
        if len(preferred) >= max_videos:
            break

    refs = preferred
    for ref in leftovers:
        if len(refs) >= max_videos:
            break
        refs.append(ref)
    return refs[:max_videos]
