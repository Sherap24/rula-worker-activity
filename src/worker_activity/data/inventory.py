"""Build clip/file inventory from external data directories."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pandas as pd

from worker_activity.config import PathsConfig, source_data_path
from worker_activity.data.cml_metadata import enrich_cml_row
from worker_activity.data.cwpv_label_map import apply_cwpv_label_mapping, load_cwpv_label_map
from worker_activity.data.cwpv_metadata import enrich_cwpv_row, parse_cwpv_filename
from worker_activity.data.integrity import quick_integrity_check
from worker_activity.data.schema import empty_inventory_frame, inventory_row, normalize_inventory_frame
from worker_activity.data.smartphone_metadata import enrich_smartphone_row
from worker_activity.data.source_registry import DataSource, load_source_registry
from worker_activity.video.metadata import VIDEO_EXTENSIONS, extract_video_metadata

SKELETON_EXTENSIONS = {".csv", ".json", ".npy", ".npz", ".txt", ".mat", ".skeleton"}


@dataclass
class InventoryBuildResult:
    frame: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    files_scanned: int = 0
    sources_scanned: list[str] = field(default_factory=list)


def iter_files(root: Path, extensions: set[str]) -> Iterator[Path]:
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def audit_source_directory(
    source: DataSource,
    paths: PathsConfig,
    *,
    extract_metadata: bool = True,
    compute_checksums: bool = False,
) -> tuple[list[dict], list[str]]:
    """Scan one source directory and return inventory rows + warnings."""
    warnings: list[str] = []
    rows: list[dict] = []

    base = source_data_path(paths, source.id, "extracted")
    if base is None:
        warnings.append(
            f"RULA_DATA_ROOT not set — cannot scan source '{source.id}'."
        )
        return rows, warnings
    if not base.is_dir():
        warnings.append(
            f"Source '{source.id}' directory not found (not_downloaded): {base.name}/"
        )
        return rows, warnings

    if source.source_type == "skeleton":
        extensions = SKELETON_EXTENSIONS | VIDEO_EXTENSIONS
        metadata_default = "not_applicable"
    elif source.source_type in {"video", "video_and_imu", "video_unknown_structure", "local_video"}:
        extensions = VIDEO_EXTENSIONS
        metadata_default = "pending"
    else:
        extensions = SKELETON_EXTENSIONS | VIDEO_EXTENSIONS
        metadata_default = "pending"

    data_root = paths.data_root
    assert data_root is not None

    for file_path in iter_files(base, extensions):
        try:
            relative = file_path.relative_to(data_root).as_posix()
        except ValueError:
            relative = file_path.name
            warnings.append(f"File outside data root skipped: {file_path.name}")

        integrity_status, checksum_or_note = quick_integrity_check(
            file_path, compute_checksum=compute_checksums
        )
        row = inventory_row(
            source=source.id,
            source_version=None,
            source_type=source.source_type,
            relative_path=relative,
            file_name=file_path.name,
            extension=file_path.suffix.lower(),
            size_bytes=file_path.stat().st_size,
            checksum=checksum_or_note if integrity_status == "ok" else None,
            metadata_status=metadata_default,
            integrity_status=integrity_status,
            notes=checksum_or_note if integrity_status == "failed" else None,
        )

        if (
            extract_metadata
            and file_path.suffix.lower() in VIDEO_EXTENSIONS
            and integrity_status == "ok"
        ):
            meta = extract_video_metadata(
                file_path,
                provider=paths.metadata_provider,
                ffprobe_binary=paths.ffprobe_binary,
            )
            row.update(
                {
                    "fps": meta.fps,
                    "frame_count": meta.frame_count,
                    "duration_seconds": meta.duration_seconds,
                    "width": meta.width,
                    "height": meta.height,
                    "codec": meta.codec,
                    "metadata_status": meta.status,
                    "notes": meta.notes or row.get("notes"),
                }
            )

        if source.id == "cml" and file_path.suffix.lower() == ".json" and integrity_status == "ok":
            row = enrich_cml_row(row, file_path, relative)
        elif (
            source.id == "cwpv"
            and file_path.suffix.lower() in VIDEO_EXTENSIONS
            and integrity_status == "ok"
        ):
            row = enrich_cwpv_row(row, file_path)
            parsed = parse_cwpv_filename(file_path.name)
            if parsed is not None:
                mapping = apply_cwpv_label_mapping(
                    parsed.motion_id,
                    load_cwpv_label_map(),
                )
                row.update(mapping)
                row["label_mapping_status"] = (
                    "mapped" if mapping.get("canonical_activity") else "unmapped"
                )
        elif (
            source.id == "local_smartphone"
            and file_path.suffix.lower() in VIDEO_EXTENSIONS
            and integrity_status == "ok"
        ):
            row = enrich_smartphone_row(row, file_path)

        rows.append(row)

    if not rows:
        warnings.append(
            f"Source '{source.id}' directory exists but contains no recognized files yet."
        )

    return rows, warnings


def detect_local_status(source: DataSource, paths: PathsConfig) -> str:
    """Infer local_status from filesystem without claiming validation."""
    base = source_data_path(paths, source.id, "extracted")
    if base is None or not base.is_dir():
        archives = source_data_path(paths, source.id, "archives")
        if archives and archives.is_dir() and any(archives.iterdir()):
            return "present_unvalidated"
        return "not_downloaded"
    if any(base.rglob("*")):
        return "present_unvalidated"
    archives = source_data_path(paths, source.id, "archives")
    if archives and archives.is_dir() and any(archives.iterdir()):
        return "present_unvalidated"
    return "not_downloaded"


def build_inventory(
    paths: PathsConfig,
    *,
    sources: list[DataSource] | None = None,
    enabled_only: bool = True,
    extract_metadata: bool = True,
    compute_checksums: bool = False,
) -> InventoryBuildResult:
    """Build inventory across registered sources."""
    sources = sources or load_source_registry()
    if enabled_only:
        sources = [s for s in sources if s.enabled]

    all_warnings: list[str] = []
    all_rows: list[dict] = []
    scanned: list[str] = []

    if paths.data_root is None:
        all_warnings.append(
            "RULA_DATA_ROOT is not set. Inventory will be empty. "
            "See docs/DATA_ACQUISITION.md."
        )
        return InventoryBuildResult(
            frame=empty_inventory_frame(),
            warnings=all_warnings,
            files_scanned=0,
            sources_scanned=[],
        )

    for source in sources:
        scanned.append(source.id)
        rows, warnings = audit_source_directory(
            source,
            paths,
            extract_metadata=extract_metadata,
            compute_checksums=compute_checksums,
        )
        all_rows.extend(rows)
        all_warnings.extend(warnings)

    frame = normalize_inventory_frame(pd.DataFrame(all_rows)) if all_rows else empty_inventory_frame()
    return InventoryBuildResult(
        frame=frame,
        warnings=all_warnings,
        files_scanned=len(all_rows),
        sources_scanned=scanned,
    )


def audit_data_sources(paths: PathsConfig) -> list[dict]:
    """Return audit records for each registered source."""
    records = []
    for source in load_source_registry():
        detected = detect_local_status(source, paths)
        base = source_data_path(paths, source.id, "extracted")
        archives = source_data_path(paths, source.id, "archives")
        records.append(
            {
                "id": source.id,
                "enabled": source.enabled,
                "registry_status": source.local_status,
                "detected_status": detected,
                "extracted_dir_exists": base.is_dir() if base else False,
                "archives_dir_exists": archives.is_dir() if archives else False,
                "extracted_relative": source.expected_local_relative_dir,
            }
        )
    return records
