"""Load and validate the data source registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worker_activity.config import find_repo_root, load_yaml

REQUIRED_SOURCE_FIELDS = [
    "id",
    "display_name",
    "official_url",
    "source_type",
    "license",
    "acquisition_method",
    "expected_local_relative_dir",
    "enabled",
    "priority",
    "intended_use",
    "known_limitations",
    "local_status",
]

VALID_SOURCE_TYPES = {
    "skeleton",
    "video_and_imu",
    "video",
    "egocentric_pose",
    "video_unknown_structure",
    "local_video",
}

VALID_LOCAL_STATUS = {
    "not_downloaded",
    "present_unvalidated",
    "validated",
    "invalid",
}


class SourceRegistryError(Exception):
    """Raised when data_sources.yaml is invalid."""


@dataclass
class DataSource:
    id: str
    display_name: str
    official_url: str | None
    doi: str | None
    supporting_repository: str | None
    source_type: str
    expected_size_gb: float | None
    license: str
    license_notes: str
    acquisition_method: str
    expected_local_relative_dir: str
    archives_relative_dir: str | None
    enabled: bool
    priority: int
    intended_use: str
    known_limitations: list[str]
    local_status: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DataSource":
        missing = [f for f in REQUIRED_SOURCE_FIELDS if f not in raw]
        if missing:
            raise SourceRegistryError(f"Source missing required fields {missing}: {raw.get('id')}")
        if raw["source_type"] not in VALID_SOURCE_TYPES:
            raise SourceRegistryError(
                f"Invalid source_type '{raw['source_type']}' for source {raw['id']}"
            )
        if raw["local_status"] not in VALID_LOCAL_STATUS:
            raise SourceRegistryError(
                f"Invalid local_status '{raw['local_status']}' for source {raw['id']}"
            )
        limitations = raw.get("known_limitations") or []
        if isinstance(limitations, str):
            limitations = [limitations]
        return cls(
            id=str(raw["id"]),
            display_name=str(raw["display_name"]),
            official_url=raw.get("official_url"),
            doi=raw.get("doi"),
            supporting_repository=raw.get("supporting_repository"),
            source_type=str(raw["source_type"]),
            expected_size_gb=raw.get("expected_size_gb"),
            license=str(raw["license"]),
            license_notes=str(raw.get("license_notes") or "").strip(),
            acquisition_method=str(raw["acquisition_method"]),
            expected_local_relative_dir=str(raw["expected_local_relative_dir"]),
            archives_relative_dir=raw.get("archives_relative_dir"),
            enabled=bool(raw["enabled"]),
            priority=int(raw["priority"]),
            intended_use=str(raw.get("intended_use") or "").strip(),
            known_limitations=[str(x) for x in limitations],
            local_status=str(raw["local_status"]),
        )


def load_source_registry(path: Path | None = None) -> list[DataSource]:
    root = find_repo_root()
    registry_path = path or root / "configs" / "data_sources.yaml"
    data = load_yaml(registry_path)
    sources_raw = data.get("sources")
    if not isinstance(sources_raw, list) or not sources_raw:
        raise SourceRegistryError("data_sources.yaml must contain a non-empty 'sources' list")
    sources = [DataSource.from_dict(item) for item in sources_raw]
    ids = [s.id for s in sources]
    if len(ids) != len(set(ids)):
        raise SourceRegistryError("Duplicate source IDs in data_sources.yaml")
    return sorted(sources, key=lambda s: s.priority)


def validate_source_registry(sources: list[DataSource]) -> list[str]:
    issues: list[str] = []
    for source in sources:
        if source.enabled and source.local_status == "not_downloaded":
            issues.append(
                f"Source '{source.id}' is enabled but local_status is not_downloaded (expected)."
            )
    return issues


def format_source_summary(source: DataSource) -> str:
    size = f"{source.expected_size_gb} GB" if source.expected_size_gb is not None else "unknown"
    lines = [
        f"ID:           {source.id}",
        f"Name:         {source.display_name}",
        f"Type:         {source.source_type}",
        f"Priority:     {source.priority}",
        f"Enabled:      {source.enabled}",
        f"License:      {source.license}",
        f"Size (est.):  {size}",
        f"Local status: {source.local_status}",
        f"Local dir:    {source.expected_local_relative_dir}",
        f"Official URL: {source.official_url or '(none)'}",
        f"DOI:          {source.doi or '(none)'}",
        f"Use:          {source.intended_use}",
    ]
    return "\n".join(lines)
