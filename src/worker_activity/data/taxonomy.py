"""Candidate taxonomy loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worker_activity.config import find_repo_root, load_yaml


@dataclass
class TaxonomyCandidate:
    version: str
    status: str
    activity_classes: list[str]
    freeze_blocked_until: list[str]
    raw: dict[str, Any]


def load_taxonomy_candidate(path: Path | None = None) -> TaxonomyCandidate:
    root = find_repo_root()
    tax_path = path or root / "configs" / "taxonomy_candidate.yaml"
    data = load_yaml(tax_path)
    classes_raw = (
        data.get("workstreams", {})
        .get("activity_classification", {})
        .get("classes", [])
    )
    class_ids = [item["id"] for item in classes_raw if isinstance(item, dict) and "id" in item]
    return TaxonomyCandidate(
        version=str(data.get("version", "candidate")),
        status=str(data.get("status", "not_frozen")),
        activity_classes=class_ids,
        freeze_blocked_until=list(data.get("freeze_blocked_until") or []),
        raw=data,
    )
