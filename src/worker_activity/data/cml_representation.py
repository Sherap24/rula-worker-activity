"""CML logical sample and 15/20-node representation grouping."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import pandas as pd

_EXPECTED_REPRESENTATIONS = 2


def extract_notes_field(notes: str | None, field: str) -> str | None:
    if notes is None or pd.isna(notes):
        return None
    match = re.search(rf"{re.escape(field)}=([^;]+)", str(notes))
    return match.group(1).strip() if match else None


def skeleton_layout_from_view_id(view_id: str | None) -> int | None:
    if view_id == "15_nodes":
        return 15
    if view_id == "20_nodes":
        return 20
    return None


def build_logical_sample_id(
    source_dataset: str | None,
    source_file: str | None,
    raw_activity_label: str | None,
) -> str | None:
    if not source_dataset or not source_file or not raw_activity_label:
        return None
    key = "|".join(
        [
            str(source_dataset).strip().lower(),
            str(source_file).strip().lower(),
            str(raw_activity_label).strip().lower(),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"cml_{digest}"


def assign_representation_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add logical_sample_id, representation_group_id, skeleton_layout."""
    out = df.copy()
    out["skeleton_layout"] = out["view_id"].map(skeleton_layout_from_view_id)
    out["logical_sample_id"] = out.apply(
        lambda r: build_logical_sample_id(
            r.get("source_dataset"),
            r.get("source_file"),
            r.get("raw_activity_label"),
        ),
        axis=1,
    )
    out["representation_group_id"] = out["logical_sample_id"]
    return out


def audit_representations(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit 15-node / 20-node pairing per logical sample."""
    if df.empty:
        return pd.DataFrame(), {"total_logical_samples": 0}

    grouped = df.groupby("logical_sample_id", dropna=False)
    rows: list[dict[str, Any]] = []
    for logical_id, group in grouped:
        if logical_id is None or (isinstance(logical_id, float) and pd.isna(logical_id)):
            continue
        layouts = sorted(group["skeleton_layout"].dropna().unique().tolist())
        labels = group["raw_activity_label"].dropna().unique().tolist()
        sources = group["source_dataset"].dropna().unique().tolist()
        count = len(group)
        status = "ok"
        if count != _EXPECTED_REPRESENTATIONS:
            status = "unexpected_count"
        if layouts != [15, 20]:
            status = "missing_layout"
        if len(labels) > 1:
            status = "inconsistent_label"
        rows.append(
            {
                "logical_sample_id": logical_id,
                "representation_count": count,
                "skeleton_layouts": ",".join(str(x) for x in layouts),
                "raw_activity_labels": "|".join(labels),
                "source_datasets": "|".join(sources),
                "status": status,
            }
        )

    audit_df = pd.DataFrame(rows)
    summary = {
        "total_logical_samples": len(audit_df),
        "ok_pairs": int((audit_df["status"] == "ok").sum()) if not audit_df.empty else 0,
        "unexpected_count": int((audit_df["status"] == "unexpected_count").sum())
        if not audit_df.empty
        else 0,
        "missing_layout": int((audit_df["status"] == "missing_layout").sum())
        if not audit_df.empty
        else 0,
        "inconsistent_label": int((audit_df["status"] == "inconsistent_label").sum())
        if not audit_df.empty
        else 0,
    }
    return audit_df, summary
