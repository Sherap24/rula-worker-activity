"""CWPV baseline pipeline: mapping, subject splits, manifests, and reports."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from worker_activity.config import find_repo_root, resolve_paths_config
from worker_activity.data.cml_baseline import LeakageAuditError, audit_leakage, split_subjects
from worker_activity.data.cwpv_label_map import apply_cwpv_label_mapping, load_cwpv_label_map
from worker_activity.data.cwpv_metadata import parse_cwpv_filename
from worker_activity.reporting.markdown import write_markdown_report

MIN_SUBJECTS_PER_BASELINE_CLASS = 3
SPLIT_SEED = 42
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


@dataclass
class CwpvBaselineResult:
    inventory_path: Path
    summary: dict[str, Any] = field(default_factory=dict)
    reports: dict[str, Path] = field(default_factory=dict)
    manifests: dict[str, Path] = field(default_factory=dict)


def enrich_cwpv_inventory_rows(df: pd.DataFrame, label_map: dict | None = None) -> pd.DataFrame:
    """Add CWPV baseline fields: motion_id, logical_sample_id, label mapping."""
    label_map = label_map or load_cwpv_label_map()
    cwpv_mask = df["source"] == "cwpv"
    out = df.copy()

    for col in [
        "motion_id",
        "logical_sample_id",
        "representation_group_id",
        "subject_id",
        "view_id",
        "canonical_activity",
        "raw_activity_label",
        "mapping_confidence",
        "mapping_status",
        "include_in_baseline",
        "exclusion_reason",
        "subject_parse_status",
    ]:
        if col not in out.columns:
            out[col] = None

    def _enrich_row(row: pd.Series) -> dict[str, Any]:
        parsed = parse_cwpv_filename(str(row.get("file_name") or ""))
        if parsed is None:
            return {
                "motion_id": None,
                "logical_sample_id": None,
                "representation_group_id": None,
                "view_id": None,
                "raw_activity_label": None,
                "subject_parse_status": "unresolved_pattern",
            }
        if parsed.block_id is not None:
            logical_id = (
                f"P{parsed.participant_id}_M{parsed.motion_id}_"
                f"B{parsed.block_id}_T{parsed.trial_id}"
            )
        else:
            logical_id = (
                f"P{parsed.participant_id}_M{parsed.motion_id}_T{parsed.trial_id}"
            )
        mapping = apply_cwpv_label_mapping(parsed.motion_id, label_map)
        return {
            "motion_id": parsed.motion_id,
            "logical_sample_id": logical_id,
            "representation_group_id": logical_id,
            "subject_id": parsed.participant_id,
            "view_id": f"camera_{parsed.camera_id}",
            "raw_activity_label": parsed.raw_motion_label,
            "subject_parse_status": "parsed_from_filename",
            **mapping,
        }

    if cwpv_mask.any():
        enriched = out.loc[cwpv_mask].apply(_enrich_row, axis=1, result_type="expand")
        for col in enriched.columns:
            out[col] = out[col].astype("object")
            out.loc[cwpv_mask, col] = enriched[col].values

    return out


def build_cwpv_splits(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Subject-disjoint splits for CWPV video clips."""
    cwpv = df[df["source"] == "cwpv"].copy()
    baseline_eligible = cwpv[cwpv["include_in_baseline"] == True].copy()
    excluded_not_baseline = cwpv[cwpv["include_in_baseline"] != True]

    unresolved = baseline_eligible[
        baseline_eligible["subject_id"].isna()
        | baseline_eligible["logical_sample_id"].isna()
        | baseline_eligible["subject_parse_status"].eq("unresolved_pattern")
    ]
    resolved = baseline_eligible[~baseline_eligible.index.isin(unresolved.index)]

    logical = (
        resolved.drop_duplicates("logical_sample_id")
        .groupby("canonical_activity")["subject_id"]
        .nunique()
        .to_dict()
    )
    excluded_classes: list[str] = []
    included_classes: list[str] = []
    for cls, n_subj in logical.items():
        if n_subj < MIN_SUBJECTS_PER_BASELINE_CLASS:
            excluded_classes.append(cls)
        else:
            included_classes.append(cls)

    split_pool = resolved[resolved["canonical_activity"].isin(included_classes)]
    class_excluded_rows = resolved[resolved["canonical_activity"].isin(excluded_classes)]

    subjects = sorted(split_pool["subject_id"].dropna().unique().tolist())
    subject_to_split = split_subjects(subjects)

    split_pool = split_pool.copy()
    split_pool["split"] = split_pool["subject_id"].map(subject_to_split)

    train = split_pool[split_pool["split"] == "train"]
    val = split_pool[split_pool["split"] == "validation"]
    test = split_pool[split_pool["split"] == "test"]

    excluded_frames = pd.concat(
        [excluded_not_baseline, class_excluded_rows],
        ignore_index=True,
    )

    meta = {
        "included_classes": included_classes,
        "excluded_classes_insufficient_subjects": excluded_classes,
        "unique_subjects_split": len(subjects),
        "unresolved_rows": len(unresolved),
        "excluded_rows": len(excluded_frames),
        "total_cwpv_videos": len(cwpv),
        "baseline_eligible_videos": len(baseline_eligible),
    }
    manifests = {
        "train": train,
        "validation": val,
        "test": test,
        "unresolved_subject": unresolved,
        "excluded": excluded_frames,
    }
    return manifests, meta


def class_balance_tables(df: pd.DataFrame, manifests: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cwpv = df[df["source"] == "cwpv"].copy()
    rows = []
    for canonical in sorted(cwpv["canonical_activity"].dropna().unique()):
        class_df = cwpv[cwpv["canonical_activity"] == canonical]
        logical = class_df.drop_duplicates("logical_sample_id")
        mapped_raw = sorted(class_df["raw_activity_label"].dropna().unique().tolist())
        row: dict[str, Any] = {
            "canonical_activity": canonical,
            "mapped_raw_labels": "; ".join(mapped_raw),
            "video_file_count": len(class_df),
            "unique_logical_sample_count": len(logical),
            "unique_parsed_subject_count": logical["subject_id"].nunique(),
            "include_in_baseline": bool(class_df["include_in_baseline"].any()),
        }
        for split_name, manifest in [
            ("train", manifests["train"]),
            ("validation", manifests["validation"]),
            ("test", manifests["test"]),
        ]:
            split_logical = manifest[manifest["canonical_activity"] == canonical].drop_duplicates(
                "logical_sample_id"
            )
            row[f"{split_name}_logical_samples"] = len(split_logical)
            row[f"{split_name}_videos"] = len(
                manifest[manifest["canonical_activity"] == canonical]
            )
            row[f"{split_name}_subjects"] = split_logical["subject_id"].nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def write_inventory_atomic(df: pd.DataFrame, path: Path) -> Path:
    backup = path.with_suffix(path.suffix + ".bak")
    if path.is_file():
        shutil.copy2(path, backup)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    return backup if backup.is_file() else path


def run_cwpv_baseline_pipeline(
    inventory_path: Path | None = None,
    *,
    repo_root: Path | None = None,
) -> CwpvBaselineResult:
    root = repo_root or find_repo_root()
    paths = resolve_paths_config(repo_root=root)
    inv_path = inventory_path or paths.manifests_dir / "clip_inventory.csv"
    reports_dir = paths.reports_dir
    manifests_dir = paths.manifests_dir

    label_map = load_cwpv_label_map()
    df = pd.read_csv(inv_path, low_memory=False)
    enriched = enrich_cwpv_inventory_rows(df, label_map)
    backup = write_inventory_atomic(enriched, inv_path)

    manifests, split_meta = build_cwpv_splits(enriched)
    report_paths: dict[str, Path] = {}

    if manifests["train"].empty and manifests["validation"].empty and manifests["test"].empty:
        split_meta["warning"] = (
            "All baseline splits are empty — likely insufficient subjects per class. "
            "Re-extract full CWPV and re-run build-inventory."
        )
    else:
        leakage_df, violations = audit_leakage(manifests)
        if violations:
            raise LeakageAuditError("; ".join(violations))
        leakage_csv = reports_dir / "cwpv_split_leakage_audit.csv"
        leakage_df.to_csv(leakage_csv, index=False)
        report_paths["leakage_csv"] = leakage_csv

        write_markdown_report(
            reports_dir / "cwpv_split_leakage_audit.md",
            "CWPV Split Leakage Audit",
            {
                "Result": "PASSED — no leakage detected",
                "Checks": _dataframe_to_markdown(leakage_df) if not leakage_df.empty else "No checks",
            },
        )
        report_paths["leakage_md"] = reports_dir / "cwpv_split_leakage_audit.md"

    balance_df = class_balance_tables(enriched, manifests)

    manifest_paths: dict[str, Path] = {}
    for name, frame in manifests.items():
        filename = f"cwpv_{name}.csv" if name != "validation" else "cwpv_validation.csv"
        out = manifests_dir / filename
        frame.to_csv(out, index=False)
        manifest_paths[name] = out

    logical_summary = (
        enriched[enriched["source"] == "cwpv"]
        .drop_duplicates("logical_sample_id")[
            ["logical_sample_id", "canonical_activity", "subject_id", "raw_activity_label"]
        ]
    )
    logical_out = manifests_dir / "cwpv_logical_samples.csv"
    logical_summary.to_csv(logical_out, index=False)
    manifest_paths["logical_samples"] = logical_out

    balance_csv = reports_dir / "class_balance_cwpv_baseline.csv"
    balance_df.to_csv(balance_csv, index=False)
    report_paths["class_balance_csv"] = balance_csv

    _write_class_balance_md(balance_df, split_meta, reports_dir / "class_balance_cwpv_baseline.md")
    report_paths["class_balance_md"] = reports_dir / "class_balance_cwpv_baseline.md"

    summary = {
        "cwpv_video_files": int((enriched["source"] == "cwpv").sum()),
        "unique_logical_samples": int(
            enriched[enriched["source"] == "cwpv"]["logical_sample_id"].nunique()
        ),
        "split_meta": split_meta,
        "leakage_passed": split_meta.get("warning") is None,
        "inventory_backup": str(backup),
    }
    return CwpvBaselineResult(
        inventory_path=inv_path,
        summary=summary,
        reports=report_paths,
        manifests=manifest_paths,
    )


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join([headers, sep, *body])


def _write_class_balance_md(balance_df: pd.DataFrame, split_meta: dict, path: Path) -> None:
    lines = [
        "# CWPV Baseline Class Balance Report",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_",
        "",
        "**Note:** Each logical sample (participant × motion × trial) has up to 4 camera views.",
        "Splits are **subject-disjoint** by participant ID.",
        "",
        "## Baseline split classes",
        "",
        f"Included: {', '.join(split_meta.get('included_classes', [])) or '(none)'}",
        "",
        f"Excluded (insufficient subjects): "
        f"{', '.join(split_meta.get('excluded_classes_insufficient_subjects', [])) or '(none)'}",
        "",
        f"Total CWPV videos indexed: {split_meta.get('total_cwpv_videos', 0)}",
        "",
        "## Per-class summary",
        "",
    ]
    if not balance_df.empty:
        lines.append(_dataframe_to_markdown(balance_df))
    path.write_text("\n".join(lines), encoding="utf-8")
