"""CML baseline pipeline: mapping, subjects, splits, reports."""

from __future__ import annotations

import random
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from worker_activity.config import find_repo_root, load_yaml, resolve_paths_config
from worker_activity.data.cml_label_map import apply_label_mapping, is_calibration_label, load_cml_label_map
from worker_activity.data.cml_representation import (
    assign_representation_fields,
    audit_representations,
    extract_notes_field,
)
from worker_activity.data.cml_subject import parse_subject_id
from worker_activity.reporting.markdown import bullet_list, write_markdown_report

MIN_SUBJECTS_PER_BASELINE_CLASS = 3
SPLIT_SEED = 42
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


class LeakageAuditError(Exception):
    """Raised when split leakage is detected."""


@dataclass
class CmlBaselineResult:
    inventory_path: Path
    summary: dict[str, Any] = field(default_factory=dict)
    reports: dict[str, Path] = field(default_factory=dict)
    manifests: dict[str, Path] = field(default_factory=dict)


def _parse_inventory_notes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["source_dataset"] = out["notes"].apply(lambda n: extract_notes_field(n, "data_source"))
    out["source_file"] = out["video_id"]
    out["construction_subset"] = out["notes"].str.contains("subset=construction_related", na=False)
    return out


def enrich_cml_inventory_rows(df: pd.DataFrame, label_map: dict) -> pd.DataFrame:
    cml_mask = df["source"] == "cml"
    out = df.copy()

    for col in [
        "canonical_activity",
        "label_mapping_status",
        "subject_id",
        "mapping_confidence",
        "mapping_status",
        "include_in_baseline",
        "exclusion_reason",
        "subject_parse_status",
        "source_file",
        "source_dataset",
        "skeleton_layout",
        "logical_sample_id",
        "representation_group_id",
        "construction_subset",
    ]:
        if col not in out.columns:
            out[col] = None

    out.loc[cml_mask, "source_file"] = out.loc[cml_mask, "video_id"]
    out = _parse_inventory_notes(out)

    mapping_rows = out.loc[cml_mask, "raw_activity_label"].apply(
        lambda lbl: apply_label_mapping(lbl, label_map)
    )
    if not mapping_rows.empty:
        mapping_df = pd.DataFrame(mapping_rows.tolist(), index=mapping_rows.index)
        for col in mapping_df.columns:
            if col not in out.columns:
                out[col] = None
            out[col] = out[col].astype("object")
            out.loc[cml_mask, col] = mapping_df[col].values

    def _parse_row(row: pd.Series) -> dict[str, Any]:
        result = parse_subject_id(row.get("source_dataset"), row.get("source_file"))
        return {
            "subject_id": result.subject_id,
            "subject_parse_status": result.subject_parse_status,
        }

    subject_data = out.loc[cml_mask].apply(_parse_row, axis=1, result_type="expand")
    if "subject_id" not in out.columns:
        out["subject_id"] = None
    if "subject_parse_status" not in out.columns:
        out["subject_parse_status"] = None
    out["subject_id"] = out["subject_id"].astype("object")
    out["subject_parse_status"] = out["subject_parse_status"].astype("object")
    out.loc[cml_mask, "subject_id"] = subject_data["subject_id"]
    out.loc[cml_mask, "subject_parse_status"] = subject_data["subject_parse_status"]

    cml_enriched = assign_representation_fields(out.loc[cml_mask].copy())
    for col in cml_enriched.columns:
        out.loc[cml_mask, col] = cml_enriched[col].values
    return out


def build_source_path_profile(df: pd.DataFrame) -> pd.DataFrame:
    con = df[(df["source"] == "cml") & (df["construction_subset"] == True)].copy()
    rows = []
    for _, row in con.drop_duplicates(["source_dataset", "source_file"]).iterrows():
        sf = str(row.get("source_file") or "")
        rows.append(
            {
                "source_dataset": row.get("source_dataset"),
                "source_file": sf,
                "path_depth": len(Path(sf).parts) if sf else 0,
                "filename": Path(sf).name if sf else "",
                "raw_activity_label": row.get("raw_activity_label"),
                "subject_id": row.get("subject_id"),
                "subject_parse_status": row.get("subject_parse_status"),
            }
        )
    return pd.DataFrame(rows)


def split_subjects(
    subjects: list[str],
    *,
    seed: int = SPLIT_SEED,
    ratios: dict[str, float] | None = None,
) -> dict[str, str]:
    ratios = ratios or SPLIT_RATIOS
    rng = random.Random(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    assignment: dict[str, str] = {}
    for sid in shuffled[:n_train]:
        assignment[sid] = "train"
    for sid in shuffled[n_train : n_train + n_val]:
        assignment[sid] = "validation"
    for sid in shuffled[n_train + n_val :]:
        assignment[sid] = "test"
    return assignment


def build_cml_splits(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    con = df[(df["source"] == "cml") & (df["construction_subset"] == True)].copy()
    calibration_mask = con["raw_activity_label"].apply(is_calibration_label)
    excluded_cal = con[calibration_mask]
    con = con[~calibration_mask]

    baseline_eligible = con[con["include_in_baseline"] == True].copy()
    excluded_not_baseline = con[con["include_in_baseline"] != True]

    unresolved = baseline_eligible[
        baseline_eligible["subject_id"].isna()
        | baseline_eligible["subject_parse_status"].isin(
            ["unresolved_pattern", "missing_source_file", "invalid_source_file"]
        )
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
        [
            excluded_not_baseline,
            excluded_cal,
            class_excluded_rows,
        ],
        ignore_index=True,
    )

    meta = {
        "included_classes": included_classes,
        "excluded_classes_insufficient_subjects": excluded_classes,
        "unique_subjects_split": len(subjects),
        "unresolved_rows": len(unresolved),
        "excluded_rows": len(excluded_frames),
    }
    manifests = {
        "train": train,
        "validation": val,
        "test": test,
        "unresolved_subject": unresolved,
        "excluded": excluded_frames,
    }
    return manifests, meta


def audit_leakage(manifests: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    violations: list[str] = []
    rows: list[dict[str, Any]] = []

    train = manifests["train"]
    val = manifests["validation"]
    test = manifests["test"]

    def _collect_ids(frame: pd.DataFrame, split: str, column: str) -> set[str]:
        return set(frame[column].dropna().astype(str).unique())

    for column in ["subject_id", "logical_sample_id", "representation_group_id"]:
        train_ids = _collect_ids(train, "train", column)
        val_ids = _collect_ids(val, "validation", column)
        test_ids = _collect_ids(test, "test", column)
        overlap_tv = train_ids & val_ids
        overlap_tt = train_ids & test_ids
        overlap_vt = val_ids & test_ids
        rows.append(
            {
                "check": f"{column}_overlap",
                "train_val_overlap": len(overlap_tv),
                "train_test_overlap": len(overlap_tt),
                "val_test_overlap": len(overlap_vt),
                "passed": len(overlap_tv) == 0 and len(overlap_tt) == 0 and len(overlap_vt) == 0,
            }
        )
        if overlap_tv:
            violations.append(f"{column} overlap train/val: {len(overlap_tv)}")
        if overlap_tt:
            violations.append(f"{column} overlap train/test: {len(overlap_tt)}")
        if overlap_vt:
            violations.append(f"{column} overlap val/test: {len(overlap_vt)}")

    if (manifests["train"]["include_in_baseline"] == False).any():
        violations.append("include_in_baseline=false rows in train manifest")
    if manifests["train"]["raw_activity_label"].apply(is_calibration_label).any():
        violations.append("calibration rows in train manifest")

    audit_df = pd.DataFrame(rows)
    return audit_df, violations


def class_balance_tables(df: pd.DataFrame, manifests: dict[str, pd.DataFrame]) -> pd.DataFrame:
    con = df[(df["source"] == "cml") & (df["construction_subset"] == True)].copy()
    rows = []
    for canonical in sorted(con["canonical_activity"].dropna().unique()):
        class_df = con[con["canonical_activity"] == canonical]
        logical = class_df.drop_duplicates("logical_sample_id")
        mapped_raw = sorted(class_df["raw_activity_label"].dropna().unique().tolist())
        row: dict[str, Any] = {
            "canonical_activity": canonical,
            "mapped_raw_labels": "; ".join(mapped_raw),
            "raw_json_file_count": len(class_df),
            "unique_logical_sample_count": len(logical),
            "unique_parsed_subject_count": logical["subject_id"].nunique(),
            "unresolved_subject_logical_samples": int(
                logical["subject_id"].isna().sum()
            ),
            "count_skeleton_15": int((class_df["skeleton_layout"] == 15).sum()),
            "count_skeleton_20": int((class_df["skeleton_layout"] == 20).sum()),
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


def run_cml_baseline_pipeline(
    inventory_path: Path | None = None,
    *,
    repo_root: Path | None = None,
) -> CmlBaselineResult:
    root = repo_root or find_repo_root()
    paths = resolve_paths_config(repo_root=root)
    inv_path = inventory_path or paths.manifests_dir / "clip_inventory.csv"
    reports_dir = paths.reports_dir
    manifests_dir = paths.manifests_dir

    label_map = load_cml_label_map()
    df = pd.read_csv(inv_path)
    enriched = enrich_cml_inventory_rows(df, label_map)
    backup = write_inventory_atomic(enriched, inv_path)

    con = enriched[(enriched["source"] == "cml") & (enriched["construction_subset"] == True)]
    rep_audit_df, rep_summary = audit_representations(con)
    anomalies = rep_audit_df[rep_audit_df["status"] != "ok"] if not rep_audit_df.empty else rep_audit_df

    profile_df = build_source_path_profile(enriched)
    manifests, split_meta = build_cml_splits(enriched)
    leakage_df, violations = audit_leakage(manifests)
    if violations:
        raise LeakageAuditError("; ".join(violations))

    balance_df = class_balance_tables(enriched, manifests)
    logical_summary = (
        con.drop_duplicates("logical_sample_id")[["logical_sample_id", "canonical_activity", "subject_id"]]
    )

    # Write manifests
    manifest_paths: dict[str, Path] = {}
    for name, frame in manifests.items():
        filename = f"cml_{name}.csv" if name != "validation" else "cml_validation.csv"
        out = manifests_dir / filename
        frame.to_csv(out, index=False)
        manifest_paths[name] = out

    logical_out = manifests_dir / "cml_logical_samples.csv"
    logical_summary.to_csv(logical_out, index=False)
    manifest_paths["logical_samples"] = logical_out

    # Reports
    report_paths: dict[str, Path] = {}
    profile_path = reports_dir / "cml_source_path_patterns.csv"
    profile_df.to_csv(profile_path, index=False)
    report_paths["source_path_patterns"] = profile_path

    unresolved_paths = con[
        (con["include_in_baseline"] == True) & con["subject_id"].isna()
    ][["source_dataset", "source_file", "raw_activity_label", "subject_parse_status"]].drop_duplicates()
    unresolved_csv = reports_dir / "cml_unresolved_subject_paths.csv"
    unresolved_paths.to_csv(unresolved_csv, index=False)
    report_paths["unresolved_subject_paths"] = unresolved_csv

    rep_audit_path = reports_dir / "cml_representation_audit.md"
    write_markdown_report(
        rep_audit_path,
        "CML Representation Audit",
        {
            "Summary": (
                f"Logical samples: {rep_summary.get('total_logical_samples', 0)}\n"
                f"OK pairs (15+20): {rep_summary.get('ok_pairs', 0)}\n"
                f"Unexpected count: {rep_summary.get('unexpected_count', 0)}\n"
                f"Missing layout: {rep_summary.get('missing_layout', 0)}\n"
                f"Inconsistent labels: {rep_summary.get('inconsistent_label', 0)}"
            ),
        },
    )
    report_paths["representation_audit"] = rep_audit_path

    anomalies_path = reports_dir / "cml_representation_anomalies.csv"
    anomalies.to_csv(anomalies_path, index=False)
    report_paths["representation_anomalies"] = anomalies_path

    balance_csv = reports_dir / "class_balance_cml.csv"
    balance_df.to_csv(balance_csv, index=False)
    report_paths["class_balance_csv"] = balance_csv

    _write_class_balance_md(balance_df, split_meta, reports_dir / "class_balance_cml.md")
    report_paths["class_balance_md"] = reports_dir / "class_balance_cml.md"

    leakage_csv = reports_dir / "cml_split_leakage_audit.csv"
    leakage_df.to_csv(leakage_csv, index=False)
    report_paths["leakage_csv"] = leakage_csv

    write_markdown_report(
        reports_dir / "cml_split_leakage_audit.md",
        "CML Split Leakage Audit",
        {
            "Result": "PASSED — no leakage detected",
            "Checks": _dataframe_to_markdown(leakage_df) if not leakage_df.empty else "No checks",
        },
    )
    report_paths["leakage_md"] = reports_dir / "cml_split_leakage_audit.md"

    _write_subject_parsing_md(enriched, reports_dir / "cml_subject_parsing.md")
    report_paths["subject_parsing"] = reports_dir / "cml_subject_parsing.md"

    _write_mapping_review_md(label_map, root, reports_dir / "cml_mapping_decisions_for_review.md")
    report_paths["mapping_review"] = reports_dir / "cml_mapping_decisions_for_review.md"

    parsed = con["subject_id"].notna().sum()
    summary = {
        "cml_representation_files": int(len(con)),
        "unique_logical_samples": int(con["logical_sample_id"].nunique()),
        "parsed_subject_rows": int(parsed),
        "parsed_subject_pct": round(100.0 * parsed / max(len(con), 1), 2),
        "split_meta": split_meta,
        "representation_summary": rep_summary,
        "leakage_passed": True,
        "inventory_backup": str(backup),
    }
    return CmlBaselineResult(inventory_path=inv_path, summary=summary, reports=report_paths, manifests=manifest_paths)


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
        "# CML Class Balance Report",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_",
        "",
        "**Note:** Raw JSON file counts double-count 15-node and 20-node representations.",
        "Use `unique_logical_sample_count` for independent motion samples.",
        "",
        "## Baseline split classes",
        "",
        f"Included: {', '.join(split_meta.get('included_classes', [])) or '(none)'}",
        "",
        f"Excluded (insufficient subjects): {', '.join(split_meta.get('excluded_classes_insufficient_subjects', [])) or '(none)'}",
        "",
        "## Per-class summary",
        "",
    ]
    if not balance_df.empty:
        lines.append(_dataframe_to_markdown(balance_df))
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_subject_parsing_md(df: pd.DataFrame, path: Path) -> None:
    con = df[(df["source"] == "cml") & (df["construction_subset"] == True)]
    lines = ["# CML Subject Parsing Report", ""]
    by_source = (
        con.groupby("source_dataset")
        .agg(
            rows=("source_file", "count"),
            parsed=("subject_id", lambda s: int(s.notna().sum())),
            unique_subjects=("subject_id", "nunique"),
        )
        .reset_index()
    )
    lines.append(_dataframe_to_markdown(by_source))
    lines.append("")
    lines.append("## Parse statuses")
    lines.append(_dataframe_to_markdown(con["subject_parse_status"].value_counts().reset_index(name="count")))
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_mapping_review_md(label_map: dict, root: Path, path: Path) -> None:
    data = load_yaml(root / "configs" / "label_map_cml.yaml")
    lines = [
        "# CML Mapping Decisions for Review",
        "",
        "**Status:** Provisional CML-only — NOT user-confirmed. CWPV may revise taxonomy.",
        "",
        "## Decision history (previously ambiguous)",
        "",
    ]
    for item in data.get("decision_history", []):
        lines.extend(
            [
                f"### `{item['cml_label']}`",
                f"- Previous candidates: {item.get('previous_candidates')}",
                f"- Provisional canonical: `{item.get('provisional_canonical')}`",
                f"- Include in baseline: {item.get('include_in_baseline')}",
                f"- Rationale: {item.get('rationale')}",
                "",
            ]
        )
    lines.append("## Warning")
    lines.append("These mappings are provisional. Do not freeze taxonomy until CWPV is inspected.")
    path.write_text("\n".join(lines), encoding="utf-8")
