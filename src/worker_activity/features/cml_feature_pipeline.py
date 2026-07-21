"""CML skeleton feature extraction pipeline (train manifest only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from worker_activity.config import find_repo_root, load_yaml, resolve_paths_config
from worker_activity.features.cml_skeleton_features import (
    LayoutConfig,
    extract_sequence_features,
    layout_config_from_yaml,
)
from worker_activity.features.cml_skeleton_io import SkeletonLoadError, load_cml_skeleton


@dataclass
class FeatureExtractionResult:
    features_15: pd.DataFrame
    features_20: pd.DataFrame
    comparison_report_path: Path
    output_paths: dict[str, Path] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)


METADATA_COLUMNS = [
    "logical_sample_id",
    "representation_group_id",
    "skeleton_layout",
    "canonical_activity",
    "raw_activity_label",
    "subject_id",
    "source_dataset",
    "relative_path",
    "source_file",
    "frame_count",
]


def load_layout_configs(path: Path | None = None) -> dict[int, LayoutConfig]:
    root = find_repo_root()
    cfg_path = path or root / "configs" / "cml_skeleton_layouts.yaml"
    raw = load_yaml(cfg_path)
    return {
        int(entry["skeleton_layout"]): layout_config_from_yaml(entry)
        for entry in raw["layouts"].values()
    }


def _extract_manifest_row(
    row: pd.Series,
    data_root: Path,
    layout_cfg: LayoutConfig,
) -> dict[str, Any]:
    rel = str(row["relative_path"])
    json_path = data_root / rel.replace("/", "\\") if "\\" in str(data_root) else data_root / rel
    skeleton = load_cml_skeleton(json_path)
    features = extract_sequence_features(skeleton["bdata"], layout_cfg)
    out: dict[str, Any] = {col: row.get(col) for col in METADATA_COLUMNS}
    out.update(features)
    out["extraction_status"] = "ok"
    return out


def extract_train_features(
    train_manifest: Path | None = None,
    *,
    data_root: Path | None = None,
) -> FeatureExtractionResult:
    root = find_repo_root()
    paths = resolve_paths_config(repo_root=root, require_data_root=True)
    data_root = data_root or paths.data_root
    assert data_root is not None

    manifest_path = train_manifest or paths.manifests_dir / "cml_train.csv"
    train_df = pd.read_csv(manifest_path)
    layouts = load_layout_configs()

    errors: list[dict[str, str]] = []
    rows_by_layout: dict[int, list[dict[str, Any]]] = {15: [], 20: []}

    for _, row in train_df.iterrows():
        layout = int(row["skeleton_layout"])
        if layout not in rows_by_layout:
            continue
        try:
            feat_row = _extract_manifest_row(row, data_root, layouts[layout])
            rows_by_layout[layout].append(feat_row)
        except (SkeletonLoadError, KeyError, ValueError) as exc:
            errors.append(
                {
                    "logical_sample_id": str(row.get("logical_sample_id")),
                    "skeleton_layout": str(layout),
                    "relative_path": str(row.get("relative_path")),
                    "error": str(exc),
                }
            )

    df15 = pd.DataFrame(rows_by_layout[15])
    df20 = pd.DataFrame(rows_by_layout[20])

    # Verify one row per logical sample per layout
    for name, df in [("15", df15), ("20", df20)]:
        if not df.empty:
            dupes = df["logical_sample_id"].duplicated().sum()
            if dupes:
                raise ValueError(f"Duplicate logical_sample_id in {name}-node features: {dupes}")

    cfg = load_yaml(root / "configs" / "cml_skeleton_layouts.yaml")
    out_cfg = cfg.get("feature_output", {})
    out_dir = root / "data" / "processed" / "cml"
    out_dir.mkdir(parents=True, exist_ok=True)

    path15 = root / out_cfg.get("train_15", "data/processed/cml/features_15_nodes_train.parquet")
    path20 = root / out_cfg.get("train_20", "data/processed/cml/features_20_nodes_train.parquet")
    path15.parent.mkdir(parents=True, exist_ok=True)
    path20.parent.mkdir(parents=True, exist_ok=True)
    df15.to_parquet(path15, index=False)
    df20.to_parquet(path20, index=False)

    report_path = root / out_cfg.get(
        "comparison_report", "reports/cml_skeleton_feature_comparison.md"
    )
    _write_comparison_report(df15, df20, errors, report_path)

    return FeatureExtractionResult(
        features_15=df15,
        features_20=df20,
        comparison_report_path=report_path,
        output_paths={"15": path15, "20": path20},
        errors=errors,
    )


def _feature_columns(df: pd.DataFrame) -> list[str]:
    skip = set(METADATA_COLUMNS) | {"extraction_status"}
    return [c for c in df.columns if c not in skip]


def _write_comparison_report(
    df15: pd.DataFrame,
    df20: pd.DataFrame,
    errors: list[dict[str, str]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CML Skeleton Feature Comparison (15-node vs 20-node)",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_",
        "",
        "**Scope:** Train manifest only (`cml_train.csv`). Features are **not combined**.",
        "Each layout has separate outputs. Logical samples are paired by `logical_sample_id`.",
        "",
        "## Sample counts",
        "",
        f"- 15-node feature rows: {len(df15)} (unique logical samples: {df15['logical_sample_id'].nunique() if not df15.empty else 0})",
        f"- 20-node feature rows: {len(df20)} (unique logical samples: {df20['logical_sample_id'].nunique() if not df20.empty else 0})",
        f"- Extraction errors: {len(errors)}",
        "",
    ]

    if df15.empty or df20.empty:
        lines.append("_Insufficient data for paired comparison._")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    merged = df15.merge(
        df20,
        on="logical_sample_id",
        suffixes=("_15", "_20"),
        how="inner",
    )
    lines.append(f"- Paired logical samples compared: {len(merged)}")
    lines.append("")
    lines.append("## Paired feature comparison (15 vs 20, same logical sample)")
    lines.append("")
    lines.append(
        "Common features present in both layouts are compared. "
        "20-node-only features (e.g. wrist angles) are listed separately."
    )
    lines.append("")

    common_feats = []
    for col in _feature_columns(df15):
        if col in _feature_columns(df20):
            common_feats.append(col)

    comp_rows = []
    for feat in sorted(common_feats):
        a = merged[f"{feat}_15"].astype(float)
        b = merged[f"{feat}_20"].astype(float)
        valid = ~(a.isna() | b.isna())
        if valid.sum() == 0:
            continue
        diff = (a[valid] - b[valid]).abs()
        corr = np.corrcoef(a[valid], b[valid])[0, 1] if valid.sum() > 1 else float("nan")
        comp_rows.append(
            {
                "feature": feat,
                "paired_n": int(valid.sum()),
                "mean_abs_diff": float(diff.mean()),
                "max_abs_diff": float(diff.max()),
                "pearson_r": float(corr),
            }
        )

    if comp_rows:
        comp_df = pd.DataFrame(comp_rows).sort_values("mean_abs_diff", ascending=False)
        lines.append("| feature | paired_n | mean_abs_diff | max_abs_diff | pearson_r |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for _, r in comp_df.head(25).iterrows():
            lines.append(
                f"| {r['feature']} | {r['paired_n']} | {r['mean_abs_diff']:.4f} | "
                f"{r['max_abs_diff']:.4f} | {r['pearson_r']:.4f} |"
            )
        lines.append("")

    only_20 = [c for c in _feature_columns(df20) if c not in _feature_columns(df15)]
    if only_20:
        lines.append("## 20-node-only features")
        lines.append("")
        for feat in only_20:
            lines.append(f"- `{feat}`")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Features are label-agnostic; `canonical_activity` is metadata only.")
    lines.append("- Do not combine layouts for training until comparison is reviewed.")
    lines.append("- Taxonomy is not frozen; CWPV inspection may revise mappings.")
    if errors:
        lines.append("")
        lines.append("## Extraction errors")
        for err in errors[:20]:
            lines.append(f"- `{err['logical_sample_id']}` layout {err['skeleton_layout']}: {err['error']}")

    path.write_text("\n".join(lines), encoding="utf-8")
