"""Generate CWPV label inspection and class-balance reports."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from worker_activity.config import find_repo_root, resolve_paths_config
from worker_activity.data.cwpv_metadata import CWPV_MOTION_LABELS
from worker_activity.reporting.markdown import write_markdown_report


def build_cwpv_inspection_reports(
    inventory: pd.DataFrame,
    *,
    reports_dir: Path | None = None,
) -> dict[str, Path]:
    root = find_repo_root()
    paths = resolve_paths_config(repo_root=root)
    out_dir = reports_dir or paths.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cwpv = inventory[inventory["source"] == "cwpv"].copy()
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    subjects = cwpv["subject_id"].dropna() if "subject_id" in cwpv.columns else pd.Series(dtype=object)
    raw_labels = (
        cwpv["raw_activity_label"].dropna()
        if "raw_activity_label" in cwpv.columns
        else pd.Series(dtype=object)
    )
    # Drop string "nan" / empty placeholders from CSV round-trips
    subjects = subjects[subjects.astype(str).str.lower().isin(["", "nan", "none"]) == False]
    raw_labels = raw_labels[raw_labels.astype(str).str.lower().isin(["", "nan", "none"]) == False]

    label_lines = [
        "# CWPV Label Inspection",
        "",
        f"_Generated: {generated}_",
        "",
        "Motion IDs follow CWPV README: filename `PPPMCamera_C.ext` where P=participant, M=motion, T=trial.",
        "",
        "## Motion catalog (from README)",
        "",
        "| Motion | Description |",
        "| --- | --- |",
    ]
    descriptions = {
        "1": "Standing, twist torso, overhead hammer",
        "2": "Standing, lean/twist, overhead hammer",
        "3": "Semi-squat shoveling",
        "4": "Squat lift to chest, hold, lower",
        "5": "Squat hammer at 50 cm",
        "6": "Kneel hammer at ground",
        "7": "Squat one-handed brick carry",
        "8": "Standing lean-forward brick transport",
    }
    for mid, label in CWPV_MOTION_LABELS.items():
        label_lines.append(f"| M{mid} ({label}) | {descriptions.get(mid, '')} |")

    label_lines.extend(
        [
            "",
            "## Inventory summary",
            "",
            f"- Video files indexed: **{len(cwpv)}**",
            f"- Unique subjects: **{subjects.nunique()}**",
            f"- Unique motions (raw labels): **{raw_labels.nunique()}**",
            "",
            "Provisional canonical mappings are in `configs/label_map_cwpv.yaml` (not frozen).",
            "",
            "Note: run `build-cwpv-baseline` after inventory to populate canonical labels and splits.",
        ]
    )
    label_path = out_dir / "cwpv_label_inspection.md"
    label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    balance_path = out_dir / "class_balance_cwpv.md"
    if len(cwpv) == 0:
        balance_path.write_text(
            "\n".join(
                [
                    "# CWPV Class Balance Report",
                    "",
                    f"_Generated: {generated}_",
                    "",
                    "No CWPV videos in inventory yet. Download and extract Video Data.rar, then run `build-inventory`.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        grouped = (
            cwpv.groupby(["canonical_activity", "raw_activity_label"], dropna=False)
            .agg(
                video_files=("file_name", "count"),
                subjects=("subject_id", "nunique"),
                cameras=("view_id", "nunique"),
            )
            .reset_index()
            .sort_values(["canonical_activity", "raw_activity_label"])
        )
        lines = [
            "# CWPV Class Balance Report",
            "",
            f"_Generated: {generated}_",
            "",
            f"Total video files: **{len(cwpv)}**",
            "",
            "| canonical_activity | raw_activity_label | video_files | subjects | cameras |",
            "| --- | --- | --- | --- | --- |",
        ]
        for _, row in grouped.iterrows():
            lines.append(
                f"| {row['canonical_activity']} | {row['raw_activity_label']} | "
                f"{int(row['video_files'])} | {int(row['subjects'])} | {int(row['cameras'])} |"
            )
        balance_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"label_inspection": label_path, "class_balance": balance_path}
