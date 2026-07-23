"""End-to-end Week 6 report tying activity + ergonomics + zone streams."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from worker_activity.config import find_repo_root, resolve_paths_config
from worker_activity.ergonomics.screening import screen_ergonomics
from worker_activity.reporting.markdown import (
    bullet_list,
    dataframe_to_markdown,
    write_markdown_report,
)
from worker_activity.zones.events import detect_zone_events


@dataclass
class Week6ReportResult:
    report_path: Path
    ergonomics_path: Path | None = None
    zone_events_path: Path | None = None
    warnings: list[str] = field(default_factory=list)


def _load_activity_section(paths) -> tuple[str, list[str]]:
    warnings: list[str] = []
    parts: list[str] = [
        "Activity classification is **workstream A** (separate from ergonomic indicators and zone events).",
        "No retraining in Week 6; predictions reuse existing CWPV-trained baselines.",
        "",
    ]

    phone_pred = paths.reports_dir / "domain_transfer_predictions.csv"
    if phone_pred.is_file():
        df = pd.read_csv(phone_pred)
        # Prefer random_forest if present
        if "model" in df.columns and (df["model"] == "random_forest").any():
            show = df[df["model"] == "random_forest"]
        else:
            show = df
        cols = [c for c in ["file_name", "canonical_activity", "predicted", "correct", "model"] if c in show.columns]
        parts.append("### Smartphone predictions (domain transfer)")
        parts.append("")
        parts.append(dataframe_to_markdown(show[cols]) if cols else "_No columns._")
        parts.append("")
    else:
        warnings.append(
            "domain_transfer_predictions.csv missing — run evaluate-domain-transfer for phone activity section."
        )
        parts.append("_Smartphone activity predictions not found._")
        parts.append("")

    cwpv_pred = paths.reports_dir / "cwpv_baseline_val_predictions.csv"
    if not cwpv_pred.is_file():
        # try alternate name from classifier
        alt = paths.reports_dir / "cwpv_baseline_predictions.csv"
        cwpv_pred = alt if alt.is_file() else cwpv_pred
    if cwpv_pred.is_file():
        df = pd.read_csv(cwpv_pred)
        if "model" in df.columns and (df["model"] == "random_forest").any():
            show = df[df["model"] == "random_forest"].head(15)
        else:
            show = df.head(15)
        cols = [
            c
            for c in ["file_name", "canonical_activity", "predicted", "correct", "model", "split"]
            if c in show.columns
        ]
        parts.append("### CWPV validation predictions (sample)")
        parts.append("")
        parts.append(dataframe_to_markdown(show[cols]) if cols else "_No columns._")
        parts.append("")
        parts.append("_CWPV test split was not used._")
    else:
        warnings.append("CWPV validation predictions CSV not found — activity section limited to phone.")
        parts.append("_CWPV activity predictions not found._")

    return "\n".join(parts), warnings


def build_week6_report(
    *,
    source: str = "both",
    max_videos: int | None = None,
    skip_pipelines: bool = False,
) -> Week6ReportResult:
    """Optionally re-run screening/zones, then write the combined markdown report."""
    root = find_repo_root()
    paths = resolve_paths_config(repo_root=root)
    warnings: list[str] = [
        "Workstreams A (activity), B (ergonomics), and C (zones) are reported separately and must not be conflated.",
        "CWPV test split was not used for training, tuning, or Week 6 demos.",
    ]

    ergo_path: Path | None = None
    zone_path: Path | None = None
    ergo_df = pd.DataFrame()
    zone_summary = pd.DataFrame()
    zone_events = pd.DataFrame()

    if not skip_pipelines:
        ergo = screen_ergonomics(source=source, max_videos=max_videos)
        ergo_path = ergo.output_path
        ergo_df = ergo.metrics
        for err in ergo.errors:
            warnings.append(f"Ergonomics: {err['relative_path']}: {err['error']}")

        zones = detect_zone_events(source=source, max_videos=max_videos)
        zone_path = zones.events_path
        zone_summary = zones.summary
        zone_events = zones.events
        for err in zones.errors:
            warnings.append(f"Zones: {err['relative_path']}: {err['error']}")
    else:
        ergo_csv = paths.processed_dir / "week6" / "ergonomic_screening.csv"
        zone_csv = paths.processed_dir / "week6" / "zone_events.csv"
        zone_sum = paths.processed_dir / "week6" / "zone_events_summary.csv"
        if ergo_csv.is_file():
            ergo_df = pd.read_csv(ergo_csv)
            ergo_path = ergo_csv
        else:
            warnings.append("Skipped pipelines and no ergonomic_screening.csv found.")
        if zone_csv.is_file():
            zone_events = pd.read_csv(zone_csv)
            zone_path = zone_csv
        if zone_sum.is_file():
            zone_summary = pd.read_csv(zone_sum)

    activity_body, act_warnings = _load_activity_section(paths)
    warnings.extend(act_warnings)

    if not ergo_df.empty:
        ergo_cols = [
            c
            for c in [
                "source",
                "file_name",
                "canonical_activity",
                "repeated_bending_count",
                "overhead_duration_s",
                "kneeling_duration_s",
                "squatting_duration_s",
                "awkward_posture_duration_s",
            ]
            if c in ergo_df.columns
        ]
        ergo_body = (
            "Ergonomic indicators are **workstream B** (duration/frequency proxies from joint angles).\n\n"
            + dataframe_to_markdown(ergo_df[ergo_cols])
            + "\n\n_Not activity labels; not certified ergonomic assessment._"
        )
    else:
        ergo_body = "_No ergonomic screening rows._"

    if not zone_summary.empty:
        n_entry = int((zone_events["event_type"] == "restricted_zone_entry").sum()) if not zone_events.empty else 0
        n_exit = int((zone_events["event_type"] == "restricted_zone_exit").sum()) if not zone_events.empty else 0
        zone_body = (
            "Zone events are **workstream C** (polygon crossings — not activity classes).\n\n"
            f"Total entry events: **{n_entry}**; exit events: **{n_exit}**.\n\n"
            + dataframe_to_markdown(zone_summary)
            + "\n\n_Demo polygons from `configs/zones_demo.yaml` (normalized image coords)._"
        )
    else:
        zone_body = "_No zone-event summary rows._"

    report_path = paths.reports_dir / "week6_end_to_end.md"
    write_markdown_report(
        report_path,
        "Week 6 End-to-End Report",
        {
            "Purpose": (
                "Show how activity outputs, screening-level ergonomic indicators, and restricted-zone "
                "events fit together from ordinary construction / phone video **without conflating** them."
            ),
            "A — Activity recognition": activity_body,
            "B — Ergonomic screening": ergo_body,
            "C — Zone events": zone_body,
            "Hard constraints": bullet_list(
                [
                    "Do not treat zone events or ergonomic indicators as activity class labels.",
                    "Do not use CWPV test for training or hyperparameter tuning.",
                    "Screening outputs are research prototypes only.",
                ]
            ),
            "Warnings / notes": bullet_list(warnings),
            "Artifacts": bullet_list(
                [
                    f"Report: {report_path}",
                    f"Ergonomics CSV: {ergo_path}" if ergo_path else "Ergonomics CSV: (missing)",
                    f"Zone events CSV: {zone_path}" if zone_path else "Zone events CSV: (missing)",
                    f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
                ]
            ),
        },
    )
    return Week6ReportResult(
        report_path=report_path,
        ergonomics_path=ergo_path,
        zone_events_path=zone_path,
        warnings=warnings,
    )
