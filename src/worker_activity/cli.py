"""Command-line interface."""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

from worker_activity import __version__
from worker_activity.config import (
    DATA_ROOT_ENV_VAR,
    ConfigurationError,
    ensure_repo_output_dirs,
    resolve_paths_config,
    validate_paths_config,
)
from worker_activity.data.cml_baseline import LeakageAuditError, run_cml_baseline_pipeline
from worker_activity.data.cwpv_baseline import run_cwpv_baseline_pipeline
from worker_activity.data.cwpv_inspection import build_cwpv_inspection_reports
from worker_activity.features.cml_feature_pipeline import extract_train_features
from worker_activity.features.cwpv_feature_pipeline import extract_cwpv_features
from worker_activity.features.smartphone_feature_pipeline import extract_smartphone_features
from worker_activity.models.baseline_classifier import train_cwpv_baseline_classifiers
from worker_activity.models.domain_transfer import evaluate_domain_transfer
from worker_activity.data.inventory import audit_data_sources, build_inventory
from worker_activity.data.manifests import write_inventory_csv, write_inventory_parquet
from worker_activity.data.schema import validate_inventory_frame
from worker_activity.data.source_registry import (
    format_source_summary,
    load_source_registry,
    validate_source_registry,
)
from worker_activity.data.taxonomy import load_taxonomy_candidate
from worker_activity.pose.pipeline import extract_pose_from_inventory
from worker_activity.reporting.markdown import bullet_list, write_markdown_report
from worker_activity.viz import render_annotated_video
from worker_activity.ergonomics.screening import screen_ergonomics
from worker_activity.zones.events import detect_zone_events
from worker_activity.week6_report import build_week6_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="worker_activity",
        description="Construction worker activity recognition — data foundation tools",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("show-sources", help="Display registered data sources")

    sub.add_parser("audit-environment", help="Check Python environment and tooling")

    sub.add_parser("audit-data", help="Audit external dataset directories")

    sub.add_parser("validate-config", help="Validate YAML configuration files")

    sub.add_parser(
        "build-cml-baseline",
        help="Apply CML mapping, subject parsing, splits, and reports",
    )

    sub.add_parser(
        "build-cwpv-baseline",
        help="Apply CWPV mapping, subject-disjoint splits, and manifests",
    )

    sub.add_parser(
        "extract-cml-features",
        help="Extract skeleton features from cml_train.csv (15 and 20 node separately)",
    )

    extract_cwpv_feat = sub.add_parser(
        "extract-cwpv-features",
        help="Extract MediaPipe pose features from CWPV train/validation manifests",
    )
    extract_cwpv_feat.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit videos per split (for smoke tests)",
    )
    extract_cwpv_feat.add_argument(
        "--view",
        type=str,
        default=None,
        help="Optional view_id filter (e.g. camera_1) to reduce runtime",
    )

    sub.add_parser(
        "train-cwpv-baseline",
        help="Train baseline classifiers on train split; evaluate validation only",
    )

    extract_phone = sub.add_parser(
        "extract-smartphone-features",
        help="Extract MediaPipe pose features from local smartphone clips",
    )
    extract_phone.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit videos (for smoke tests)",
    )

    sub.add_parser(
        "evaluate-domain-transfer",
        help="Score smartphone features with CWPV-trained baselines (eval only)",
    )

    screen_ergo = sub.add_parser(
        "screen-ergonomics",
        help="Screening-level ergonomic duration/frequency indicators from pose",
    )
    screen_ergo.add_argument(
        "--source",
        choices=["both", "phone", "cwpv"],
        default="both",
        help="Which videos to screen (default: both phone + CWPV sample)",
    )
    screen_ergo.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit videos per selected source (smoke tests)",
    )

    zone_events = sub.add_parser(
        "detect-zone-events",
        help="Detect restricted-zone entry/exit vs demo polygons",
    )
    zone_events.add_argument(
        "--source",
        choices=["both", "phone", "cwpv"],
        default="both",
        help="Which videos to process (default: both phone + CWPV sample)",
    )
    zone_events.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit videos per selected source (smoke tests)",
    )

    week6 = sub.add_parser(
        "build-week6-report",
        help="End-to-end Week 6 report (activity + ergonomics + zones)",
    )
    week6.add_argument(
        "--source",
        choices=["both", "phone", "cwpv"],
        default="both",
        help="Which videos to include in B/C pipelines",
    )
    week6.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit videos per selected source",
    )
    week6.add_argument(
        "--skip-pipelines",
        action="store_true",
        help="Reuse existing week6 CSVs instead of re-running screening/zones",
    )

    build_inv = sub.add_parser(
        "build-inventory",
        help="Build clip inventory from external data directories",
    )
    build_inv.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip video metadata extraction (faster directory scan)",
    )
    build_inv.add_argument(
        "--checksums",
        action="store_true",
        help="Compute SHA-256 checksums (slow on large datasets)",
    )
    build_inv.add_argument(
        "--all-sources",
        action="store_true",
        help="Include disabled sources in scan",
    )

    inspect_cwpv = sub.add_parser(
        "inspect-cwpv",
        help="Generate CWPV label inspection and class-balance reports from inventory",
    )
    inspect_cwpv.add_argument(
        "--inventory",
        type=Path,
        default=None,
        help="Path to clip_inventory.csv (default: data/manifests/clip_inventory.csv)",
    )

    extract_pose = sub.add_parser(
        "extract-pose",
        help="Extract MediaPipe pose landmarks from CWPV videos (subset)",
    )
    extract_pose.add_argument("--subject", default=None, help="Participant ID, e.g. 09")
    extract_pose.add_argument("--motion", default=None, help="Motion ID 1-8")
    extract_pose.add_argument("--camera", default=None, help="Camera ID 1-4")
    extract_pose.add_argument("--max-videos", type=int, default=2)
    extract_pose.add_argument("--max-frames", type=int, default=None)

    render_ann = sub.add_parser(
        "render-annotated",
        help="Render annotated video with pose landmarks drawn",
    )
    render_ann.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Absolute path to input video",
    )
    render_ann.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path (default: outputs/cwpv/annotated/)",
    )
    render_ann.add_argument("--max-frames", type=int, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "show-sources": cmd_show_sources,
        "audit-environment": cmd_audit_environment,
        "audit-data": cmd_audit_data,
        "validate-config": cmd_validate_config,
        "build-cml-baseline": cmd_build_cml_baseline,
        "build-cwpv-baseline": cmd_build_cwpv_baseline,
        "extract-cml-features": cmd_extract_cml_features,
        "extract-cwpv-features": cmd_extract_cwpv_features,
        "train-cwpv-baseline": cmd_train_cwpv_baseline,
        "extract-smartphone-features": cmd_extract_smartphone_features,
        "evaluate-domain-transfer": cmd_evaluate_domain_transfer,
        "screen-ergonomics": cmd_screen_ergonomics,
        "detect-zone-events": cmd_detect_zone_events,
        "build-week6-report": cmd_build_week6_report,
        "build-inventory": cmd_build_inventory,
        "inspect-cwpv": cmd_inspect_cwpv,
        "extract-pose": cmd_extract_pose,
        "render-annotated": cmd_render_annotated,
    }
    try:
        return handlers[args.command](args)
    except ConfigurationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def cmd_show_sources(_args: argparse.Namespace) -> int:
    sources = load_source_registry()
    print(f"Registered data sources ({len(sources)}):\n")
    for source in sources:
        print("-" * 72)
        print(format_source_summary(source))
        if source.known_limitations:
            print("Limitations:")
            for item in source.known_limitations:
                print(f"  - {item}")
        print()
    print(f"Registry file: {resolve_paths_config().repo_root / 'configs' / 'data_sources.yaml'}")
    return 0


def cmd_audit_environment(_args: argparse.Namespace) -> int:
    paths = resolve_paths_config()
    ensure_repo_output_dirs(paths)
    report_path = paths.reports_dir / "environment_audit_after_setup.txt"

    lines = [
        "Environment audit",
        f"Python: {platform.python_version()} ({sys.executable})",
        f"Platform: {platform.platform()}",
        f"Package version: {__version__}",
        "",
        "Dependency import check:",
    ]

    packages = ["numpy", "pandas", "cv2", "yaml", "pyarrow", "mediapipe", "tqdm", "pytest"]
    for name in packages:
        module = "cv2" if name == "cv2" else name
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            lines.append(f"  {name}: OK ({version})")
        except ImportError as exc:
            lines.append(f"  {name}: MISSING ({exc})")

    lines.extend(
        [
            "",
            "Tooling:",
            f"  ffmpeg:  {'found' if shutil.which('ffmpeg') else 'NOT FOUND'}",
            f"  ffprobe: {'found' if shutil.which('ffprobe') else 'NOT FOUND'}",
            "",
            f"  {DATA_ROOT_ENV_VAR}: {paths.data_root or 'NOT SET'}",
            f"  Metadata provider: {paths.metadata_provider}",
            "",
            "Installed distribution (worker-activity):",
        ]
    )
    try:
        dist_version = importlib_metadata.version("worker-activity")
        lines.append(f"  worker-activity {dist_version} (editable install)")
    except importlib_metadata.PackageNotFoundError:
        lines.append("  worker-activity not installed as package (use pip install -e .)")

    text = "\n".join(lines)
    print(text)
    report_path.write_text(text + "\n", encoding="utf-8")
    print(f"\nReport written: {report_path}")
    return 0


def cmd_audit_data(_args: argparse.Namespace) -> int:
    paths = resolve_paths_config()
    ensure_repo_output_dirs(paths)
    report_path = paths.reports_dir / "data_audit.md"

    warnings = validate_paths_config(paths)
    records = audit_data_sources(paths)

    sections = {
        "Summary": (
            f"{DATA_ROOT_ENV_VAR} = `{paths.data_root}`\n\n"
            + ("No datasets downloaded yet (expected at foundation stage)." if paths.data_root is None else "")
        ),
        "Warnings": bullet_list(warnings),
        "Source status": "",
    }

    table_lines = [
        "| Source | Enabled | Registry | Detected | Extracted dir | Archives dir |",
        "|--------|---------|----------|----------|---------------|--------------|",
    ]
    for rec in records:
        table_lines.append(
            f"| {rec['id']} | {rec['enabled']} | {rec['registry_status']} | "
            f"{rec['detected_status']} | {rec['extracted_dir_exists']} | "
            f"{rec['archives_dir_exists']} |"
        )
    sections["Source status"] = "\n".join(table_lines)

    write_markdown_report(report_path, "Data Directory Audit", sections)
    print(f"Data audit report: {report_path}")

    for warning in warnings:
        print(f"WARNING: {warning}")

    for rec in records:
        status = rec["detected_status"]
        print(
            f"  {rec['id']}: {status} "
            f"(extracted={rec['extracted_dir_exists']}, archives={rec['archives_dir_exists']})"
        )

    if paths.data_root is None:
        print(
            f"\nNo action required until {DATA_ROOT_ENV_VAR} is set. "
            "See docs/DATA_ACQUISITION.md."
        )
    return 0


def cmd_validate_config(_args: argparse.Namespace) -> int:
    paths = resolve_paths_config()
    issues: list[str] = []

    try:
        sources = load_source_registry()
        issues.extend(validate_source_registry(sources))
    except Exception as exc:
        print(f"ERROR: data_sources.yaml — {exc}", file=sys.stderr)
        return 1

    try:
        taxonomy = load_taxonomy_candidate()
        if taxonomy.status != "not_frozen":
            issues.append(f"Taxonomy status is '{taxonomy.status}' (expected not_frozen).")
    except Exception as exc:
        print(f"ERROR: taxonomy_candidate.yaml — {exc}", file=sys.stderr)
        return 1

    issues.extend(validate_paths_config(paths))

    print("Configuration validation:")
    print(f"  Repo root:     {paths.repo_root}")
    print(f"  Data root:     {paths.data_root or 'NOT SET'}")
    print(f"  Manifests dir: {paths.manifests_dir}")
    print(f"  Taxonomy:      {len(taxonomy.activity_classes)} candidate classes ({taxonomy.status})")
    print(f"  Sources:       {len(sources)} registered")

    if issues:
        print("\nWarnings:")
        for item in issues:
            print(f"  - {item}")
    else:
        print("\nNo configuration issues detected.")

    return 0


def cmd_build_inventory(args: argparse.Namespace) -> int:
    paths = resolve_paths_config()
    ensure_repo_output_dirs(paths)

    result = build_inventory(
        paths,
        enabled_only=not args.all_sources,
        extract_metadata=not args.no_metadata,
        compute_checksums=args.checksums,
    )

    csv_path = paths.manifests_dir / paths.inventory_filenames["csv"]
    parquet_path = paths.manifests_dir / paths.inventory_filenames["parquet"]
    md_path = paths.reports_dir / paths.inventory_filenames["markdown"]

    write_inventory_csv(result.frame, csv_path)
    write_inventory_parquet(result.frame, parquet_path)

    validate_inventory_frame(result.frame, strict=False)

    summary = (
        f"Files scanned: {result.files_scanned}\n"
        f"Sources scanned: {', '.join(result.sources_scanned) or '(none)'}\n"
        f"Inventory rows: {len(result.frame)}"
    )
    write_markdown_report(
        md_path,
        "Dataset Inventory",
        {
            "Summary": summary,
            "Warnings": bullet_list(result.warnings),
            "Outputs": bullet_list(
                [
                    f"CSV: {csv_path}",
                    f"Parquet: {parquet_path}",
                ]
            ),
        },
    )

    print(f"Inventory rows: {len(result.frame)}")
    print(f"CSV output:     {csv_path}")
    print(f"Parquet output: {parquet_path}")
    print(f"Report:         {md_path}")

    for warning in result.warnings:
        print(f"WARNING: {warning}")

    if result.files_scanned > 0:
        cwpv_reports = build_cwpv_inspection_reports(result.frame)
        for name, path in cwpv_reports.items():
            print(f"CWPV report [{name}]: {path}")

    if result.files_scanned == 0:
        print(
            "\nInventory is empty (expected when datasets are not downloaded). "
            f"Set {DATA_ROOT_ENV_VAR} and place files under the external layout."
        )
    return 0


def cmd_build_cml_baseline(_args: argparse.Namespace) -> int:
    paths = resolve_paths_config()
    ensure_repo_output_dirs(paths)
    try:
        result = run_cml_baseline_pipeline()
    except LeakageAuditError as exc:
        print(f"ERROR: Leakage audit failed: {exc}", file=sys.stderr)
        return 1

    print("CML baseline pipeline complete.")
    print(f"  Inventory updated: {result.inventory_path}")
    print(f"  Backup: {result.summary.get('inventory_backup')}")
    print(f"  Representation files (construction): {result.summary.get('cml_representation_files')}")
    print(f"  Unique logical samples: {result.summary.get('unique_logical_samples')}")
    print(f"  Parsed subject rows: {result.summary.get('parsed_subject_rows')} "
          f"({result.summary.get('parsed_subject_pct')}%)")

    split_meta = result.summary.get("split_meta", {})
    print(f"  Baseline classes: {', '.join(split_meta.get('included_classes', []))}")
    print(f"  Excluded classes (insufficient subjects): "
          f"{', '.join(split_meta.get('excluded_classes_insufficient_subjects', []))}")

    for name, path in result.manifests.items():
        print(f"  Manifest [{name}]: {path}")
    for name, path in result.reports.items():
        print(f"  Report [{name}]: {path}")
    return 0


def cmd_build_cwpv_baseline(_args: argparse.Namespace) -> int:
    paths = resolve_paths_config()
    ensure_repo_output_dirs(paths)
    try:
        result = run_cwpv_baseline_pipeline()
    except LeakageAuditError as exc:
        print(f"ERROR: Leakage audit failed: {exc}", file=sys.stderr)
        return 1

    print("CWPV baseline pipeline complete.")
    print(f"  Inventory updated: {result.inventory_path}")
    print(f"  Backup: {result.summary.get('inventory_backup')}")
    print(f"  CWPV video files: {result.summary.get('cwpv_video_files')}")
    print(f"  Unique logical samples: {result.summary.get('unique_logical_samples')}")

    split_meta = result.summary.get("split_meta", {})
    if split_meta.get("warning"):
        print(f"  WARNING: {split_meta['warning']}")
    print(f"  Baseline classes: {', '.join(split_meta.get('included_classes', []))}")
    print(f"  Excluded classes (insufficient subjects): "
          f"{', '.join(split_meta.get('excluded_classes_insufficient_subjects', []))}")

    for name, path in result.manifests.items():
        print(f"  Manifest [{name}]: {path}")
    for name, path in result.reports.items():
        print(f"  Report [{name}]: {path}")
    return 0


def cmd_extract_cml_features(_args: argparse.Namespace) -> int:
    paths = resolve_paths_config(require_data_root=True)
    ensure_repo_output_dirs(paths)
    try:
        result = extract_train_features()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("CML skeleton feature extraction complete (train only).")
    print(f"  15-node rows: {len(result.features_15)} -> {result.output_paths.get('15')}")
    print(f"  20-node rows: {len(result.features_20)} -> {result.output_paths.get('20')}")
    print(f"  Comparison report: {result.comparison_report_path}")
    if result.errors:
        print(f"  Errors: {len(result.errors)} (see comparison report)")
    return 0


def cmd_extract_cwpv_features(args: argparse.Namespace) -> int:
    paths = resolve_paths_config(require_data_root=True)
    ensure_repo_output_dirs(paths)
    try:
        result = extract_cwpv_features(max_videos=args.max_videos, view_id=args.view)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("CWPV pose feature extraction complete (train/validation only).")
    print(f"  Feature rows: {len(result.features)} -> {result.output_path}")
    print(f"  Report: {result.report_path}")
    if result.errors:
        print(f"  Errors: {len(result.errors)}")
    return 0


def cmd_train_cwpv_baseline(_args: argparse.Namespace) -> int:
    paths = resolve_paths_config()
    ensure_repo_output_dirs(paths)
    try:
        result = train_cwpv_baseline_classifiers()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("CWPV baseline classifier training complete.")
    for name, metrics in result.metrics.items():
        print(f"  {name}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    print(f"  Report: {result.report_path}")
    print(f"  Val predictions: {result.predictions_path}")
    if result.model_dir:
        print(f"  Saved models: {result.model_dir}")
    for warning in result.warnings:
        print(f"  WARNING: {warning}")
    return 0


def cmd_extract_smartphone_features(args: argparse.Namespace) -> int:
    paths = resolve_paths_config(require_data_root=True)
    ensure_repo_output_dirs(paths)
    try:
        result = extract_smartphone_features(max_videos=args.max_videos)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Smartphone pose feature extraction complete.")
    print(f"  Feature rows: {len(result.features)} -> {result.output_path}")
    print(f"  Report: {result.report_path}")
    if result.errors:
        print(f"  Errors: {len(result.errors)}")
    return 0


def cmd_evaluate_domain_transfer(_args: argparse.Namespace) -> int:
    paths = resolve_paths_config()
    ensure_repo_output_dirs(paths)
    try:
        result = evaluate_domain_transfer()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Domain-transfer evaluation complete.")
    for name, metrics in result.metrics.items():
        print(f"  {name}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    print(f"  Report: {result.report_path}")
    print(f"  Predictions: {result.predictions_path}")
    for warning in result.warnings:
        print(f"  NOTE: {warning}")
    return 0


def cmd_screen_ergonomics(args: argparse.Namespace) -> int:
    paths = resolve_paths_config()
    ensure_repo_output_dirs(paths)
    try:
        result = screen_ergonomics(source=args.source, max_videos=args.max_videos)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Ergonomic screening complete.")
    print(f"  Videos: {len(result.metrics)}")
    print(f"  CSV: {result.output_path}")
    print(f"  Report: {result.report_path}")
    if result.errors:
        print(f"  Errors: {len(result.errors)}")
    return 0


def cmd_detect_zone_events(args: argparse.Namespace) -> int:
    paths = resolve_paths_config()
    ensure_repo_output_dirs(paths)
    try:
        result = detect_zone_events(source=args.source, max_videos=args.max_videos)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    n_events = len(result.events)
    print("Zone-event detection complete.")
    print(f"  Videos: {len(result.summary)}")
    print(f"  Events: {n_events}")
    print(f"  Events CSV: {result.events_path}")
    print(f"  Summary CSV: {result.summary_path}")
    print(f"  Report: {result.report_path}")
    if result.errors:
        print(f"  Errors: {len(result.errors)}")
    return 0


def cmd_build_week6_report(args: argparse.Namespace) -> int:
    paths = resolve_paths_config()
    ensure_repo_output_dirs(paths)
    try:
        result = build_week6_report(
            source=args.source,
            max_videos=args.max_videos,
            skip_pipelines=args.skip_pipelines,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Week 6 end-to-end report complete.")
    print(f"  Report: {result.report_path}")
    if result.ergonomics_path:
        print(f"  Ergonomics: {result.ergonomics_path}")
    if result.zone_events_path:
        print(f"  Zone events: {result.zone_events_path}")
    for warning in result.warnings:
        print(f"  NOTE: {warning}")
    return 0


def cmd_inspect_cwpv(args: argparse.Namespace) -> int:
    paths = resolve_paths_config()
    ensure_repo_output_dirs(paths)
    inv_path = args.inventory or paths.manifests_dir / "clip_inventory.csv"
    if not inv_path.is_file():
        print(f"ERROR: Inventory not found: {inv_path}", file=sys.stderr)
        return 1
    import pandas as pd

    inventory = pd.read_csv(inv_path, low_memory=False)
    reports = build_cwpv_inspection_reports(inventory)
    for name, path in reports.items():
        print(f"CWPV report [{name}]: {path}")
    return 0


def cmd_extract_pose(args: argparse.Namespace) -> int:
    paths = resolve_paths_config(require_data_root=True)
    ensure_repo_output_dirs(paths)
    try:
        result = extract_pose_from_inventory(
            subject_id=args.subject,
            motion_id=args.motion,
            camera_id=args.camera,
            max_videos=args.max_videos,
            max_frames=args.max_frames,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Pose extraction complete.")
    print(f"  Outputs: {len(result.outputs)}")
    for path in result.outputs:
        print(f"    {path}")
    if result.report_path:
        print(f"  Report: {result.report_path}")
    if result.errors:
        print(f"  Errors: {len(result.errors)}")
        for err in result.errors:
            print(f"    {err['relative_path']}: {err['error']}")
    return 0 if result.outputs or not result.errors else 1


def cmd_render_annotated(args: argparse.Namespace) -> int:
    paths = resolve_paths_config()
    ensure_repo_output_dirs(paths)
    video_path = args.video.resolve()
    if not video_path.is_file():
        print(f"ERROR: Video not found: {video_path}", file=sys.stderr)
        return 1
    if args.output:
        output_path = args.output.resolve()
    else:
        output_path = paths.outputs_dir / "cwpv" / "annotated" / f"{video_path.stem}_annotated.mp4"
    try:
        render_annotated_video(
            video_path,
            output_path,
            max_frames=args.max_frames,
        )
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Annotated video: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
