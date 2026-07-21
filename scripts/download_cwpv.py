"""Download and extract CWPV archives from Figshare."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

FIGSHARE_FILES = {
    "video": {
        "url": "https://ndownloader.figshare.com/files/50810442",
        "name": "Video_Data.rar",
        "size_bytes": 12_266_253_286,
    },
    "readme": {
        "url": "https://ndownloader.figshare.com/files/50846403",
        "name": "README.pdf",
        "size_bytes": 670_122,
    },
}


def _download(url: str, dest: Path, *, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected_size = None
    for spec in FIGSHARE_FILES.values():
        if spec["name"] == dest.name:
            expected_size = spec.get("size_bytes")
            break

    if dest.is_file() and not force:
        size = dest.stat().st_size
        if expected_size is None or abs(size - expected_size) < 1_000_000:
            print(f"Already exists: {dest} ({size} bytes)")
            return
        print(f"Incomplete/wrong size ({size} bytes) — re-downloading.")

    part_path = dest.with_suffix(dest.suffix + ".part")
    if force and dest.is_file():
        print(f"Removing existing file: {dest}")
        try:
            dest.unlink()
        except PermissionError:
            print(
                f"WARNING: Could not delete {dest} (file locked). "
                f"Will write to {part_path} and replace when complete.",
                file=sys.stderr,
            )

    if force or not dest.is_file():
        if part_path.is_file():
            part_path.unlink(missing_ok=True)

    write_path = part_path
    print(f"Downloading {url} -> {write_path}")
    req = urllib.request.Request(url, headers={"User-Agent": "worker-activity/0.1"})
    total = 0
    with urllib.request.urlopen(req, timeout=120) as response, write_path.open("wb") as handle:
        while True:
            chunk = response.read(4 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            total += len(chunk)
            if total % (500 * 1024 * 1024) < 4 * 1024 * 1024:
                print(f"  {total / 1e9:.2f} GB", flush=True)
    print(f"Downloaded {total} bytes to {write_path}")

    if dest.is_file():
        try:
            dest.unlink()
        except PermissionError as exc:
            raise RuntimeError(
                f"Download complete but cannot replace locked file {dest}. "
                f"Close other programs using it, then rename {write_path} manually."
            ) from exc
    write_path.replace(dest)
    print(f"Final archive: {dest} ({dest.stat().st_size} bytes)")


def _find_7z() -> Path | None:
    candidates = [
        Path(r"C:\Program Files\7-Zip\7z.exe"),
        Path(r"C:\Program Files (x86)\7-Zip\7z.exe"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    which = shutil.which("7z")
    return Path(which) if which else None


def _test_archive(archive: Path) -> bool:
    seven_zip = _find_7z()
    if seven_zip is None:
        print("WARNING: 7-Zip not found — skipping archive integrity test.", file=sys.stderr)
        return True
    cmd = [str(seven_zip), "t", str(archive)]
    print("Testing archive integrity:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return False
    if "ERROR" in result.stdout.upper() or "ERRORS" in result.stdout.upper():
        print(result.stdout, file=sys.stderr)
        return False
    print("Archive integrity test passed.")
    return True


def _extract_rar(archive: Path, output_dir: Path) -> None:
    seven_zip = _find_7z()
    if seven_zip is None:
        raise RuntimeError("7-Zip not found. Install 7-Zip to extract Video_Data.rar.")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(seven_zip), "x", str(archive), f"-o{output_dir}", "-y"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and extract CWPV from Figshare")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Users\asahi\Datasets\RULA"),
        help="RULA_DATA_ROOT path",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download archives even if they already exist (use after corrupt RAR)",
    )
    parser.add_argument(
        "--skip-integrity-test",
        action="store_true",
        help="Skip 7z integrity test before extraction",
    )
    args = parser.parse_args(argv)

    archives = args.data_root / "cwpv" / "archives"
    extracted = args.data_root / "cwpv" / "extracted"

    if not args.skip_download:
        for spec in FIGSHARE_FILES.values():
            _download(spec["url"], archives / spec["name"], force=args.force_download)

    video_rar = archives / "Video_Data.rar"
    if not args.skip_extract:
        if not video_rar.is_file():
            print(f"ERROR: Missing archive {video_rar}", file=sys.stderr)
            return 1
        if not args.skip_integrity_test and not _test_archive(video_rar):
            print(
                "ERROR: Archive failed integrity test. "
                "Re-run with --force-download to fetch a clean copy.",
                file=sys.stderr,
            )
            return 1
        _extract_rar(video_rar, extracted)

    print("CWPV acquisition step complete.")
    print(f"  Archives:  {archives}")
    print(f"  Extracted: {extracted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
