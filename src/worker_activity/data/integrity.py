"""File integrity helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def quick_integrity_check(path: Path, *, compute_checksum: bool = True) -> tuple[str, str | None]:
    """
    Return (integrity_status, notes).

    For zero-byte files returns failed. Otherwise returns ok with checksum when enabled.
    """
    if not path.is_file():
        return "failed", "file not found"
    size = path.stat().st_size
    if size == 0:
        return "failed", "zero-byte file"
    if not compute_checksum:
        return "ok", None
    try:
        checksum = sha256_file(path)
        return "ok", checksum
    except OSError as exc:
        return "failed", str(exc)
