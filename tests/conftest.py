"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Create a tiny MP4 with OpenCV (not committed)."""
    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter unavailable on this platform")
    for i in range(5):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :] = (i * 40 % 255, 64, 128)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def external_data_root(tmp_path: Path) -> Path:
    root = tmp_path / "RULA"
    for sub in [
        "cml/archives",
        "cml/extracted",
        "cwpv/archives",
        "cwpv/extracted",
    ]:
        (root / sub).mkdir(parents=True)
    return root
