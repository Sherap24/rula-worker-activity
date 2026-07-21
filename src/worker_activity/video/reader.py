"""Frame iteration utilities."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2


def iter_frames(
    video_path: Path,
    *,
    max_frames: int | None = None,
    frame_stride: int = 1,
) -> Iterator:
    """Yield frames from a video file using OpenCV.

    ``frame_stride`` > 1 keeps every Nth frame (1 = all frames).
    """
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Cannot open video: {video_path}")
    try:
        yielded = 0
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % frame_stride == 0:
                yield frame
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            index += 1
    finally:
        capture.release()
