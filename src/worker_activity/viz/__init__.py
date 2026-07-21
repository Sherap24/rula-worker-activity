"""Annotated video rendering."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from worker_activity.pose.mediapipe_estimator import MediaPipePoseConfig, MediaPipePoseEstimator
from worker_activity.video.reader import iter_frames


def _draw_pose_landmarks(
    frame: np.ndarray,
    landmarks: list,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    from mediapipe.tasks.python.vision import pose_landmarker

    height, width = frame.shape[:2]
    points: list[tuple[int, int] | None] = []
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), 3, color, -1)

    for connection in pose_landmarker.PoseLandmarksConnections.POSE_LANDMARKS:
        start = points[connection.start]
        end = points[connection.end]
        if start is None or end is None:
            continue
        cv2.line(frame, start, end, color, 2)


def render_annotated_video(
    video_path: Path,
    output_path: Path,
    *,
    max_frames: int | None = None,
    pose_config: MediaPipePoseConfig | None = None,
) -> Path:
    """Write a copy of *video_path* with MediaPipe pose landmarks drawn."""
    from mediapipe import Image, ImageFormat

    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise OSError(f"Cannot open video writer: {output_path}")

    try:
        with MediaPipePoseEstimator(pose_config) as estimator:
            landmarker = estimator._ensure_landmarker()
            for idx, frame in enumerate(iter_frames(video_path, max_frames=max_frames)):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = Image(
                    image_format=ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb),
                )
                result = landmarker.detect(mp_image)
                annotated = frame.copy()
                if result.pose_landmarks:
                    _draw_pose_landmarks(annotated, result.pose_landmarks[0])
                writer.write(annotated)
                if max_frames is not None and idx + 1 >= max_frames:
                    break
    finally:
        writer.release()

    return output_path
