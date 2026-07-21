"""MediaPipe Pose estimator (tasks API, MediaPipe >= 0.10)."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from worker_activity.config import find_repo_root
from worker_activity.pose.base import PoseFrame, PoseSequence
from worker_activity.video.reader import iter_frames

DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


@dataclass
class MediaPipePoseConfig:
    model_path: Path | None = None
    model_url: str = DEFAULT_MODEL_URL
    num_poses: int = 1
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


def ensure_pose_model(config: MediaPipePoseConfig) -> Path:
    root = find_repo_root()
    model_path = config.model_path or root / "configs" / "models" / "pose_landmarker_lite.task"
    if model_path.is_file() and model_path.stat().st_size > 0:
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model to {model_path}")
    req = urllib.request.Request(config.model_url, headers={"User-Agent": "worker-activity/0.1"})
    with urllib.request.urlopen(req, timeout=120) as response, model_path.open("wb") as handle:
        handle.write(response.read())
    return model_path


class MediaPipePoseEstimator:
    """Extract pose landmarks from video frames using MediaPipe PoseLandmarker."""

    def __init__(self, config: MediaPipePoseConfig | None = None) -> None:
        self.config = config or MediaPipePoseConfig()
        self._landmarker = None

    def _ensure_landmarker(self) -> object:
        if self._landmarker is None:
            from mediapipe.tasks.python.core import base_options as base_options_module
            from mediapipe.tasks.python.vision import pose_landmarker
            from mediapipe.tasks.python.vision.core import vision_task_running_mode

            model_path = ensure_pose_model(self.config)
            options = pose_landmarker.PoseLandmarkerOptions(
                base_options=base_options_module.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
                num_poses=self.config.num_poses,
                min_pose_detection_confidence=self.config.min_pose_detection_confidence,
                min_pose_presence_confidence=self.config.min_pose_presence_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
            self._landmarker = pose_landmarker.PoseLandmarker.create_from_options(options)
        return self._landmarker

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> MediaPipePoseEstimator:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def extract_from_video(
        self,
        video_path: Path,
        *,
        relative_path: str | None = None,
        max_frames: int | None = None,
        fps: float | None = None,
        frame_stride: int = 1,
    ) -> PoseSequence:
        from mediapipe import Image, ImageFormat

        landmarker = self._ensure_landmarker()
        rel = relative_path or video_path.name
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise OSError(f"Cannot open video: {video_path}")
        try:
            video_fps = capture.get(cv2.CAP_PROP_FPS)
            fps_val = fps or (float(video_fps) if video_fps and video_fps > 0 else None)
        finally:
            capture.release()

        sequence = PoseSequence(video_relative_path=rel, fps=fps_val)
        for idx, frame in enumerate(
            iter_frames(video_path, max_frames=max_frames, frame_stride=frame_stride)
        ):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(
                image_format=ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb),
            )
            result = landmarker.detect(mp_image)
            landmarks: dict[str, dict[str, float]] = {}
            if result.pose_landmarks:
                pose = result.pose_landmarks[0]
                for landmark_idx, landmark in enumerate(pose):
                    name = self.landmark_name(landmark_idx)
                    landmarks[name] = {
                        "x": float(landmark.x),
                        "y": float(landmark.y),
                        "z": float(landmark.z),
                        "visibility": float(getattr(landmark, "visibility", 0.0) or 0.0),
                    }
            timestamp = (idx / fps_val) if fps_val else None
            sequence.frames.append(
                PoseFrame(
                    frame_index=idx,
                    timestamp_seconds=timestamp,
                    landmarks=landmarks,
                    detection_confidence=(
                        landmarks.get("LEFT_HIP", {}).get("visibility")
                        if landmarks
                        else None
                    ),
                )
            )
        return sequence

    @staticmethod
    def landmark_names() -> list[str]:
        from mediapipe.tasks.python.vision import pose_landmarker

        return [pose_landmarker.PoseLandmark(i).name for i in range(33)]

    @staticmethod
    def landmark_name(index: int) -> str:
        from mediapipe.tasks.python.vision import pose_landmarker

        return pose_landmarker.PoseLandmark(index).name
