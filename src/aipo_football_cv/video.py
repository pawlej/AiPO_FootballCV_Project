from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np
import supervision as sv


def iter_video_frames(video_path: str | Path, stride: int = 1) -> Iterator[np.ndarray]:
    """Yield frames with Supervision's frame generator.

    This avoids the manual OpenCV read loop from the Colab notebook and is easier
    to reuse in local scripts. ``stride`` lets you process every n-th frame.
    """
    video_path = str(video_path)
    for i, frame in enumerate(sv.get_video_frames_generator(video_path)):
        if i % stride == 0:
            yield frame


def load_video_frames(video_path: str | Path, max_frames: int | None = None, stride: int = 1) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for frame in iter_video_frames(video_path, stride=stride):
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    return frames


def save_video(frames: Iterable[np.ndarray], output_path: str | Path, fps: float, codec: str = "mp4v") -> None:
    """Save frames to a video file with OpenCV.

    Reading is handled by Supervision's generator; writing is intentionally kept
    conservative because OpenCV's writer API is stable across environments.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    iterator = iter(frames)
    try:
        first_frame = next(iterator)
    except StopIteration as exc:
        raise ValueError("No frames to save.") from exc

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    writer.write(first_frame)
    for frame in iterator:
        writer.write(frame)
    writer.release()


def get_video_fps(video_path: str | Path, default: float = 24.0) -> float:
    try:
        info = sv.VideoInfo.from_video_path(str(video_path))
        return float(info.fps or default)
    except Exception:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(fps or default)
