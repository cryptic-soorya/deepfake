"""Sample frames from a video at a fixed rate/strategy for the frame-level classifiers."""
from collections.abc import Iterator

import cv2
import numpy as np


def sample_frames(video_path: str, fps: float = 2.0) -> Iterator[np.ndarray]:
    """Yields BGR frames sampled at approximately `fps` frames/sec.

    If the container's own fps can't be read (corrupt/missing header), falls
    back to yielding every decoded frame rather than dividing by zero.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    step = max(1, round(source_fps / fps)) if source_fps > 0 else 1

    try:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                yield frame
            frame_index += 1
    finally:
        cap.release()
