"""Sample frames from a video at a fixed rate for the frame-level classifiers.

Interim, visible design choice (per CLAUDE.md's substitution-flagging rule):
this does fixed-rate sampling, not scene-change-aware or face-track-aware
sampling. PLAN.md's temporal classifier (GRU/attention head over per-frame
embeddings) and app/pipeline/face_track.py both eventually want a
face-tracked frame *sequence*, not independent fixed-rate frames -- revisit
this once face_track.py exists. For Round 2 (single-frame classifiers,
non-temporal), independent fixed-rate sampling is sufficient.
"""
import tempfile
from pathlib import Path
from typing import Any

import cv2


def _write_to_tempfile(video: bytes) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        tmp.write(video)
    finally:
        tmp.close()
    return Path(tmp.name)


def sample_frames(video: Any, fps: float = 2.0, max_frames: int = 32) -> list[dict]:
    """Sample frames from a video at roughly `fps` frames/sec.

    video: a file path (str/Path) or raw video bytes (e.g. straight from
    `download_media()`).
    fps: target sampling rate. Frames are picked at the nearest available
    native frame to each `1/fps` interval -- this is *not* re-encoding or
    interpolating, just choosing which decoded frames to keep.
    max_frames: hard cap on frames returned, so a long video doesn't blow up
    per-scan inference cost. Sampling proceeds from the start at the given
    fps and stops once the cap is hit, rather than pre-computing a
    duration-wide even spread -- for the ~seconds-long clips this pipeline
    targets that difference doesn't matter in practice, but callers passing
    in long recordings should not assume the cap gives uniform coverage of
    the whole clip.

    Returns a list of {"index", "timestamp_sec", "frame"} dicts, `frame`
    being an HxWx3 BGR numpy array (same convention as SCRFDFaceDetector /
    EfficientNetB4SBI). Returns an empty list if the video has no frames.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")

    tmp_path: Path | None = None
    if isinstance(video, (bytes, bytearray)):
        tmp_path = _write_to_tempfile(bytes(video))
        path = str(tmp_path)
    elif isinstance(video, Path):
        path = str(video)
    elif isinstance(video, str):
        path = video
    else:
        raise TypeError(f"unsupported input type for sample_frames: {type(video)!r}")

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError("could not open video for frame sampling")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if native_fps <= 0:
            # Some containers/codecs don't report a reliable fps -- fall
            # back to a conservative guess rather than dividing by zero.
            # This only affects the reported timestamp_sec, not which
            # frames get sampled (the step is computed relative to
            # whatever native_fps we end up using).
            native_fps = 25.0

        step = max(1, round(native_fps / fps))

        results: list[dict] = []
        read_index = 0
        while len(results) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if read_index % step == 0:
                results.append(
                    {
                        "index": len(results),
                        "timestamp_sec": read_index / native_fps,
                        "frame": frame,
                    }
                )
            read_index += 1

        return results
    finally:
        cap.release()
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
