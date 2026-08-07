"""Per-frame face detection + tracking across a video, producing the
face-cropped, continuous frame sequence SyncNet (lip-sync) needs.

This is deliberately *not* the same job as frame_sampler.py: frame_sampler
picks sparse, independent frames for the single-frame classifier (any gap
between samples is fine, they're scored independently). SyncNet needs a
temporally continuous crop sequence at a fixed fps so its audio/video offset
search means something -- a gap or a jump in identity between frames would
corrupt the sync estimate.

Single-face assumption: track_faces follows *one* face per video (the most
confident detection, held across brief detection dropouts) rather than doing
full multi-object tracking (e.g. SORT/DeepSORT) with identity switching.
This matches how this platform's live-session and identity-verification
flows are framed around one primary subject per session (per CLAUDE.md).
Revisit if multi-face scenes become a real requirement.
"""
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.models.face_detector import SCRFDFaceDetector


def _write_to_tempfile(video: bytes) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        tmp.write(video)
    finally:
        tmp.close()
    return Path(tmp.name)


def _crop_and_resize(frame: np.ndarray, bbox_xyxy: list[float], crop_size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in bbox_xyxy]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("degenerate bbox after clamping to frame bounds")
    crop = frame[y0:y1, x0:x1]
    return cv2.resize(crop, (crop_size, crop_size))


def track_faces(
    video: Any,
    target_fps: int = 25,
    crop_size: int = 224,
    detector: SCRFDFaceDetector | None = None,
    max_frames: int = 250,
) -> dict:
    """Decode `video` at (as close as possible to) `target_fps`, detect the
    most confident face each frame, and return a continuous face-crop
    sequence.

    video: a file path (str/Path) or raw video bytes.
    detector: an already-loaded SCRFDFaceDetector to reuse (avoids spinning
    up a second ONNX session when a caller already has one loaded); if
    omitted, this loads its own.
    max_frames: hard cap (~10s at 25fps) bounding tracking/inference cost per
    scan, same rationale as frame_sampler.py's cap.

    Tracking behaviour: when a frame has no detected face, the last known
    bbox is reused ("held") rather than dropping the frame, so the output
    stays a continuous sequence with no timing gaps -- SyncNet's offset
    search assumes fixed-interval frames. If no face has been seen yet, the
    frame is skipped from the output entirely (nothing to hold).

    Returns {"frames": (T,crop_size,crop_size,3) uint8 BGR array,
    "fps": target_fps, "num_frames": T, "num_frames_with_face": int,
    "held_frames": int}. `frames` has T=0 (empty array) if no face was ever
    found.
    """
    owns_detector = detector is None
    if detector is None:
        detector = SCRFDFaceDetector()
        detector.load()

    tmp_path: Path | None = None
    if isinstance(video, (bytes, bytearray)):
        tmp_path = _write_to_tempfile(bytes(video))
        path = str(tmp_path)
    elif isinstance(video, Path):
        path = str(video)
    elif isinstance(video, str):
        path = video
    else:
        raise TypeError(f"unsupported input type for track_faces: {type(video)!r}")

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError("could not open video for face tracking")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if native_fps <= 0:
            native_fps = float(target_fps)

        step = max(1, round(native_fps / target_fps))

        crops: list[np.ndarray] = []
        num_with_face = 0
        num_held = 0
        last_bbox: list[float] | None = None

        read_index = 0
        while len(crops) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if read_index % step == 0:
                faces = detector.predict(frame)["raw"]["faces"]
                if faces:
                    best = max(faces, key=lambda f: f["det_score"])
                    last_bbox = best["bbox"]
                    num_with_face += 1
                elif last_bbox is not None:
                    num_held += 1
                else:
                    read_index += 1
                    continue  # no face yet seen at all -- nothing to hold, skip

                crops.append(_crop_and_resize(frame, last_bbox, crop_size))
            read_index += 1

        frames_array = np.stack(crops, axis=0) if crops else np.empty((0, crop_size, crop_size, 3), dtype=np.uint8)

        return {
            "frames": frames_array,
            "fps": target_fps,
            "num_frames": int(frames_array.shape[0]),
            "num_frames_with_face": num_with_face,
            "held_frames": num_held,
        }
    finally:
        cap.release()
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        if owns_detector:
            # SCRFDFaceDetector has no explicit teardown -- the ONNX session
            # is garbage-collected with the object. Nothing to release here;
            # kept as an explicit branch so a future close()/unload() is an
            # obvious one-line addition.
            pass
