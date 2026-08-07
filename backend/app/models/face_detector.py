from typing import Any

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.models.base import ModelWrapper
from app.models.device import get_device


def _onnx_providers(device) -> list[str]:
    """Map the shared torch device to onnxruntime execution providers.

    onnxruntime has no MPS provider (Apple's analogue is CoreML), so `mps`
    falls through to CoreML-then-CPU rather than being treated as CUDA.
    """
    if device.type == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device.type == "mps":
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class SCRFDFaceDetector(ModelWrapper):
    """Face detection + landmarks via SCRFD (InsightFace).

    Uses InsightFace's `buffalo_l` model pack (same pack CLAUDE.md specifies
    for ArcFace) restricted to the `detection` module only, so this doesn't
    pull in the recognition weights that `face_recognition.py` owns.
    """

    model_pack = "buffalo_l"
    det_size = (640, 640)
    det_thresh = 0.5

    def __init__(self, model_pack: str | None = None) -> None:
        super().__init__()
        if model_pack is not None:
            self.model_pack = model_pack

    def load(self) -> None:
        self.device = get_device()
        providers = _onnx_providers(self.device)
        self._app = FaceAnalysis(
            name=self.model_pack,
            allowed_modules=["detection"],
            providers=providers,
        )
        # ctx_id selects the onnxruntime provider's device index; -1 means
        # "use whatever CPU/CoreML provider was passed in", 0 means GPU 0.
        ctx_id = 0 if self.device.type == "cuda" else -1
        self._app.prepare(ctx_id=ctx_id, det_size=self.det_size, det_thresh=self.det_thresh)
        self._loaded = True

    def predict(self, input: Any) -> dict:
        """input: raw image bytes, a file path, or an HxWx3 BGR numpy array."""
        image = self._to_bgr_array(input)
        faces = self._app.get(image)

        faces_out = [
            {
                "bbox": face.bbox.tolist(),
                "landmarks_5pt": face.kps.tolist(),
                "det_score": float(face.det_score),
            }
            for face in faces
        ]
        det_scores = [f["det_score"] for f in faces_out]

        return {
            "score": max(det_scores) if det_scores else 0.0,
            "confidence": max(det_scores) if det_scores else 0.0,
            "raw": {"faces": faces_out},
            "metadata": {"num_faces": len(faces_out), "image_shape": list(image.shape)},
        }

    @staticmethod
    def _to_bgr_array(input: Any) -> np.ndarray:
        if isinstance(input, np.ndarray):
            return input
        if isinstance(input, (bytes, bytearray)):
            buffer = np.frombuffer(input, dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("could not decode image bytes")
            return image
        if isinstance(input, str):
            image = cv2.imread(input)
            if image is None:
                raise ValueError(f"could not read image at path: {input}")
            return image
        raise TypeError(f"unsupported input type for SCRFDFaceDetector: {type(input)!r}")
