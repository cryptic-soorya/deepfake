from typing import Any

import numpy as np
from insightface.app import FaceAnalysis

from app.models.base import ModelWrapper
from app.models.device import get_device
from app.models.face_detector import SCRFDFaceDetector, _onnx_providers


class ArcFaceEmbedder(ModelWrapper):
    """Face recognition / identity embedding via ArcFace (InsightFace buffalo_l).

    Shares the `buffalo_l` model pack with SCRFDFaceDetector (same on-disk
    cache, single download) but also loads the `recognition` submodule,
    since ArcFace needs SCRFD's landmarks to produce an aligned 112x112 crop
    before embedding.
    """

    model_pack = "buffalo_l"
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
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        ctx_id = 0 if self.device.type == "cuda" else -1
        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=self.det_thresh)
        self._loaded = True

    def predict(self, input: Any) -> dict:
        """input: a single image (enrollment: embed every detected face), or a
        (probe_image, reference_image) pair (verification: cosine similarity
        between each image's best face).
        """
        if isinstance(input, (tuple, list)) and len(input) == 2:
            return self._verify(input[0], input[1])
        return self._enroll(input)

    def _embed_faces(self, input: Any) -> list[dict]:
        image = SCRFDFaceDetector._to_bgr_array(input)
        faces = self._app.get(image)
        return [
            {
                "bbox": face.bbox.tolist(),
                "det_score": float(face.det_score),
                "embedding": face.normed_embedding.tolist(),
            }
            for face in faces
        ]

    def _enroll(self, input: Any) -> dict:
        faces = self._embed_faces(input)
        det_scores = [f["det_score"] for f in faces]
        return {
            "score": max(det_scores) if det_scores else 0.0,
            "confidence": max(det_scores) if det_scores else 0.0,
            "raw": {"faces": faces},
            "metadata": {"mode": "enroll", "num_faces": len(faces), "embedding_dim": 512},
        }

    def _verify(self, probe: Any, reference: Any) -> dict:
        probe_faces = self._embed_faces(probe)
        reference_faces = self._embed_faces(reference)

        if not probe_faces or not reference_faces:
            return {
                "score": 0.0,
                "confidence": 0.0,
                "raw": {"probe_faces": probe_faces, "reference_faces": reference_faces},
                "metadata": {"mode": "verify", "error": "no face detected in probe and/or reference image"},
            }

        probe_face = max(probe_faces, key=lambda f: f["det_score"])
        reference_face = max(reference_faces, key=lambda f: f["det_score"])
        # embeddings are already L2-normalized, so dot product == cosine similarity
        similarity = float(np.dot(probe_face["embedding"], reference_face["embedding"]))
        # map cosine similarity [-1, 1] to a [0, 1] match score
        match_score = (similarity + 1.0) / 2.0

        return {
            "score": match_score,
            "confidence": min(probe_face["det_score"], reference_face["det_score"]),
            "raw": {"cosine_similarity": similarity, "probe_faces": probe_faces, "reference_faces": reference_faces},
            "metadata": {"mode": "verify", "interpretation": "score is identity match confidence in [0, 1]"},
        }
