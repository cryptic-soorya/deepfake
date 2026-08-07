from typing import Any

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

from app.models.base import ModelWrapper
from app.models.face_detector import SCRFDFaceDetector
from app.models.vendor.weights import fetch

# Google's official hosted checkpoint for the MediaPipe Face Landmarker task
# (478 landmarks incl. iris, plus blendshapes) — see
# https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


class MediaPipeFaceMesh(ModelWrapper):
    """Micro-expression / mesh landmarks via MediaPipe Face Mesh (Face Landmarker task)."""

    max_num_faces = 1

    def load(self) -> None:
        model_path = fetch(MODEL_URL, "face_landmarker.task")
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_faces=self.max_num_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._loaded = True

    def predict(self, input: Any) -> dict:
        """input: raw image bytes, a file path, or an HxWx3 BGR numpy array."""
        bgr = SCRFDFaceDetector._to_bgr_array(input)
        rgb = bgr[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        result = self._landmarker.detect(mp_image)

        faces_out = []
        for face_landmarks, blendshapes in zip(
            result.face_landmarks,
            result.face_blendshapes or [[]] * len(result.face_landmarks),
        ):
            faces_out.append(
                {
                    "landmarks_3d": [[lm.x, lm.y, lm.z] for lm in face_landmarks],
                    "blendshapes": {b.category_name: float(b.score) for b in blendshapes},
                }
            )

        # micro-expression signal proxy: mean activation across all blendshape
        # categories for the primary face, used downstream as a coarse
        # expressiveness feature for the fusion layer, not a fake/real verdict
        mean_blendshape_activation = 0.0
        if faces_out and faces_out[0]["blendshapes"]:
            mean_blendshape_activation = float(np.mean(list(faces_out[0]["blendshapes"].values())))

        return {
            "score": mean_blendshape_activation,
            "confidence": 1.0 if faces_out else 0.0,
            "raw": {"faces": faces_out},
            "metadata": {
                "num_faces": len(faces_out),
                "num_landmarks": len(faces_out[0]["landmarks_3d"]) if faces_out else 0,
                "interpretation": "not a deepfake score — mesh/blendshape features feed the fusion layer",
            },
        }
