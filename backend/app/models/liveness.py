"""Liveness / anti-spoofing via Silent-Face-Anti-Spoofing (MiniFASNet).

Reimplements the original repo's `test.py` inference exactly (crop face to
each model's declared scale, run both scale-specific models, sum softmax
outputs, argmax over {fake, real} with the summed score halved) — see
app/models/vendor/minifasnet.py for the vendored architecture and provenance.
Weights are the authors' own pretrained checkpoints, downloaded lazily from
the repo (same "fetch on first run, cache after" pattern insightface uses
for its own model packs) rather than a caffemodel-based face detector; face
localization here reuses our own SCRFDFaceDetector instead of the original
repo's bundled RetinaFace-caffe detector, since that avoids duplicating a
second face detector in the stack.
"""
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.models.base import ModelWrapper
from app.models.device import get_device
from app.models.face_detector import SCRFDFaceDetector
from app.models.vendor.minifasnet import MODEL_MAPPING
from app.models.vendor.weights import fetch

WEIGHTS_BASE_URL = (
    "https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/master/"
    "resources/anti_spoof_models"
)
# (checkpoint filename, crop scale). Scale is baked into the filename per the
# original repo's naming convention (`{scale}_{h}x{w}_{ModelType}.pth`).
CHECKPOINTS = [
    ("2.7_80x80_MiniFASNetV2.pth", 2.7),
    ("4_0_0_80x80_MiniFASNetV1SE.pth", 4.0),
]


def _parse_model_type(filename: str) -> str:
    return filename.split(".pth")[0].split("_")[-1]


def _get_new_box(src_w, src_h, bbox, scale):
    x, y, box_w, box_h = bbox
    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
    new_width, new_height = box_w * scale, box_h * scale
    center_x, center_y = box_w / 2 + x, box_h / 2 + y

    left_top_x = center_x - new_width / 2
    left_top_y = center_y - new_height / 2
    right_bottom_x = center_x + new_width / 2
    right_bottom_y = center_y + new_height / 2

    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0
    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0
    if right_bottom_x > src_w - 1:
        left_top_x -= right_bottom_x - src_w + 1
        right_bottom_x = src_w - 1
    if right_bottom_y > src_h - 1:
        left_top_y -= right_bottom_y - src_h + 1
        right_bottom_y = src_h - 1

    return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)


def _crop(image: np.ndarray, bbox, scale: float, out_w: int = 80, out_h: int = 80) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    l, t, r, b = _get_new_box(src_w, src_h, bbox, scale)
    patch = image[t : b + 1, l : r + 1]
    return cv2.resize(patch, (out_w, out_h))


class MiniFASNetLiveness(ModelWrapper):
    """Liveness / anti-spoofing via Silent-Face-Anti-Spoofing (MiniFASNet)."""

    def load(self) -> None:
        self.device = get_device()
        self._detector = SCRFDFaceDetector()
        self._detector.load()

        self._models = []
        for filename, scale in CHECKPOINTS:
            model_type = _parse_model_type(filename)
            model = MODEL_MAPPING[model_type](conv6_kernel=(5, 5))  # get_kernel(80, 80) = (5, 5)
            weights_path = fetch(f"{WEIGHTS_BASE_URL}/{filename}", filename)
            state_dict = torch.load(weights_path, map_location=self.device)
            state_dict = {
                (k[len("module.") :] if k.startswith("module.") else k): v for k, v in state_dict.items()
            }
            model.load_state_dict(state_dict)
            model.to(self.device).eval()
            self._models.append((model, scale))
        self._loaded = True

    def predict(self, input: Any) -> dict:
        """input: raw image bytes, a file path, or an HxWx3 BGR numpy array containing one face."""
        image = SCRFDFaceDetector._to_bgr_array(input)
        faces = self._detector.predict(image)["raw"]["faces"]
        if not faces:
            return {
                "score": 0.0,
                "confidence": 0.0,
                "raw": {"prediction": None},
                "metadata": {"error": "no face detected"},
            }
        bbox_xyxy = max(faces, key=lambda f: f["det_score"])["bbox"]
        x0, y0, x1, y1 = bbox_xyxy
        bbox_xywh = [x0, y0, x1 - x0, y1 - y0]

        prediction = np.zeros((1, 3))
        with torch.no_grad():
            for model, scale in self._models:
                patch = _crop(image, bbox_xywh, scale)
                tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                logits = model(tensor)
                prediction += F.softmax(logits, dim=1).cpu().numpy()

        label = int(np.argmax(prediction))
        real_score = float(prediction[0][1] / len(self._models))
        is_real = label == 1

        return {
            "score": real_score,
            "confidence": float(prediction[0][label] / len(self._models)),
            "raw": {"prediction": prediction.tolist(), "label": label},
            "metadata": {
                "is_live": is_real,
                "bbox": bbox_xywh,
                "interpretation": "score is P(real/live face); low score indicates spoof/presentation attack",
            },
        }
