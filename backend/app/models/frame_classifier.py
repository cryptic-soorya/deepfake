"""Frame-level deepfake classifier: EfficientNet-B4 trained via the Self-Blended
Images (SBI) recipe (Shiohara & Yamasaki, CVPR 2022).

Uses the authors' own pretrained checkpoint (trained on FF-c23) rather than
fine-tuning from scratch, per CLAUDE.md's "start from the authors' own
pretrained checkpoint" pattern already used by app/models/liveness.py. Face
localization/crop reuses our own SCRFDFaceDetector instead of the original
repo's bundled RetinaFace detector, for the same reason liveness.py does:
avoid a second face detector in the stack.
"""
from typing import Any

import cv2
import numpy as np
import torch
from efficientnet_pytorch import EfficientNet

from app.models.base import ModelWrapper
from app.models.device import get_device
from app.models.face_detector import SCRFDFaceDetector
from app.models.vendor.weights import fetch_gdrive

# Google Drive file IDs from the authors' README (mapooon/SelfBlendedImages),
# "Pretrained model" section — trained on FF-c23 (compressed, closer to
# real-world video than the "raw" checkpoint).
CHECKPOINT_GDRIVE_ID = "1X0-NYT8KPursLZZdxduRQju6E52hauV0"
CHECKPOINT_FILENAME = "sbi_efficientnet_b4_ffc23.tar"

IMAGE_SIZE = (380, 380)


def _crop_with_margin(image: np.ndarray, bbox_xyxy: list[float], margin_ratio: float = 0.125) -> np.ndarray:
    """Reproduces the original repo's test-time crop: bbox + 1/4 margin per
    side, halved (*0.5) at test time vs. train time -> effectively 1/8 (0.125)
    of the box width/height added to each side."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox_xyxy
    box_w, box_h = x1 - x0, y1 - y0
    w_margin, h_margin = box_w * margin_ratio, box_h * margin_ratio

    x0_new = max(0, int(x0 - w_margin))
    y0_new = max(0, int(y0 - h_margin))
    x1_new = min(w, int(x1 + w_margin) + 1)
    y1_new = min(h, int(y1 + h_margin) + 1)
    return image[y0_new:y1_new, x0_new:x1_new]


class EfficientNetB4SBI(ModelWrapper):
    """Frame-level deepfake classifier: EfficientNet-B4 trained via the Self-Blended Images (SBI) recipe."""

    def load(self) -> None:
        self.device = get_device()

        self._detector = SCRFDFaceDetector()
        self._detector.load()

        self._net = EfficientNet.from_pretrained("efficientnet-b4", advprop=True, num_classes=2)

        weights_path = fetch_gdrive(CHECKPOINT_GDRIVE_ID, CHECKPOINT_FILENAME)
        checkpoint = torch.load(weights_path, map_location=self.device)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        # The authors' checkpoint was saved from a `Detector` wrapper whose
        # only submodule is `self.net = EfficientNet(...)` (see model.py in
        # mapooon/SelfBlendedImages) -> every key is prefixed "net.".
        state_dict = {
            (k[len("net.") :] if k.startswith("net.") else k): v for k, v in state_dict.items()
        }
        self._net.load_state_dict(state_dict)
        self._net.to(self.device).eval()
        self._loaded = True

    def predict(self, input: Any) -> dict:
        """input: raw image/video-frame bytes, a file path, or an HxWx3 BGR numpy array containing one face."""
        image = SCRFDFaceDetector._to_bgr_array(input)
        faces = self._detector.predict(image)["raw"]["faces"]
        if not faces:
            return {
                "score": None,
                "confidence": None,
                "raw": None,
                "metadata": {"error": "no face detected"},
            }

        best_face = max(faces, key=lambda f: f["det_score"])
        crop_bgr = _crop_with_margin(image, best_face["bbox"])
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_rgb = cv2.resize(crop_rgb, IMAGE_SIZE)

        # Matches the SBI authors' own inference code exactly (src/inference/
        # inference_video.py in mapooon/SelfBlendedImages): torch.tensor(...).
        # float()/255, nothing else. `advprop=True` in load() only picks which
        # ImageNet backbone from_pretrained() initializes from before the SBI
        # checkpoint's state_dict overwrites every weight -- it does NOT mean
        # the checkpoint was trained on [-1, 1] inputs. (A prior fix here
        # added a (x-0.5)/0.5 step on that mistaken assumption; verified
        # against upstream and reverted.)
        tensor = torch.from_numpy(crop_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self._net(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        fake_score = float(probs[1])

        return {
            "score": fake_score,
            "confidence": float(max(probs).item()),
            "raw": {"probs": probs.tolist(), "bbox": best_face["bbox"]},
            "metadata": {
                "det_score": best_face["det_score"],
                "interpretation": "score is P(fake); high score indicates a deepfake frame",
            },
        }
