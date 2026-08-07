"""Visual explainability via Grad-CAM over the frame classifier's last conv
layer (EfficientNet-B4/SBI's `_conv_head`, the standard Grad-CAM target: the
last feature map before global pooling, still spatially aligned with the
input crop).

Deliberately reuses `EfficientNetB4SBI` rather than loading its own copy of
the network -- same checkpoint, same face crop, so the heatmap lines up with
the score `frame_classifier.py` already reports for a given frame.
"""
import base64
from typing import Any

import cv2
import numpy as np
import torch

from app.models.base import ModelWrapper
from app.models.face_detector import SCRFDFaceDetector
from app.models.frame_classifier import IMAGE_SIZE, EfficientNetB4SBI, _crop_with_margin


class GradCAMExplainer(ModelWrapper):
    """Returns a heatmap overlay, not a score — kept in the same interface for pipeline uniformity."""

    def load(self) -> None:
        self._classifier = EfficientNetB4SBI()
        self._classifier.load()
        self._target_layer = self._classifier._net._conv_head
        self._loaded = True

    def predict(self, input: Any) -> dict:
        """input: same raw image/video-frame bytes, file path, or HxWx3 BGR
        array the frame classifier accepts. Returns a base64-encoded PNG
        heatmap overlay in `metadata.heatmap_png_base64`, plus the
        classifier's own score/confidence for the same crop so a caller
        doesn't need to invoke both wrappers to get score + explanation."""
        classifier = self._classifier
        image = SCRFDFaceDetector._to_bgr_array(input)
        faces = classifier._detector.predict(image)["raw"]["faces"]
        if not faces:
            return {
                "score": None,
                "confidence": None,
                "raw": None,
                "metadata": {"error": "no face detected"},
            }

        best_face = max(faces, key=lambda f: f["det_score"])
        crop_bgr = _crop_with_margin(image, best_face["bbox"])
        crop_bgr = cv2.resize(crop_bgr, IMAGE_SIZE)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(crop_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        tensor = tensor.to(classifier.device)

        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []
        fwd_handle = self._target_layer.register_forward_hook(lambda m, i, o: activations.append(o))
        bwd_handle = self._target_layer.register_full_backward_hook(
            lambda m, grad_in, grad_out: gradients.append(grad_out[0])
        )

        try:
            classifier._net.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(True):
                logits = classifier._net(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                fake_score = probs[1]
                fake_score.backward()
        finally:
            fwd_handle.remove()
            bwd_handle.remove()

        activation = activations[0].detach()[0]  # (C, H, W)
        gradient = gradients[0].detach()[0]  # (C, H, W)
        weights = gradient.mean(dim=(1, 2))  # (C,)
        cam = torch.relu((weights[:, None, None] * activation).sum(dim=0))
        cam = cam / (cam.max().clamp(min=1e-8))
        cam_np = cam.cpu().numpy()
        cam_resized = cv2.resize(cam_np, IMAGE_SIZE)

        heatmap_bgr = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(crop_bgr, 0.55, heatmap_bgr, 0.45, 0)
        _, encoded = cv2.imencode(".png", overlay)
        heatmap_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

        return {
            "score": float(fake_score.item()),
            "confidence": float(max(probs).item()),
            "raw": {"cam": cam_resized.tolist(), "bbox": best_face["bbox"]},
            "metadata": {
                "heatmap_png_base64": heatmap_b64,
                "det_score": best_face["det_score"],
                "target_layer": "_conv_head",
                "interpretation": "brighter regions contributed more to the P(fake) score",
            },
        }
