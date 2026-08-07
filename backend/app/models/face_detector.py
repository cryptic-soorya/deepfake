from typing import Any

from app.models.base import ModelWrapper
from app.models.device import get_device


class SCRFDFaceDetector(ModelWrapper):
    """Face detection + landmarks via SCRFD (InsightFace)."""

    def load(self) -> None:
        self.device = get_device()
        raise NotImplementedError

    def predict(self, input: Any) -> dict:
        raise NotImplementedError
