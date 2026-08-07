from typing import Any

from app.models.base import ModelWrapper
from app.models.device import get_device


class UnivFDDetector(ModelWrapper):
    """Unseen-generator / generalization detector: frozen CLIP ViT-L/14 + linear probe."""

    def load(self) -> None:
        self.device = get_device()
        raise NotImplementedError

    def predict(self, input: Any) -> dict:
        raise NotImplementedError
