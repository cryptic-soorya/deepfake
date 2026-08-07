from typing import Any

from app.models.base import ModelWrapper
from app.models.device import get_device


class EfficientNetB4SBI(ModelWrapper):
    """Frame-level deepfake classifier: EfficientNet-B4 trained via the Self-Blended Images (SBI) recipe."""

    def load(self) -> None:
        self.device = get_device()
        raise NotImplementedError

    def predict(self, input: Any) -> dict:
        raise NotImplementedError
