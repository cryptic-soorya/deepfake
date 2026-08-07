from typing import Any

from app.models.base import ModelWrapper
from app.models.device import get_device


class SyncNetLipSync(ModelWrapper):
    """Lip-sync consistency via SyncNet (pretrained inference only)."""

    def load(self) -> None:
        self.device = get_device()
        raise NotImplementedError

    def predict(self, input: Any) -> dict:
        raise NotImplementedError
