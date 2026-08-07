from typing import Any

from app.models.base import ModelWrapper
from app.models.device import get_device


class MiniFASNetLiveness(ModelWrapper):
    """Liveness / anti-spoofing via Silent-Face-Anti-Spoofing (MiniFASNet)."""

    def load(self) -> None:
        self.device = get_device()
        raise NotImplementedError

    def predict(self, input: Any) -> dict:
        raise NotImplementedError
