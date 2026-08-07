from typing import Any

from app.models.base import ModelWrapper
from app.models.device import get_device


class ArcFaceEmbedder(ModelWrapper):
    """Face recognition / identity embedding via ArcFace (InsightFace buffalo_l)."""

    def load(self) -> None:
        self.device = get_device()
        raise NotImplementedError

    def predict(self, input: Any) -> dict:
        raise NotImplementedError
