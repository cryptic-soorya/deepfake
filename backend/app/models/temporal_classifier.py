from typing import Any

from app.models.base import ModelWrapper
from app.models.device import get_device


class TemporalAttentionClassifier(ModelWrapper):
    """Video-level temporal classifier: GRU/temporal-attention head over per-frame embeddings."""

    def load(self) -> None:
        self.device = get_device()
        raise NotImplementedError

    def predict(self, input: Any) -> dict:
        raise NotImplementedError
