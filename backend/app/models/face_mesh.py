from typing import Any

from app.models.base import ModelWrapper


class MediaPipeFaceMesh(ModelWrapper):
    """Micro-expression / mesh landmarks via MediaPipe Face Mesh."""

    def load(self) -> None:
        raise NotImplementedError

    def predict(self, input: Any) -> dict:
        raise NotImplementedError
