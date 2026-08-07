from typing import Any

from app.models.base import ModelWrapper


class GradCAMExplainer(ModelWrapper):
    """Visual explainability via Grad-CAM / Grad-CAM++ over the frame classifier."""

    def load(self) -> None:
        raise NotImplementedError

    def predict(self, input: Any) -> dict:
        """Returns a heatmap overlay, not a score — kept in the same interface for pipeline uniformity."""
        raise NotImplementedError
