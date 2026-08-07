from abc import ABC, abstractmethod
from typing import Any


class ModelWrapper(ABC):
    """Common interface every model wrapper must implement.

    This is what lets the fusion layer treat every detector uniformly
    instead of special-casing each model's I/O shape.
    """

    def __init__(self) -> None:
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load weights and move the model to the shared device."""

    @abstractmethod
    def predict(self, input: Any) -> dict:
        """Return {score, confidence, raw, metadata}."""
