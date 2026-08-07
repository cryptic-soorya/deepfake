import os
from typing import Any

from app.models.base import ModelWrapper


class ClaudeExplainer(ModelWrapper):
    """Narrative explanation generation via the Claude API (Sonnet)."""

    def load(self) -> None:
        self.api_key = os.environ["ANTHROPIC_API_KEY"]
        raise NotImplementedError

    def predict(self, input: Any) -> dict:
        """input: fused verdict + per-model scores/metadata; returns a narrative explanation string."""
        raise NotImplementedError
