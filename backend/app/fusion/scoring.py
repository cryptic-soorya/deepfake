"""Small MLP/logistic-regression stacking head that fuses per-model scores into one verdict."""

from typing import Any


def fuse(model_outputs: dict[str, dict[str, Any]]) -> dict:
    """model_outputs: {model_name: {score, confidence, raw, metadata}} -> fused verdict."""
    raise NotImplementedError
