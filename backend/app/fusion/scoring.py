"""Fusion head over per-modality detector scores.

Target per CLAUDE.md/PLAN.md is a trained MLP/logistic-regression stacking
head with temperature-scaling calibration. No labeled fusion training set
exists yet, so this is an interim, explicitly-flagged substitute: an
equal-weighted average of each modality's fake-probability, direction-
normalized per detector (see `_fake_probability`). Replace with a trained
stacking head once a small labeled fusion set exists (PLAN.md Round 2
checklist item "fusion head").
"""
from typing import Any

FAKE_VERDICT_THRESHOLD = 0.5

# Each model wrapper's raw `score` means something different (see each
# wrapper's docstring/metadata["interpretation"]). This maps every model's
# score onto a common axis: P(media is fake/spoofed), 0=authentic, 1=fake.
_INVERTED_MODELS = {
    # audio_deepfake's score is P(real/bonafide voice); low score = cloned.
    "audio_deepfake",
}


def _fake_probability(model_name: str, output: dict[str, Any]) -> float | None:
    score = output.get("score")
    if score is None:
        return None
    return 1.0 - score if model_name in _INVERTED_MODELS else score


def fuse(model_outputs: dict[str, dict[str, Any]]) -> dict:
    """model_outputs: {model_name: {score, confidence, raw, metadata}} -> fused verdict.

    Skips any modality whose score is None (blocked/no-detection) rather than
    treating it as evidence of anything. Raises NotImplementedError if no
    modality produced a usable score, so callers can surface that honestly
    instead of fabricating a verdict from nothing.
    """
    fake_probs: dict[str, float] = {}
    for model_name, output in model_outputs.items():
        prob = _fake_probability(model_name, output)
        if prob is not None:
            fake_probs[model_name] = prob

    if not fake_probs:
        raise NotImplementedError("no modality produced a usable score for fusion")

    fused_score = sum(fake_probs.values()) / len(fake_probs)
    verdict = "fake" if fused_score >= FAKE_VERDICT_THRESHOLD else "authentic"

    return {
        "score": fused_score,
        "verdict": verdict,
        "raw": {"per_model_fake_probability": fake_probs},
        "metadata": {
            "method": "equal_weighted_average_interim_pending_stacking_head",
            "contributing_models": sorted(fake_probs),
            "interpretation": "score is fused P(fake); >= 0.5 -> fake verdict",
        },
    }
