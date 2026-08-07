"""Fuses per-model scores into one verdict.

Interim, visible substitution (per CLAUDE.md's substitution-flagging rule):
PLAN.md/CLAUDE.md call for a *learned* stacking head (small MLP or logistic
regression) plus temperature-scaling calibration, trained on a labeled
held-out set. No labeled fusion-training set exists yet, so this is a fixed
weighted average instead -- a real, documented interim step, not a stub.
Swap in a trained LogisticRegression/small MLP (scikit-learn or a 2-layer
torch head is enough) once a labeled set of {model scores -> real/fake
label} is available; keep this function's signature and output contract
identical so `fusion_worker.py` doesn't need to change.

Each per-modality model reports its score in its own native direction (see
each model wrapper's `metadata["interpretation"]`) -- fusion first normalizes
every score to a common "P(fake)" direction before combining them.
"""
from typing import Any

# score -> P(fake), per model_name. Must match each wrapper's own
# `metadata["interpretation"]` (see app/models/*.py docstrings).
_TO_FAKE_PROB: dict[str, Any] = {
    "frame_classifier": lambda score: score,  # already P(fake)
    "audio_deepfake": lambda score: 1.0 - score,  # score is P(real/bonafide)
    "lipsync": lambda score: 1.0 - score,  # score is A/V sync confidence; low = desynced/manipulated
}

# Relative trust in each modality when more than one is present. Renormalized
# over whichever modalities actually produced a usable (non-None) score for
# this scan, so a missing/failed modality doesn't silently zero out the
# fused score -- it's just excluded and noted in metadata. Lip-sync gets the
# lowest weight of the three: it's the newest-integrated signal, it's silent
# on image scans and audio-only manipulations, and its sync-confidence score
# hasn't been benchmarked against a labeled desync/sync set yet (unlike
# frame_classifier and audio_deepfake, which ship with published benchmark
# numbers on their training sets).
_MODEL_WEIGHTS: dict[str, float] = {
    "frame_classifier": 0.45,
    "audio_deepfake": 0.30,
    "lipsync": 0.25,
}

# Fused P(fake) -> verdict label. Round-2 placeholder thresholds -- revisit
# once benchmark numbers (AUROC/EER, per PLAN.md ยง6) exist to calibrate
# these against an actual false-positive-rate target.
FAKE_THRESHOLD = 0.7
AUTHENTIC_THRESHOLD = 0.3


def _verdict_for(fake_prob: float) -> str:
    if fake_prob >= FAKE_THRESHOLD:
        return "fake"
    if fake_prob <= AUTHENTIC_THRESHOLD:
        return "authentic"
    return "suspicious"


def fuse(model_outputs: dict[str, dict[str, Any]]) -> dict:
    """model_outputs: {model_name: {score, confidence, raw, metadata}} -> fused verdict.

    Returns {"score": P(fake) in [0,1] or None, "verdict": str, "metadata": {...}}.
    `score` is None only when every contributing model itself returned score=None
    (e.g. no face detected in any sampled frame *and* no audio decoded) --
    fusion_worker treats a None fused score as still COMPLETED (a real "we
    could not form an opinion" answer), not BLOCKED (which is reserved for
    "the model literally isn't wired in yet").
    """
    contributions: dict[str, float] = {}
    skipped: dict[str, str] = {}

    for model_name, output in model_outputs.items():
        to_fake_prob = _TO_FAKE_PROB.get(model_name)
        if to_fake_prob is None:
            skipped[model_name] = "no fake-direction mapping registered for this model"
            continue

        raw_score = output.get("score") if output else None
        if raw_score is None:
            skipped[model_name] = output.get("metadata", {}).get("error", "score unavailable") if output else "no output"
            continue

        contributions[model_name] = to_fake_prob(raw_score)

    if not contributions:
        return {
            "score": None,
            "verdict": "inconclusive",
            "metadata": {"reason": "no modality produced a usable score", "skipped": skipped},
        }

    weights = {name: _MODEL_WEIGHTS.get(name, 1.0) for name in contributions}
    total_weight = sum(weights.values())
    fused_fake_prob = sum(contributions[name] * weights[name] for name in contributions) / total_weight

    return {
        "score": fused_fake_prob,
        "verdict": _verdict_for(fused_fake_prob),
        "metadata": {
            "method": "fixed_weighted_average",
            "per_model_fake_prob": contributions,
            "weights_used": {name: weights[name] / total_weight for name in contributions},
            "skipped": skipped,
            "note": "interim fixed weights -- not yet a trained stacking head, see module docstring",
        },
    }
