"""Narrative explanation generation via the Gemini API.

Deliberate substitution for the "Claude API (Sonnet)" slot CLAUDE.md's model
stack table specifies — the team asked for Gemini 3.1 Flash-Lite instead
(cost/latency tradeoff for a narrative-generation task that doesn't need
Sonnet-level reasoning). Per CLAUDE.md's "no silent substitutions" rule,
this is called out explicitly here and in PROGRESS.md, not swapped quietly.
CLAUDE.md's model-stack table should be updated to match.
"""
import os
from typing import Any

from google import genai
from google.genai.errors import APIError

from app.models.base import ModelWrapper

DEFAULT_MODEL = "gemini-3.1-flash-lite"


class GeminiExplainer(ModelWrapper):
    """Turns a fused verdict + per-modality scores into a plain-English narrative.

    Fails over from GEMINI_API_KEY to GEMINI_API_KEY_2 on any API error
    (invalid/revoked key, quota exhaustion, transient Gemini-side failure) --
    team runs two keys so a single key's rate limit doesn't block scans
    mid-demo. Only the primary key is required; the second is optional.
    """

    def load(self) -> None:
        primary_key = os.environ["GEMINI_API_KEY"]
        secondary_key = os.environ.get("GEMINI_API_KEY_2") or None
        self.model_name = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

        self._clients = [genai.Client(api_key=primary_key)]
        if secondary_key:
            self._clients.append(genai.Client(api_key=secondary_key))
        self._loaded = True

    def predict(self, input: Any) -> dict:
        """input: {fused_score, fused_verdict, model_outputs} -- returns a
        2-3 paragraph narrative in metadata.narrative. Tries each configured
        key in order, returning on the first success; re-raises the last
        key's error if every key fails, so the caller's existing
        `except APIError` handling still degrades cleanly."""
        prompt = _build_prompt(input)

        last_error: APIError | None = None
        for index, client in enumerate(self._clients):
            try:
                response = client.models.generate_content(model=self.model_name, contents=prompt)
            except APIError as exc:
                last_error = exc
                continue

            return {
                "score": None,
                "confidence": None,
                "raw": {"prompt": prompt},
                "metadata": {
                    "narrative": response.text,
                    "model": self.model_name,
                    "key_used": "primary" if index == 0 else "fallback",
                },
            }

        raise last_error


def _build_prompt(input: dict) -> str:
    fused_score = input.get("fused_score")
    fused_verdict = input.get("fused_verdict")
    model_outputs = input.get("model_outputs", {})

    lines = [
        "You are a forensic analyst explaining an AI-generated media detection "
        "verdict to a non-technical reader. Be precise, do not hedge beyond what "
        "the numbers support, and do not invent evidence that isn't listed below.",
        "",
        f"Fused verdict: {fused_verdict}",
        f"Fused score (P(fake), 0-1): {fused_score}",
        "",
        "Per-modality detector outputs:",
    ]
    for name, output in model_outputs.items():
        lines.append(
            f"- {name}: score={output.get('score')}, confidence={output.get('confidence')}, "
            f"metadata={output.get('metadata')}"
        )
    lines.append(
        "\nWrite a 2-3 paragraph plain-English explanation of this verdict: what each "
        "modality found, how confident the system is overall, and what a human reviewer "
        "should look at next. No markdown headers, no bullet lists."
    )
    return "\n".join(lines)
