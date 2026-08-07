"""Unit test for app/models/gemini_explainer.py.

Mocks the google-genai client rather than hitting the real Gemini API --
this repo has no live GEMINI_API_KEY in CI/dev by default (the team
provides it out of band). Proves the prompt-building and response-shaping
logic works; does not prove the API contract with a real key. Run manually
against a real key if you want that assurance.
"""
from unittest.mock import MagicMock

import pytest
from google.genai.errors import APIError

from app.models.gemini_explainer import GeminiExplainer, _build_prompt


def _fake_client(response_text=None, error: APIError | None = None) -> MagicMock:
    client = MagicMock()
    if error is not None:
        client.models.generate_content.side_effect = error
    else:
        response = MagicMock()
        response.text = response_text
        client.models.generate_content.return_value = response
    return client


def test_build_prompt_includes_verdict_and_modalities():
    prompt = _build_prompt(
        {
            "fused_score": 0.87,
            "fused_verdict": "likely_fake",
            "model_outputs": {
                "frame_classifier": {"score": 0.9, "confidence": 0.8, "metadata": {}},
                "audio_deepfake": {"score": 0.1, "confidence": 0.7, "metadata": {}},
            },
        }
    )

    assert "likely_fake" in prompt
    assert "0.87" in prompt
    assert "frame_classifier" in prompt
    assert "audio_deepfake" in prompt


def test_predict_calls_gemini_client_and_shapes_output(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_API_KEY_2", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

    explainer = GeminiExplainer()
    fake_client = _fake_client(response_text="This scan is likely fake because...")
    monkeypatch.setattr("app.models.gemini_explainer.genai.Client", lambda api_key: fake_client)

    explainer.load()
    output = explainer.predict(
        {"fused_score": 0.9, "fused_verdict": "likely_fake", "model_outputs": {}}
    )

    assert output["metadata"]["narrative"] == "This scan is likely fake because..."
    assert output["metadata"]["model"] == "gemini-3.1-flash-lite"
    assert output["metadata"]["key_used"] == "primary"
    fake_client.models.generate_content.assert_called_once()
    _, kwargs = fake_client.models.generate_content.call_args
    assert kwargs["model"] == "gemini-3.1-flash-lite"


def test_predict_falls_back_to_second_key_when_primary_errors(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "primary-key")
    monkeypatch.setenv("GEMINI_API_KEY_2", "secondary-key")

    primary_error = APIError(429, {"error": {"message": "quota exceeded"}})
    primary_client = _fake_client(error=primary_error)
    secondary_client = _fake_client(response_text="Fallback narrative")

    clients_by_key = {"primary-key": primary_client, "secondary-key": secondary_client}
    monkeypatch.setattr(
        "app.models.gemini_explainer.genai.Client",
        lambda api_key: clients_by_key[api_key],
    )

    explainer = GeminiExplainer()
    explainer.load()
    output = explainer.predict({"fused_score": 0.5, "fused_verdict": "uncertain", "model_outputs": {}})

    assert output["metadata"]["narrative"] == "Fallback narrative"
    assert output["metadata"]["key_used"] == "fallback"
    primary_client.models.generate_content.assert_called_once()
    secondary_client.models.generate_content.assert_called_once()


def test_predict_raises_last_error_when_both_keys_fail(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "primary-key")
    monkeypatch.setenv("GEMINI_API_KEY_2", "secondary-key")

    primary_error = APIError(429, {"error": {"message": "quota exceeded"}})
    secondary_error = APIError(401, {"error": {"message": "invalid key"}})
    clients_by_key = {
        "primary-key": _fake_client(error=primary_error),
        "secondary-key": _fake_client(error=secondary_error),
    }
    monkeypatch.setattr(
        "app.models.gemini_explainer.genai.Client",
        lambda api_key: clients_by_key[api_key],
    )

    explainer = GeminiExplainer()
    explainer.load()

    with pytest.raises(APIError) as exc_info:
        explainer.predict({"fused_score": 0.5, "fused_verdict": "uncertain", "model_outputs": {}})
    assert exc_info.value is secondary_error


def test_load_raises_key_error_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    explainer = GeminiExplainer()

    try:
        explainer.load()
        assert False, "expected KeyError"
    except KeyError:
        pass
