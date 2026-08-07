from typing import Any

from app.models.base import ModelWrapper
from app.models.device import get_device


class AASISTVoiceDetector(ModelWrapper):
    """Audio deepfake / voice-clone detector: AASIST + wav2vec2-XLSR-53 front end."""

    def load(self) -> None:
        self.device = get_device()
        raise NotImplementedError

    def predict(self, input: Any) -> dict:
        raise NotImplementedError
