"""Audio deepfake / voice-clone detector.

CLAUDE.md specifies AASIST with a wav2vec2-XLSR-53 front end (the SSL-AASIST
variant from TakHemlata/SSL_Anti-spoofing). That variant's pretrained
checkpoint lives in a Google Drive *folder* (no directly resolvable file ID)
and pulls in a pinned fairseq fork vendored into that repo.

Interim, visible substitution: this wrapper uses plain AASIST (clovaai/aasist)
instead — same graph-attention back end, but its own sinc-conv front end in
place of XLSR-53. It has a directly downloadable pretrained checkpoint
(ASVspoof2019 LA) and no fairseq dependency. Swap in the XLSR-53 front end
once its checkpoint can be fetched and the fairseq dependency verified on
real hardware.
"""
import tempfile
from typing import Any

import librosa
import numpy as np
import torch

from app.models.base import ModelWrapper
from app.models.device import get_device
from app.models.vendor.aasist import Model as AASISTModel
from app.models.vendor.weights import fetch

WEIGHTS_URL = "https://raw.githubusercontent.com/clovaai/aasist/main/models/weights/AASIST.pth"
WEIGHTS_FILENAME = "AASIST.pth"

SAMPLE_RATE = 16000
NUM_SAMPLES = 64600  # ~4.04s at 16kHz, per config/AASIST.conf

# model_config block from clovaai/aasist's config/AASIST.conf, required to
# reconstruct the architecture the checkpoint's state dict matches.
MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": NUM_SAMPLES,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


def _pad(waveform: np.ndarray, max_len: int = NUM_SAMPLES) -> np.ndarray:
    """Eval-time padding from the original repo's data_utils.py: truncate if
    long enough, otherwise tile-repeat up to max_len."""
    length = waveform.shape[0]
    if length >= max_len:
        return waveform[:max_len]
    num_repeats = int(max_len / length) + 1
    return np.tile(waveform, num_repeats)[:max_len]


class AASISTVoiceDetector(ModelWrapper):
    """Audio deepfake / voice-clone detector: AASIST (see module docstring re: XLSR-53 substitution)."""

    def load(self) -> None:
        self.device = get_device()
        self._net = AASISTModel(MODEL_CONFIG)

        weights_path = fetch(WEIGHTS_URL, WEIGHTS_FILENAME)
        state_dict = torch.load(weights_path, map_location=self.device)
        self._net.load_state_dict(state_dict)
        self._net.to(self.device).eval()
        self._loaded = True

    def predict(self, input: Any) -> dict:
        """input: raw audio/video bytes or a file path. Any container ffmpeg
        can demux (mp4, wav, ...) works — librosa decodes and resamples to
        16kHz mono."""
        waveform = self._load_waveform(input)
        if waveform.size == 0:
            return {
                "score": None,
                "confidence": None,
                "raw": None,
                "metadata": {"error": "no audio samples decoded from input"},
            }

        padded = _pad(waveform)
        tensor = torch.from_numpy(padded).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, logits = self._net(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        real_score = float(probs[1])  # class 1 = bonafide, per genSpoof_list's label convention

        return {
            "score": real_score,
            "confidence": float(max(probs).item()),
            "raw": {"probs": probs.tolist()},
            "metadata": {
                "num_samples": int(waveform.shape[0]),
                "interpretation": "score is P(real/bonafide voice); low score indicates a cloned/synthetic voice",
            },
        }

    @staticmethod
    def _load_waveform(input: Any) -> np.ndarray:
        if isinstance(input, np.ndarray):
            return input.astype(np.float32)
        if isinstance(input, str):
            waveform, _ = librosa.load(input, sr=SAMPLE_RATE, mono=True)
            return waveform
        if isinstance(input, (bytes, bytearray)):
            with tempfile.NamedTemporaryFile(suffix=".media") as tmp:
                tmp.write(input)
                tmp.flush()
                waveform, _ = librosa.load(tmp.name, sr=SAMPLE_RATE, mono=True)
            return waveform
        raise TypeError(f"unsupported input type for AASISTVoiceDetector: {type(input)!r}")
