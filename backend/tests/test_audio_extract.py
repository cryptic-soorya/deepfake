"""Unit tests for audio extraction. Uses a synthetically generated .wav (pure
sine tone) so this needs no network and no video codec support -- just
librosa/soundfile, which are already hard requirements of this project.
"""
import numpy as np
import soundfile as sf

from app.pipeline.audio_extract import extract_audio


def _write_tone(path, duration_sec=1.0, sr=22050, freq=440.0):
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    sf.write(str(path), tone, sr)


def test_extract_audio_resamples_from_path(tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_tone(wav_path, duration_sec=1.0, sr=22050)

    waveform = extract_audio(str(wav_path), target_sr=16000)

    assert waveform.dtype == np.float32
    # 1s of audio resampled to 16kHz should be close to 16000 samples.
    assert abs(len(waveform) - 16000) < 200


def test_extract_audio_from_bytes(tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_tone(wav_path, duration_sec=0.5, sr=16000)
    data = wav_path.read_bytes()

    waveform = extract_audio(data, target_sr=16000)

    assert waveform.dtype == np.float32
    assert abs(len(waveform) - 8000) < 200


def test_extract_audio_returns_empty_array_for_undecodable_input():
    waveform = extract_audio(b"not audio data at all", target_sr=16000)
    assert waveform.size == 0
    assert waveform.dtype == np.float32
