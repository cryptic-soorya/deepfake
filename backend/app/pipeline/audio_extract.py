"""Extract and resample the audio track from a video for the voice-clone
(AASIST) and lip-sync (SyncNet) detectors.

Both those model wrappers currently decode audio themselves rather than
going through this helper -- AASISTVoiceDetector._load_waveform duplicates
most of what's here. That duplication is a known, deliberate non-fix for
Round 2 (touching a working, tested wrapper wasn't worth the churn); this
module exists as the canonical pipeline-layer extractor per CLAUDE.md's
repo-structure convention, and is what the new lipsync_worker.py uses.
Converging AASISTVoiceDetector onto this helper is a good small cleanup for
Round 3, not urgent before then.
"""
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np


def extract_audio(video: Any, target_sr: int = 16000) -> np.ndarray:
    """video: a file path (str/Path) or raw media bytes (video or audio-only,
    anything librosa/audioread's ffmpeg backend can demux).

    Returns the full-length 1D float32 mono waveform at `target_sr` Hz -- no
    padding or truncation here, since the two callers (AASIST, SyncNet) each
    need a different fixed length and already handle that themselves.

    Returns an empty array (not an exception) if the input has no decodable
    audio track (e.g. a silent/video-only clip), matching how
    AASISTVoiceDetector.predict already treats zero decoded samples as a
    normal, reportable case rather than a crash.
    """
    if isinstance(video, (bytes, bytearray)):
        with tempfile.NamedTemporaryFile(suffix=".media") as tmp:
            tmp.write(video)
            tmp.flush()
            try:
                waveform, _ = librosa.load(tmp.name, sr=target_sr, mono=True)
            except Exception:
                return np.array([], dtype=np.float32)
        return waveform.astype(np.float32)

    path = str(video) if isinstance(video, Path) else video
    try:
        waveform, _ = librosa.load(path, sr=target_sr, mono=True)
    except Exception:
        return np.array([], dtype=np.float32)
    return waveform.astype(np.float32)
