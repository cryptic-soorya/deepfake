"""Lip-sync consistency via SyncNet (pretrained inference only).

Reimplements joonson/syncnet_python's `calc_pdist`/offset-confidence
inference exactly (see app/models/vendor/syncnet.py for the vendored
architecture and provenance) against the authors' own `syncnet_v2.model`
checkpoint, downloaded lazily the same way liveness.py fetches MiniFASNet's
weights. Frame/audio extraction (ffmpeg, face tracking, mouth-region
cropping) is the pipeline layer's job (app/pipeline/*) — this wrapper takes
already-cropped mouth frames and a mono 16kHz waveform, matching how
face_detector.py takes decoded images rather than raw video files.
"""
from typing import Any

import numpy as np
import python_speech_features
import torch
import torch.nn.functional as F

from app.models.base import ModelWrapper
from app.models.device import get_device
from app.models.vendor.syncnet import SyncNetS
from app.models.vendor.weights import fetch

WEIGHTS_URL = "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model"
VIDEO_FPS = 25
AUDIO_SAMPLE_RATE = 16000
LIP_FRAME_SIZE = 224  # network expects 224x224 mouth-centered crops


def _calc_pdist(feat1: torch.Tensor, feat2: torch.Tensor, vshift: int = 15) -> list[torch.Tensor]:
    win_size = vshift * 2 + 1
    feat2p = F.pad(feat2, (0, 0, vshift, vshift))
    return [
        F.pairwise_distance(feat1[[i], :].repeat(win_size, 1), feat2p[i : i + win_size, :])
        for i in range(len(feat1))
    ]


class SyncNetLipSync(ModelWrapper):
    """Lip-sync consistency via SyncNet (pretrained inference only)."""

    def __init__(self, num_layers_in_fc_layers: int = 1024, vshift: int = 15) -> None:
        super().__init__()
        self.num_layers_in_fc_layers = num_layers_in_fc_layers
        self.vshift = vshift

    def load(self) -> None:
        self.device = get_device()
        self._net = SyncNetS(num_layers_in_fc_layers=self.num_layers_in_fc_layers).to(self.device)
        weights_path = fetch(WEIGHTS_URL, "syncnet_v2.model")
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self._net.load_state_dict(state_dict)
        self._net.eval()
        self._loaded = True

    def predict(self, input: Any) -> dict:
        """input: {"lip_frames": (T,224,224,3) uint8 BGR mouth-cropped frames
        at 25fps, "audio": 1D int16/float mono waveform at 16kHz}.
        """
        lip_frames = np.asarray(input["lip_frames"])
        audio = np.asarray(input["audio"], dtype=np.float64)

        if lip_frames.ndim != 4 or lip_frames.shape[1:3] != (LIP_FRAME_SIZE, LIP_FRAME_SIZE):
            raise ValueError(f"lip_frames must be (T, {LIP_FRAME_SIZE}, {LIP_FRAME_SIZE}, 3), got {lip_frames.shape}")

        # T,H,W,C -> 1,C,T,H,W
        video_tensor = torch.from_numpy(lip_frames).permute(3, 0, 1, 2).unsqueeze(0).float()

        mfcc = python_speech_features.mfcc(audio, AUDIO_SAMPLE_RATE)
        mfcc = np.stack(list(zip(*mfcc)))  # (num_ceps, num_frames)
        audio_tensor = torch.from_numpy(mfcc).float().unsqueeze(0).unsqueeze(0)  # 1,1,ceps,frames

        min_length = min(lip_frames.shape[0], audio_tensor.shape[3] // 4)
        lastframe = min_length - 5
        if lastframe < 1:
            return {
                "score": 0.5,
                "confidence": 0.0,
                "raw": {},
                "metadata": {"error": "clip too short for a sync estimate (need >= 6 aligned frames)"},
            }

        im_feats, cc_feats = [], []
        with torch.no_grad():
            for i in range(lastframe):
                im_in = video_tensor[:, :, i : i + 5, :, :].to(self.device)
                im_feats.append(self._net.forward_lip(im_in).cpu())

                cc_in = audio_tensor[:, :, :, i * 4 : i * 4 + 20].to(self.device)
                cc_feats.append(self._net.forward_aud(cc_in).cpu())

        im_feat = torch.cat(im_feats, 0)
        cc_feat = torch.cat(cc_feats, 0)

        dists = _calc_pdist(im_feat, cc_feat, vshift=self.vshift)
        mdist = torch.mean(torch.stack(dists, 1), 1)
        minval, minidx = torch.min(mdist, 0)

        offset = int(self.vshift - minidx.item())
        confidence = float(torch.median(mdist).item() - minval.item())
        # confidence is an unbounded distance margin in the original repo;
        # squash to [0, 1] so this wrapper matches the shared model contract
        sync_score = float(torch.sigmoid(torch.tensor(confidence - 1.0)).item())

        return {
            "score": sync_score,
            "confidence": sync_score,
            "raw": {"av_offset_frames": offset, "raw_confidence": confidence, "min_dist": float(minval.item())},
            "metadata": {
                "interpretation": (
                    "score is A/V sync confidence in [0, 1]; low score or "
                    "|av_offset_frames| >> 0 suggests dubbed/lip-sync-manipulated audio"
                )
            },
        }
