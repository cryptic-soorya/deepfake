# Benchmark Report

Filled in once models are trained/evaluated. Per CLAUDE.md, Round 2 is not
"done" until there's a benchmark number (not an adjective) for at least the
visual and audio detectors.

## Visual (frame-level + video-level)

| Model | Dataset | Metric | Value |
|---|---|---|---|
| EfficientNet-B4 (SBI) | Hemg/deepfake-and-real-images sample (61 held-out images, 19 skipped — no face detected; see substitution note below) | AUC | 0.7519 |
| Temporal attention head | TBD | AUC | TBD |
| UnivFD | TBD | AUC | TBD |

**Honesty note:** PLAN.md's Round 2 checklist asks for FF++/DFDC. That download is
gated behind an EULA form the team hasn't completed, so this number is instead
from a small (~80-image) slice of a public, no-auth Hugging Face real/fake face
dataset (`scripts/download_hemg_sample.py` / `scripts/eval_hemg_images.py`) —
single images, not the FF++ manipulation-method video split, and well below the
PLAN.md target sample size. 0.7519 AUC is a real, reproducible number from the
actual deployed checkpoint (not an adjective), but it should be presented to
judges as "in-progress, on a substitute open dataset" — not treated as the
FF++/DFDC benchmark PLAN.md specifies. Re-run against real FF++/DFDC once EULA
access lands to get the number that actually satisfies the target table below.

## Audio

| Model | Dataset | Metric | Value |
|---|---|---|---|
| AASIST + XLSR-53 | TBD | EER | TBD |

## Lip-sync

| Model | Dataset | Metric | Value |
|---|---|---|---|
| SyncNet | TBD | TBD | TBD |

## Fusion

| Metric | Value |
|---|---|
| AUC | TBD |
| Calibration (ECE) | TBD |
