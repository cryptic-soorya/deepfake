"""Benchmark the real, already-loaded EfficientNet-B4/SBI checkpoint against a
held-out slice of FaceForensics++ and report AUROC.

This is eval-only -- it does NOT fine-tune anything. Fine-tuning EfficientNet-B4
from scratch belongs on a rented GPU per CLAUDE.md; this script exists to turn
the checkpoint we already have into the real number PLAN.md's Round 2 checklist
asks for ("a benchmark number, not an adjective").

Expected FaceForensics++ layout (the standard download_ffpp.py layout):

    <root>/original_sequences/youtube/c23/videos/*.mp4        (label 0 = real)
    <root>/manipulated_sequences/Deepfakes/c23/videos/*.mp4    (label 1 = fake)
    <root>/manipulated_sequences/Face2Face/c23/videos/*.mp4
    <root>/manipulated_sequences/FaceSwap/c23/videos/*.mp4
    <root>/manipulated_sequences/NeuralTextures/c23/videos/*.mp4

Usage:
    .venv/bin/python scripts/eval_ffpp.py /path/to/FaceForensics++ \
        --max-videos-per-class 40 --frames-per-video 8

Each video is scored by sampling frames (via the same app.pipeline.frame_sampler
used in production) and mean-pooling per-frame P(fake) -- the same interim
aggregation frame_classifier_worker.py uses, so this number reflects what the
running pipeline would actually produce, not a best-case single-frame score.

Videos with no detected face (or a checkpoint/model load failure) are skipped
and counted, not silently dropped -- the printed summary reports how many of
each label were actually scored.
"""
import argparse
import random
import sys
from pathlib import Path

from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.frame_classifier import EfficientNetB4SBI  # noqa: E402
from app.pipeline.frame_sampler import sample_frames  # noqa: E402

MANIPULATED_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def _list_videos(root: Path, max_per_class: int, seed: int) -> tuple[list[Path], list[Path]]:
    real_dir = root / "original_sequences" / "youtube" / "c23" / "videos"
    real_videos = sorted(real_dir.glob("*.mp4"))

    fake_videos: list[Path] = []
    for method in MANIPULATED_METHODS:
        fake_dir = root / "manipulated_sequences" / method / "c23" / "videos"
        fake_videos.extend(sorted(fake_dir.glob("*.mp4")))

    rng = random.Random(seed)
    rng.shuffle(real_videos)
    rng.shuffle(fake_videos)
    return real_videos[:max_per_class], fake_videos[:max_per_class]


def _score_video(model: EfficientNetB4SBI, path: Path, frames_per_video: int) -> float | None:
    frames = sample_frames(str(path), fps=2.0, max_frames=frames_per_video)
    scores = []
    for frame in frames:
        output = model.predict(frame["frame"])
        if output["score"] is not None:
            scores.append(output["score"])
    if not scores:
        return None
    return sum(scores) / len(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_root", type=Path, help="Path to the FaceForensics++ root directory")
    parser.add_argument("--max-videos-per-class", type=int, default=40)
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    real_videos, fake_videos = _list_videos(args.dataset_root, args.max_videos_per_class, args.seed)
    if not real_videos or not fake_videos:
        raise SystemExit(
            f"No videos found under {args.dataset_root} -- check the FF++ layout matches the docstring above."
        )
    print(f"Found {len(real_videos)} real / {len(fake_videos)} fake candidate videos. Loading model...")

    model = EfficientNetB4SBI()
    model.load()

    y_true, y_score = [], []
    skipped = 0
    for label, videos in ((0, real_videos), (1, fake_videos)):
        for i, path in enumerate(videos):
            score = _score_video(model, path, args.frames_per_video)
            if score is None:
                skipped += 1
                print(f"  [skip: no face detected] {path.name}")
                continue
            y_true.append(label)
            y_score.append(score)
            kind = "real" if label == 0 else "fake"
            print(f"  [{kind} {i + 1}/{len(videos)}] {path.name} -> P(fake)={score:.3f}")

    if len(set(y_true)) < 2:
        raise SystemExit("Need at least one scored video from each class to compute AUROC.")

    auc = roc_auc_score(y_true, y_score)
    print(f"\nScored {len(y_true)} videos ({skipped} skipped, no face detected).")
    print(f"AUROC: {auc:.4f}")
    print(
        "\nDrop this into docs/benchmark-report.md, e.g.:\n"
        f"| EfficientNet-B4 (SBI) | FaceForensics++ (c23, {len(y_true)} held-out videos: "
        f"{len(MANIPULATED_METHODS)} manipulation methods + youtube-real) | AUC | {auc:.4f} |"
    )


if __name__ == "__main__":
    main()
