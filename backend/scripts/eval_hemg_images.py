"""Benchmark the real, already-loaded EfficientNet-B4/SBI checkpoint against the
Hemg/deepfake-and-real-images sample pulled down by download_hemg_sample.py.

Companion to eval_ffpp.py: that script expects FaceForensics++'s video-folder
layout, which this sample (single JPEGs under fake/ and real/) doesn't match,
so a separate image-native eval lives here instead of overloading eval_ffpp.py
with a second input format.

Usage:
    .venv/bin/python scripts/eval_hemg_images.py scripts/data/hemg_deepfake
"""
import argparse
import sys
from pathlib import Path

import cv2
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.frame_classifier import EfficientNetB4SBI  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_root", type=Path, help="Path to the hemg_deepfake sample (contains fake/ and real/)")
    args = parser.parse_args()

    real_images = sorted((args.dataset_root / "real").glob("*.jpg"))
    fake_images = sorted((args.dataset_root / "fake").glob("*.jpg"))
    if not real_images or not fake_images:
        raise SystemExit(f"No images found under {args.dataset_root} -- expected fake/ and real/ subfolders.")
    print(f"Found {len(real_images)} real / {len(fake_images)} fake images. Loading model...")

    model = EfficientNetB4SBI()
    model.load()

    y_true, y_score = [], []
    skipped = 0
    for label, images in ((0, real_images), (1, fake_images)):
        for i, path in enumerate(images):
            frame = cv2.imread(str(path))
            if frame is None:
                skipped += 1
                print(f"  [skip: unreadable] {path.name}")
                continue
            output = model.predict(frame)
            if output["score"] is None:
                skipped += 1
                print(f"  [skip: {output.get('metadata', {}).get('error', 'no score')}] {path.name}")
                continue
            y_true.append(label)
            y_score.append(output["score"])
            kind = "real" if label == 0 else "fake"
            print(f"  [{kind} {i + 1}/{len(images)}] {path.name} -> P(fake)={output['score']:.3f}")

    if len(set(y_true)) < 2:
        raise SystemExit("Need at least one scored image from each class to compute AUROC.")

    auc = roc_auc_score(y_true, y_score)
    print(f"\nScored {len(y_true)} images ({skipped} skipped).")
    print(f"AUROC: {auc:.4f}")
    print(
        "\nDrop this into docs/benchmark-report.md, e.g.:\n"
        f"| EfficientNet-B4 (SBI) | Hemg/deepfake-and-real-images sample ({len(y_true)} held-out images) "
        f"| AUC | {auc:.4f} |"
    )


if __name__ == "__main__":
    main()
