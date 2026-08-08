"""Download a small held-out slice of the Hemg/deepfake-and-real-images
dataset (Hugging Face) for the visual-detector benchmark.

Substitution note (flagged per CLAUDE.md, not silent): PLAN.md's Round 2
checklist calls for FaceForensics++/DFDC, but FF++'s download script is
gated behind an EULA form the team hasn't completed yet, and this machine
has too little free disk for the full dataset regardless. This pulls a
small (~a few hundred image) slice of a freely-downloadable, no-auth
Hugging Face face real/fake dataset instead, via the public
datasets-server `/rows` API (small hosted JPEG thumbnails, no bulk
parquet/tar download) so it fits in a few dozen MB.

Label convention on the source dataset: 0 = Fake, 1 = Real. Row order is
label-sorted (all Fake rows first, then all Real), not shuffled -- offsets
below were picked by manually probing where the label flips (see
PROGRESS.md), not from any documented split boundary, so treat them as
approximate and re-probe if the dataset is ever updated.

Usage:
    .venv/bin/python scripts/download_hemg_sample.py --n-per-class 60
"""
import argparse
import sys
import time
import urllib.request
from pathlib import Path

DATASET = "Hemg/deepfake-and-real-images"
API = "https://datasets-server.huggingface.co/rows"
OUT_ROOT = Path(__file__).resolve().parent / "data" / "hemg_deepfake"

# Empirically probed row offsets (see docstring) where each label is dense,
# to avoid paging through ~50k rows of the other class first.
FAKE_OFFSET = 0
REAL_OFFSET = 100_000

PAGE_SIZE = 100


def _fetch_rows(offset: int, length: int) -> list[dict]:
    url = f"{API}?dataset={DATASET}&config=default&split=train&offset={offset}&length={length}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        import json

        return json.load(resp)["rows"]


def _download_class(label_name: str, expected_label: int, start_offset: int, n: int) -> int:
    out_dir = OUT_ROOT / label_name
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    offset = start_offset
    while saved < n:
        rows = _fetch_rows(offset, PAGE_SIZE)
        if not rows:
            break
        for row in rows:
            if saved >= n:
                break
            r = row["row"]
            if r["label"] != expected_label:
                continue
            dest = out_dir / f"{row['row_idx']}.jpg"
            if dest.exists():
                saved += 1
                continue
            src_url = r["image"]["src"]
            try:
                urllib.request.urlretrieve(src_url, dest)
            except Exception as exc:  # pragma: no cover - network dependent
                print(f"  [skip] row {row['row_idx']}: {exc}", file=sys.stderr)
                continue
            saved += 1
            if saved % 10 == 0:
                print(f"  {label_name}: {saved}/{n}")
            time.sleep(0.05)  # be polite to the shared datasets-server cache
        offset += PAGE_SIZE
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-per-class", type=int, default=60)
    args = parser.parse_args()

    print(f"Downloading {args.n_per_class} fake images...")
    n_fake = _download_class("fake", expected_label=0, start_offset=FAKE_OFFSET, n=args.n_per_class)
    print(f"Downloading {args.n_per_class} real images...")
    n_real = _download_class("real", expected_label=1, start_offset=REAL_OFFSET, n=args.n_per_class)

    print(f"\nSaved {n_fake} fake / {n_real} real images under {OUT_ROOT}")


if __name__ == "__main__":
    main()
