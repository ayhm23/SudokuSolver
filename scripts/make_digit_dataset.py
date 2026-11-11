"""Build the 32x32 digit dataset from rendered puzzles, one render directory per font split.

    python scripts/make_digit_dataset.py --train data/synthetic/train --val data/synthetic/val \
        --test data/synthetic/test_clean --out data/digits/digits.npz

Crops are cut with grid detection + vision.cell_extract.extract_cells (the
pipeline's own code path, see perception/dataset.py). Prints skipped images,
fonts per split and class counts per split.
"""

import argparse
from pathlib import Path

import numpy as np

from sudoku_ns.perception.dataset import build_split, class_counts, save_npz


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--train", type=Path, default=Path("data/synthetic/train"))
    ap.add_argument("--val", type=Path, default=Path("data/synthetic/val"))
    ap.add_argument("--test", type=Path, default=Path("data/synthetic/test_clean"))
    ap.add_argument("--out", type=Path, default=Path("data/digits/digits.npz"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    ds = {}
    for split in ("train", "val", "test"):
        data, counts = build_split(getattr(args, split), split, workers=args.workers)
        ds[split] = data
        print(
            f"{split}: {counts['images']} images, {counts['not_found']} grid not found, "
            f"{counts['bad_corners']} corners off by > 8 px -> {len(data['y'])} crops"
        )
    save_npz(ds, args.out)
    print(f"\nwrote {args.out}\n")
    names = ["empty", *map(str, range(1, 10))]
    print(f"{'split':<6} {'n':>7}  " + " ".join(f"{k:>6}" for k in names) + "   empty%")
    for split, d in ds.items():
        counts = class_counts(d["y"])
        print(
            f"{split:<6} {len(d['y']):>7}  "
            + " ".join(f"{c:>6}" for c in counts)
            + f"   {counts[0] / max(counts.sum(), 1):.1%}"
        )
    print()
    for split, d in ds.items():
        print(f"{split} fonts: {', '.join(sorted(map(str, np.unique(d['font']))))}")


if __name__ == "__main__":
    main()
