"""Build the 32x32 digit dataset from rendered puzzles, split by font family.

    python scripts/make_digit_dataset.py --renders data/synthetic/clean --out data/digits/digits.npz

Prints which fonts landed in each split and the class counts per split.
"""

import argparse
from pathlib import Path

import numpy as np

from sudoku_ns.perception.dataset import build_dataset, class_counts, save_npz


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--renders", type=Path, default=Path("data/synthetic/clean"))
    ap.add_argument("--out", type=Path, default=Path("data/digits/digits.npz"))
    ap.add_argument("--aug", type=int, default=2, help="augmented copies per train crop")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = build_dataset(args.renders, n_aug=args.aug, seed=args.seed)
    save_npz(ds, args.out)
    print(f"wrote {args.out}\n")
    print(
        f"{'split':<6} {'n':>6}  " + " ".join(f"{k:>5}" for k in ["empty", *map(str, range(1, 10))])
    )
    for split, d in ds.items():
        counts = class_counts(d["y"])
        print(f"{split:<6} {len(d['y']):>6}  " + " ".join(f"{c:>5}" for c in counts))
    print()
    for split, d in ds.items():
        print(f"{split} fonts: {', '.join(sorted(np.unique(d['font'])))}")


if __name__ == "__main__":
    main()
