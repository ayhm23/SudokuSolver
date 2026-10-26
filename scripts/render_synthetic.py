"""Render N clean synthetic puzzle photos with ground truth.

    python scripts/render_synthetic.py --n 200 --seed 0

Writes data/synthetic/clean/{00000.png, 00000.json, ...}. Puzzle i uses
generator seed (seed + i), so renders are reproducible one by one.
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sudoku_ns.reasoning.generator import generate
from sudoku_ns.synth.render import list_fonts, render_puzzle, save_render


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fonts", type=Path, default=Path("data/fonts"))
    ap.add_argument("--out", type=Path, default=Path("data/synthetic/clean"))
    args = ap.parse_args()

    fonts = list_fonts(args.fonts)
    print(f"{len(fonts)} fonts")
    for i in tqdm(range(args.n), desc="rendering"):
        givens, solution = generate(args.seed + i)
        rng = np.random.default_rng([args.seed, i])
        save_render(render_puzzle(givens, solution, rng, fonts), args.out, f"{i:05d}")
    print(f"wrote {args.n} renders to {args.out}")


if __name__ == "__main__":
    main()
