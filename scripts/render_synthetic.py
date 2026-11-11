"""Render N synthetic puzzle photos with ground truth, from one font split, optionally noisy.

    # Week 2 data (fonts never shared between splits; seeds never shared either)
    python scripts/render_synthetic.py --split train --n 2000 --seed 10000 --noise-frac 0.4 \
        --out data/synthetic/train
    python scripts/render_synthetic.py --split val --n 300 --seed 20000 --noise-frac 0.4 \
        --out data/synthetic/val
    python scripts/render_synthetic.py --split test --n 200 --seed 30000 \
        --out data/synthetic/test_clean
    # Noisy test sets: SAME seeds as test_clean, so image i is test_clean image i plus noise
    python scripts/render_synthetic.py --split test --n 50 --seed 30000 --noise blur=0.5 \
        --out data/synthetic/test_noise/blur_0.5

Writes <out>/{00000.png, 00000.json, ...}. Puzzle i uses generator seed (seed + i) and render rng
[seed, i], so renders are reproducible one by one. The JSON also records the split and the noise
spec; corners are moved with the image when the noise includes rotation.

--noise-frac f: each image gets random noise (1-2 types, severity 0.1-1, see synth/noise.py
random_spec) with probability f. --noise name=s[,name=s]: the same fixed noise on every image.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sudoku_ns.perception.dataset import fonts_for_split, load_manifest
from sudoku_ns.reasoning.generator import generate
from sudoku_ns.synth.noise import apply_noise, noisy_corners, random_spec
from sudoku_ns.synth.render import Render, list_fonts, render_puzzle, save_render


def parse_spec(text: str | None) -> dict[str, float] | None:
    if not text:
        return None
    return {k: float(v) for k, v in (part.split("=") for part in text.split(","))}


def render_one(job: tuple) -> None:
    i, seed, fonts, out, split, noise_frac, fixed = job
    givens, solution = generate(seed + i)
    rng = np.random.default_rng([seed, i])
    r = render_puzzle(givens, solution, rng, fonts)
    # A separate rng for noise, so adding noise never changes the clean render.
    nrng = np.random.default_rng([seed, i, 1])
    spec = fixed if fixed is not None else (random_spec(nrng) if nrng.random() < noise_frac else {})
    if spec:
        img = apply_noise(r.image, spec, seed=seed + i)
        r = Render(img, r.givens, r.solution, noisy_corners(r.image.shape, r.corners, spec), r.font)
    save_render(r, out, f"{i:05d}", {"split": split, "seed": seed + i, "noise": spec})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fonts", type=Path, default=Path("data/fonts"))
    ap.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    ap.add_argument("--noise-frac", type=float, default=0.0)
    ap.add_argument("--noise", type=str, default=None, help="fixed spec, e.g. blur=0.5")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("data/synthetic/clean"))
    args = ap.parse_args()

    fonts = list_fonts(args.fonts)
    if args.split != "all":
        fonts = fonts_for_split(fonts, args.split, load_manifest(args.fonts / "fonts.json"))
    print(f"{len(fonts)} fonts ({args.split}): {', '.join(f.name for f in fonts)}")
    fixed = parse_spec(args.noise)
    jobs = [
        (i, args.seed, fonts, args.out, args.split, args.noise_frac, fixed) for i in range(args.n)
    ]
    with ProcessPoolExecutor(args.workers) as ex:
        list(tqdm(ex.map(render_one, jobs, chunksize=4), total=args.n, desc="rendering"))
    print(f"wrote {args.n} renders to {args.out}")


if __name__ == "__main__":
    main()
