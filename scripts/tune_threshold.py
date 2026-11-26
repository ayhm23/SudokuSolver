"""Choose tau for the threshold_then_sat baseline on the VALIDATION renders.

    python scripts/tune_threshold.py --checkpoint outputs/models/cnn_v1/model.pt \
        --renders data/synthetic/val

A baseline is only a fair comparison if its one hyper-parameter is tuned as
well as our method's. Week 1 used tau = 0.8, which sat in a gap of the
SIMULATOR's confidence ranges; the CNN's confidences look different. Each val
image is detected and classified once, then solved with every tau; the tau
with the most correctly solved images wins (ties: the smaller tau, i.e. the
one that drops fewer digits). Test images are never used.

Several directories (or globs) can be given, e.g. the noisy val sets:

    python scripts/tune_threshold.py --checkpoint outputs/models/cnn_v1/model.pt \
        --renders "data/synthetic/val_noise/*" --out outputs/tune_threshold_val_noise.json
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sudoku_ns.perception.perceiver import CNNPerceiver
from sudoku_ns.pipeline import perceive
from sudoku_ns.reasoning.baselines import threshold_then_sat
from sudoku_ns.synth.render import load_render
from sudoku_ns.vision.grid_detect import GridNotFoundError

TAUS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--renders", nargs="+", default=["data/synthetic/val"], help="dirs or globs")
    ap.add_argument("--out", type=Path, default=None, help="also save the table as JSON")
    args = ap.parse_args()
    dirs = sorted({Path(p) for pat in args.renders for p in glob.glob(pat) if Path(p).is_dir()})
    if not dirs:
        raise SystemExit(f"no render directories match {args.renders}")
    if any("test" in d.as_posix() for d in dirs):
        raise SystemExit("refusing to tune on test renders")
    jsons = [jp for d in dirs for jp in sorted(d.glob("*.json"))]

    perceiver = CNNPerceiver(args.checkpoint)
    correct = {t: 0 for t in TAUS}
    n = 0
    for jp in tqdm(jsons, desc="val images"):
        r = load_render(jp)
        n += 1
        try:
            _, _, prob = perceive(r.image, perceiver)
        except GridNotFoundError:
            continue
        for t in TAUS:
            res = threshold_then_sat(prob, t, check_unique=False)
            correct[t] += res.solution is not None and bool(
                np.array_equal(res.solution, r.solution)
            )
    print(f"\n{n} val images; % solved correctly by threshold_then_sat:")
    for t in TAUS:
        print(f"  tau {t:<6} {100 * correct[t] / n:5.1f}%")
    best = max(TAUS, key=lambda t: (correct[t], -t))
    print(f'\nbest tau = {best} (put it in the configs\' "tau")')
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "renders": [d.as_posix() for d in dirs],
            "n": n,
            "solved_pct": {str(t): 100 * correct[t] / n for t in TAUS},
            "best_tau": best,
        }
        args.out.write_text(json.dumps(record, indent=2), "utf-8")


if __name__ == "__main__":
    main()
