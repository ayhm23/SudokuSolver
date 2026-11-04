"""Time solve() with pairwise vs sequential-counter at-most-one encodings.

    python scripts/bench_encoding.py --n 100 --seed 0

Same simulated grids for both encodings (error model as in sweep_simulated.py:
misread e, missed e/4, hallucinated e/20). The two encodings are run back to
back on each grid, so slow drift of the machine affects both equally. Only the
MaxSAT solve is timed (check_unique=False), since uniqueness checks use a
fixed pairwise CNF regardless of this setting. Writes outputs/bench_encoding.csv.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sudoku_ns.eval.simulate import simulate_prob_grid
from sudoku_ns.reasoning.encoding import AMO_ENCODINGS
from sudoku_ns.reasoning.generator import generate
from sudoku_ns.reasoning.solver import solve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    ap.add_argument("--out", type=Path, default=Path("outputs/bench_encoding.csv"))
    args = ap.parse_args()

    puzzles = [generate(args.seed + i)[0] for i in tqdm(range(args.n), desc="puzzles")]
    rows = []
    for e in args.rates:
        for i, givens in enumerate(tqdm(puzzles, desc=f"e={e}")):
            prob, _ = simulate_prob_grid(
                givens, e, e / 4, e / 20, seed=args.seed * 1000 + i + int(e * 1e4)
            )
            costs = set()
            for enc in AMO_ENCODINGS:
                res = solve(prob, check_unique=False, amo_encoding=enc)
                costs.add(res.cost)
                rows.append({"rate": e, "puzzle": i, "encoding": enc, "time_s": res.solve_time_s})
            assert len(costs) == 1, f"encodings disagree on the optimum: {costs}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {args.out}  (n = {args.n} grids per rate, times in ms)\n")
    print(f"{'rate':>6} {'encoding':<11} {'median':>8} {'p95':>8} {'max':>8}")
    for e in args.rates:
        for enc in AMO_ENCODINGS:
            t = np.array([r["time_s"] for r in rows if r["rate"] == e and r["encoding"] == enc])
            t *= 1000
            print(
                f"{e:>6.3f} {enc:<11} {np.median(t):>8.1f} "
                f"{np.percentile(t, 95):>8.1f} {t.max():>8.1f}"
            )


if __name__ == "__main__":
    main()
