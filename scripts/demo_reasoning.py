"""Show MaxSAT fixing a corrupted CNN reading, next to the argmax baseline.

python scripts/demo_reasoning.py --seed 0 --no-color
"""

import argparse

import numpy as np

from sudoku_ns.eval.simulate import simulate_prob_grid
from sudoku_ns.reasoning.baselines import argmax_then_sat
from sudoku_ns.reasoning.explain import describe_result, format_grid
from sudoku_ns.reasoning.generator import generate
from sudoku_ns.reasoning.solver import solve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--error-rate", type=float, default=0.1)
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--require-unique", action="store_true", help="solve(..., require_unique=True)")
    args = ap.parse_args()
    color = not args.no_color

    givens, solution = generate(args.seed)
    prob, record = simulate_prob_grid(
        givens,
        error_rate=args.error_rate,
        empty_miss_rate=0.05,
        hallucination_rate=0.02,
        seed=args.seed,
    )
    argmax = prob.argmax(axis=2).astype(np.int8)
    corrupted = {(r, c) for r, c, *_ in record.all_cells()}

    print("True givens:")
    print(format_grid(givens, color=color))
    print(f"\nCNN argmax (simulated, corrupted cells marked): {record.summary()}")
    print(format_grid(argmax, corrupted, color=color))

    base = argmax_then_sat(prob)
    print(f"\nBaseline argmax -> SAT: {base.status.value}")

    res = solve(prob, require_unique=args.require_unique)
    changed = {(o.row, o.col) for o in res.overrides}
    label = "MaxSAT + require_unique" if args.require_unique else "MaxSAT"
    print(f"\n{label} reading (overridden cells marked):")
    print(format_grid(res.read_givens, changed, color=color))
    print(describe_result(res))
    print(f"\n{label} solution:")
    print(format_grid(res.solution, color=color))
    print(f"\nSolution correct: {bool(np.array_equal(res.solution, solution))}")


if __name__ == "__main__":
    main()
