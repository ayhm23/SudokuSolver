"""Non-neurosymbolic baselines: trust the CNN, then solve exactly.

These are what a "pipeline" system does: perception makes hard decisions,
reasoning only consumes them. They are the comparison point for the MaxSAT
solver, which lets the rules feed back into what was read.
"""

from __future__ import annotations

import time

import numpy as np

from sudoku_ns.interfaces import Override, ProbGrid, SolveResult, SolveStatus, validate_prob_grid
from sudoku_ns.reasoning.solver import is_unique, sat_solve


def _solve_givens(
    prob: ProbGrid, givens: np.ndarray, overrides: list[Override], check_unique: bool
):
    t0 = time.perf_counter()
    solution = sat_solve(givens)
    elapsed = time.perf_counter() - t0
    if solution is None:
        return SolveResult(
            SolveStatus.UNSAT, read_givens=givens, overrides=overrides, solve_time_s=elapsed
        )
    return SolveResult(
        status=SolveStatus.SOLVED,
        solution=solution,
        read_givens=givens,
        overrides=overrides,
        unique=is_unique(givens) if check_unique else None,
        solve_time_s=elapsed,
    )


def argmax_then_sat(prob: ProbGrid, check_unique: bool = True) -> SolveResult:
    """Take the CNN's top class per cell as the givens, then solve exactly.

    One misread digit that clashes with a row, column or box makes the whole
    puzzle UNSAT: the baseline has no way to doubt its input.
    """
    validate_prob_grid(prob)
    givens = prob.argmax(axis=2).astype(np.int8)
    return _solve_givens(prob, givens, [], check_unique)


def threshold_then_sat(prob: ProbGrid, tau: float, check_unique: bool = True) -> SolveResult:
    """Like argmax, but drop any digit whose confidence is below tau.

    A common heuristic fix: uncertain digits are treated as empty. It removes
    some misreads, but also removes correct givens (possibly making the puzzle
    non-unique), and cannot fix a confident misread or a missed digit.
    """
    validate_prob_grid(prob)
    argmax = prob.argmax(axis=2)
    conf = prob.max(axis=2)
    drop = (argmax >= 1) & (conf < tau)
    givens = np.where(drop, 0, argmax).astype(np.int8)
    overrides = [
        Override(
            row=int(r),
            col=int(c),
            cnn_label=int(argmax[r, c]),
            cnn_prob=float(conf[r, c]),
            final_label=0,
            final_prob=float(prob[r, c, 0]),
            reason=f"confidence {conf[r, c]:.2f} < tau {tau}",
        )
        for r, c in zip(*np.nonzero(drop), strict=True)
    ]
    return _solve_givens(prob, givens, overrides, check_unique)
