"""Metrics comparing a SolveResult against ground truth.

"Before reasoning" always means the CNN argmax; "after" means the solver's
read_givens. Comparing the two isolates what reasoning added, independent of
how good the CNN is.
"""

from __future__ import annotations

import numpy as np

from sudoku_ns.interfaces import ProbGrid, SolveResult, SolveStatus


def evaluate(
    result: SolveResult, prob: ProbGrid, true_givens: np.ndarray, true_solution: np.ndarray
) -> dict:
    """Per-puzzle metrics. Cell counts are over all 81 cells (empties included)."""
    argmax = prob.argmax(axis=2)
    before_ok = argmax == true_givens
    solved = result.status is SolveStatus.SOLVED
    read = result.read_givens if result.read_givens is not None else argmax
    after_ok = read == true_givens
    return {
        "status": result.status.value,
        "solved_correct": bool(solved and np.array_equal(result.solution, true_solution)),
        "unsat": result.status is SolveStatus.UNSAT,
        "unique": result.unique,
        "cell_acc_before": float(before_ok.mean()),
        "cell_acc_after": float(after_ok.mean()),
        "n_misreads": int((~before_ok).sum()),
        "misreads_corrected": int((~before_ok & after_ok).sum()),
        "correct_overridden": int((before_ok & ~after_ok).sum()),
        "solve_time_s": result.solve_time_s,
    }


def aggregate(rows: list[dict]) -> dict:
    """Mean of each per-puzzle metric (rates for booleans)."""
    if not rows:
        return {}
    out = {"n": len(rows)}
    for key in ("solved_correct", "unsat", "cell_acc_before", "cell_acc_after", "solve_time_s"):
        out[key] = float(np.mean([r[key] for r in rows]))
    for key in ("n_misreads", "misreads_corrected", "correct_overridden"):
        out[key] = int(sum(r[key] for r in rows))
    return out
