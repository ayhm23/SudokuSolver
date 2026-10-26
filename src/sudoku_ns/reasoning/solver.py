"""MaxSAT solver: turn a ProbGrid into a solved grid plus an explanation.

Two solvers are used on purpose:
- RC2 (core-guided MaxSAT) for the weighted problem, because it is exact: the
  answer is provably the minimum-cost reading, which is what lets us call it a
  MAP estimate rather than "a good guess".
- A plain SAT solver (Glucose 4) for exact givens and uniqueness checks. Those
  are decision problems with no weights, and plain SAT is much faster.
"""

from __future__ import annotations

import time

import numpy as np
from pysat.examples.rc2 import RC2
from pysat.solvers import Solver

from sudoku_ns.interfaces import (
    Override,
    ProbGrid,
    SolveResult,
    SolveStatus,
    validate_prob_grid,
)
from sudoku_ns.reasoning.encoding import (
    build_wcnf,
    decode_model,
    decode_solution,
    sudoku_cnf,
    x_var,
)

SAT_SOLVER = "g4"  # Glucose 4


def sat_solve(givens: np.ndarray) -> np.ndarray | None:
    """Exact SAT solve with givens fixed. Returns the solution or None if UNSAT."""
    with Solver(name=SAT_SOLVER, bootstrap_with=sudoku_cnf(givens)) as s:
        if not s.solve():
            return None
        return decode_solution(s.get_model())


def count_solutions(givens: np.ndarray, limit: int = 2) -> int:
    """Count solutions up to `limit` by repeatedly blocking the last one found.

    Blocking clause: OR over free cells of NOT x(r, c, solution[r, c]), i.e.
    "at least one free cell must differ". Given cells can be left out because
    they are fixed anyway. With limit=2 this is the standard uniqueness test.
    """
    free = list(zip(*np.nonzero(givens == 0), strict=True))
    n = 0
    with Solver(name=SAT_SOLVER, bootstrap_with=sudoku_cnf(givens)) as s:
        while n < limit and s.solve():
            n += 1
            sol = decode_solution(s.get_model())
            if not free:
                break  # a full grid has at most one solution
            s.add_clause([-x_var(r, c, int(sol[r, c])) for r, c in free])
    return n


def is_unique(givens: np.ndarray) -> bool:
    return count_solutions(givens, limit=2) == 1


def _clash(grid: np.ndarray, r: int, c: int, d: int) -> str | None:
    """Name the first row/column/box where digit d at (r, c) repeats a given in grid."""
    for cc in range(9):
        if cc != c and grid[r, cc] == d:
            return f"{d} clashes with {d} at ({r}, {cc}) in row {r}"
    for rr in range(9):
        if rr != r and grid[rr, c] == d:
            return f"{d} clashes with {d} at ({rr}, {c}) in column {c}"
    br, bc = 3 * (r // 3), 3 * (c // 3)
    for rr in range(br, br + 3):
        for cc in range(bc, bc + 3):
            if (rr, cc) != (r, c) and grid[rr, cc] == d:
                return f"{d} clashes with {d} at ({rr}, {cc}) in box {(r // 3) * 3 + c // 3}"
    return None


def find_overrides(prob: ProbGrid, read_givens: np.ndarray) -> list[Override]:
    """Cells where the solver's reading differs from the CNN argmax.

    The clash is checked against the other FINAL read givens: those are the
    digits the solver kept, so a clash with one of them is the concrete reason
    the argmax could not stay. If there is no direct clash the argmax was still
    inconsistent (or more expensive) once the whole grid is considered, which we
    report as "lower total cost".
    """
    argmax = prob.argmax(axis=2)
    out = []
    for r, c in zip(*np.nonzero(argmax != read_givens), strict=True):
        a, f = int(argmax[r, c]), int(read_givens[r, c])
        reason = (_clash(read_givens, r, c, a) if a >= 1 else None) or "lower total cost"
        out.append(
            Override(
                row=int(r),
                col=int(c),
                cnn_label=a,
                cnn_prob=float(prob[r, c, a]),
                final_label=f,
                final_prob=float(prob[r, c, f]),
                reason=reason,
            )
        )
    return out


def solve(prob: ProbGrid, check_unique: bool = True, amo_encoding: str = "pairwise") -> SolveResult:
    """Most probable consistent reading of the image, and its solution.

    solve_time_s covers building the formula and the MaxSAT solve only, not the
    optional uniqueness check, so it is comparable with the baselines.
    """
    t0 = time.perf_counter()
    try:
        validate_prob_grid(prob)
        wcnf = build_wcnf(prob, amo_encoding)
        with RC2(wcnf) as rc2:
            model = rc2.compute()
            cost = rc2.cost
    except ValueError:
        # Invalid ProbGrid (wrong shape, NaN, rows not summing to 1). Callers
        # that want the message should call validate_prob_grid themselves.
        return SolveResult(SolveStatus.ERROR, solve_time_s=time.perf_counter() - t0)
    elapsed = time.perf_counter() - t0
    if model is None:
        # Cannot happen with a valid ProbGrid (reading all cells as empty is
        # always consistent), but kept so a future encoding bug is reported.
        return SolveResult(SolveStatus.UNSAT, solve_time_s=elapsed)

    solution, read_givens = decode_model(model)
    return SolveResult(
        status=SolveStatus.SOLVED,
        solution=solution,
        read_givens=read_givens,
        overrides=find_overrides(prob, read_givens),
        unique=is_unique(read_givens) if check_unique else None,
        cost=int(cost),
        solve_time_s=elapsed,
    )
