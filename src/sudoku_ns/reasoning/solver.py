"""MaxSAT solver: turn a ProbGrid into a solved grid plus an explanation.

Two solvers are used on purpose:
- RC2 (core-guided MaxSAT) for the weighted problem, because it is exact: the
  answer is provably the minimum-cost reading, which is what lets us call it a
  MAP estimate rather than "a good guess". We use its stratified variant with
  adapt/exhaust/minz: stratification solves the high-weight (confident) soft
  clauses first, and the three options shrink and reuse conflict cores. All
  of them are speed-ups that keep the result exactly optimal.
- A plain SAT solver (Glucose 4) for exact givens and uniqueness checks. Those
  are decision problems with no weights, and plain SAT is much faster.
"""

from __future__ import annotations

import time

import numpy as np
from pysat.examples.rc2 import RC2Stratified
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
    y_var,
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


MAX_UNIQUE_ROUNDS = 50


def _block_reading(read_givens: np.ndarray) -> list[int]:
    """Hard clause 'at least one cell is read differently from read_givens'."""
    return [-y_var(r, c, int(read_givens[r, c])) for r in range(9) for c in range(9)]


def solve(
    prob: ProbGrid,
    check_unique: bool = True,
    amo_encoding: str = "pairwise",
    shift: bool = True,
    require_unique: bool = False,
) -> SolveResult:
    """Most probable consistent reading of the image, and its solution.

    cost is the MaxSAT objective; with shift=True (default) 0 means "the CNN
    argmax was kept everywhere" (see encoding.py, "Per-cell shift").

    require_unique: also use the fact that printed puzzles have exactly one
    solution. The MAP reading can be cheaper than the truth precisely because
    it drops or changes a given and leaves the puzzle ambiguous (in the Week 1
    simulated sweep, 14 of 15 MaxSAT failures at 5% error were of this kind).
    Uniqueness ("no second solution exists") is not expressible as a plain
    clause, so we add it lazily: solve, check uniqueness with SAT, and if the
    reading is ambiguous add a hard clause forbidding exactly that reading
    and re-solve. Each round
    returns the next-cheapest reading, so the first unique one is the MAP
    reading under the prior "uniform over readings with a unique solution".
    Gives up after MAX_UNIQUE_ROUNDS and returns the last (ambiguous) reading
    with unique=False.

    solve_time_s covers building the formula and the MaxSAT solve(s), plus the
    uniqueness checks only when require_unique=True (they are part of the
    method then), so it is comparable with the baselines.
    """
    t0 = time.perf_counter()
    try:
        validate_prob_grid(prob)
        wcnf = build_wcnf(prob, amo_encoding, shift)
    except ValueError:
        # Invalid ProbGrid (wrong shape, NaN, rows not summing to 1). Callers
        # that want the message should call validate_prob_grid themselves.
        return SolveResult(SolveStatus.ERROR, solve_time_s=time.perf_counter() - t0)

    unique = None
    for _ in range(MAX_UNIQUE_ROUNDS if require_unique else 1):
        # A fresh solver per round: pysat's RC2Stratified returned "no model"
        # after hard clauses were added between compute() calls (seen on 12 of
        # 30 puzzles; plain RC2 handled it but is ~100x slower on noisy grids).
        # Rebuilding costs ~0.1 s per round and is exact.
        with RC2Stratified(wcnf, adapt=True, exhaust=True, minz=True) as rc2:
            model = rc2.compute()
            cost = rc2.cost
        if model is None:
            break
        solution, read_givens = decode_model(model)
        if not require_unique:
            break
        unique = is_unique(read_givens)
        if unique:
            break
        wcnf.append(_block_reading(read_givens))
    elapsed = time.perf_counter() - t0
    if model is None:
        # Without require_unique this cannot happen for a valid ProbGrid
        # (reading every cell as empty is always consistent).
        return SolveResult(SolveStatus.UNSAT, solve_time_s=elapsed)

    if unique is None and check_unique:
        unique = is_unique(read_givens)
    return SolveResult(
        status=SolveStatus.SOLVED,
        solution=solution,
        read_givens=read_givens,
        overrides=find_overrides(prob, read_givens),
        unique=unique,
        cost=int(cost),
        solve_time_s=elapsed,
    )
