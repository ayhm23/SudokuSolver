"""Baselines for the weighted MaxSAT solver.

argmax_then_sat and threshold_then_sat are what a "pipeline" system does:
perception makes hard decisions, reasoning only consumes them.

unweighted_maxsat is an ablation, not a pipeline: it lets the rules overrule
the CNN like the weighted solver does, but ignores HOW confident the CNN was.
Comparing it with the weighted solver isolates the value of the probabilities.
"""

from __future__ import annotations

import time

import numpy as np
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

from sudoku_ns.interfaces import Override, ProbGrid, SolveResult, SolveStatus, validate_prob_grid
from sudoku_ns.reasoning.encoding import decode_model, reading_clauses, sudoku_clauses, y_var
from sudoku_ns.reasoning.explain import find_overrides
from sudoku_ns.reasoning.solver import is_unique, sat_solve


def _solve_givens(
    prob: ProbGrid, givens: np.ndarray, overrides: list[Override], check_unique: bool
):
    t0 = time.perf_counter()
    solution = sat_solve(givens)
    elapsed = time.perf_counter() - t0
    if solution is None:
        return SolveResult(
            SolveStatus.UNSAT,
            read_givens=givens,
            overrides=overrides,
            solve_time_s=elapsed,
            message="reasoning: the read givens break a Sudoku rule or have no solution",
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


def unweighted_maxsat(prob: ProbGrid, check_unique: bool = True) -> SolveResult:
    """Keep as many CNN argmax digits as possible; every digit counts the same.

    Same hard clauses as the weighted solver (encoding.py). The soft part is
    one clause y(r, c, argmax) of weight 1 per cell whose argmax is a digit,
    and no soft clauses for cells read as empty. So the optimum is "drop the
    fewest read digits so that a valid Sudoku remains", a maximum consistent
    subset of the argmax, blind to the 0.99 vs 0.51 difference.

    Two extra hard clauses make read_givens well defined without changing the
    optimum: a cell read as empty stays empty, and a digit cell is either kept
    or read as empty (never as another digit). Both are safe because reading
    a cell as empty never creates a conflict, so with equal weights neither a
    hallucinated digit nor a substituted one can lower the cost; without these
    clauses the solver could fill them in arbitrarily at zero cost.

    cost = number of argmax digits dropped.
    """
    t0 = time.perf_counter()
    validate_prob_grid(prob)
    argmax = prob.argmax(axis=2)
    wcnf = WCNF()
    for cl in sudoku_clauses() + reading_clauses():
        wcnf.append(cl)
    for r in range(9):
        for c in range(9):
            a = int(argmax[r, c])
            if a == 0:
                wcnf.append([y_var(r, c, 0)])
            else:
                wcnf.append([y_var(r, c, a), y_var(r, c, 0)])
                wcnf.append([y_var(r, c, a)], weight=1)
    # Plain RC2: with all weights equal, stratification has nothing to stratify.
    with RC2(wcnf) as rc2:
        model = rc2.compute()
        cost = rc2.cost
    elapsed = time.perf_counter() - t0
    if model is None:  # pragma: no cover - dropping every digit is always consistent
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
