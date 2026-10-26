"""Weighted MaxSAT encoding of "read the image and solve the Sudoku".

Two families of Boolean variables
---------------------------------
x(r, c, d)  d in 1..9 : the SOLUTION has digit d at cell (r, c).
y(r, c, k)  k in 0..9 : the IMAGE shows class k at (r, c) (k = 0 means empty).

Keeping "what is printed" (y) separate from "what the answer is" (x) is the
central design choice. A CNN only ever tells us about y, and a printed digit
constrains the answer (y = d forces x = d), but an empty cell constrains
nothing. With a single set of variables we could not express "this cell is
empty in the image but 4 in the solution", which is the normal case for ~50
of the 81 cells.

Hard clauses (must hold in every model)
--------------------------------------
1. Each cell holds exactly one solution digit:      exactly-one_d x(r,c,d)
2. Each digit once per row / column / 3x3 box:      exactly-one over the 9 cells
3. Each cell shows exactly one visual class:        exactly-one_k y(r,c,k)
4. A printed digit must agree with the solution:    y(r,c,d) -> x(r,c,d), d >= 1

(1) + (2) is the standard Sudoku encoding: any assignment satisfying them is a
valid completed grid. (4) is the only link between perception and rules.

Soft clauses (may be violated at a price)
----------------------------------------
For every cell and every class k: the unit clause  NOT y(r,c,k)  with weight

    w(r,c,k) = round(SCALE * -log(max(p(r,c,k), EPS)))

Violating "NOT y(r,c,k)" means asserting the cell shows k, which costs
-log p(r,c,k). Because (3) forces exactly one y per cell to be true, every
model violates exactly one soft clause per cell, so the total cost of a model
is

    cost = SCALE * sum_cells -log p(cell, chosen class)
         = -SCALE * log prod_cells p(cell, chosen class).

Why the optimum is a MAP reading
--------------------------------
Treat the CNN outputs as independent per-cell likelihoods of each visual class
(the CNN looks at one crop at a time, so this is the model it implements).
The probability of a full reading K = (k_cell) of the image is then
prod_cells p(cell, k_cell). Minimising the cost above is exactly maximising
that product, i.e. choosing the most probable reading. The hard clauses
restrict the search to readings that can be extended to at least one valid
Sudoku solution (via the x variables). So the MaxSAT optimum is

    argmax_K  P(K | image)   subject to   K is consistent with some solution,

which is the maximum a posteriori (MAP) interpretation under a prior that is
uniform over consistent readings and zero elsewhere. A misread digit that
clashes with a row is therefore "corrected" only when the cheapest consistent
alternative costs less than all other ways of fixing the clash; the solver
never has a hand-written rule for corrections.

Note the optimum is a reading consistent with SOME solution, not necessarily
a unique one; uniqueness is checked separately in solver.py.

Why these constants
-------------------
SCALE = 1000: RC2 needs integer weights. Rounding changes each cell's cost by
at most 0.5/1000 nats, so the total error over 81 cells is at most ~0.04
nats; ties closer than that are broken arbitrarily, which is acceptable.
EPS = 1e-6: p = 0 would give infinite weight, turning a soft belief into a
hard constraint. Clipping caps the cost of any single override at
-log(1e-6) ~ 13.8 nats, so the solver can still overrule a CNN that is
confidently wrong, but only if nothing cheaper exists.
Zero-weight clauses (p ~ 1 after rounding) are skipped: they cost nothing to
violate, so they only make the formula bigger.

At-most-one encoding
--------------------
"Exactly one" = one at-least-one clause + an at-most-one (AMO) constraint.
Pairwise AMO adds n(n-1)/2 binary clauses and no new variables (36 per group
of 9), which is small for Sudoku (~12k clauses in total) and propagates
well. The sequential counter (Sinz 2005) uses O(n) clauses plus auxiliary
variables; it is available behind ``amo_encoding`` for comparison.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from functools import lru_cache

import numpy as np
from pysat.card import CardEnc, EncType
from pysat.formula import WCNF, IDPool

from sudoku_ns.interfaces import ProbGrid, validate_prob_grid

SCALE = 1000
EPS = 1e-6

N_X = 9 * 9 * 9  # 729 solution variables, ids 1..729
N_Y = 9 * 9 * 10  # 810 reading variables, ids 730..1539
MAX_VAR = N_X + N_Y

AMO_ENCODINGS = ("pairwise", "seqcounter")


# ---------------------------------------------------------------- numbering
# DIMACS variable ids must be positive integers. Fixed arithmetic numbering
# (instead of an IDPool) means any id can be decoded without carrying state
# around, which makes debugging models and writing tests much easier.


def x_var(r: int, c: int, d: int) -> int:
    """Variable id for 'solution has digit d (1..9) at (r, c)'."""
    if not (0 <= r < 9 and 0 <= c < 9 and 1 <= d <= 9):
        raise ValueError(f"x_var: bad index r={r}, c={c}, d={d}")
    return int(1 + (9 * r + c) * 9 + (d - 1))  # int(): numpy ints are rejected by pysat


def y_var(r: int, c: int, k: int) -> int:
    """Variable id for 'image shows class k (0..9, 0 = empty) at (r, c)'."""
    if not (0 <= r < 9 and 0 <= c < 9 and 0 <= k <= 9):
        raise ValueError(f"y_var: bad index r={r}, c={c}, k={k}")
    return int(N_X + 1 + (9 * r + c) * 10 + k)


def decode_var(v: int) -> tuple[str, int, int, int]:
    """Inverse of x_var / y_var: returns ('x', r, c, d) or ('y', r, c, k)."""
    if 1 <= v <= N_X:
        i = v - 1
        cell, d0 = divmod(i, 9)
        return ("x", cell // 9, cell % 9, d0 + 1)
    if N_X < v <= MAX_VAR:
        i = v - N_X - 1
        cell, k = divmod(i, 10)
        return ("y", cell // 9, cell % 9, k)
    raise ValueError(f"decode_var: {v} is not an x or y variable (1..{MAX_VAR})")


# ------------------------------------------------------------ hard clauses


def _exactly_one(lits: list[int], amo_encoding: str, pool: IDPool) -> list[list[int]]:
    clauses = [list(lits)]  # at least one
    if amo_encoding == "pairwise":
        clauses += [[-a, -b] for i, a in enumerate(lits) for b in lits[i + 1 :]]
    elif amo_encoding == "seqcounter":
        enc = CardEnc.atmost(lits, bound=1, vpool=pool, encoding=EncType.seqcounter)
        clauses += enc.clauses
    else:
        raise ValueError(f"amo_encoding must be one of {AMO_ENCODINGS}, got {amo_encoding!r}")
    return clauses


def sudoku_units() -> list[list[tuple[int, int]]]:
    """The 27 rows, columns and 3x3 boxes as lists of (r, c)."""
    rows = [[(r, c) for c in range(9)] for r in range(9)]
    cols = [[(r, c) for r in range(9)] for c in range(9)]
    boxes = [
        [(br + i, bc + j) for i in range(3) for j in range(3)]
        for br in range(0, 9, 3)
        for bc in range(0, 9, 3)
    ]
    return rows + cols + boxes


def _new_pool() -> IDPool:
    # Auxiliary variables (seqcounter only) start above our fixed ids.
    return IDPool(start_from=MAX_VAR + 1)


def sudoku_clauses(amo_encoding: str = "pairwise", pool: IDPool | None = None) -> list[list[int]]:
    """Hard clauses (1) and (2): x describes a valid completed Sudoku."""
    pool = pool or _new_pool()
    clauses: list[list[int]] = []
    for r in range(9):
        for c in range(9):
            clauses += _exactly_one([x_var(r, c, d) for d in range(1, 10)], amo_encoding, pool)
    for unit in sudoku_units():
        for d in range(1, 10):
            clauses += _exactly_one([x_var(r, c, d) for r, c in unit], amo_encoding, pool)
    return clauses


def reading_clauses(amo_encoding: str = "pairwise", pool: IDPool | None = None) -> list[list[int]]:
    """Hard clauses (3) and (4): one visual class per cell, printed digits bind x."""
    pool = pool or _new_pool()
    clauses: list[list[int]] = []
    for r in range(9):
        for c in range(9):
            clauses += _exactly_one([y_var(r, c, k) for k in range(10)], amo_encoding, pool)
            for d in range(1, 10):
                clauses.append([-y_var(r, c, d), x_var(r, c, d)])
    return clauses


def givens_units(givens: np.ndarray) -> list[list[int]]:
    """Unit clauses fixing x for every nonzero given (used by exact SAT solves)."""
    return [[x_var(r, c, int(givens[r, c]))] for r, c in zip(*np.nonzero(givens), strict=True)]


@lru_cache(maxsize=1)
def _pairwise_rules() -> tuple[tuple[int, ...], ...]:
    # The rule clauses never change, and the puzzle generator runs ~80
    # uniqueness checks per puzzle, so building them once matters (~10x faster).
    return tuple(tuple(cl) for cl in sudoku_clauses("pairwise"))


def sudoku_cnf(givens: np.ndarray | None = None) -> list:
    """Plain CNF clause list: Sudoku rules plus optional fixed givens. No perception."""
    clauses = list(_pairwise_rules())
    if givens is not None:
        clauses += givens_units(givens)
    return clauses


# ------------------------------------------------------------ soft clauses


def weight(p: float) -> int:
    """Integer cost of asserting a class the CNN gave probability p."""
    return round(SCALE * -math.log(max(float(p), EPS)))


def soft_clauses(prob: ProbGrid) -> Iterable[tuple[list[int], int]]:
    """(clause, weight) pairs: NOT y(r,c,k) weighted by -log p(r,c,k)."""
    for r in range(9):
        for c in range(9):
            for k in range(10):
                w = weight(prob[r, c, k])
                if w > 0:
                    yield [-y_var(r, c, k)], w


def build_wcnf(prob: ProbGrid, amo_encoding: str = "pairwise") -> WCNF:
    """Full weighted formula: hard Sudoku + reading clauses, soft CNN beliefs."""
    validate_prob_grid(prob)
    pool = _new_pool()
    wcnf = WCNF()
    for cl in sudoku_clauses(amo_encoding, pool) + reading_clauses(amo_encoding, pool):
        wcnf.append(cl)
    for cl, w in soft_clauses(prob):
        wcnf.append(cl, weight=w)
    return wcnf


# ------------------------------------------------------------- decoding


def decode_model(model: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Turn a SAT model into (solution, read_givens) int8 grids."""
    solution = np.zeros((9, 9), np.int8)
    reading = np.full((9, 9), -1, np.int8)
    for lit in model:
        if lit <= 0 or lit > MAX_VAR:
            continue
        kind, r, c, v = decode_var(lit)
        if kind == "x":
            solution[r, c] = v
        else:
            reading[r, c] = v
    return solution, reading


def decode_solution(model: Iterable[int]) -> np.ndarray:
    return decode_model(model)[0]
