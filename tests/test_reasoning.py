"""Reasoning tests on hand-written probability grids built from generated puzzles."""

import numpy as np
import pytest

from sudoku_ns.interfaces import SolveStatus
from sudoku_ns.reasoning.baselines import argmax_then_sat, threshold_then_sat
from sudoku_ns.reasoning.encoding import (
    MAX_VAR,
    build_wcnf,
    decode_var,
    weight,
    x_var,
    y_var,
)
from sudoku_ns.reasoning.generator import generate, redundant_givens
from sudoku_ns.reasoning.solver import count_solutions, solve


@pytest.fixture(scope="module")
def puzzle():
    return generate(seed=1, target_givens=30)


def one_hot(grid, conf=0.97):
    """Each cell puts `conf` on its label and spreads the rest evenly."""
    p = np.full((9, 9, 10), (1 - conf) / 9, np.float32)
    for r in range(9):
        for c in range(9):
            p[r, c, grid[r, c]] = conf
    return p


def set_cell(p, r, c, dist):
    """Overwrite one cell: dist maps class -> prob, the remainder is spread evenly."""
    rest = (1 - sum(dist.values())) / (10 - len(dist))
    p[r, c, :] = rest
    for k, v in dist.items():
        p[r, c, k] = v


def test_clean_grid_solves_without_overrides(puzzle):
    givens, solution = puzzle
    res = solve(one_hot(givens))
    assert res.status is SolveStatus.SOLVED
    np.testing.assert_array_equal(res.solution, solution)
    np.testing.assert_array_equal(res.read_givens, givens)
    assert res.overrides == []
    assert res.unique is True


def test_misread_digit_is_corrected(puzzle):
    givens, solution = puzzle
    # Find a given (r, c) and a wrong digit w that already appears as a given in row r,
    # so reading w at (r, c) clashes with that row.
    r, c, w = next(
        (r, c, int(givens[r, cc]))
        for r, c in zip(*np.nonzero(givens), strict=True)
        for cc in range(9)
        if cc != c and givens[r, cc] != 0
    )
    t = int(givens[r, c])
    p = one_hot(givens)
    set_cell(p, r, c, {w: 0.6, t: 0.3})

    res = solve(p)
    assert res.status is SolveStatus.SOLVED
    np.testing.assert_array_equal(res.solution, solution)
    assert res.read_givens[r, c] == t
    [o] = res.overrides
    assert (o.row, o.col, o.cnn_label, o.final_label) == (r, c, w, t)
    assert f"row {r}" in o.reason

    assert argmax_then_sat(p).status is SolveStatus.UNSAT


def test_missed_digit_still_solves(puzzle):
    givens, solution = puzzle
    redundant = redundant_givens(givens)
    assert redundant, "generator should produce non-minimal puzzles"
    r, c = redundant[0]
    p = one_hot(givens)
    set_cell(p, r, c, {0: 0.55, int(givens[r, c]): 0.35})

    res = solve(p)
    assert res.status is SolveStatus.SOLVED
    np.testing.assert_array_equal(res.solution, solution)
    assert res.unique is True


def test_hallucinated_digit_is_removed(puzzle):
    givens, solution = puzzle
    # An empty cell (r, c) and a digit h that is already a given in row r.
    r, c, h = next(
        (r, c, int(givens[r, cc]))
        for r, c in zip(*np.nonzero(givens == 0), strict=True)
        for cc in range(9)
        if givens[r, cc] != 0
    )
    p = one_hot(givens)
    set_cell(p, r, c, {h: 0.6, 0: 0.35})

    res = solve(p)
    assert res.status is SolveStatus.SOLVED
    np.testing.assert_array_equal(res.solution, solution)
    assert res.read_givens[r, c] == 0
    assert [(o.row, o.col, o.final_label) for o in res.overrides] == [(r, c, 0)]
    assert argmax_then_sat(p).status is SolveStatus.UNSAT


def test_variable_numbering_round_trips():
    seen = set()
    for r in range(9):
        for c in range(9):
            for d in range(1, 10):
                v = x_var(r, c, d)
                assert decode_var(v) == ("x", r, c, d)
                seen.add(v)
            for k in range(10):
                v = y_var(r, c, k)
                assert decode_var(v) == ("y", r, c, k)
                seen.add(v)
    assert seen == set(range(1, MAX_VAR + 1))  # dense and collision-free


def test_weight_is_monotone_in_confidence():
    ps = [1e-9, 1e-6, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0]
    ws = [weight(p) for p in ps]
    # weight(p) is the cost of ASSERTING class k (violating NOT y), so it must
    # fall as p rises. The "cost to deny" view is tested separately below.
    assert ws == sorted(ws, reverse=True)
    assert weight(1e-9) == weight(1e-6)  # EPS clipping
    assert weight(1.0) == 0


def test_denying_confident_reading_costs_more():
    # Cost to deny a class k = cost of the best alternative. Higher p(k) leaves
    # less mass for alternatives, so denying it costs more.
    def deny_cost(pk):
        return weight((1 - pk) / 9)

    costs = [deny_cost(pk) for pk in (0.1, 0.5, 0.9, 0.99)]
    assert costs == sorted(costs)


def test_seqcounter_matches_pairwise(puzzle):
    givens, solution = puzzle
    p = one_hot(givens)
    a = solve(p, check_unique=False, amo_encoding="pairwise")
    b = solve(p, check_unique=False, amo_encoding="seqcounter")
    assert a.cost == b.cost
    np.testing.assert_array_equal(b.solution, solution)
    assert len(build_wcnf(p, "seqcounter").hard) < len(build_wcnf(p, "pairwise").hard)


def test_threshold_baseline_drops_low_confidence(puzzle):
    givens, solution = puzzle
    p = one_hot(givens)
    r, c = map(int, np.argwhere(givens)[0])
    set_cell(p, r, c, {int(givens[r, c]): 0.5})
    res = threshold_then_sat(p, tau=0.8)
    assert res.read_givens[r, c] == 0
    assert len(res.overrides) == 1


def test_count_solutions():
    assert count_solutions(np.zeros((9, 9), np.int8), limit=3) == 3
    _, solution = generate(seed=2)
    assert count_solutions(solution) == 1
    bad = solution.copy()
    bad[0, 0], bad[0, 1] = bad[0, 1], bad[0, 1]  # duplicate in row 0
    assert count_solutions(bad) == 0


def test_generator_is_seeded():
    a, _ = generate(seed=5)
    b, _ = generate(seed=5)
    np.testing.assert_array_equal(a, b)
    assert (a > 0).sum() == 30


def test_per_cell_shift_keeps_the_optimum(puzzle):
    # Same misread as test 2: shifted and textbook weights must pick the same
    # reading, and their costs differ by exactly the constant sum of per-cell minima.
    givens, _ = puzzle
    p = one_hot(givens)
    r, c = map(int, np.argwhere(givens)[0])
    w = next(d for d in range(1, 10) if d != givens[r, c])
    set_cell(p, r, c, {w: 0.5, int(givens[r, c]): 0.4})
    a = solve(p, check_unique=False, shift=True)
    b = solve(p, check_unique=False, shift=False)
    np.testing.assert_array_equal(a.read_givens, b.read_givens)
    const = sum(min(weight(p[i, j, k]) for k in range(10)) for i in range(9) for j in range(9))
    assert b.cost - a.cost == const


def test_require_unique_recovers_essential_missed_digit(puzzle):
    # Miss a given that is ESSENTIAL (its removal makes the puzzle ambiguous).
    # Plain MAP happily reads it as empty (cheaper) and returns some solution
    # of the ambiguous puzzle; require_unique pays to read the digit back.
    givens, solution = puzzle
    redundant = set(redundant_givens(givens))
    r, c = next(
        (int(r), int(c))
        for r, c in zip(*np.nonzero(givens), strict=True)
        if (r, c) not in redundant
    )
    p = one_hot(givens)
    set_cell(p, r, c, {0: 0.55, int(givens[r, c]): 0.35})

    plain = solve(p)
    assert plain.read_givens[r, c] == 0 and plain.unique is False

    res = solve(p, require_unique=True)
    assert res.unique is True and res.read_givens[r, c] == givens[r, c]
    np.testing.assert_array_equal(res.solution, solution)
