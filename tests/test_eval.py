import numpy as np

from sudoku_ns.eval.metrics import aggregate, evaluate
from sudoku_ns.eval.simulate import simulate_prob_grid
from sudoku_ns.reasoning.baselines import argmax_then_sat
from sudoku_ns.reasoning.generator import generate
from sudoku_ns.reasoning.solver import solve

GIVENS, SOLUTION = generate(seed=3)


def test_zero_error_is_correct_but_not_one_hot():
    prob, rec = simulate_prob_grid(GIVENS, error_rate=0, seed=0)
    assert rec.n_corrupted() == 0
    np.testing.assert_array_equal(prob.argmax(2), GIVENS)
    assert prob.max() < 1.0  # realistic: never fully certain
    assert len(np.unique(prob.max(2).round(3))) > 20  # confidence varies per cell


def test_record_matches_corruptions():
    prob, rec = simulate_prob_grid(
        GIVENS, error_rate=0.3, empty_miss_rate=0.2, hallucination_rate=0.1, seed=1
    )
    argmax = prob.argmax(2)
    for r, c, true, read in rec.misread:
        assert GIVENS[r, c] == true and argmax[r, c] == read != true
        assert np.argsort(prob[r, c])[-2] == true  # true class is the runner-up
    for r, c, _ in rec.missed:
        assert argmax[r, c] == 0
    for r, c, read in rec.hallucinated:
        assert GIVENS[r, c] == 0 and argmax[r, c] == read
    corrupted = {(r, c) for r, c, *_ in rec.all_cells()}
    assert int((argmax != GIVENS).sum()) == len(corrupted)


def test_seeded():
    a, _ = simulate_prob_grid(GIVENS, error_rate=0.1, seed=7)
    b, _ = simulate_prob_grid(GIVENS, error_rate=0.1, seed=7)
    np.testing.assert_array_equal(a, b)


def test_metrics_on_clean_and_corrupted():
    prob, _ = simulate_prob_grid(GIVENS, error_rate=0, seed=0)
    m = evaluate(solve(prob), prob, GIVENS, SOLUTION)
    assert m["solved_correct"] and m["cell_acc_after"] == 1.0 and m["correct_overridden"] == 0

    prob, rec = simulate_prob_grid(GIVENS, error_rate=0.2, seed=4)
    rows = [evaluate(f(prob), prob, GIVENS, SOLUTION) for f in (solve, argmax_then_sat)]
    assert rows[0]["n_misreads"] == rec.n_corrupted()
    assert rows[1]["misreads_corrected"] == 0  # argmax never changes a reading
    agg = aggregate(rows)
    assert agg["n"] == 2 and 0 <= agg["solved_correct"] <= 1
