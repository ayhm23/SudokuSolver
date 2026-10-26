# Reasoning: how MaxSAT reads and solves the grid

Code: [`src/sudoku_ns/reasoning/`](../src/sudoku_ns/reasoning/). Tests: [`tests/test_reasoning.py`](../tests/test_reasoning.py).

## The problem in plain words

The CNN looks at each of the 81 cells separately and says something like:

> cell (3, 4): 60% it's a 7, 30% it's a 1, 10% spread over the rest.

If we just take the most likely class everywhere ("argmax") and one cell is wrong, the puzzle can become
impossible (two 7s in a row), and an ordinary solver gives up. We want a solver that says instead:

> "Reading a 7 there breaks the row. Reading a 1 costs a bit more confidence, but then everything fits.
> I'll read a 1."

Weighted MaxSAT does exactly that.

## Variables: "what the image shows" vs "what the answer is"

| Variable | Meaning | Count |
|---|---|---|
| `x(r, c, d)` | the **solution** has digit d (1..9) at (r, c) | 729 |
| `y(r, c, k)` | the **image shows** class k at (r, c); k = 0 means empty | 810 |

Why two sets? An empty cell in the image says nothing about the answer, but a printed 5 forces the answer to be 5.
One set of variables cannot express "empty in the picture, 4 in the solution".

Numbering is plain arithmetic (`x_var`, `y_var`, `decode_var` in `encoding.py`), so any variable id can be decoded
back to (r, c, digit) without a lookup table.

## Hard rules (always true)

1. every cell has exactly one solution digit
2. every digit appears exactly once in each row, column and 3x3 box
3. every cell shows exactly one visual class (empty or a digit)
4. if the image shows digit d, the solution has d: `y(r,c,d) → x(r,c,d)`

"Exactly one" = "at least one" (one clause) + "at most one" (pairwise: no two can both be true, 36 clauses for 9
options). `amo_encoding="seqcounter"` swaps in a sequential counter (fewer clauses, extra variables).

## Soft rules (CNN beliefs, can be broken at a price)

For each cell and class k: "the cell does NOT show k", with weight `round(1000 * -log p)`.

- the solver must pick exactly one class per cell (rule 3), so it breaks exactly one soft rule per cell
- breaking it costs `-log p(chosen class)`: cheap if the CNN was confident, expensive if not
- total cost = `-log` of the product of the chosen probabilities

**Minimum cost = maximum probability reading that still allows a valid Sudoku.** That is a MAP (maximum a
posteriori) estimate. Nothing in the code says "fix misreads"; it falls out of the optimisation.

Constants: `SCALE = 1000` because the solver needs integers; `EPS = 1e-6` so a probability of 0 does not become an
unbreakable rule (cost is capped at ~13.8 nats).

## Worked example (test 2 in `test_reasoning.py`)

A given 3 is misread: CNN says 7 (p=0.6), 3 (p=0.3), but row 0 already has a 7.

| Reading of that cell | Cost (×1000 nats) | Allowed? |
|---|---|---|
| 7 | -ln 0.6 = 511 | no: two 7s in the row |
| 3 | -ln 0.3 = 1204 | yes |
| empty | -ln 0.0125 ≈ 4382 | yes, but costlier |

The solver reads 3 and reports an `Override` with reason "7 clashes with 7 at (0, 5) in row 0".
`argmax_then_sat` on the same input returns UNSAT.

## Files

| File | What it does |
|---|---|
| `encoding.py` | variables, hard and soft clauses, `build_wcnf(prob)`, `decode_model` |
| `solver.py` | `solve(prob)`: RC2 MaxSAT, overrides with reasons, uniqueness check; `sat_solve`, `count_solutions` for exact givens |
| `baselines.py` | `argmax_then_sat` (trust CNN fully), `threshold_then_sat` (drop digits below confidence tau) |
| `generator.py` | seeded random solutions (backtracking) and unique puzzles with ~30 givens; `redundant_givens` |
| `explain.py` | text rendering of grids and overrides for the demo |

## Uniqueness check

Solve with the read givens fixed, add a clause "at least one free cell must differ from this solution", solve
again. UNSAT the second time means the solution was unique. (`count_solutions` in `solver.py`.)

## Why puzzles are not minimal

A minimal puzzle loses uniqueness if any single given is removed. Then a CNN that *misses* one digit makes the
puzzle ambiguous no matter how good the solver is. The generator stops removing givens at `target_givens` (default
30), well above the typical minimum (~22-26), so most puzzles have several redundant givens
(`redundant_givens(puzzle)` lists them).

## Speed (measured on this laptop, Week 1)

- one MaxSAT solve: ~0.1 s
- generating one 30-given puzzle: ~0.75 s (about 80 uniqueness checks; the rule clauses are cached)
