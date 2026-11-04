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

## Worked example (test 2 in `test_reasoning.py`, real output)

Puzzle seed 1: cell (0, 2) truly shows 5. The CNN says 9 (p=0.60), 5 (p=0.30), and 0.0125 for each other class.
Row 0 already has a given 9 at (0, 6).

| Reading of that cell | Weight `round(1000 * -ln p)` | Allowed? |
|---|---|---|
| 9 | 511 | no: two 9s in row 0 |
| 5 | 1204 | yes |
| empty / other digits | 4382 | yes, but costlier |

The solver reads 5. With the per-cell shift (below) the reported cost is 1204 - 511 = **693**: the price of
overruling the CNN once. The override printed by `explain.describe_override`:

```
(0, 2): CNN read 9 (p=0.60) -> solver reads 5 (p=0.30); conflicts with given 9 at (0, 6) in row 0
```

`argmax_then_sat` on the same input returns UNSAT. `unweighted_maxsat` (below) drops the 9 (cost 1) but cannot
know that 5 was the runner-up, so the cell becomes empty, the puzzle becomes ambiguous and its solution is
**wrong**. The weighted solver reads the 5 back in and gets the right solution: that is what the probabilities
buy.

## Per-cell shift: why `cost = 0` means "agreed with the CNN"

Each cell's ten weights have that cell's smallest weight (its argmax class) subtracted. Every reading pays one
weight per cell, so all readings get cheaper by the same constant and the best reading does not change
(`test_per_cell_shift_keeps_the_optimum` checks this). Two benefits:

- the cost is readable: 0 = kept the CNN's answer everywhere; 693 = overruled one 60/30 call
- speed: RC2 raises its lower bound from 0 one conflict at a time. Unshifted, it had to "discover" the
  unavoidable cost of all 81 cells first: one noisy grid took **68 s** unshifted vs **0.1 s** shifted.

We also use `RC2Stratified` (handles confident clauses first) with `adapt`, `exhaust`, `minz`. These are speed
options; the answer is still exactly optimal.

## `require_unique`: using "real puzzles have one solution"

Week 1 finding: at 5% simulated error, plain MaxSAT solved 15/30 puzzles. In **14 of the 15 failures** the
solver found a reading that was *cheaper than the truth* but had more than one solution. Typical case: a missed
digit. Reading it as empty is cheaper than reading the faint digit, and the puzzle is still solvable, just not
uniquely, so the solver returns one of several solutions.

Real printed puzzles always have exactly one solution, so that is legitimate prior knowledge.
`solve(prob, require_unique=True)` adds it lazily:

1. solve MaxSAT, get reading K
2. check if K has a unique solution (plain SAT + blocking clause)
3. if not: add the hard clause "at least one cell is read differently from K", go to 1

Each round returns the next-cheapest reading (costs rise monotonically), so the first unique one is the most
probable reading *among readings with a unique solution*. In the sweep ([evaluation.md](evaluation.md)): 50% ->
**80%** solved at 5% misreads, 10% -> **36.7%** at 20%.

Costs: slower when the image is very noisy (sweep mean per puzzle: 0.76 s at 5%, 8.4 s at 20%; the worst single
puzzle seen in a separate 20% run took 167 s), because many cheap ambiguous readings must be ruled out one by
one; capped at 50 rounds. It can also break a correct reading to reach a unique one (1-5 cells per 30 puzzles). Each round rebuilds the RC2 solver,
because pysat's `RC2Stratified` returned "no model" when clauses were added between `compute()` calls.

## Explaining overrides with unsat cores (`explain.py`)

Every override comes with a reason a person can check. For a cell where the CNN read digit `a` and the solver
read something else, we ask a plain SAT solver:

> Is there a valid Sudoku with `a` at this cell **and** every other digit the solver finally kept?

The kept digits are passed as *assumptions* on top of the Sudoku rules, so one solver instance is reused for
all overrides.

- **UNSAT**: the solver returns a *core*, a subset of the assumptions that is already contradictory. We shrink it
  to a **minimal** core: drop one given, re-check, and keep the drop if it is still UNSAT. Every given listed is
  then needed for the contradiction.
  - core of size 1 → `conflicts with given 9 at (0, 6) in row 0` (a direct clash)
  - larger core → `no valid Sudoku has 2 here together with givens 7 at (0, 1), 5 at (0, 2), 2 at (2, 3), 2 at
    (7, 0)` (real output, `demo_reasoning.py --seed 3 --error-rate 0.15`). Here no single given rules the 2
    out; these four together do. The Week 1 row/column/box scan could not find this kind of reason at all.
- **SAT**: keeping the CNN's digit was possible, so the solver dropped it for cost (or, with `require_unique`,
  for uniqueness). The reason is `lower total cost (no direct rule conflict)`. Cells the CNN read as *empty* always
  get this reason, because an empty reading never conflicts with anything.

Why use the solver for explanations rather than a hand-written check: it covers direct and indirect conflicts with
one method, and the minimal core is a proof, not a heuristic. Tests: `test_explain_direct_conflict_names_one_given`,
`test_explain_indirect_conflict_is_minimal` (checks that removing any given from the core makes it satisfiable),
`test_explain_falls_back_without_conflict`.

## The unweighted ablation (`baselines.unweighted_maxsat`)

Same hard clauses as the weighted solver. The soft part: one clause "keep the argmax digit" with weight **1** for
every cell the CNN read as a digit, and nothing for cells read as empty. The optimum is therefore "drop the
fewest read digits so that a valid Sudoku remains", blind to whether a digit had p = 0.99 or p = 0.51.

Two extra hard clauses make the reading well defined without changing the optimum: an empty-read cell stays
empty, and a digit cell is either kept or emptied (never changed into another digit). Both are safe because
emptying a cell never creates a conflict, so with equal weights neither a new nor a substituted digit can lower
the cost. Comparing `maxsat_weighted` with `maxsat_unweighted` isolates the value of the CNN's confidence scores.

## At-most-one encoding benchmark (`scripts/bench_encoding.py`)

`solve(prob, check_unique=False)` on 100 simulated grids per error rate (same grids for both encodings, run back to
back). Times in ms on this laptop; both encodings found the same optimum on all 400 grids.

| misread rate | pairwise median | pairwise p95 | seqcounter median | seqcounter p95 |
|---|---|---|---|---|
| 0% | 53.7 | 63.4 | 51.7 | 67.2 |
| 5% | 57.3 | 86.4 | 55.1 | 126.2 |
| 10% | 64.9 | 130.7 | 71.6 | 268.4 |
| 20% | 96.5 | 266.8 | 144.2 | 384.1 |

**Pairwise stays the default.** The medians are equal at low noise, and pairwise is clearly faster in the tail as
noise grows. A plausible reason: for 9 options the pairwise encoding is only 36 binary clauses with no auxiliary
variables, which propagate directly. The sequential counter's auxiliary variables give RC2 more to search
through. A maximum of about 2 s appears for both at 10% noise.

## Files

| File | What it does |
|---|---|
| `encoding.py` | variables, hard and soft clauses, `build_wcnf(prob)`, `decode_model` |
| `solver.py` | `solve(prob, require_unique=False)`: RC2 MaxSAT, uniqueness check or loop; `sat_solve`, `count_solutions` for exact givens |
| `baselines.py` | `argmax_then_sat` (trust CNN fully), `threshold_then_sat` (drop digits below confidence tau), `unweighted_maxsat` (ablation: rules without confidences) |
| `generator.py` | seeded random solutions (backtracking) and unique puzzles with ~30 givens; `redundant_givens` |
| `explain.py` | `find_overrides` with unsat-core reasons (`conflict_core`), text rendering for the demo |

## Uniqueness check

Solve with the read givens fixed, add a clause "at least one free cell must differ from this solution", solve
again. UNSAT the second time means the solution was unique. (`count_solutions` in `solver.py`.)

## Why puzzles are not minimal

A minimal puzzle loses uniqueness if any single given is removed. Then a CNN that *misses* one digit makes the
puzzle ambiguous no matter how good the solver is. The generator stops removing givens at `target_givens` (default
30), well above the typical minimum (~22-26), so most puzzles have several redundant givens
(`redundant_givens(puzzle)` lists them).

## Speed (measured on this laptop, Week 1)

- one MaxSAT solve: ~0.06-0.12 s (shifted weights, stratified RC2)
- generating one 30-given puzzle: ~0.75 s (about 80 uniqueness checks; the rule clauses are cached)
