# Interface contracts

Source of truth: [`src/sudoku_ns/interfaces.py`](../src/sudoku_ns/interfaces.py). If you change one, change both.

Every module imports its types from `sudoku_ns.interfaces`. Call the `validate_*` function at each module
boundary; each raises `ValueError` with a message naming the bad cell, shape or dtype.

## Conventions

- Cells are row-major: flat index = `9 * r + c`, with `r, c` in `0..8`.
- Label `0` = empty cell, `1..9` = digit.

## Arrays

| Name         | dtype     | shape          | Values                                                     | Validator                |
|--------------|-----------|----------------|------------------------------------------------------------|--------------------------|
| `CellImages` | `float32` | `(81, 32, 32)` | in `[0, 1]`, row-major cell order                          | `validate_cell_images`   |
| `ProbGrid`   | `float32` | `(9, 9, 10)`   | class 0 = empty, 1..9 = digits; each cell sums to 1 (±1e-4) | `validate_prob_grid`     |
| `Grid`       | `int8`    | `(9, 9)`       | 0 = empty (puzzles), 1..9 (solutions)                      | `validate_grid(allow_empty=...)` |

## Dataclasses

### `Override`
A cell where the solver's reading differs from the CNN argmax.

| Field         | Type  | Meaning                                                     |
|---------------|-------|-------------------------------------------------------------|
| `row`, `col`  | int   | cell position                                               |
| `cnn_label`   | int   | CNN argmax class (0 = empty)                                |
| `cnn_prob`    | float | CNN probability of that class                               |
| `final_label` | int   | class the solver decided the cell shows                     |
| `final_prob`  | float | CNN probability of the final class                          |
| `reason`      | str   | from a minimal unsat core: `"conflicts with given 9 at (0, 6) in row 0"`, `"no valid Sudoku has 2 here together with givens ..."`, or `"lower total cost (no direct rule conflict)"` ([reasoning.md](reasoning.md#explaining-overrides-with-unsat-cores-explainpy)) |

### `SolveResult`

| Field          | Type                 | Meaning                                                       |
|----------------|----------------------|---------------------------------------------------------------|
| `status`       | `SolveStatus`        | `SOLVED`, `UNSAT` or `ERROR`                                   |
| `solution`     | `Grid \| None`       | full solution (no zeros)                                       |
| `read_givens`  | `Grid \| None`       | what the solver decided the image shows (0 = empty)           |
| `overrides`    | `list[Override]`     | cells where `read_givens` differs from the CNN argmax         |
| `unique`       | `bool \| None`       | whether `read_givens` has exactly one solution (None = not checked) |
| `cost`         | `int \| None`        | MaxSAT objective: total weight of violated soft clauses       |
| `solve_time_s` | `float`              | wall-clock solve time                                         |
| `message`      | `str \| None`        | **new in Week 2.** Why the status is ERROR/UNSAT, prefixed with the stage: `"vision: ..."`, `"perception: ..."`, `"reasoning: ..."`. None when SOLVED |

## Perceiver

Any callable `CellImages -> ProbGrid`.

| Perceiver | Where | What |
|---|---|---|
| `CNNPerceiver(checkpoint_path, use_temperature=True)` | `perception/perceiver.py` | the trained CNN: one batch of 81 crops, logits / T, softmax, clipped at `EPS = 1e-6` and renormalised, validated |
| `OraclePerceiver(givens, seed, error_rate=..., ...)` | `pipeline.py` | simulated CNN from the true givens (Week 1); an upper bound or a controlled-error stand-in |

## Pipeline

`pipeline.run(image_path, perceiver, method="maxsat_weighted", tau=0.8) -> PipelineOutput(corners, cells, prob,
result)`. If the grid is not found, `corners`, `cells` and `prob` are None and `result.status` is `ERROR` with
`message="vision: <reason>"`. Nothing is raised. Methods: `maxsat_weighted`, `maxsat_unique`,
`maxsat_unweighted`, `argmax_then_sat`, `threshold_then_sat`.
