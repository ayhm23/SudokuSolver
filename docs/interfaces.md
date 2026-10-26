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
| `reason`      | str   | e.g. `"7 clashes with given 7 in row 3"` or `"lower total cost"` |

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

## Perceiver

Any callable `CellImages -> ProbGrid`. The Week 1 placeholder is `pipeline.OraclePerceiver`; the CNN replaces it in Week 2.
