# Code and data flow

This is the "map" of the project: what data exists, which file produces it, and which file consumes it.
Read this first, then the per-area explainers:

- [reasoning.md](reasoning.md): how the MaxSAT solver turns probabilities into a solved grid
- [evaluation.md](evaluation.md): simulated CNN, metrics, the sweep experiment
- [synthetic-data.md](synthetic-data.md): fonts, rendered puzzles, noise, the digit dataset
- [vision.md](vision.md): finding the grid in a photo and cutting out cells
- [interfaces.md](interfaces.md): exact shapes and dtypes of every array passed between modules
- [progress.md](progress.md): what each commit added

## The whole pipeline in one picture

```
 photo (PNG/JPG)
      │
      ▼
 vision/grid_detect.py      find_grid_corners(img) ─────────► 4 corners (tl, tr, br, bl)
      │
      ▼
 vision/cell_extract.py     extract_cells(img, corners) ────► CellImages  float32 (81, 32, 32)
      │
      ▼
 perceiver (any callable)   CNN in Week 2, OraclePerceiver ─► ProbGrid    float32 (9, 9, 10)
      │                     in Week 1                         P(cell shows empty / 1..9)
      ▼
 reasoning/solver.py        solve(prob) ────────────────────► SolveResult
      │                       └ encoding.py builds the           ├ solution     (9, 9) digits
      │                         weighted MaxSAT formula          ├ read_givens  what the image shows
      │                                                          ├ overrides    where it disagreed with the CNN
      ▼                                                          └ unique, cost, solve_time_s
 pipeline.py run(image_path, perceiver) wires all of the above together
```

The key idea: the CNN never makes a hard decision. It hands over a probability for every class in every cell, and
the solver picks the most probable reading that is also a legal Sudoku. That is how a misread digit gets corrected.

## Two ways data enters the system

Real photos only arrive in Week 2, so in Week 1 every input is generated:

```
 reasoning/generator.py  generate(seed) ─► (puzzle Grid, solution Grid)
         │
         ├──► eval/simulate.py  simulate_prob_grid(puzzle, error_rate, ...) ─► ProbGrid + CorruptionRecord
         │        (a fake CNN with controlled mistakes; used to test reasoning on its own)
         │
         └──► synth/render.py   render_puzzle(puzzle, ...) ─► image + corners + font
                  │                                            (a fake photo; used to test vision)
                  ├──► synth/noise.py  apply_noise(img, spec)   (blur, rotation, JPEG, sensor noise)
                  └──► perception/dataset.py  crops of cells ─► digit dataset .npz (CNN training, Week 2)
```

## Who calls whom (Week 1)

| File | Main function(s) | Reads | Produces | Used by |
|---|---|---|---|---|
| `interfaces.py` | `validate_*`, `Override`, `SolveResult` | - | type contracts | everything |
| `reasoning/encoding.py` | `x_var`, `y_var`, `build_wcnf`, `sudoku_cnf` | ProbGrid | PySAT formula | solver, baselines, generator |
| `reasoning/solver.py` | `solve`, `sat_solve`, `is_unique` | ProbGrid / Grid | SolveResult | pipeline, eval, demo |
| `reasoning/baselines.py` | `argmax_then_sat`, `threshold_then_sat` | ProbGrid | SolveResult | sweep |
| `reasoning/generator.py` | `generate`, `redundant_givens` | seed | puzzle + solution | tests, sweep, render |
| `reasoning/explain.py` | `format_grid`, `describe_result` | Grid, SolveResult | text | demo |

The table grows as each task lands; see [progress.md](progress.md).
