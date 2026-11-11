# Code and data flow

This is the "map" of the project: what data exists, which file produces it, and which file consumes it.
Read this first, then the per-area explainers:

- [reasoning.md](reasoning.md): how the MaxSAT solver turns probabilities into a solved grid
- [evaluation.md](evaluation.md): simulated CNN, metrics, the sweep experiment
- [synthetic-data.md](synthetic-data.md): fonts, rendered puzzles, noise, the digit dataset
- [vision.md](vision.md): finding the grid in a photo and cutting out cells
- [perception.md](perception.md): the CNN, training, temperature scaling, accuracy and calibration numbers
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
 perceiver (any callable)   CNNPerceiver (trained CNN, ──────► ProbGrid    float32 (9, 9, 10)
      │                     logits / T), or OraclePerceiver    P(cell shows empty / 1..9)
      ▼
 reasoning/solver.py        solve(prob) ────────────────────► SolveResult
      │                       ├ encoding.py builds the           ├ solution     (9, 9) digits
      │                       │ weighted MaxSAT formula          ├ read_givens  what the image shows
      │                       └ explain.py: unsat-core reason    ├ overrides    where it disagreed with the CNN, + why
      ▼                         for every override               └ unique, cost, solve_time_s, message
 pipeline.py run(image_path, perceiver, method) wires all of the above together;
             vision failure -> SolveResult(ERROR, message="vision: ...") instead of an exception
             python -m sudoku_ns.pipeline photo.jpg --checkpoint ...  prints + saves a visualisation
```

The key idea: the CNN never makes a hard decision. It hands over a probability for every class in every cell, and
the solver picks the most probable reading that is also a legal Sudoku. That is how a misread digit gets corrected.

## How data enters the system

```
 reasoning/generator.py  generate(seed) ─► (puzzle Grid, solution Grid)
         │
         ├──► eval/simulate.py  simulate_prob_grid(puzzle, error_rate, ...) ─► ProbGrid + CorruptionRecord
         │        (a fake CNN with controlled mistakes; tests reasoning on its own)
         │
         └──► synth/render.py   render_puzzle(puzzle, fonts of ONE split) ─► image + corners + font
                  │   scripts/render_synthetic.py --split train|val|test  ─► data/synthetic/<split>/
                  ├──► synth/noise.py  apply_noise (lighting, rotate, blur, noise, JPEG), clutter
                  │
                  └──► perception/dataset.py ─ find_grid_corners + extract_cells (THE PIPELINE'S OWN CODE)
                           ─► data/digits/digits.npz ─► perception/train.py ─► outputs/models/<run>/model.pt
                                                        perception/calibrate.py (T on val) ─┘

 real photos: data/real/*.jpg ─► scripts/label_real.py ─► <photo>.json (same keys as a render)
```

Evaluation: `scripts/run_e2e.py --config configs/<name>.json` runs every image through vision → perceiver → each
method and writes `outputs/e2e/<name>.csv`; `scripts/summarize_e2e.py` turns it into tables.

## Who calls whom

| File | Main function(s) | Reads | Produces | Used by |
|---|---|---|---|---|
| `interfaces.py` | `validate_*`, `Override`, `SolveResult` | - | type contracts | everything |
| `pipeline.py` | `run(image_path, perceiver, method)`, `perceive`, `make_method`, `visualize`, `OraclePerceiver`, CLI | image file | `PipelineOutput` (corners, cells, prob, result) | CLI, `run_e2e.py` |
| `vision/grid_detect.py` | `find_grid_corners`, `binarize`, `grid_score`, `refine_corners` | RGB image | 4 corners | pipeline, dataset, `eval_grid_detect.py` |
| `vision/cell_extract.py` | `extract_cells`, `clean_cell`, `overlay`, `save_debug` | image + corners | CellImages | pipeline, perception/dataset |
| `perception/dataset.py` | `build_split`, `crops_from_render`, `augment`, `torch_dataset` | renders (PNG + JSON) | `digits.npz` | `train.py` |
| `perception/model.py` | `DigitCNN` | (N, 1, 32, 32) | (N, 10) logits | train, perceiver |
| `perception/train.py` | `train`, `load_model`, `classification_report` | `digits.npz` | `outputs/models/<run>/` | - |
| `perception/calibrate.py` | `fit_temperature`, `ece`, `calibrate_run` | checkpoint + val split | T in checkpoint, reliability plots | - |
| `perception/perceiver.py` | `CNNPerceiver` | CellImages | ProbGrid | pipeline, `run_e2e.py` |
| `reasoning/encoding.py` | `x_var`, `y_var`, `build_wcnf`, `sudoku_cnf` | ProbGrid | PySAT formula | solver, baselines, generator |
| `reasoning/solver.py` | `solve`, `sat_solve`, `is_unique` | ProbGrid / Grid | SolveResult | pipeline, eval, demo |
| `reasoning/baselines.py` | `argmax_then_sat`, `threshold_then_sat`, `unweighted_maxsat` | ProbGrid | SolveResult | pipeline, sweep |
| `reasoning/generator.py` | `generate`, `redundant_givens` | seed | puzzle + solution | tests, sweep, render |
| `reasoning/explain.py` | `find_overrides`, `conflict_core`, `format_grid`, `describe_result` | ProbGrid + reading | overrides with reasons, text | solver, baselines, demo |
| `synth/render.py` | `render_puzzle`, `save_render`, `load_render` | puzzle + fonts | image + ground truth | render script, tests |
| `synth/noise.py` | `blur`, `rotate`, `jpeg`, `gaussian_noise`, `lighting`, `clutter`, `apply_noise`, `noisy_corners` | image | degraded image | render script, grid-detect eval |
| `eval/simulate.py` | `simulate_prob_grid` | puzzle | ProbGrid + CorruptionRecord | sweep, demo, OraclePerceiver |
| `eval/metrics.py` | `evaluate`, `aggregate` | SolveResult + truth | metric dicts | sweep |

## Scripts and what they write

| Script | Writes |
|---|---|
| `download_fonts.py` | `data/fonts/` |
| `render_synthetic.py` | `data/synthetic/{train,val,test_clean,test_noise/<type>_<s>}/` |
| `make_digit_dataset.py` | `data/digits/digits.npz` |
| `python -m sudoku_ns.perception.train` | `outputs/models/<run>/{model.pt,config.json,history.json,metrics.json}` |
| `python -m sudoku_ns.perception.calibrate` | T into `model.pt` and `metrics.json`, `reliability_{val,test}.png` |
| `tune_threshold.py` | stdout: best tau on val |
| `run_e2e.py` | `outputs/e2e/<config name>.csv` |
| `summarize_e2e.py` | stdout tables (or markdown) |
| `label_real.py` | `data/real/<photo>.json` |
| `eval_grid_detect.py` | stdout table, `outputs/vision_failures/{grid_detect.csv,<noise>/*.png}` |
| `bench_encoding.py` | `outputs/bench_encoding.csv` |
| `sweep_simulated.py` | `outputs/sweep_simulated.{csv,png}` |
| `demo_reasoning.py` | stdout |
| `python -m sudoku_ns.pipeline` | stdout + `<photo>_solved.png` |

What each commit added: [progress.md](progress.md).
