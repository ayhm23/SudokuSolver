# Neurosymbolic Sudoku Solver

[![CI](https://github.com/ayhm23/SudokuSolver/actions/workflows/ci.yml/badge.svg)](https://github.com/ayhm23/SudokuSolver/actions/workflows/ci.yml)

AID 851 (Machine Reasoning) course project. Final deadline: 27 November 2026.

Photo of a Sudoku → OpenCV finds the grid → a CNN gives a probability distribution over {empty, 1..9} for each
cell → **weighted MaxSAT** combines the Sudoku rules (hard clauses) with the CNN's beliefs (soft clauses, weight
−log p) → a solved grid, where misread digits get corrected because the rules overrule low-confidence guesses.
We compare against "argmax then solve" baselines across image-noise levels.

**New here? Start with [docs/dataflow.md](docs/dataflow.md)**: one page showing every module, what it reads and
what it produces.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
# CPU-only torch (~200 MB instead of ~2 GB with CUDA); skip this line if you have a GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,ml]"           # core + torch + pytest, ruff, pre-commit
pre-commit install
```

Python 3.10+ (CI uses 3.11). Always run things with the venv's Python.

## Tests

```bash
pytest -q                 # everything (CI skips tests marked slow: pytest -m "not slow")
ruff check . && ruff format --check .
```

## Solve a photo

```bash
python -m sudoku_ns.pipeline photo.jpg --checkpoint outputs/models/cnn_v1/model.pt
python -m sudoku_ns.pipeline photo.jpg --checkpoint outputs/models/cnn_v1/model.pt --method maxsat_unique --out vis.png
```

This prints the CNN's reading, the solver's reading with an explanation for every override, and the solution. It
also saves a picture: the detected grid, the CNN reads with confidences, and the solution with overrides in red.
Methods: `maxsat_weighted` (default), `maxsat_unique`, `maxsat_unweighted`, `argmax_then_sat`,
`threshold_then_sat`.

## Reproduce everything

All data goes to `data/` and all results to `outputs/` (both gitignored). In this order:

| Step | Command | Time (this laptop, CPU) |
|---|---|---|
| fonts | `python scripts/download_fonts.py` | ~5 min |
| renders | the 4 `render_synthetic.py` commands at the top of the script (train 2000 / val 300 / test_clean 200 / 15 noisy test sets of 50) | ~5 min |
| digit dataset | `python scripts/make_digit_dataset.py` (crops via detection + `extract_cells`) | ~2 min |
| train | `python -m sudoku_ns.perception.train --run cnn_v1` | see [docs/perception.md](docs/perception.md) |
| calibrate | `python -m sudoku_ns.perception.calibrate --run outputs/models/cnn_v1` | ~1 min |
| tune threshold baseline | `python scripts/tune_threshold.py --checkpoint outputs/models/cnn_v1/model.pt` (val only) | ~1 min |
| end to end | `python scripts/run_e2e.py --config configs/e2e_test_clean_cnn.json`, then `python scripts/summarize_e2e.py outputs/e2e/e2e_test_clean_cnn.csv` | ~10 min |
| noisy sets | the same with `configs/e2e_test_noise_cnn.json` | ~30 min |
| grid detection | `python scripts/eval_grid_detect.py --renders data/synthetic/test_clean --n 50` | ~2 min |
| encoding benchmark | `python scripts/bench_encoding.py --n 100` | ~3 min |
| reasoning demo / Week 1 sweep | `python scripts/demo_reasoning.py --seed 0`, `python scripts/sweep_simulated.py --n 30` | 2 s / ~30 min |
| real photos | protocol in [data/real/README.md](data/real/README.md), label with `python scripts/label_real.py data/real/ --checkpoint ...` | |

## Repo map

```
src/sudoku_ns/
  interfaces.py      data contracts (CellImages, ProbGrid, Grid, Override, SolveResult) + validators
  pipeline.py        run(image_path, perceiver, method), CLI + visualisation, OraclePerceiver
  vision/            grid_detect.py (4 corners), cell_extract.py (81 cleaned 32x32 cells)
  perception/        dataset.py, model.py (CNN), train.py, calibrate.py (temperature), perceiver.py
  reasoning/         encoding.py (MaxSAT), solver.py, baselines.py, generator.py, explain.py (unsat cores)
  synth/             render.py (synthetic photos), noise.py (blur, rotate, jpeg, noise, lighting, clutter)
  eval/              simulate.py (fake CNN output), metrics.py
configs/             experiment configs for scripts/run_e2e.py
scripts/             command-line entry points (table above)
tests/               pytest suite
data/real/README.md  capture protocol for the real-photo test set
docs/                explainers: dataflow, interfaces, reasoning, vision, perception, synthetic data,
                     evaluation, progress
notebooks/ report/ demo/
```

## Interfaces

Full details: [docs/interfaces.md](docs/interfaces.md).

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `CellImages` | float32 | (81, 32, 32) | cell crops in [0, 1], row-major (index = 9r + c) |
| `ProbGrid` | float32 | (9, 9, 10) | per-cell distribution; class 0 = empty, 1-9 = digits |
| `Grid` | int8 | (9, 9) | 0 = empty, 1-9 = digit |
| `Override` | dataclass | | a cell where the solver disagreed with the CNN, with a reason |
| `SolveResult` | dataclass | | status, solution, read_givens, overrides, unique, cost, solve_time_s, message |

## Module owners

| Area | Owner | Code |
|---|---|---|
| Vision | `<name>` | `vision/`, `synth/` |
| Perception | `<name>` | `perception/` |
| Reasoning | `<name>` | `reasoning/` |
| Evaluation & Demo | `<name>` | `eval/`, `scripts/`, `demo/` |

## Roadmap

| Week | Goal | Status |
|---|---|---|
| 1 | Repo, interfaces, MaxSAT reasoning + baselines, simulated sweep, renderer, noise, digit dataset, grid detection on clean renders | done, see [docs/progress.md](docs/progress.md) |
| 2 | CNN training + temperature scaling; robust grid detection and cell extraction; end-to-end harness; real-photo tooling | done, see [docs/progress.md](docs/progress.md) |
| 3 | Full noise sweeps; ablations; real photos; failure gallery; demo UI ([Week 3 milestone](https://github.com/ayhm23/SudokuSolver/milestone/3)) | |
| 4 | Report, viva preparation, final demo (deadline 27 Nov) | |

## Week 2 results (real CNN, end to end)

200 held-out test renders (fonts never seen in training), real grid detection, trained CNN (test cell accuracy
99.26%), % of puzzles solved correctly:

| maxsat_unique | maxsat_weighted | maxsat_unweighted | argmax_then_sat | threshold_then_sat |
|---|---|---|---|---|
| **98.0%** | **89.5%** | 77.5% | 74.5% (23.5% UNSAT) | 74.5% |

Grid detection: 100% under every noise type and severity tested. Details, caveats and noisy-set numbers:
[docs/evaluation.md](docs/evaluation.md), [docs/perception.md](docs/perception.md), [docs/vision.md](docs/vision.md).

## Week 1 results (simulated perception)

See [docs/evaluation.md](docs/evaluation.md) for the full table. In short: argmax→SAT fails as soon as one misread
clashes with a rule. Plain MaxSAT never returns UNSAT and corrects many misreads, but often lands on a cheaper,
ambiguous reading. `require_unique=True` fixes most of those.
