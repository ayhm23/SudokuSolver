# Neurosymbolic Sudoku Solver

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
pip install -e ".[dev]"              # core + pytest, ruff, pre-commit
pip install -e ".[ml]"               # only for Week 2 CNN work (torch, torchvision)
pre-commit install
```

Python 3.10+ (CI uses 3.11).

## Tests

```bash
pytest -q                 # everything (CI skips tests marked slow: pytest -m "not slow")
ruff check . && ruff format --check .
```

## Scripts

All data goes to `data/` and all results to `outputs/` (both gitignored). Run in this order the first time:

| Command | What it does | Time |
|---|---|---|
| `python scripts/demo_reasoning.py --seed 0` | corrupts a puzzle with a simulated CNN, shows MaxSAT vs argmax | 2 s |
| `python scripts/sweep_simulated.py --n 30` | MaxSAT vs baselines across 0-20% error → `outputs/sweep_simulated.{csv,png}` | ~30 min |
| `python scripts/download_fonts.py` | 15 OFL fonts → `data/fonts/` | ~5 min |
| `python scripts/render_synthetic.py --n 200 --seed 0` | rendered puzzle photos + ground truth → `data/synthetic/clean/` | ~3 min |
| `python scripts/make_digit_dataset.py` | 32x32 cell crops split by font family → `data/digits/digits.npz` | 1 min |
| `python scripts/eval_grid_detect.py` | corner-detection accuracy on the renders | 30 s |

## Repo map

```
src/sudoku_ns/
  interfaces.py      data contracts (CellImages, ProbGrid, Grid, Override, SolveResult) + validators
  pipeline.py        run(image_path, perceiver); OraclePerceiver placeholder for the CNN
  vision/            grid_detect.py (4 corners), cell_extract.py (81 cells of 32x32)
  perception/        dataset.py (digit crops); model.py, train.py, calibrate.py = Week 2 stubs
  reasoning/         encoding.py (MaxSAT), solver.py, baselines.py, generator.py, explain.py
  synth/             render.py (synthetic photos), noise.py (blur, rotate, jpeg, noise)
  eval/              simulate.py (fake CNN output), metrics.py
scripts/             command-line entry points (table above)
tests/               pytest suite
docs/                explainers: dataflow, interfaces, reasoning, vision, synthetic data, evaluation, progress
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
| `SolveResult` | dataclass | | status, solution, read_givens, overrides, unique, cost, solve_time_s |

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
| 2 | CNN training + temperature scaling; robust cell extraction under noise; noise-sweep harness; real-photo test set | open issues |
| 3 | End-to-end evaluation on synthetic + real photos across noise levels; error analysis; demo | |
| 4 | Report, viva preparation, final demo (deadline 27 Nov) | |

## Week 1 results (simulated perception)

See [docs/evaluation.md](docs/evaluation.md) for the full table. In short: argmax→SAT fails as soon as one misread
clashes with a rule. Plain MaxSAT never returns UNSAT and corrects many misreads, but often lands on a cheaper,
ambiguous reading. `require_unique=True` fixes most of those.
