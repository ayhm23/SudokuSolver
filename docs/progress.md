# Progress log

One entry per commit, newest at the bottom. Each says what changed and where to look.

## Week 1

1. **chore: scaffold package, tooling and CI**
   `pyproject.toml` (src layout, `pip install -e ".[dev]"`), `.gitignore` (data/, outputs/, checkpoints never
   committed), pre-commit with ruff, GitHub Actions running ruff + pytest on Python 3.11 (skips `slow` tests).

2. **feat(interfaces): data contracts and validators**
   `interfaces.py` defines `CellImages`, `ProbGrid`, `Grid`, `Override`, `SolveResult` with `validate_*`
   functions. Mirrored in [interfaces.md](interfaces.md).

3. **feat(reasoning): MaxSAT encoding, solver, baselines, generator**
   The core of the project. See [reasoning.md](reasoning.md). 11 reasoning tests: clean grid, misread corrected,
   missed digit, hallucination removed, numbering round-trip, weight monotonicity, plus encoding/baseline checks.
   Added [dataflow.md](dataflow.md) as the project map.

4. **perf(reasoning): per-cell weight shift, stratified RC2, `require_unique`**
   Found while running the sweep: plain RC2 took 68 s on one noisy grid. Shifting each cell's weights by its argmax
   cost (same optimum, proven by a test) plus `RC2Stratified` brought it to ~0.1 s. Diagnosed sweep failures: 14 of
   15 were cheap-but-ambiguous readings, so added `solve(..., require_unique=True)` (next-best reading until unique).
   Details and measured numbers in [reasoning.md](reasoning.md).

5. **feat(synth): font download, puzzle renderer, noise functions**
   15 OFL fonts, renders with exact corner ground truth, 4 noise types with physical severity units.
   See [synthetic-data.md](synthetic-data.md). Tests: `test_noise.py` (17), `test_render.py` (3).

6. **feat(vision): grid detection and cell extraction**
   Contour-based corner finder with a grid-line check (fixes "found the paper instead of the grid"), perspective
   warp to 81 cells. 200/200 clean renders within 5 px, mean error 1.74 px. See [vision.md](vision.md).

7. **feat(perception): digit dataset split by font family**
   Crops cut with the pipeline's own cell extractor, augmentation for train, held-out font families for val/test,
   lazy torch wrapper, Week 2 stubs (`model.py`, `train.py`, `calibrate.py`). Counts in
   [synthetic-data.md](synthetic-data.md#step-4-digit-dataset-perceptiondatasetpy-scriptsmake_digit_datasetpy).

8. **feat(eval): simulated perception, metrics, sweep, reasoning demo**
   `simulate_prob_grid` (realistic fake CNN output with a corruption record), metrics before/after reasoning,
   the 4-method sweep (n = 30, ~28 min) and `demo_reasoning.py`. Results and caveats in
   [evaluation.md](evaluation.md). Headline at 5% misreads: argmax 6.7%, threshold 23.3%, maxsat 50.0%,
   maxsat_unique 80.0% of puzzles solved correctly.
