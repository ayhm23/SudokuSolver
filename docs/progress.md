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

9. **feat(pipeline): end-to-end run, README, CONTRIBUTING**
   `pipeline.run(image_path, perceiver)` wires vision -> perceiver -> solver; `OraclePerceiver` stands in for the
   CNN. README (setup, scripts, repo map, owners, roadmap), CONTRIBUTING (branches, reviews, conventional
   commits, docs-with-every-change). Full module table in [dataflow.md](dataflow.md).
   On GitHub: labels `vision`, `perception`, `reasoning`, `evaluation`; milestones Week 1-4; Week 2 issues #1-#4.

10. **docs: CI badge in README**
    Shows the status of the latest GitHub Actions run (ruff + pytest) at the top of the README.

## Week 2

One PR per area, merged in this order (perception's dataset is cut with the new vision code). The README,
[dataflow.md](dataflow.md) and this log were updated in the last PR.

11. **feat(reasoning): unsat-core explanations, unweighted ablation, encoding benchmark**
    `explain.find_overrides` gives each override a reason from a minimal unsat core (direct clash or indirect
    conflict), replacing the Week 1 row/column/box scan. `baselines.unweighted_maxsat` = rules without confidences.
    `bench_encoding.py`: pairwise stays default (p95 267 ms vs 384 ms at 20% error). `SolveResult` gets a
    `message` field saying which stage failed and why. See [reasoning.md](reasoning.md), [interfaces.md](interfaces.md).

12. **feat(vision): robust detection, cell cleaning, noise-type evaluation** (closes #2)
    3 binarisations, shape ranking, line-structure verification, line-centre corner refinement; `clean_cell`
    erases border line remnants and centres digits, never decides "empty". New `noise.lighting` and `noise.clutter`.
    100% detection in all 25 conditions on held-out renders (Week 1: 22% at noise severity 1.0); corner error
    1.7 → about 1.1 px. See [vision.md](vision.md).

13. **feat(perception): dataset from the pipeline's own crops, CNN, temperature scaling, CNNPerceiver** (closes #1)
    2,000 / 300 / 200 renders in disjoint font families; crops via detection + `extract_cells`. `cnn_v1`: val 100.00%,
    test 99.26% (all errors: Space Mono 5→6 and 9→8); T = 1.10, test NLL 0.0231 → 0.0217. CI installs CPU torch.
    See [perception.md](perception.md).

14. **feat(eval): end-to-end harness, CLI, real-photo tooling** (closes #3; #4 stays open until the photos exist)
    `run_e2e.py` / `summarize_e2e.py` / `tune_threshold.py`, `configs/`, `pipeline` CLI with visualisation, vision
    failures → `SolveResult(ERROR, message="vision: ...")` instead of an exception, capture protocol
    `data/real/README.md` and `label_real.py`. test_clean: maxsat_unique 98.0%, maxsat_weighted 89.5%, unweighted 77.5%, argmax 74.5% solved.
    See [evaluation.md](evaluation.md#week-2-end-to-end-with-the-real-cnn-scriptsrun_e2epy).
