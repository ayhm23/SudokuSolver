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

## Week 3

15. **feat(eval): ablation knobs and final-sweep configs** (no behaviour change)
    `encoding.weight / soft_clauses / build_wcnf` and `solver.solve` take `scale` / `eps` (defaults = `SCALE` /
    `EPS`; a test checks the defaults give the identical formula and result). `run_e2e.py` configs take
    `method_kwargs`, and a resumed run cuts off a half-written last CSV row. `tune_threshold.py` takes several val
    dirs and refuses test dirs. `sweep_simulated.py` gets `--rates`, `--methods` (adds `maxsat_unweighted`),
    `--config`, and a per-puzzle CSV. New sets from `scripts/render_final_sets.sh`: `test_final/` (TEST fonts, new
    seed 40000, 100 images x 16 sets incl. clean) and `val_noise/` (VAL fonts, 50 x 15). Configs:
    `configs/e2e_final_noise_cnn*.json`, `configs/ablations/`. `scripts/run_detached.ps1` = nohup for Windows.

16. **feat(eval): report generator for the final results**
    `scripts/make_report.py` turns the raw run CSVs into small summary CSVs (95% Wilson intervals, cell accuracy
    before/after, corrections vs wrong overrides, UNSAT rate, median/p95 time, failure stage), the 7 figures
    (PNG 300 dpi + PDF, one fixed style per method, readable in black and white) and `report/results/RESULTS.md`,
    which contains tables only, generated from the CSVs so no number is typed by hand. `scripts/dump_cell_logits.py`
    saves per-cell logits on the test sets for the reliability diagram. `run_detached.ps1` now logs every
    command of an `&&` chain (before, cmd redirected only the last one).

17. **results: Week 3 final sweep and ablations**
    `report/results/` (summary CSVs + `RESULTS.md`, tables only) and `report/figures/` (7 figures, PNG + PDF),
    all produced by `make_report.py` from the runs listed in `configs/README.md`. Main sweep: 1,600 test images x
    5 methods; ablations: temperature off, SCALE 10-10000, EPS 1e-3-1e-9 (maxsat_weighted, same images), tau on
    noisy VAL (0.5), simulated sweep (300 puzzles x 9 error rates). How to reproduce:
    [evaluation.md](evaluation.md#week-3-final-sweep-and-ablations).

18. **analysis: why unweighted MaxSAT fixes 0 misreads** (read-only, no method or metric change)
    `scripts/check_unweighted.py` re-runs `unweighted_maxsat` on the probabilities rebuilt from the PR #9 logit
    dump, reproduces every CSV row, and shows the 0 is forced by construction: the method can only keep or blank a
    CNN digit, and every misread in test_final is a digit substitution. Report: `report/results/unweighted_check.md`.

19. **analysis: CNN confusion pairs and noisy-vs-clean cell flips** (read-only)
    `scripts/analyse_cells.py` (from the PR #9 logit dump): top (true -> read) pairs overall, per noise type and per
    font (`report/results/confusions.md`, figure `fig8_confusions`), and per noise set the cells that flip relative
    to the clean render of the same puzzle, with the Space Mono "5" cells broken out (`noise_flips.md`).

20. **results: cluster-bootstrap intervals for pooled solve rates**
    The 16 test_final sets share the same 100 puzzles, so pooled Wilson intervals are too narrow. `make_report.py`
    now also resamples puzzles (10,000 resamples, seed 0) and reports both intervals side by side, plus the paired
    difference maxsat_weighted - argmax_then_sat on identical resamples (RESULTS.md sections 1d-1e,
    `e2e_pooled_bootstrap.csv`, `e2e_paired_diff_bootstrap.csv`). Existing tables are unchanged.

21. **analysis: maxsat_unique runtime tail and a timeout what-if** (report only, default unchanged)
    `scripts/unique_tail.py` re-solves the images whose recorded maxsat_unique time is over 10 s to recover their
    cost and solver-call counts (counted by wrapping the solver classes from the script), and checks against the
    recorded data what a T-second timeout with fallback to the maxsat_weighted answer would change.
    `report/results/unique_tail.md`.

## Week 4: real photos (public dataset)

22. **feat(reasoning): optional time limit for maxsat_unique**
    `solve(..., require_unique=True, time_limit=T)` stops the uniqueness rounds after T seconds and returns the
    first-round reading (= the maxsat_weighted answer), flagged with a `timeout:` message. Set per run with
    `method_kwargs: {"maxsat_unique": {"time_limit": 10}}`. Off by default. Test:
    `test_require_unique_time_limit_falls_back_to_first_round`. See [reasoning.md](reasoning.md).

23. **feat(eval): import the public real-photo dataset; vision check without corners**
    `scripts/import_wichtounet.py` turns the 200 v2 photos of github.com/wichtounet/sudoku_dataset into our
    real-photo JSON format (original split, phone, declared and actual resolution, `puzzle_id`, SAT solution;
    unsolvable or ambiguous labels would be flagged, none were). Findings recorded in `data/real/README.md`: only 76
    distinct puzzles, 44 photos smaller than their `.dat` says, 9 EXIF tags that contradict the labels, a
    third-party outline file stored for cross-checking only. `load_render` accepts `"corners": null`.
    `run_e2e.py`: for such images "vision failed" = grid not found, or >= 12 cells where the CNN's empty / digit
    decision disagrees with the truth (threshold fixed from synthetic test_final, before any real run; new column
    `occ_disagree`), and optional `save_perception` writes each ProbGrid + detected corners to `.npz`. Tests:
    `test_import_wichtounet.py`, `test_run_e2e_without_corners_uses_occupancy_check`.


24. **feat(eval): contact sheet for checking the imported labels by eye**
    `scripts/real_label_sheet.py` draws the imported givens on 20 random photos (seed 0), placed with the
    third-party outline, plus a plain-grid panel per photo, into `outputs/real_label_check/`. Draws only; never
    edits a label.

25. **feat(eval): real-photo run config and report script**
    `configs/e2e_real_wichtounet_cnn.json` (all 5 methods, tau 0.5, maxsat_unique time limit 10 s, ProbGrids saved)
    and `scripts/real_photos_report.py`: per-method tables with Wilson and puzzle-bootstrap intervals, cell accuracy
    before/after, vision check details with a cross-check against the third-party outlines, synthetic vs real,
    resolution / phone breakdowns, confusion pairs, one figure per photo and 6 figures picked by fixed rules. It
    re-solves maxsat_weighted from the saved ProbGrids and checks that every result matches the run CSV. How to
    run: [evaluation.md](evaluation.md#week-4-real-photos-public-dataset).

26. **results: real photos (public wichtounet v2, 200 photos)**
    `report/results/real_photos.md` (+ `real_photos_*.csv`) and `report/figures/real/` (6 per-photo figures), all
    generated by `real_photos_report.py` from the run of `configs/e2e_real_wichtounet_cnn.json`. Facts only; nothing
    was tuned on these photos.

27. **feat(report): failure and success gallery**
    `scripts/make_gallery.py` picks real examples from the run CSVs by fixed rules, re-runs each one like
    `run_e2e.py` (the CSVs have no explanation text) and keeps it only if the re-run reproduces the CSV row. Writes
    `report/results/gallery.md` (dataset, noise, what happened, the system's override explanations verbatim, a
    cost breakdown for the wrong override) and 11 figures in `report/figures/gallery/`. Found: every category except
    "missed digit recovered", which has no example in any run (census in gallery.md). Details:
    [evaluation.md](evaluation.md#gallery-real-successes-and-failures-scriptsmake_gallerypy).

