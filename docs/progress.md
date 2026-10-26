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
