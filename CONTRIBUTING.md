# Contributing

## Branches

- One branch per feature, named `role/short-name`: `vision/robust-extract`, `perception/cnn-v1`,
  `reasoning/seqcounter`, `eval/noise-sweep`.
- Never push directly to `main`. Open a pull request into `main`; it needs **one reviewer's approval** and green CI.
- Keep PRs small enough to review in 15 minutes. Rebase on `main` before asking for review.

## Commits

[Conventional commits](https://www.conventionalcommits.org/):

```
feat(reasoning): add sequential-counter AMO encoding
fix(vision): handle grids touching the image border
test(eval): cover hallucination metrics
docs: explain temperature scaling
chore: bump ruff
perf(reasoning): cache rule clauses
```

Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`. Scope = the package area.

## Before you push

```bash
ruff check . && ruff format .
pytest -q
```

`pre-commit install` runs ruff for you on every commit.

## Docs are part of the change

Every PR that changes behaviour also updates:

- [docs/progress.md](docs/progress.md): one entry saying what changed and where to look
- the explainer for that area (`docs/reasoning.md`, `docs/vision.md`, ...) and [docs/dataflow.md](docs/dataflow.md)
  if inputs/outputs moved
- [docs/interfaces.md](docs/interfaces.md) **and** `interfaces.py` together if a contract changes, and tell the
  other owners

## Numbers

Never paste a number into docs, the report or a PR that a script did not produce. Say which script and seed.

## Large files

`data/` and `outputs/` are gitignored. Do not commit datasets, renders or model checkpoints (`*.pt`). Share them
outside git and document how to regenerate them.
