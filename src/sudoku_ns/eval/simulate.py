"""Simulated CNN output: a ProbGrid with controlled, recorded mistakes.

Until the real CNN exists (Week 2) this lets us test the reasoning layer on
its own, at any error rate we like, with ground truth for every corruption.

What "realistic" means here, and why each choice was made:
- Confidence is not 1.0. A trained CNN's top-class probability varies by cell,
  so clean cells draw it from `confidence_range`.
- Wrong cells are usually LESS confident than right ones, but not always: CNNs
  are known to be overconfident when wrong. So corrupted cells draw their top
  probability from `corrupted_range`, which overlaps the clean range.
- The true class is usually the runner-up when the CNN is wrong (a blurred 3
  looks most like an 8, then a 3). We give it 40-80% of the leftover mass.
- Mistakes follow visual similarity (1<->7, 3<->8, 5<->6, ...), not uniform
  noise; see CONFUSABLE. Remaining mass is spread with a Dirichlet so each
  cell's tail looks different, instead of a flat floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sudoku_ns.interfaces import ProbGrid, validate_grid, validate_prob_grid

# Classes that printed digits are commonly mistaken for (0 = empty). Empty cells
# are mostly confused with 1 (a stray grid-line remnant looks like a thin 1).
CONFUSABLE: dict[int, list[int]] = {
    0: [1],
    1: [7, 4, 0],
    2: [7, 3],
    3: [8, 5, 2],
    4: [9, 1],
    5: [6, 3],
    6: [5, 8],
    7: [1, 2],
    8: [3, 6, 9],
    9: [4, 8],
}


@dataclass
class CorruptionRecord:
    """Ground truth about what the simulator did to each cell."""

    misread: list[tuple[int, int, int, int]] = field(default_factory=list)  # r, c, true, read
    missed: list[tuple[int, int, int]] = field(default_factory=list)  # r, c, true digit
    hallucinated: list[tuple[int, int, int]] = field(default_factory=list)  # r, c, read digit

    def all_cells(self) -> list[tuple]:
        return [*self.misread, *self.missed, *self.hallucinated]

    def n_corrupted(self) -> int:
        return len(self.misread) + len(self.missed) + len(self.hallucinated)

    def summary(self) -> str:
        return (
            f"{len(self.misread)} misread, {len(self.missed)} missed, "
            f"{len(self.hallucinated)} hallucinated"
        )


def _wrong_digit(true: int, rng: np.random.Generator) -> int:
    """A visually plausible wrong digit: a confusable one 80% of the time."""
    conf = [d for d in CONFUSABLE[true] if d != 0]
    if conf and rng.random() < 0.8:
        return int(rng.choice(conf))
    return int(rng.choice([d for d in range(1, 10) if d != true]))


def _cell_distribution(shown: int, true: int, top: float, rng: np.random.Generator) -> np.ndarray:
    """Distribution with argmax `shown` at probability `top`."""
    p = np.zeros(10, np.float64)
    p[shown] = top
    left = 1.0 - top
    others = [k for k in range(10) if k != shown]
    if true != shown:
        # Runner-up is the true class, capped so it never overtakes `shown`.
        share = min(rng.uniform(0.4, 0.8) * left, 0.95 * top)
        p[true] = share
        left -= share
        others.remove(true)
    else:
        # Correct cell: confusable classes take a bigger slice of the tail.
        conf = [k for k in CONFUSABLE[shown] if k in others]
        if conf:
            share = rng.uniform(0.3, 0.7) * left
            p[conf] += share * rng.dirichlet(np.ones(len(conf)))
            left -= share
    p[others] += left * rng.dirichlet(np.full(len(others), 0.5))
    # Guard against the tail accidentally overtaking the top class.
    if p.argmax() != shown:
        j = p.argmax()
        p[shown], p[j] = p[j], p[shown]
    return p


def simulate_prob_grid(
    givens: np.ndarray,
    error_rate: float = 0.05,
    empty_miss_rate: float = 0.0,
    hallucination_rate: float = 0.0,
    confidence_range: tuple[float, float] = (0.85, 0.999),
    seed: int | None = None,
    corrupted_range: tuple[float, float] = (0.4, 0.9),
) -> tuple[ProbGrid, CorruptionRecord]:
    """Fake CNN output for a puzzle.

    Per given cell (mutually exclusive): misread as another digit with prob
    error_rate, read as empty with prob empty_miss_rate, else correct.
    Per empty cell: read as a digit with prob hallucination_rate.
    """
    validate_grid(givens)
    if error_rate + empty_miss_rate > 1:
        raise ValueError("error_rate + empty_miss_rate must be <= 1")
    rng = np.random.default_rng(seed)
    prob = np.zeros((9, 9, 10), np.float64)
    rec = CorruptionRecord()
    for r in range(9):
        for c in range(9):
            true = int(givens[r, c])
            u = rng.random()
            shown = true
            if true != 0 and u < error_rate:
                shown = _wrong_digit(true, rng)
                rec.misread.append((r, c, true, shown))
            elif true != 0 and u < error_rate + empty_miss_rate:
                shown = 0
                rec.missed.append((r, c, true))
            elif true == 0 and u < hallucination_rate:
                shown = _wrong_digit(0, rng) if rng.random() < 0.5 else int(rng.integers(1, 10))
                rec.hallucinated.append((r, c, shown))
            lo, hi = confidence_range if shown == true else corrupted_range
            prob[r, c] = _cell_distribution(shown, true, rng.uniform(lo, hi), rng)
    prob = prob.astype(np.float32)
    prob /= prob.sum(axis=2, keepdims=True)  # re-normalise after float32 cast
    return validate_prob_grid(prob), rec
