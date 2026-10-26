"""Data contracts shared by every module.

Each stage of the pipeline (vision -> perception -> reasoning -> eval) hands
plain numpy arrays to the next one. Keeping them as arrays (not custom classes)
means any team member can produce or inspect them without importing anything,
and the ``validate_*`` functions below are the single place where shapes,
dtypes and value ranges are checked. Call them at module boundaries so a bug
shows up where it was introduced, not three stages later.

Conventions used everywhere:
- Cells are indexed row-major: flat index = 9 * r + c, with r, c in 0..8.
- Label 0 means "empty cell"; 1..9 are digits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# Type aliases. They are documentation only: numpy arrays are not checked by
# the type system, which is why the validate_* functions exist.
CellImages = np.ndarray  # float32, (81, 32, 32), values in [0, 1]
ProbGrid = np.ndarray  # float32, (9, 9, 10), each cell a distribution over {empty, 1..9}
Grid = np.ndarray  # int8, (9, 9), 0 = empty, 1..9 = digit

CELL_SIZE = 32
NUM_CLASSES = 10  # class 0 = empty, 1..9 = digits
PROB_SUM_TOL = 1e-4


def _check_array(x: object, name: str, dtype: type, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise ValueError(f"{name}: expected np.ndarray, got {type(x).__name__}")
    if x.dtype != dtype:
        raise ValueError(f"{name}: expected dtype {np.dtype(dtype)}, got {x.dtype}")
    if x.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {x.shape}")
    return x


def validate_cell_images(x: object) -> CellImages:
    """Check 81 grayscale 32x32 cell crops in [0, 1], row-major."""
    x = _check_array(x, "CellImages", np.float32, (81, CELL_SIZE, CELL_SIZE))
    if not np.all(np.isfinite(x)):
        raise ValueError("CellImages: contains NaN or inf")
    lo, hi = float(x.min()), float(x.max())
    if lo < 0.0 or hi > 1.0:
        raise ValueError(f"CellImages: values must be in [0, 1], got range [{lo:.4f}, {hi:.4f}]")
    return x


def validate_prob_grid(x: object) -> ProbGrid:
    """Check a (9, 9, 10) per-cell probability distribution over {empty, 1..9}."""
    x = _check_array(x, "ProbGrid", np.float32, (9, 9, NUM_CLASSES))
    if not np.all(np.isfinite(x)):
        raise ValueError("ProbGrid: contains NaN or inf")
    if np.any(x < 0):
        r, c, k = np.argwhere(x < 0)[0]
        raise ValueError(f"ProbGrid: negative probability at cell ({r}, {c}), class {k}")
    sums = x.sum(axis=2)
    bad = np.abs(sums - 1.0) > PROB_SUM_TOL
    if np.any(bad):
        r, c = np.argwhere(bad)[0]
        raise ValueError(
            f"ProbGrid: cell ({r}, {c}) sums to {sums[r, c]:.6f}, "
            f"expected 1 (tol {PROB_SUM_TOL}); {int(bad.sum())} bad cell(s) in total"
        )
    return x


def validate_grid(x: object, allow_empty: bool = True) -> Grid:
    """Check a 9x9 int8 grid. Puzzles may contain 0 (empty); solutions may not."""
    x = _check_array(x, "Grid", np.int8, (9, 9))
    lo = 0 if allow_empty else 1
    if np.any((x < lo) | (x > 9)):
        r, c = np.argwhere((x < lo) | (x > 9))[0]
        raise ValueError(f"Grid: value {x[r, c]} at ({r}, {c}) outside [{lo}, 9]")
    return x


class SolveStatus(Enum):
    SOLVED = "solved"
    UNSAT = "unsat"  # no valid Sudoku is consistent with the hard constraints
    ERROR = "error"  # bad input or solver failure


@dataclass
class Override:
    """A cell where the reasoner's reading disagrees with the CNN's argmax.

    cnn_label/cnn_prob: the CNN's top class and its probability.
    final_label/final_prob: what the solver decided the cell shows, and the
    probability the CNN gave to that class (so a viewer can see how much
    the solver had to "pay" to override).
    """

    row: int
    col: int
    cnn_label: int
    cnn_prob: float
    final_label: int
    final_prob: float
    reason: str


@dataclass
class SolveResult:
    status: SolveStatus
    solution: Grid | None = None
    read_givens: Grid | None = None  # what the solver decided the image shows (0 = empty)
    overrides: list[Override] = field(default_factory=list)
    unique: bool | None = None  # None when uniqueness was not checked
    cost: int | None = None  # MaxSAT objective (sum of violated soft weights)
    solve_time_s: float = 0.0
