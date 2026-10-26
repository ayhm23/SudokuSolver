"""End to end: image file -> grid corners -> cell images -> probabilities -> solved grid.

The perceiver is any callable CellImages -> ProbGrid. Passing it in (rather
than importing a model here) keeps vision and reasoning testable before the
CNN exists, and lets us swap the CNN, a calibrated CNN or a simulator without
touching this file.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from sudoku_ns.eval.simulate import CorruptionRecord, simulate_prob_grid
from sudoku_ns.interfaces import (
    CellImages,
    ProbGrid,
    SolveResult,
    validate_cell_images,
    validate_prob_grid,
)
from sudoku_ns.reasoning.solver import solve
from sudoku_ns.vision.cell_extract import extract_cells
from sudoku_ns.vision.grid_detect import find_grid_corners

Perceiver = Callable[[CellImages], ProbGrid]


@dataclass
class PipelineOutput:
    corners: np.ndarray  # (4, 2) tl, tr, br, bl
    cells: CellImages
    prob: ProbGrid
    result: SolveResult


class OraclePerceiver:
    """Week 1 stand-in for the CNN: ignores the pixels and returns a simulated
    ProbGrid built from the ground-truth givens (see eval/simulate.py).

    It lets the full pipeline run, and lets us dial in perception error, until
    the real CNN is trained in Week 2. `last_record` keeps what was corrupted.
    """

    def __init__(self, givens: np.ndarray, seed: int | None = None, **simulate_kwargs):
        self.givens = givens
        self.seed = seed
        self.simulate_kwargs = {"error_rate": 0.0} | simulate_kwargs
        self.last_record: CorruptionRecord | None = None

    def __call__(self, cells: CellImages) -> ProbGrid:
        validate_cell_images(cells)
        prob, self.last_record = simulate_prob_grid(
            self.givens, seed=self.seed, **self.simulate_kwargs
        )
        return prob


def run(
    image_path: Path | str, perceiver: Perceiver, require_unique: bool = False
) -> PipelineOutput:
    """Run the whole pipeline on one image. Raises GridNotFoundError if no grid is found."""
    image = np.asarray(Image.open(image_path).convert("RGB"))
    corners = find_grid_corners(image)
    cells = extract_cells(image, corners)
    prob = validate_prob_grid(perceiver(cells))
    return PipelineOutput(corners, cells, prob, solve(prob, require_unique=require_unique))
