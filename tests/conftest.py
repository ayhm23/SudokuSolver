from pathlib import Path

import matplotlib
import numpy as np
import pytest

from sudoku_ns.reasoning.generator import generate
from sudoku_ns.synth.render import render_puzzle


@pytest.fixture(scope="session")
def test_fonts() -> list[Path]:
    # data/fonts is gitignored and absent in CI; matplotlib (a core dependency)
    # always ships DejaVu Sans, so rendering tests never need a download.
    return [Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"]


@pytest.fixture(scope="session")
def renders(test_fonts):
    """A handful of clean renders shared by synth and vision tests."""
    out = []
    for i in range(6):
        givens, solution = generate(seed=100 + i)
        out.append(render_puzzle(givens, solution, np.random.default_rng(i), test_fonts))
    return out
