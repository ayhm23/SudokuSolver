import numpy as np
import pytest

from sudoku_ns.interfaces import validate_cell_images, validate_grid, validate_prob_grid


def test_valid_inputs_pass():
    validate_cell_images(np.zeros((81, 32, 32), np.float32))
    p = np.full((9, 9, 10), 0.1, np.float32)
    validate_prob_grid(p)
    validate_grid(np.zeros((9, 9), np.int8))


def test_cell_images_rejects_wrong_range_and_dtype():
    with pytest.raises(ValueError, match="range"):
        validate_cell_images(np.full((81, 32, 32), 2.0, np.float32))
    with pytest.raises(ValueError, match="dtype"):
        validate_cell_images(np.zeros((81, 32, 32), np.float64))


def test_prob_grid_rejects_bad_sum():
    p = np.full((9, 9, 10), 0.1, np.float32)
    p[3, 4, 0] = 0.5
    with pytest.raises(ValueError, match=r"cell \(3, 4\)"):
        validate_prob_grid(p)


def test_grid_rejects_empty_when_solution_expected():
    with pytest.raises(ValueError, match="outside"):
        validate_grid(np.zeros((9, 9), np.int8), allow_empty=False)
    with pytest.raises(ValueError, match="shape"):
        validate_grid(np.zeros((9, 8), np.int8))
