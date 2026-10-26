import numpy as np
import pytest

from sudoku_ns.interfaces import validate_cell_images
from sudoku_ns.vision.cell_extract import extract_cells
from sudoku_ns.vision.grid_detect import GridNotFoundError, find_grid_corners, order_corners


def test_order_corners():
    pts = np.array([[90, 95], [10, 12], [12, 88], [95, 8]], np.float32)
    np.testing.assert_array_equal(order_corners(pts), [[10, 12], [95, 8], [90, 95], [12, 88]])


def test_corner_error_on_clean_renders(renders):
    errs = []
    for r in renders:
        corners = find_grid_corners(r.image)
        errs.append(np.linalg.norm(corners - r.corners, axis=1).mean())
    assert np.mean(errs) < 5.0, errs


def test_blank_image_raises_with_reason():
    with pytest.raises(GridNotFoundError, match="contour"):
        find_grid_corners(np.full((200, 200, 3), 230, np.uint8))


def test_paper_without_grid_is_rejected():
    # A light sheet on a dark background is a big quadrilateral but not a grid.
    img = np.full((300, 300, 3), 40, np.uint8)
    img[50:250, 60:240] = 235
    with pytest.raises(GridNotFoundError, match="grid lines"):
        find_grid_corners(img)


def test_extract_cells_contract_and_content(renders):
    r = renders[0]
    cells = extract_cells(r.image, r.corners)
    validate_cell_images(cells)
    # Cells with a digit are darker on average than empty cells.
    dark = 1 - cells.mean(axis=(1, 2))
    labels = r.givens.ravel()
    assert dark[labels > 0].min() > dark[labels == 0].mean()
