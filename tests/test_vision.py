import cv2
import numpy as np
import pytest

from sudoku_ns.interfaces import validate_cell_images
from sudoku_ns.synth.noise import apply_noise, clutter, rotation_matrix
from sudoku_ns.vision.cell_extract import clean_cell, extract_cells
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


def _cell_with(digit_box=None, line=None, side=32):
    c = np.full((side, side), 230, np.uint8)
    if digit_box:
        x0, y0, x1, y1 = digit_box
        c[y0:y1, x0:x1] = 20
    if line == "left":
        c[:, 0:2] = 30
    return c


def test_clean_cell_removes_border_line_and_centres_digit():
    cell = _cell_with(digit_box=(6, 8, 14, 26), line="left")  # digit left of centre
    out = clean_cell(cell)
    assert out[:, 0:2].min() > 200  # remnant erased
    ys, xs = np.nonzero(out < 128)
    assert abs((xs.min() + xs.max() + 1) / 2 - 16) <= 1  # centred horizontally
    assert abs((ys.min() + ys.max() + 1) / 2 - 16) <= 1
    assert len(xs) == 8 * 18  # same digit size, nothing erased from it


def test_clean_cell_never_erases_the_centre():
    # A faint small mark in the middle is not "digit-sized" but must survive:
    # deciding empty vs digit is the CNN's job.
    cell = _cell_with(digit_box=(15, 15, 17, 17))
    cell[15:17, 15:17] = 120
    out = clean_cell(cell)
    assert out[15:17, 15:17].max() < 200


def test_clean_cell_empty_is_paper():
    out = clean_cell(_cell_with(line="left"))
    assert out.min() > 200


def test_extract_cells_debug_image(renders, tmp_path):
    r = renders[1]
    path = tmp_path / "dbg.png"
    extract_cells(r.image, r.corners, debug_path=path)
    assert path.exists() and path.stat().st_size > 0


@pytest.mark.parametrize(
    "spec", [{"gaussian_noise": 1.0}, {"blur": 1.0}, {"lighting": 1.0}, {"rotate": 0.5}]
)
def test_detection_under_noise(renders, spec):
    for i, r in enumerate(renders[:3]):
        img = apply_noise(r.image, spec, seed=i)
        gt = r.corners
        if "rotate" in spec:
            gt = cv2.transform(gt[None], rotation_matrix(r.image.shape, spec["rotate"]))[0]
        err = np.linalg.norm(find_grid_corners(img) - gt, axis=1).mean()
        assert err < 5.0, (spec, i, err)


def test_detection_ignores_enclosing_clutter(renders):
    # Frames drawn around the grid are bigger quadrilaterals than the grid itself.
    for i, r in enumerate(renders[:3]):
        img = clutter(r.image, r.corners, 1.0, seed=i)
        err = np.linalg.norm(find_grid_corners(img) - r.corners, axis=1).mean()
        assert err < 5.0, (i, err)
