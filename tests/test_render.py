import numpy as np

from sudoku_ns.synth.render import load_render, save_render


def test_render_shapes_and_ground_truth(renders):
    for r in renders:
        assert r.image.dtype == np.uint8 and r.image.ndim == 3
        h, w = r.image.shape[:2]
        assert r.corners.shape == (4, 2)
        assert np.all((r.corners >= 0) & (r.corners < [w, h]))
        # Corners are ordered tl, tr, br, bl.
        tl, tr, br, bl = r.corners
        assert tl[0] < tr[0] and bl[0] < br[0] and tl[1] < bl[1] and tr[1] < br[1]
        assert (r.solution[r.givens > 0] == r.givens[r.givens > 0]).all()


def test_corners_sit_on_dark_border(renders):
    # Ground truth claims to be the centre of the outer border line: the pixel
    # there should be ink (dark), unlike the paper just inside the grid.
    for r in renders:
        gray = r.image.mean(axis=2)
        cx, cy = r.corners.mean(axis=0)
        for x, y in r.corners:
            on_line = gray[round(y), round(x)]
            inside = gray[round(y + 0.06 * (cy - y)), round(x + 0.06 * (cx - x))]  # 6% inwards
            assert on_line < inside


def test_save_load_round_trip(renders, tmp_path):
    save_render(renders[0], tmp_path, "00000")
    back = load_render(tmp_path / "00000.json")
    np.testing.assert_array_equal(back.image, renders[0].image)
    np.testing.assert_array_equal(back.givens, renders[0].givens)
    np.testing.assert_allclose(back.corners, renders[0].corners, atol=0.01)
    assert back.font == renders[0].font
