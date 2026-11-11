import numpy as np
import pytest

from sudoku_ns.interfaces import SolveStatus
from sudoku_ns.pipeline import METHODS, OraclePerceiver, run, visualize
from sudoku_ns.synth.render import save_render


def test_pipeline_end_to_end_with_oracle(renders, tmp_path):
    r = renders[1]
    save_render(r, tmp_path, "img")
    out = run(tmp_path / "img.png", OraclePerceiver(r.givens, seed=0))
    assert np.linalg.norm(out.corners - r.corners, axis=1).mean() < 5
    assert out.result.status is SolveStatus.SOLVED
    np.testing.assert_array_equal(out.result.solution, r.solution)


def test_oracle_with_errors_records_them(renders, tmp_path):
    r = renders[2]
    save_render(r, tmp_path, "img")
    oracle = OraclePerceiver(r.givens, seed=3, error_rate=0.2)
    out = run(tmp_path / "img.png", oracle)
    assert oracle.last_record is not None and len(oracle.last_record.misread) > 0
    assert out.result.status is SolveStatus.SOLVED


def test_vision_failure_is_an_error_result_not_a_crash(tmp_path):
    from PIL import Image

    Image.fromarray(np.full((200, 200, 3), 230, np.uint8)).save(tmp_path / "blank.png")
    out = run(tmp_path / "blank.png", OraclePerceiver(np.zeros((9, 9), np.int8)))
    assert out.result.status is SolveStatus.ERROR
    assert out.result.message.startswith("vision:")
    assert out.corners is None and out.prob is None


@pytest.mark.parametrize("method", METHODS)
def test_every_method_runs(renders, tmp_path, method):
    r = renders[1]
    save_render(r, tmp_path, "img")
    out = run(tmp_path / "img.png", OraclePerceiver(r.givens, seed=0), method=method)
    assert out.result.status is SolveStatus.SOLVED
    np.testing.assert_array_equal(out.result.solution, r.solution)


def test_visualize_writes_png(renders, tmp_path):
    r = renders[1]
    save_render(r, tmp_path, "img")
    out = run(tmp_path / "img.png", OraclePerceiver(r.givens, seed=0, error_rate=0.2))
    visualize(r.image, out, tmp_path / "vis.png")
    assert (tmp_path / "vis.png").stat().st_size > 0
