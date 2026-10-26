import numpy as np

from sudoku_ns.interfaces import SolveStatus
from sudoku_ns.pipeline import OraclePerceiver, run
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
