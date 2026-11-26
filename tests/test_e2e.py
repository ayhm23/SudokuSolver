"""The end-to-end harness on a tiny rendered set with the oracle perceiver."""

import csv
import json
import subprocess
import sys
from pathlib import Path

from sudoku_ns.synth.render import save_render

ROOT = Path(__file__).resolve().parents[1]


def _run(script, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )  # fmt: skip


def test_run_e2e_is_complete_and_resumable(renders, tmp_path):
    for i, r in enumerate(renders[:3]):
        save_render(r, tmp_path / "ds", f"{i:05d}")
    out = tmp_path / "e2e.csv"
    cfg = {
        "datasets": [str(tmp_path / "ds")],
        "perceiver": {"type": "oracle", "error_rate": 0.1},
        "methods": ["maxsat_weighted", "argmax_then_sat"],
        "out": str(out),
    }
    (tmp_path / "c.json").write_text(json.dumps(cfg), encoding="utf-8")
    _run("run_e2e.py", "--config", tmp_path / "c.json", "--limit", 2)
    with out.open(encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 2 * 2
    _run("run_e2e.py", "--config", tmp_path / "c.json")  # resumes: only image 3 is new
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * 2
    assert len({(r["image"], r["method"]) for r in rows}) == 6
    for r in rows:
        assert (r["correct"] == "True") == (r["failure_stage"] == "")
        assert r["failure_stage"] in ("", "vision", "perception", "reasoning")
    summary = _run("summarize_e2e.py", out).stdout
    assert "maxsat_weighted" in summary and "argmax_then_sat" in summary


def test_run_e2e_resume_drops_partial_last_row(renders, tmp_path):
    for i, r in enumerate(renders[:2]):
        save_render(r, tmp_path / "ds", f"{i:05d}")
    out = tmp_path / "e2e.csv"
    cfg = {
        "datasets": [str(tmp_path / "ds")],
        "perceiver": {"type": "oracle", "error_rate": 0.1},
        "methods": ["maxsat_weighted"],
        "method_kwargs": {"maxsat_weighted": {"scale": 100}},
        "out": str(out),
    }
    (tmp_path / "c.json").write_text(json.dumps(cfg), encoding="utf-8")
    _run("run_e2e.py", "--config", tmp_path / "c.json", "--limit", 1)
    with out.open("a", encoding="utf-8", newline="") as f:
        f.write("ds,00001,maxsat_wei")  # simulate a run killed mid-write
    _run("run_e2e.py", "--config", tmp_path / "c.json")
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [(r["image"], r["method"]) for r in rows] == [
        ("00000", "maxsat_weighted"),
        ("00001", "maxsat_weighted"),
    ]


def test_run_e2e_without_corners_uses_occupancy_check(renders, tmp_path):
    # A real photo from the public dataset has "corners": null. Vision is then
    # judged by the empty / digit pattern of the reads (column occ_disagree).
    for i, r in enumerate(renders[:2]):
        save_render(r, tmp_path / "ds", f"{i:05d}")
        jp = tmp_path / "ds" / f"{i:05d}.json"
        meta = json.loads(jp.read_text(encoding="utf-8"))
        meta["corners"] = None
        jp.write_text(json.dumps(meta), encoding="utf-8")
    out = tmp_path / "e2e.csv"
    cfg = {
        "datasets": [str(tmp_path / "ds")],
        "perceiver": {"type": "oracle", "error_rate": 0.0},
        "methods": ["maxsat_weighted"],
        "save_perception": True,
        "out": str(out),
    }
    (tmp_path / "c.json").write_text(json.dumps(cfg), encoding="utf-8")
    _run("run_e2e.py", "--config", tmp_path / "c.json")
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    for r in rows:
        assert r["corner_err"] == "nan" and r["occ_disagree"] == "0"
        assert r["failure_stage"] != "vision"
    import numpy as np

    d = np.load(tmp_path / "e2e_perception" / "ds" / "00000.npz")
    assert d["prob"].shape == (9, 9, 10) and d["corners"].shape == (4, 2)
