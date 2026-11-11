"""End-to-end evaluation: rendered images -> vision -> perceiver -> every reasoning method.

    python scripts/run_e2e.py --config configs/e2e_test_clean_cnn.json

One CSV row per (dataset, image, method). Vision and perception run ONCE per
image and are shared by all methods, so the methods are compared on exactly
the same probabilities. The run is resumable: rows already in the output CSV
are skipped, and rows are appended image by image, so an interrupted run
continues where it stopped.

Config (JSON) keys:
  name        run name (default: config file stem)
  datasets    list of render directories or globs, e.g. ["data/synthetic/test_noise/*"]
  perceiver   {"type": "cnn", "checkpoint": "...", "use_temperature": true}
              or {"type": "oracle", "error_rate": 0.05, ...simulate_prob_grid kwargs}
  methods     subset of pipeline.METHODS
  tau         threshold for threshold_then_sat (choose it on VAL: scripts/tune_threshold.py)
  noise       optional extra noise applied on load, e.g. {"blur": 0.5} (corners move with it)
  seed        base seed (oracle perceiver: seed + image index; noise: seed + image index)
  out         CSV path (default outputs/e2e/<name>.csv)

failure_stage is the FIRST stage that went wrong, for rows that are not
solved correctly:
  vision      grid not found, or mean corner error >= 5 px
  perception  vision fine, but the CNN argmax misread at least one cell
              (and this method did not manage to repair it)
  reasoning   vision and the CNN argmax were both entirely right, and the
              method still failed (e.g. it overrode a correct reading, or the
              reading was ambiguous)
"""

import argparse
import csv
import glob
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sudoku_ns.eval.metrics import evaluate
from sudoku_ns.interfaces import SolveResult, SolveStatus
from sudoku_ns.pipeline import METHODS, OraclePerceiver, make_method, perceive
from sudoku_ns.synth.noise import apply_noise, noisy_corners
from sudoku_ns.synth.render import load_render
from sudoku_ns.vision.grid_detect import GridNotFoundError

CORNER_TOL = 5.0
FIELDS = [
    "dataset", "image", "method", "status", "correct", "failure_stage", "cell_acc_before",
    "cell_acc_after", "n_misreads", "misreads_corrected", "correct_overridden", "unique",
    "solve_time_s", "perceive_time_s", "corner_err", "noise", "message",
]  # fmt: skip


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text("utf-8"))
    cfg.setdefault("name", path.stem)
    cfg.setdefault("methods", list(METHODS))
    cfg.setdefault("tau", 0.8)
    cfg.setdefault("noise", {})
    cfg.setdefault("seed", 0)
    cfg.setdefault("out", f"outputs/e2e/{cfg['name']}.csv")
    unknown = set(cfg["methods"]) - set(METHODS)
    if unknown:
        raise ValueError(f"unknown methods {sorted(unknown)}")
    return cfg


def dataset_dirs(cfg: dict) -> list[Path]:
    dirs = []
    for pattern in cfg["datasets"]:
        dirs += sorted(Path(p) for p in glob.glob(pattern) if Path(p).is_dir())
    if not dirs:
        raise FileNotFoundError(f"no dataset directories match {cfg['datasets']}")
    return dirs


def make_perceiver(cfg: dict, givens: np.ndarray, index: int, cache: dict):
    spec = dict(cfg["perceiver"])
    kind = spec.pop("type")
    if kind == "oracle":
        return OraclePerceiver(givens, seed=cfg["seed"] + index, **spec)
    if kind == "cnn":
        if "cnn" not in cache:
            from sudoku_ns.perception.perceiver import CNNPerceiver

            cache["cnn"] = CNNPerceiver(spec["checkpoint"], spec.get("use_temperature", True))
        return cache["cnn"]
    raise ValueError(f"unknown perceiver type {kind!r}")


def failure_stage(correct: bool, vision_ok: bool, argmax_ok: bool) -> str:
    if correct:
        return ""
    if not vision_ok:
        return "vision"
    return "perception" if not argmax_ok else "reasoning"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N images per dataset")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        with out.open(encoding="utf-8") as f:
            done = {(r["dataset"], r["image"], r["method"]) for r in csv.DictReader(f)}
    new_file = not out.exists() or out.stat().st_size == 0
    methods = {m: make_method(m, cfg["tau"]) for m in cfg["methods"]}
    cache: dict = {}

    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for d in dataset_dirs(cfg):
            jsons = sorted(d.glob("*.json"))[: args.limit]
            for index, jp in enumerate(tqdm(jsons, desc=d.name)):
                todo = [m for m in methods if (d.name, jp.stem, m) not in done]
                if not todo:
                    continue
                r = load_render(jp)
                image, gt = r.image, r.corners
                if cfg["noise"]:
                    image = apply_noise(r.image, cfg["noise"], seed=cfg["seed"] + index)
                    gt = noisy_corners(r.image.shape, gt, cfg["noise"])
                meta = json.loads(jp.read_text("utf-8"))
                noise = json.dumps(meta.get("noise", {}) | cfg["noise"])
                perceiver = make_perceiver(cfg, r.givens, index, cache)
                t0 = time.perf_counter()
                try:
                    corners, _, prob = perceive(image, perceiver)
                    corner_err = float(np.linalg.norm(corners - gt, axis=1).mean())
                    vision_msg = ""
                except GridNotFoundError as e:
                    prob, corner_err, vision_msg = None, float("nan"), f"vision: {e}"
                t_perceive = time.perf_counter() - t0
                vision_ok = prob is not None and corner_err < CORNER_TOL
                argmax_ok = prob is not None and bool(np.array_equal(prob.argmax(2), r.givens))
                for m in todo:
                    if prob is None:
                        res = SolveResult(SolveStatus.ERROR, message=vision_msg)
                        row = {
                            "status": res.status.value, "correct": False,
                            "cell_acc_before": "", "cell_acc_after": "", "n_misreads": "",
                            "misreads_corrected": "", "correct_overridden": "", "unique": "",
                            "solve_time_s": 0.0,
                        }  # fmt: skip
                    else:
                        res = methods[m](prob)
                        ev = evaluate(res, prob, r.givens, r.solution)
                        row = {k: ev[k] for k in FIELDS if k in ev}
                        row["correct"] = ev["solved_correct"]
                    row |= {
                        "dataset": d.name,
                        "image": jp.stem,
                        "method": m,
                        "failure_stage": failure_stage(bool(row["correct"]), vision_ok, argmax_ok),
                        "perceive_time_s": round(t_perceive, 4),
                        "corner_err": round(corner_err, 3),
                        "noise": noise,
                        "message": res.message or "",
                    }
                    writer.writerow(row)
                f.flush()
    print(f"results: {out}\nsummarise: python scripts/summarize_e2e.py {out}")


if __name__ == "__main__":
    main()
