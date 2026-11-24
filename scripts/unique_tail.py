"""maxsat_unique runtime tail on test_final, and what a per-image timeout would change.

    python scripts/unique_tail.py [--slow 10]

Part 1: images whose recorded maxsat_unique solve time (outputs/e2e/e2e_final_noise_cnn_unique.csv)
exceeds --slow seconds are re-solved from their rebuilt ProbGrid (logit dump, same arithmetic as
CNNPerceiver) to recover what the CSV does not record: the MaxSAT cost of the returned reading and
the number of solver calls (RC2 MaxSAT solves and SAT uniqueness checks). The calls are counted by
wrapping the solver classes from this script; solver.py is not modified.

Part 2 (no re-run): for timeouts T, an image whose recorded maxsat_unique time exceeds T is given
the maxsat_weighted outcome on the same image instead. That is what a "stop after T seconds and
fall back" rule would return, because the first round of maxsat_unique is the same MaxSAT solve as
maxsat_weighted. Report only: the default is not changed.

Writes report/results/unique_tail.{csv,md}.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from check_unweighted import MAIN, RENDERS, load_cells, prob_grid  # noqa: E402

import sudoku_ns.reasoning.solver as solver_mod  # noqa: E402
from sudoku_ns.eval.metrics import evaluate  # noqa: E402

UNIQUE = Path("outputs/e2e/e2e_final_noise_cnn_unique.csv")
RES = Path("report/results")
TIMEOUTS = (1, 2, 5, 10, 30, 60)


class Counting:
    calls = {"maxsat": 0, "unique_check": 0}


def instrument():
    rc2_cls, is_unique = solver_mod.RC2Stratified, solver_mod.is_unique

    class CountingRC2(rc2_cls):
        def compute(self, *a, **k):
            Counting.calls["maxsat"] += 1
            return super().compute(*a, **k)

    def counting_is_unique(*a, **k):
        Counting.calls["unique_check"] += 1
        return is_unique(*a, **k)

    solver_mod.RC2Stratified = CountingRC2
    solver_mod.is_unique = counting_is_unique


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(str(v) for v in r) + " |" for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--slow", type=float, default=10.0)
    args = ap.parse_args()
    uq = {(r["dataset"], r["image"]): r for r in csv.DictReader(UNIQUE.open(encoding="utf-8"))}
    mw = {
        (r["dataset"], r["image"]): r
        for r in csv.DictReader(MAIN.open(encoding="utf-8"))
        if r["method"] == "maxsat_weighted"
    }

    # ---- part 1: re-solve the slow images
    instrument()
    ids, lg, t = load_cells()
    where = {k: i for i, k in enumerate(ids)}
    slow = sorted(
        (k for k, r in uq.items() if float(r["solve_time_s"]) > args.slow),
        key=lambda k: -float(uq[k]["solve_time_s"]),
    )
    rows = []
    for s, im in slow:
        meta = json.loads((RENDERS / s / f"{im}.json").read_text("utf-8"))
        givens, solution = np.array(meta["givens"]), np.array(meta["solution"])
        prob = prob_grid(lg[where[(s, im)]], t)
        Counting.calls = {"maxsat": 0, "unique_check": 0}
        t0 = time.perf_counter()
        res = solver_mod.solve(prob, require_unique=True)
        rerun_s = time.perf_counter() - t0
        ev = evaluate(res, prob, givens, solution)
        calls = dict(Counting.calls)  # before the extra maxsat_weighted solve below
        weighted = solver_mod.solve(prob)
        rec = uq[(s, im)]
        rows.append(
            {
                "set": s,
                "image": im,
                "noise": json.dumps(meta["noise"]),
                "font": meta["font"],
                "misread_cells": int(rec["n_misreads"]),
                "csv_time_s": round(float(rec["solve_time_s"]), 1),
                "rerun_time_s": round(rerun_s, 1),
                "maxsat_calls": calls["maxsat"],
                "unique_checks": calls["unique_check"],
                "cost_returned": res.cost,
                "cost_maxsat_weighted": weighted.cost,
                "unique": res.unique,
                "correct": ev["solved_correct"],
                "csv_correct": rec["correct"] == "True",
                "maxsat_weighted_correct": mw[(s, im)]["correct"] == "True",
            }
        )
    with (RES / "unique_tail.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- part 2: timeout what-if from the recorded data
    n = len(uq)
    base = sum(r["correct"] == "True" for r in uq.values())
    what_if = []
    for to in TIMEOUTS:
        over = [k for k, r in uq.items() if float(r["solve_time_s"]) > to]
        gained = sum(uq[k]["correct"] != "True" and mw[k]["correct"] == "True" for k in over)
        lost = sum(uq[k]["correct"] == "True" and mw[k]["correct"] != "True" for k in over)
        what_if.append(
            [
                to,
                len(over),
                gained,
                lost,
                base,
                base + gained - lost,
                n,
                f"{100 * (base + gained - lost) / n:.2f}",
            ]
        )

    md = [
        "# maxsat_unique: runtime tail and a timeout what-if",
        "",
        "Generated by `python scripts/unique_tail.py` (data: `unique_tail.csv`). Recorded times "
        "come from `outputs/e2e/e2e_final_noise_cnn_unique.csv` (config "
        "`configs/e2e_final_noise_cnn_unique.json`); that run shared the CPU with the ablation "
        "runs, so its times are inflated. *rerun time* is a fresh single-image re-solve in this "
        "script, for reference.",
        "",
        f"## Images over {args.slow:g} s",
        "",
        "*MaxSAT calls*: RC2 solves (one per round; each blocked ambiguous reading "
        "forces another). "
        "*uniqueness checks*: SAT checks for a second solution. *cost returned*: MaxSAT objective "
        "of the reading maxsat_unique returned; *cost weighted*: of the first-round (= "
        "maxsat_weighted) reading. *correct (rerun)* must equal *correct (CSV)*.",
        "",
        md_table(
            [
                "set",
                "image",
                "font",
                "misread cells",
                "time CSV (s)",
                "time rerun (s)",
                "MaxSAT calls",
                "uniqueness checks",
                "cost returned",
                "cost weighted",
                "unique",
                "correct (rerun)",
                "correct (CSV)",
                "maxsat_weighted correct",
            ],
            [
                [
                    r["set"],
                    r["image"],
                    r["font"],
                    r["misread_cells"],
                    r["csv_time_s"],
                    r["rerun_time_s"],
                    r["maxsat_calls"],
                    r["unique_checks"],
                    r["cost_returned"],
                    r["cost_maxsat_weighted"],
                    r["unique"],
                    r["correct"],
                    r["csv_correct"],
                    r["maxsat_weighted_correct"],
                ]
                for r in rows
            ],
        ),
        "",
        "## Timeout what-if (no re-run)",
        "",
        "If maxsat_unique stopped after T seconds and returned the maxsat_weighted answer for that "
        "image: *gained* = images maxsat_unique got wrong but maxsat_weighted got right; *lost* = "
        "the reverse. Uses the recorded (contended) times, so the set of images over T on an idle "
        "machine could be smaller.",
        "",
        md_table(
            [
                "timeout T (s)",
                "images over T",
                "gained",
                "lost",
                "solved now",
                "solved with timeout",
                "images",
                "solved % with timeout",
            ],
            what_if,
        ),
        "",
    ]
    (RES / "unique_tail.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
