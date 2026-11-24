"""Why does maxsat_unweighted show 0 misreads fixed? Cell-level check on test_final.

    python scripts/check_unweighted.py

Rebuilds every image's ProbGrid from the logit dump (outputs/e2e/cell_logits_test_final.npz,
same arithmetic as CNNPerceiver), re-runs unweighted_maxsat, and checks that the recomputed
per-image metrics equal the rows in outputs/e2e/e2e_final_noise_cnn.csv. Then it counts what
the method did with each misread cell and prints two worked examples. Read-only analysis:
no method or metric is changed.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from sudoku_ns.eval.metrics import evaluate
from sudoku_ns.reasoning.baselines import unweighted_maxsat
from sudoku_ns.reasoning.encoding import EPS

LOGITS = Path("outputs/e2e/cell_logits_test_final.npz")
MAIN = Path("outputs/e2e/e2e_final_noise_cnn.csv")
RENDERS = Path("data/synthetic/test_final")


def load_cells():
    """(ids, logits (N,81,10), T) in dump order; ids = (set, image)."""
    d = np.load(LOGITS)
    rows = {
        (r["dataset"], r["image"]): r
        for r in csv.DictReader(MAIN.open(encoding="utf-8"))
        if r["method"] == "argmax_then_sat"
    }
    ids = [
        (s.name, jp.stem)
        for s in sorted(RENDERS.iterdir())
        for jp in sorted(s.glob("*.json"))
        if rows[(s.name, jp.stem)]["status"] != "error"  # skipped by the dump too
    ]
    lg = d["logits"].reshape(-1, 81, 10)
    assert len(ids) == len(lg), (len(ids), len(lg))
    return ids, lg, float(d["temperature"])


def prob_grid(logits81: np.ndarray, t: float) -> np.ndarray:
    """Exactly CNNPerceiver.__call__ after the logits."""
    z = torch.from_numpy(logits81).double() / t
    p = torch.softmax(z, dim=1).numpy()
    p = np.maximum(p, EPS)
    p /= p.sum(axis=1, keepdims=True)
    return p.reshape(9, 9, 10).astype(np.float32)


def main() -> None:
    ids, lg, t = load_cells()
    labels = np.load(LOGITS)["labels"].astype(int).reshape(-1, 81)
    pred = lg.argmax(2)
    w = pred != labels
    print(
        f"misread cells in test_final: {int(w.sum())} = "
        f"{int((w & (labels > 0) & (pred > 0)).sum())} substituted (digit -> other digit) + "
        f"{int((w & (labels > 0) & (pred == 0)).sum())} missed (digit -> empty) + "
        f"{int((w & (labels == 0) & (pred > 0)).sum())} hallucinated (empty -> digit)"
    )
    csv_rows = {
        (r["dataset"], r["image"]): r
        for r in csv.DictReader(MAIN.open(encoding="utf-8"))
        if r["method"] == "maxsat_unweighted"
    }
    outcome = Counter()  # what happened to each misread cell
    correct_cell = Counter()  # what happened to each correctly read digit cell
    mismatches, examples = 0, []
    for (s, im), z in zip(ids, lg, strict=True):
        meta = json.loads((RENDERS / s / f"{im}.json").read_text("utf-8"))
        givens, solution = np.array(meta["givens"]), np.array(meta["solution"])
        prob = prob_grid(z, t)
        res = unweighted_maxsat(prob)
        ev = evaluate(res, prob, givens, solution)
        row = csv_rows[(s, im)]
        same = (
            ev["misreads_corrected"] == int(row["misreads_corrected"])
            and ev["correct_overridden"] == int(row["correct_overridden"])
            and str(ev["solved_correct"]) == row["correct"]
        )
        mismatches += not same
        a, read = prob.argmax(2), res.read_givens
        wrong = a != givens
        for r, c in zip(*np.nonzero(wrong), strict=True):
            outcome[
                "kept CNN digit"
                if read[r, c] == a[r, c]
                else "blanked"
                if read[r, c] == 0
                else "other"
            ] += 1
        dig_ok = (~wrong) & (givens > 0)
        correct_cell["kept"] += int((dig_ok & (read == givens)).sum())
        correct_cell["blanked"] += int((dig_ok & (read == 0)).sum())
        correct_cell["changed to another digit"] += int(
            (dig_ok & (read > 0) & (read != givens)).sum()
        )
        if wrong.any() and ev["correct_overridden"] > 0 and len(examples) < 2:
            if not examples or examples[0][0][0] != s:  # two different sets
                examples.append(((s, im), givens, a, read, prob, ev, row))
    print(f"images re-run: {len(ids)}; rows differing from the CSV: {mismatches}")
    print(f"misread cells, by what unweighted MaxSAT read there: {dict(outcome)}")
    print(f"correctly read digit cells: {dict(correct_cell)}")
    for (s, im), givens, a, read, prob, _ev, row in examples:
        print(
            f"\n=== {s}/{im}: CSV misreads_corrected={row['misreads_corrected']}, "
            f"correct_overridden={row['correct_overridden']}, correct={row['correct']}"
        )
        for r, c in zip(*np.nonzero((a != givens) | (read != a)), strict=True):
            print(
                f"  cell ({r},{c}): true {givens[r, c]}, CNN {a[r, c]} "
                f"(p={prob[r, c, a[r, c]]:.3f}), unweighted read {read[r, c]}"
            )


OUT = Path("report/results/unweighted_check.md")
HEADER = """# Unweighted MaxSAT: why 0 misreads are fixed

Generated by `python scripts/check_unweighted.py` (output below, verbatim). Source data:
`outputs/e2e/cell_logits_test_final.npz` (logit dump of the PR #9 runs) and
`outputs/e2e/e2e_final_noise_cnn.csv` (config `configs/e2e_final_noise_cnn.json`).

## Why, from the code

- `reasoning/baselines.py::unweighted_maxsat` adds two hard clauses per cell: a cell whose CNN
  argmax is empty stays empty (`[y(r,c,0)]`), and a cell whose argmax is digit a is read as a or
  as empty (`[y(r,c,a), y(r,c,0)]`). So its reading of every cell is either the CNN argmax or 0.
  It can drop a digit, never write a different one.
- `eval/metrics.py::evaluate` counts a cell as *fixed* (`misreads_corrected`) when
  `argmax != true_givens` and `read_givens == true_givens`, and as *correct overridden* when
  `argmax == true_givens` and `read_givens != true_givens`.
- A misread of a true digit t as a != t can therefore only end up as a or 0, and both differ from
  t: it can never count as fixed. The only misread the method could fix is a hallucination
  (true empty, read as a digit, then dropped to 0). The first lines below show every misread cell
  in test_final is a digit-for-digit substitution, so 0 fixed is forced by construction. It is not a
  metric bug: the metric does what its definition says. Note that it gives no credit for blanking
  a misread, even though blanking removes the clash (see the `blanked` count).
- A correct digit dropped to 0 counts as *correct overridden*. With all weights equal, the solver
  has no reason to drop the misread digit rather than the correct one it clashes with; the
  examples show both happening in the same image.

"""

if __name__ == "__main__":
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main()
    print(buf.getvalue())
    OUT.write_text(HEADER + "```\n" + buf.getvalue() + "```\n", encoding="utf-8")
    print(f"wrote {OUT}")
