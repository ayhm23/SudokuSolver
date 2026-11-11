"""Summarise run_e2e.py CSVs: one table per dataset, one row per method, plus failure stages.

    python scripts/summarize_e2e.py outputs/e2e/e2e_test_clean_cnn.csv
    python scripts/summarize_e2e.py outputs/e2e/e2e_test_noise_cnn.csv --by-dataset-only
    python scripts/summarize_e2e.py ... --markdown     # tables ready to paste into docs

Columns: n images; % solved correctly; % UNSAT; % ERROR (vision failed);
cell accuracy before (CNN argmax) and after (method's reading), over all 81
cells of images where vision worked; misreads corrected / total misreads;
correct readings broken; median and 95th-percentile solve time; failures by
stage (vision / perception / reasoning, see run_e2e.py).
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

STAGES = ("vision", "perception", "reasoning")


def _f(rows, key):
    return [float(r[key]) for r in rows if r[key] not in ("", "nan")]


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    solved = [r for r in rows if r["status"] != "error"]
    times = _f(solved, "solve_time_s")
    stages = defaultdict(int)
    for r in rows:
        if r["failure_stage"]:
            stages[r["failure_stage"]] += 1
    return {
        "n": n,
        "correct": 100 * sum(r["correct"] == "True" for r in rows) / n,
        "unsat": 100 * sum(r["status"] == "unsat" for r in rows) / n,
        "error": 100 * sum(r["status"] == "error" for r in rows) / n,
        "acc_before": 100 * np.mean(_f(solved, "cell_acc_before")) if solved else float("nan"),
        "acc_after": 100 * np.mean(_f(solved, "cell_acc_after")) if solved else float("nan"),
        "misreads": int(sum(_f(solved, "n_misreads"))),
        "corrected": int(sum(_f(solved, "misreads_corrected"))),
        "broken": int(sum(_f(solved, "correct_overridden"))),
        "t_median_ms": 1000 * np.median(times) if times else float("nan"),
        "t_p95_ms": 1000 * np.percentile(times, 95) if times else float("nan"),
        **{f"fail_{s}": stages[s] for s in STAGES},
    }


HEAD = [
    ("method", "{}"), ("n", "{}"), ("correct %", "{:.1f}"), ("unsat %", "{:.1f}"),
    ("error %", "{:.1f}"), ("cell acc before", "{:.2f}"), ("cell acc after", "{:.2f}"),
    ("fixed / misreads", "{}"), ("broken", "{}"), ("median ms", "{:.0f}"), ("p95 ms", "{:.0f}"),
    ("fail: vision", "{}"), ("perception", "{}"), ("reasoning", "{}"),
]  # fmt: skip


def table(by_method: dict[str, dict], markdown: bool) -> str:
    rows = []
    for m, s in by_method.items():
        vals = [
            m, s["n"], s["correct"], s["unsat"], s["error"], s["acc_before"], s["acc_after"],
            f"{s['corrected']} / {s['misreads']}", s["broken"], s["t_median_ms"], s["t_p95_ms"],
            s["fail_vision"], s["fail_perception"], s["fail_reasoning"],
        ]  # fmt: skip
        rows.append([fmt.format(v) for (_, fmt), v in zip(HEAD, vals, strict=True)])
    names = [h for h, _ in HEAD]
    if markdown:
        lines = ["| " + " | ".join(names) + " |", "|" + "---|" * len(names)]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(lines)
    widths = [max(len(names[i]), *(len(r[i]) for r in rows)) for i in range(len(names))]
    lines = ["  ".join(n.rjust(w) for n, w in zip(names, widths, strict=True))]
    lines += ["  ".join(v.rjust(w) for v, w in zip(r, widths, strict=True)) for r in rows]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("csvs", type=Path, nargs="+")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument(
        "--by-dataset-only", action="store_true", help="one compact row per dataset x method"
    )
    args = ap.parse_args()
    rows = []
    for p in args.csvs:
        with p.open(encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[r["dataset"]][r["method"]].append(r)

    if args.by_dataset_only:
        methods = list(dict.fromkeys(r["method"] for r in rows))
        print(
            ("| dataset | " if args.markdown else "dataset".ljust(26))
            + (
                " | ".join(methods) + " |"
                if args.markdown
                else "".join(m.rjust(20) for m in methods)
            )
        )
        if args.markdown:
            print("|" + "---|" * (len(methods) + 1))
        for ds, by in groups.items():
            cells = [f"{summarise(by[m])['correct']:.0f}%" if m in by else "-" for m in methods]
            if args.markdown:
                print(f"| {ds} | " + " | ".join(cells) + " |")
            else:
                print(ds.ljust(26) + "".join(c.rjust(20) for c in cells))
        print("\n(% of images solved correctly)")
        return

    for ds, by in groups.items():
        print(f"\n## {ds}\n" if args.markdown else f"\n=== {ds} ===")
        print(table({m: summarise(rs) for m, rs in by.items()}, args.markdown))


if __name__ == "__main__":
    main()
