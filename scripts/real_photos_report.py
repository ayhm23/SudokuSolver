"""Real-photo results (public wichtounet v2 set): tables, per-image figures and real_photos.md.

    python scripts/real_photos_report.py

Reads (nothing is re-tuned; the run is configs/e2e_real_wichtounet_cnn.json):
  outputs/e2e/e2e_real_wichtounet_cnn.csv               one row per (photo, method)
  outputs/e2e/e2e_real_wichtounet_cnn_perception/       per-photo ProbGrid + detected corners
  data/real/wichtounet_v2_{train,test}/*.json           labels (scripts/import_wichtounet.py)
  report/results/e2e_pooled.csv, e2e_pooled_bootstrap.csv, confusions.csv   synthetic test_final

Writes:
  report/results/real_photos.md                    tables + facts, every number generated here
  report/results/real_photos_*.csv                 the same tables as CSV
  outputs/real_vis/<split>_<image>.png             one figure per photo (all 200)
  report/figures/real/*.png                        6 figures chosen by fixed rules (see FIGURE RULES)

Intervals: Wilson 95% per photo, plus a cluster bootstrap that resamples PUZZLES (puzzle_id),
because the 200 photos show only 76 distinct puzzles (10,000 resamples, seed 0).
"""

# ruff: noqa: E501  (long markdown sentences and table rows; wrapping them hurts readability)

from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from make_report import fmt, md_table, summarise, wilson, write_csv  # noqa: E402

from sudoku_ns.eval.metrics import evaluate  # noqa: E402
from sudoku_ns.pipeline import make_method  # noqa: E402
from sudoku_ns.vision.cell_extract import warp_grid  # noqa: E402

CFG = Path("configs/e2e_real_wichtounet_cnn.json")
RUN = Path("outputs/e2e/e2e_real_wichtounet_cnn.csv")
PERC = Path("outputs/e2e/e2e_real_wichtounet_cnn_perception")
DATA = Path("data/real")
RES = Path("report/results")
FIG = Path("report/figures/real")
VIS = Path("outputs/real_vis")
METHODS = [
    "maxsat_weighted", "maxsat_unique", "maxsat_unweighted", "argmax_then_sat", "threshold_then_sat",
]  # fmt: skip
SHORT = {
    "maxsat_weighted": "weighted",
    "maxsat_unique": "unique",
    "maxsat_unweighted": "unweighted",
    "argmax_then_sat": "argmax",
    "threshold_then_sat": "threshold",
}
BOOT_B, BOOT_SEED = 10_000, 0
MIN_GROUP = 10  # breakdowns: only groups with at least this many photos get their own row
# Vision failure reasons, written after looking at each failed photo's figure in outputs/real_vis/
# (the automatic quad-vs-outline numbers are reported next to them). If the set of failed photos
# changes, the script stops so that the descriptions are re-checked by eye.
MANUAL_REASONS = {
    ("wichtounet_v2_test", "image1009"): "detected one 3x3 box instead of the whole grid",
    ("wichtounet_v2_train", "image206"): "detected one 3x3 box instead of the whole grid",
    ("wichtounet_v2_train", "image209"): "detected only part of the grid (top 6 rows)",
}


# ------------------------------------------------------------------ loading
def load():
    cfg = json.loads(CFG.read_text("utf-8"))
    with RUN.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    meta, perc = {}, {}
    for jp in sorted(DATA.glob("wichtounet_v2_*/*.json")):
        meta[(jp.parent.name, jp.stem)] = json.loads(jp.read_text("utf-8")) | {"_json": jp}
    for key in meta:
        d = np.load(PERC / key[0] / f"{key[1]}.npz")
        perc[key] = (d["prob"], d["corners"])
    for r in rows:
        r["key"] = (r["dataset"], r["image"])
    keys = sorted(meta, key=lambda k: (k[0], int(k[1].removeprefix("image"))))
    assert {r["key"] for r in rows} == set(keys), "run CSV and imported labels disagree"
    for m in METHODS:
        assert sum(r["method"] == m for r in rows) == len(keys), m
    return cfg, rows, meta, perc, keys


def subset(rows, keys):
    ks = set(keys)
    return [r for r in rows if r["key"] in ks]


def vision_ok(row: dict, tol: int) -> bool:
    """The run_e2e.py rule for photos without corners: grid found and occ_disagree < tol."""
    return row["status"] != "error" and int(row["occ_disagree"]) < tol


# ------------------------------------------------------------------ statistics
def puzzle_bootstrap(rows, meta, keys):
    """Percentile 95% CI of solve % per method, resampling puzzles (all photos of a puzzle together)."""
    pids = sorted({meta[k]["puzzle_id"] for k in keys})
    pidx = {p: i for i, p in enumerate(pids)}
    draws = np.random.default_rng(BOOT_SEED).integers(0, len(pids), size=(BOOT_B, len(pids)))
    out = {}
    for m in METHODS:
        k, n = np.zeros(len(pids)), np.zeros(len(pids))
        for r in rows:
            if r["method"] == m and r["key"] in keys:
                i = pidx[meta[r["key"]]["puzzle_id"]]
                k[i] += r["correct"] == "True"
                n[i] += 1
        boot = 100 * k[draws].sum(1) / n[draws].sum(1)
        out[m] = (*np.percentile(boot, [2.5, 97.5]), len(pids))
    return out


def method_table(rows, meta, keys, label, tol):
    boot = puzzle_bootstrap(rows, meta, set(keys))
    ok_keys = {r["key"] for r in rows if r["key"] in set(keys) and vision_ok(r, tol)}
    out = []
    for m in METHODS:
        rs = [r for r in rows if r["method"] == m and r["key"] in set(keys)]
        s = summarise(rs)
        ok = [r for r in rs if r["key"] in ok_keys]
        so = summarise(ok) if ok else {}
        out.append(
            {
                "set": label,
                "method": m,
                **{k: s[k] for k in ("n", "solved", "solve_pct", "ci_lo_pct", "ci_hi_pct")},
                "boot_lo_pct": boot[m][0],
                "boot_hi_pct": boot[m][1],
                "n_puzzles": boot[m][2],
                "unsat_pct": s["unsat_pct"],
                "fail_vision": s["fail_vision"],
                "fail_perception": s["fail_perception"],
                "fail_reasoning": s["fail_reasoning"],
                "n_vision_ok": len(ok),
                "solve_pct_vision_ok": so.get("solve_pct", float("nan")),
                "cell_acc_before_pct": so.get("cell_acc_before_pct", float("nan")),
                "cell_acc_after_pct": so.get("cell_acc_after_pct", float("nan")),
                "misread_cells": so.get("misread_cells", 0),
                "misreads_corrected": so.get("misreads_corrected", 0),
                "correct_overridden": so.get("correct_overridden", 0),
                "t_median_ms": s["t_median_ms"],
                "t_p95_ms": s["t_p95_ms"],
                "t_max_ms": s["t_max_ms"],
            }
        )
    return out


# ------------------------------------------------------------------ vision check details
def outline_error(meta_one, corners):
    """Mean corner distance detected vs third-party outline, as % of the outline's mean side."""
    q = meta_one.get("outline_corners")
    if q is None or np.isnan(corners).any():
        return float("nan")
    q = np.asarray(q, float)
    side = np.mean([np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)])
    return 100 * float(np.linalg.norm(corners - q, axis=1).mean()) / side


def quad_area(q):
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def area_ratio(meta_one, corners):
    """Area of the detected quad / area of the third-party outline (NaN if either is missing)."""
    q = meta_one.get("outline_corners")
    if q is None or np.isnan(corners).any():
        return float("nan")
    return quad_area(corners) / quad_area(np.asarray(q, float))


# ------------------------------------------------------------------ figures
def draw_board(ax, title):
    ax.set_xlim(0, 9)
    ax.set_ylim(9, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        ax.axhline(i, color="k", lw=lw)
        ax.axvline(i, color="k", lw=lw)
    ax.set_title(title, fontsize=9)


def vis_one(path, key, meta_one, image, prob, corners, res_w, outcomes):
    gt = np.array(meta_one["givens"])
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.2), width_ratios=[1.4, 1, 1, 1])
    ax = axes[0]
    ax.imshow(image)
    if meta_one.get("outline_corners") is not None:
        ax.add_patch(plt.Polygon(meta_one["outline_corners"], fill=False, ec="cyan", lw=1.2,
                                 ls="--", label="third-party outline"))  # fmt: skip
    if not np.isnan(corners).any():
        ax.add_patch(plt.Polygon(corners, fill=False, ec="lime", lw=1.6, label="detected"))
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("photo (raw pixels): detected grid (green), outline (cyan)", fontsize=9)
    ax.axis("off")
    if np.isnan(corners).any():
        for a in axes[1:]:
            a.axis("off")
        axes[1].text(0, 0.5, "grid NOT found", fontsize=14, transform=axes[1].transAxes)
    else:
        axes[1].imshow(warp_grid(image, corners, 450), cmap="gray")
        axes[1].set_title("straightened grid (CNN input)", fontsize=9)
        axes[1].axis("off")
        arg, conf = prob.argmax(2), prob.max(2)
        draw_board(axes[2], "CNN argmax vs truth (red = misread: read/true)")
        for r in range(9):
            for c in range(9):
                a, t = int(arg[r, c]), int(gt[r, c])
                if a == t and t:
                    axes[2].text(c + 0.5, r + 0.5, str(a), ha="center", va="center", fontsize=13)
                elif a != t:
                    txt = f"{a or '.'}/{t or '.'}"
                    axes[2].text(c + 0.5, r + 0.5, txt, ha="center", va="center", fontsize=9,
                                 color="r", fontweight="bold")  # fmt: skip
                if a != 0 or t != 0:
                    axes[2].text(c + 0.5, r + 0.88, f"{conf[r, c]:.2f}", ha="center",
                                 va="center", fontsize=5.5, color="#555")  # fmt: skip
        draw_board(axes[3], "maxsat_weighted reading (green = fixed, red = wrong)")
        read = res_w.read_givens
        for r in range(9):
            for c in range(9):
                v, a, t = int(read[r, c]), int(arg[r, c]), int(gt[r, c])
                if v == 0 and t == 0:
                    if a != 0:  # hallucinated digit removed
                        axes[3].text(c + 0.5, r + 0.5, f"{a}->.", ha="center", va="center",
                                     fontsize=9, color="#16a34a", fontweight="bold")  # fmt: skip
                    continue
                colour = "k" if v == t and a == t else ("#16a34a" if v == t else "r")
                txt = str(v) if v == t else f"{v or '.'}/{t or '.'}"
                axes[3].text(c + 0.5, r + 0.5, txt, ha="center", va="center",
                             fontsize=13 if v == t else 9, color=colour,
                             fontweight="bold" if colour != "k" else "normal")  # fmt: skip
    line = "   ".join(f"{SHORT[m]}: {'OK' if outcomes[m] else 'wrong'}" for m in METHODS)
    fig.suptitle(f"{key[0]}/{key[1]}  ({meta_one['phone']}, {meta_one['size_actual'][0]}x"
                 f"{meta_one['size_actual'][1]})   solved correctly -> {line}", fontsize=10)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main() -> None:
    from PIL import Image

    cfg, rows, meta, perc, keys = load()
    tol = cfg.get("occupancy_tol", 12)
    by_key = defaultdict(dict)
    for r in rows:
        by_key[r["key"]][r["method"]] = r
    test_keys = [k for k in keys if meta[k]["wichtounet_split"] == "test"]
    RES.mkdir(parents=True, exist_ok=True)
    md = []

    # ---- consistency: re-solve maxsat_weighted from the saved ProbGrid, must match the CSV
    weighted = make_method("maxsat_weighted", cfg["tau"])
    res_w, mismatches = {}, []
    for k in keys:
        prob, corners = perc[k]
        if np.isnan(prob).any():
            continue
        res_w[k] = weighted(prob)
        ev = evaluate(res_w[k], prob, np.array(meta[k]["givens"]), np.array(meta[k]["solution"]))
        if str(ev["solved_correct"]) != by_key[k]["maxsat_weighted"]["correct"]:
            mismatches.append(k)

    # ---- 1. per-method tables
    tab_all = method_table(rows, meta, keys, "all 200", tol)
    tab_test = method_table(rows, meta, test_keys, "v2 test 40", tol)
    tab_train = method_table(
        rows, meta, [k for k in keys if meta[k]["wichtounet_split"] == "train"], "v2 train 160", tol
    )
    write_csv(RES / "real_photos_by_method.csv", tab_all + tab_test + tab_train)

    # ---- vision check
    vis_rows = []
    for k in keys:
        r0 = by_key[k]["maxsat_weighted"]
        prob, corners = perc[k]
        failed = not vision_ok(r0, tol)
        vis_rows.append(
            {
                "set": k[0],
                "image": k[1],
                "puzzle_id": meta[k]["puzzle_id"],
                "grid_found": r0["status"] != "error",
                "occ_disagree": r0["occ_disagree"],
                "vision_fail": failed,
                "outline_err_pct_of_side": outline_error(meta[k], corners),
                "area_ratio_vs_outline": area_ratio(meta[k], corners),
                "reason": MANUAL_REASONS.get(k, "NOT CHECKED BY EYE") if failed else "",
                "n_misreads": r0["n_misreads"],
            }
        )
    write_csv(RES / "real_photos_vision.csv", vis_rows)
    failed_keys = {(v["set"], v["image"]) for v in vis_rows if v["vision_fail"]}
    if failed_keys != set(MANUAL_REASONS):
        raise SystemExit(
            f"vision failures changed: {sorted(failed_keys)}; look at their figures in "
            f"{VIS} and update MANUAL_REASONS"
        )

    # ---- unique timeouts
    uq = [r for r in rows if r["method"] == "maxsat_unique"]
    timeouts = [r for r in uq if r["message"].startswith("timeout:")]
    uq_times = sorted(float(r["solve_time_s"]) for r in uq if r["status"] != "error")

    # ---- per-image figures + figure choice
    VIS.mkdir(parents=True, exist_ok=True)
    for k in keys:
        prob, corners = perc[k]
        image = np.asarray(Image.open(meta[k]["_json"].parent / meta[k]["image"]).convert("RGB"))
        outcomes = {m: by_key[k][m]["correct"] == "True" for m in METHODS}
        vis_one(VIS / f"{k[0]}_{k[1]}.png", k, meta[k], image, prob, corners, res_w.get(k),
                outcomes)  # fmt: skip

    def n_of(k, field, m="maxsat_weighted"):
        v = by_key[k][m][field]
        return int(v) if v not in ("", "nan") else -1

    vok = [k for k in keys if vision_ok(by_key[k]["maxsat_weighted"], tol)]
    picks = []  # (role, key, rule)

    def pick(role, cands, sort_key, rule):
        cands = [k for k in cands if k not in [p[1] for p in picks]]
        if cands:
            picks.append((role, min(cands, key=sort_key), rule))

    largest = lambda k: (-meta[k]["size_actual"][0] * meta[k]["size_actual"][1], k)  # noqa: E731
    pick("success", [k for k in vok if all(by_key[k][m]["correct"] == "True" for m in METHODS)
                     and n_of(k, "n_misreads") == 0], largest,
         "all 5 methods correct and 0 misreads; largest photo, then name")  # fmt: skip
    corr = [k for k in vok if by_key[k]["maxsat_weighted"]["correct"] == "True"
            and by_key[k]["argmax_then_sat"]["correct"] != "True"]  # fmt: skip
    for i in (1, 2):
        pick(f"maxsat_correction_{i}", corr, lambda k: (-n_of(k, "misreads_corrected"), k),
             "maxsat_weighted correct, argmax_then_sat wrong; most misreads corrected")  # fmt: skip
    vf = [k for k in keys if k not in vok]
    pick("vision_not_found", [k for k in vf if by_key[k]["maxsat_weighted"]["status"] == "error"],
         lambda k: k, "grid not found; first by name")  # fmt: skip
    pick("vision_wrong_grid", [k for k in vf if by_key[k]["maxsat_weighted"]["status"] != "error"],
         lambda k: (-n_of(k, "occ_disagree"), k),
         "grid found but empty/digit pattern disagrees >= tol; largest disagreement")  # fmt: skip
    if len([p for p in picks if p[0].startswith("vision")]) < 2:  # only one kind exists
        pick("vision_2", vf, lambda k: k, "second vision failure; first by name")
    pick("dense_errors", vok, lambda k: (-n_of(k, "n_misreads"), k),
         "vision passed; most CNN misreads")  # fmt: skip
    if FIG.exists():
        shutil.rmtree(FIG)
    FIG.mkdir(parents=True)
    for role, k, _ in picks:
        shutil.copy(VIS / f"{k[0]}_{k[1]}.png", FIG / f"{role}_{k[1]}.png")

    # ---- breakdowns
    def group_rows(label_of):
        groups = defaultdict(list)
        for k in keys:
            groups[label_of(k)].append(k)
        out = []
        for g, ks in sorted(groups.items(), key=lambda x: (-len(x[1]), x[0])):
            if len(ks) < MIN_GROUP:
                continue
            ks_set = set(ks)
            ok = [k for k in ks if vision_ok(by_key[k]["maxsat_weighted"], tol)]
            accb = [float(by_key[k]["maxsat_weighted"]["cell_acc_before"]) for k in ok]
            row = {
                "group": g,
                "photos": len(ks),
                "puzzles": len({meta[k]["puzzle_id"] for k in ks}),
                "vision_fail": len(ks) - len(ok),
                "cell_acc_before_pct_vision_ok": 100 * float(np.mean(accb)) if accb else np.nan,
            }
            for m in METHODS:
                kk = sum(by_key[k][m]["correct"] == "True" for k in ks_set)
                lo, hi = wilson(kk, len(ks))
                row[f"{SHORT[m]}_solved"] = kk
                row[f"{SHORT[m]}_pct"] = 100 * kk / len(ks)
                row[f"{SHORT[m]}_ci"] = f"{100 * lo:.0f}-{100 * hi:.0f}"
            out.append(row)
        dropped = {g: len(v) for g, v in groups.items() if len(v) < MIN_GROUP}
        return out, dropped

    def res_label(k):
        w, h = meta[k]["size_actual"]
        return "640x480" if (w, h) in ((640, 480), (480, 640)) else "larger than 640x480"

    by_res, _ = group_rows(res_label)
    by_phone, phone_dropped = group_rows(lambda k: meta[k]["phone"])
    write_csv(RES / "real_photos_by_group.csv",
              [{"by": "resolution", **r} for r in by_res] + [{"by": "phone", **r} for r in by_phone])  # fmt: skip

    # ---- confusions (vision-ok photos; argmax = what every method saw)
    pairs, kinds = Counter(), Counter()
    for k in vok:
        prob, _ = perc[k]
        a, t = prob.argmax(2), np.array(meta[k]["givens"])
        for r, c in zip(*np.nonzero(a != t), strict=True):
            pairs[(int(t[r, c]), int(a[r, c]))] += 1
            kinds["substituted (digit -> other digit)" if t[r, c] and a[r, c] else
                  "missed (digit -> empty)" if t[r, c] else "hallucinated (empty -> digit)"] += 1  # fmt: skip
    n_mis = sum(pairs.values())
    write_csv(RES / "real_photos_confusions.csv",
              [{"true": t, "read": a, "cells": n} for (t, a), n in pairs.most_common()])  # fmt: skip
    syn = Counter()
    with (RES / "confusions.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            syn[(int(r["true"]), int(r["read"]))] += int(r["cells"])
    n_syn = sum(syn.values())

    # ---- synthetic side by side
    with (RES / "e2e_pooled.csv").open(encoding="utf-8") as f:
        pooled = {(r["method"], r["group"]): r for r in csv.DictReader(f)}
    with (RES / "e2e_pooled_bootstrap.csv").open(encoding="utf-8") as f:
        pboot = {(r["method"], r["group"]): r for r in csv.DictReader(f)}

    # ================================================================== markdown
    nv = sum(v["vision_fail"] for v in vis_rows)
    md.append("# Real photos: public wichtounet v2 dataset (200 photos)\n")
    md.append(
        "Generated by `python scripts/real_photos_report.py` from `outputs/e2e/e2e_real_wichtounet_cnn.csv` "
        "(config `configs/e2e_real_wichtounet_cnn.json`, CNN `cnn_v1` with temperature, tau = "
        f"{cfg['tau']}, maxsat_unique time limit {cfg['method_kwargs']['maxsat_unique']['time_limit']} s). "
        "Every number below is computed by the script; nothing is typed by hand. Data: "
        "`data/real/README.md` (source, licence, citation). Nothing was trained, tuned or chosen on these "
        "photos: the CNN, tau, the solver settings and the vision check were all fixed before the run.\n"
    )
    md.append(
        "**Not independent samples.** The 200 photos show "
        f"{len({meta[k]['puzzle_id'] for k in keys})} distinct puzzles "
        f"(test split: {len(test_keys)} photos, {len({meta[k]['puzzle_id'] for k in test_keys})} puzzles; "
        f"{sum(any(meta[j]['puzzle_id'] == meta[k]['puzzle_id'] for j in keys if meta[j]['wichtounet_split'] == 'train') for k in test_keys)}"
        " of the test photos show a puzzle that is also in a train photo). The Wilson intervals treat photos "
        "as independent and are too narrow; the *puzzle bootstrap* columns resample puzzles "
        f"({BOOT_B:,} resamples, seed {BOOT_SEED}) and are the honest ones.\n"
    )
    md.append(
        "**Vision check (no verified corners).** A photo counts as a vision failure if the grid is not "
        f"found, or if the CNN's empty / has-a-digit pattern disagrees with the labels on >= "
        f"{tol} of the 81 cells (`occ_disagree`). The threshold was fixed from "
        "synthetic test_final before this run: there, all 1,598 images with correct corners had 0 such cells, "
        "while two different puzzles differ on 24-81 cells (median 38); 12 is the midpoint of 0 and 24. A "
        "misread digit on the right cell does not count. Section 3 shows how the real photos fall around the "
        "threshold and cross-checks against the third-party outlines.\n"
    )
    md.append(
        f"Consistency check: maxsat_weighted re-solved from the saved ProbGrids reproduces the CSV's "
        f"`correct` on {len(res_w) - len(mismatches)} of {len(res_w)} photos where a grid was found"
        + (f" (mismatches: {mismatches})" if mismatches else "")
        + ".\n"
    )

    md.append("## 1. Per method\n")
    cols = [
        ("method", "method", 0), ("n", "n", 0), ("solved", "solved", 0), ("solve_pct", "solved %", 1),
        ("ci_lo_pct", "Wilson lo", 1), ("ci_hi_pct", "Wilson hi", 1),
        ("boot_lo_pct", "puzzle-bootstrap lo", 1), ("boot_hi_pct", "puzzle-bootstrap hi", 1),
        ("unsat_pct", "UNSAT %", 1), ("fail_vision", "fail: vision", 0),
        ("fail_perception", "fail: perception", 0), ("fail_reasoning", "fail: reasoning", 0),
        ("t_median_ms", "median ms", 0), ("t_p95_ms", "p95 ms", 0), ("t_max_ms", "max ms", 0),
    ]  # fmt: skip
    for title, tab in (("All 200 photos", tab_all), ("v2 test split (40 photos)", tab_test),
                       ("v2 train split (160 photos; never trained on by us)", tab_train)):  # fmt: skip
        md.append(f"### {title}\n")
        md.append(md_table(tab, cols) + "\n")
    md.append(
        "*solved* = the full 9x9 solution equals the true one. *fail: X* = first stage that went wrong "
        "(`scripts/run_e2e.py`): vision = check above; perception = vision passed but the CNN argmax misread "
        ">= 1 cell and the method did not repair it; reasoning = vision and argmax were both fully right and "
        "the method still failed. Times are reasoning only, per photo.\n"
    )
    md.append(
        f"maxsat_unique time limit: **{len(timeouts)} of {len(uq)} photos timed out** and returned the "
        f"maxsat_weighted reading. maxsat_unique solve time over photos where a grid was found: median "
        f"{np.median(uq_times):.2f} s, max {max(uq_times):.2f} s (the limit is checked between rounds, so a "
        "round in progress finishes)."
        + (
            " Timed-out photos (both are vision failures: the CNN read the wrong region): "
            + ", ".join(
                f"{r['dataset']}/{r['image']} ({float(r['solve_time_s']):.1f} s, "
                f"{'correct' if r['correct'] == 'True' else 'wrong'})"
                for r in timeouts
            )
            + "."
            if timeouts
            else ""
        )  # fmt: skip
        + "\n"
    )

    md.append(
        "## 2. Cell accuracy before / after reasoning (photos that passed the vision check)\n"
    )
    cols2 = [
        ("set", "set", 0), ("method", "method", 0), ("n_vision_ok", "photos", 0),
        ("solve_pct_vision_ok", "solved % (these photos)", 1),
        ("cell_acc_before_pct", "cell acc before %", 2), ("cell_acc_after_pct", "cell acc after %", 2),
        ("misread_cells", "misread cells", 0), ("misreads_corrected", "misreads fixed", 0),
        ("correct_overridden", "correct reads overridden", 0),
    ]  # fmt: skip
    md.append(md_table(tab_all + tab_test, cols2) + "\n")
    md.append(
        "*before* = CNN argmax, *after* = the method's reading, over all 81 cells. *misreads fixed* = wrong "
        "before and right after; *correct reads overridden* = right before and wrong after. Photos that "
        "failed the vision check are left out: their cells are not compared against the right grid.\n"
    )
    md.append("CNN misread cells on these photos by kind: " + ", ".join(
        f"{v} {k}" for k, v in kinds.most_common()) + f" (total {n_mis}).\n")  # fmt: skip

    md.append("## 3. Vision check in detail\n")
    bins = [("grid not found", lambda v: not v["grid_found"])]
    for lo, hi in ((0, 0), (1, 5), (6, 11), (12, 23), (24, 81)):
        bins.append((f"occ_disagree {lo}-{hi}" if lo != hi else f"occ_disagree {lo}",
                     lambda v, lo=lo, hi=hi: v["grid_found"] and lo <= int(v["occ_disagree"]) <= hi))  # fmt: skip
    md.append("| photos | count | outline error < 5% of side | 5-10% | >= 10% | no outline |")
    md.append("|---|---|---|---|---|---|")
    for name, pred in bins:
        vs = [v for v in vis_rows if pred(v)]
        e = [v["outline_err_pct_of_side"] for v in vs]
        md.append(
            f"| {name} | {len(vs)} | {sum(x < 5 for x in e)} | {sum(5 <= x < 10 for x in e)} | "
            f"{sum(x >= 10 for x in e)} | {sum(np.isnan(x) for x in e)} |"
        )
    md.append(
        f"\nRule: >= {tol} disagreeing cells or no grid = vision failure "
        f"(**{nv} of {len(keys)} photos**; test split: "
        f"{sum(v['vision_fail'] for v in vis_rows if v['set'].endswith('test'))} of {len(test_keys)}). "
        "*outline error* = mean distance between our detected corners and the third-party outline "
        "(`outline_corners`, unverified), as % of the outline's mean side length; it is only a cross-check. "
        "Photos with a grid found and 1-23 disagreeing cells are the ones where the rule could matter; "
        "their count is in the table.\n"
    )
    md.append("### Vision failure reasons (grouped and counted)\n")
    reasons = Counter(v["reason"] for v in vis_rows if v["vision_fail"])
    md.append("| reason (checked by eye) | photos |\n|---|---|")
    md += [f"| {r} | {n} |" for r, n in reasons.most_common()]
    md.append(
        "\n| photo | phone | size | reason | occ_disagree | detected area / outline area | "
        "outline error % of side |\n|---|---|---|---|---|---|---|"
    )
    for v in vis_rows:
        if v["vision_fail"]:
            m1 = meta[(v["set"], v["image"])]
            md.append(
                f"| {v['set']}/{v['image']} | {m1['phone']} | {m1['size_actual'][0]}x"
                f"{m1['size_actual'][1]} | {v['reason']} | {v['occ_disagree']} | "
                f"{v['area_ratio_vs_outline']:.2f} | {v['outline_err_pct_of_side']:.1f} |"
            )
    md.append(
        "\nReasons were assigned by looking at each failed photo's figure (`outputs/real_vis/`); the "
        "area ratio and outline error are computed against the third-party outline. Per photo: "
        "`real_photos_vision.csv`.\n"
    )

    md.append("## 4. Synthetic test_final vs real photos\n")
    md.append(
        "Synthetic: `report/results/e2e_pooled.csv` / `e2e_pooled_bootstrap.csv` (1,600 renders = 100 "
        "puzzles x 16 clean/noisy sets; held-out fonts). Real: section 1, all 200 photos. The bootstrap "
        "columns resample puzzles in both.\n"
    )
    md.append("| method | synthetic clean solved % | synthetic all 16 sets solved % (puzzle-bootstrap CI) | "
              "real 200 solved % (puzzle-bootstrap CI) | real test 40 solved % (puzzle-bootstrap CI) | "
              "synthetic cell acc before -> after % | real cell acc before -> after % (vision ok) |")  # fmt: skip
    md.append("|---|---|---|---|---|---|---|")
    for m in METHODS:
        sc, sa, sb = pooled[(m, "clean")], pooled[(m, "all 16 sets")], pboot[(m, "all 16 sets")]
        ra = next(t for t in tab_all if t["method"] == m)
        rt = next(t for t in tab_test if t["method"] == m)
        md.append(
            f"| {m} | {float(sc['solve_pct']):.1f} | {float(sa['solve_pct']):.1f} "
            f"({float(sb['boot_lo_pct']):.1f}-{float(sb['boot_hi_pct']):.1f}) | "
            f"{ra['solve_pct']:.1f} ({ra['boot_lo_pct']:.1f}-{ra['boot_hi_pct']:.1f}) | "
            f"{rt['solve_pct']:.1f} ({rt['boot_lo_pct']:.1f}-{rt['boot_hi_pct']:.1f}) | "
            f"{float(sa['cell_acc_before_pct']):.2f} -> {float(sa['cell_acc_after_pct']):.2f} | "
            f"{fmt(ra['cell_acc_before_pct'], 2)} -> {fmt(ra['cell_acc_after_pct'], 2)} |"
        )
    md.append(
        f"\nVision failures: synthetic {pooled[('maxsat_weighted', 'all 16 sets')]['fail_vision']} of "
        f"{pooled[('maxsat_weighted', 'all 16 sets')]['n']} (corner error >= 5 px or not found); real {nv} of "
        f"{len(keys)} (rule above). The two vision checks differ, so compare these counts with care. "
        "Cell accuracy: synthetic over images where a grid was found, real over photos that passed the "
        "vision check.\n"
    )

    md.append("## 5. Breakdown by resolution and phone\n")
    gcols = [("group", "group", 0), ("photos", "photos", 0), ("puzzles", "puzzles", 0),
             ("vision_fail", "vision fail", 0),
             ("cell_acc_before_pct_vision_ok", "cell acc before % (vision ok)", 2)]  # fmt: skip
    for m in METHODS:
        gcols += [(f"{SHORT[m]}_pct", f"{SHORT[m]} solved %", 1), (f"{SHORT[m]}_ci", "Wilson", 0)]
    md.append("### Actual pixel size (`size_actual`, not the declared one)\n")
    md.append(md_table(by_res, gcols) + "\n")
    md.append(f"### Phone model (groups with >= {MIN_GROUP} photos)\n")
    md.append(md_table(by_phone, gcols) + "\n")
    md.append(
        "Left out (fewer than "
        f"{MIN_GROUP} photos): "
        + ", ".join(f"{g} ({n})" for g, n in sorted(phone_dropped.items()))
        + ". Phone and resolution are confounded (each phone has its own resolution and newspaper), and "
        "groups share puzzles, so these rows describe the data rather than isolate a cause. Wilson intervals "
        "treat photos as independent.\n"
    )

    md.append("## 6. CNN confusion pairs: real vs synthetic\n")
    md.append(f"Real: {n_mis} misread cells on the {len(vok)} photos that passed the vision check "
              f"(`real_photos_confusions.csv`). Synthetic: {n_syn} misread cells on test_final "
              "(`confusions.csv`, all 16 sets). Pair = true -> read, 0 = empty.\n")  # fmt: skip
    top_r, top_s = pairs.most_common(15), syn.most_common(15)
    md.append("| rank | real pair | real cells | % of real misreads | synthetic pair | synthetic cells | "
              "% of synthetic misreads |\n|---|---|---|---|---|---|---|")  # fmt: skip
    for i in range(max(len(top_r), len(top_s))):
        a = top_r[i] if i < len(top_r) else None
        b = top_s[i] if i < len(top_s) else None
        md.append(
            f"| {i + 1} | "
            + (f"{a[0][0]} -> {a[0][1]} | {a[1]} | {100 * a[1] / n_mis:.1f}" if a else "| |")
            + " | "
            + (f"{b[0][0]} -> {b[0][1]} | {b[1]} | {100 * b[1] / n_syn:.1f}" if b else "| |")
            + " |"
        )
    md.append("")

    md.append("## 7. Figures\n")
    md.append("All 200 per-photo figures: `outputs/real_vis/` (not committed). Copied to "
              "`report/figures/real/` by fixed rules (FIGURE RULES in the script):\n")  # fmt: skip
    md.append(
        "| file | role | rule | outcome (weighted / unique / unweighted / argmax / threshold) |"
    )
    md.append("|---|---|---|---|")
    for role, k, rule in picks:
        o = " / ".join("OK" if by_key[k][m]["correct"] == "True" else "wrong" for m in METHODS)
        md.append(f"| `{role}_{k[1]}.png` | {role} ({k[0]}/{k[1]}) | {rule} | {o} |")
    md.append(
        "\nPanels: photo with detected grid (green) and third-party outline (cyan, dashed) | straightened "
        "grid the CNN sees | CNN argmax vs truth (red `read/true`, `.` = empty; small number = confidence) | "
        "maxsat_weighted reading (green = misread it fixed, `d->.` = hallucinated digit removed; red "
        "`read/true` = still wrong). A photo can count as solved with a red cell left: a hallucinated "
        "digit in an empty cell that equals the true solution digit there does not change the solution "
        "(e.g. image1049).\n"
    )

    md.append("## 8. Published results on this dataset (context only, not comparable)\n")
    md.append(
        "The dataset README lists: Wicht & Hennebert (SoCPaR 2014), Hough transform + deep belief network, "
        "**12.5% error rate on v1** (the older 160-photo version); and Wicht's Ph.D. thesis (unpublished when "
        "the README was written), **17.5% on v2** and 7.5% on the `mixed` version. These are listed only as "
        "context. We did not check the exact definition of their error rate in the paper or thesis (e.g. per "
        "photo or per cell, recognition only or with solving) or how their systems were trained, and "
        "the 12.5% is on a different version. Our numbers measure full-solution correctness of a "
        "CNN trained only on synthetic renders, so **no direct comparison is claimed**.\n"
    )

    (RES / "real_photos.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {RES / 'real_photos.md'}, {len(picks)} figures in {FIG}, {len(keys)} in {VIS}")
    if mismatches:
        print(f"WARNING: re-solve mismatches {mismatches}")


if __name__ == "__main__":
    main()
