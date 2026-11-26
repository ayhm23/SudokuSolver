"""Week 3 final results: summary CSVs, figures and RESULTS.md from the raw run outputs.

    python scripts/make_report.py

Reads (all produced by the configs in configs/ and configs/ablations/, see configs/README.md):
  outputs/e2e/e2e_final_noise_cnn.csv          main sweep, 4 methods
  outputs/e2e/e2e_final_noise_cnn_unique.csv   maxsat_unique on the same images
  outputs/e2e/abl_*.csv                        ablations (calibration, SCALE, EPS)
  outputs/e2e/cell_logits_test_final.npz       per-cell logits (scripts/dump_cell_logits.py)
  outputs/sim/sweep_simulated_n300_rows.csv    simulated-perception sweep
  outputs/tune_threshold_val_noise.json        tau chosen on noisy VAL

Writes report/results/*.csv (small summaries), report/results/RESULTS.md (tables only, every
number generated here, nothing typed by hand) and report/figures/*.{png,pdf}. A missing input
is reported and its tables/figures are skipped, so the script can be re-run as runs finish.

Confidence intervals: 95% Wilson score intervals for proportions. Pooled rows treat every
image as independent; the 16 test_final sets reuse the same 100 puzzles (paired design),
so pooled intervals are narrower than a puzzle-level analysis would give.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sudoku_ns.perception.calibrate import N_BINS, ece, nll, softmax  # noqa: E402

E2E = Path("outputs/e2e")
RES = Path("report/results")
FIG = Path("report/figures")
MAIN = E2E / "e2e_final_noise_cnn.csv"
UNIQUE = E2E / "e2e_final_noise_cnn_unique.csv"
LOGITS = E2E / "cell_logits_test_final.npz"
SIM_ROWS = Path("outputs/sim/sweep_simulated_n300_rows.csv")
SIM_CFG = "configs/ablations/sweep_simulated_n300.json"
SIM_TAU = json.loads(Path(SIM_CFG).read_text("utf-8"))["tau"]
TAU_JSON = Path("outputs/tune_threshold_val_noise.json")
CFG_MAIN = "configs/e2e_final_noise_cnn.json"
CFG_UNIQUE = "configs/e2e_final_noise_cnn_unique.json"
TAU = json.loads(Path(CFG_MAIN).read_text("utf-8"))["tau"]


def label(m: str) -> str:
    return f"{m} (tau {TAU})" if m == "threshold_then_sat" else m


# Fixed entity -> style mapping, used by every figure. Colours are the first five categorical
# slots of the validated default palette (dataviz skill), in fixed order; marker + line style
# make every series readable in black and white.
METHODS = [
    "maxsat_weighted",
    "maxsat_unweighted",
    "argmax_then_sat",
    "threshold_then_sat",
    "maxsat_unique",
]
STYLE = {
    "maxsat_weighted": ("#2a78d6", "o", "-"),
    "maxsat_unweighted": ("#eb6834", "s", "--"),
    "argmax_then_sat": ("#1baf7a", "^", ":"),
    "threshold_then_sat": ("#eda100", "v", "-."),
    "maxsat_unique": ("#e87ba4", "D", (0, (5, 1, 1, 1))),
}
NOISE = ["blur", "gaussian_noise", "jpeg", "lighting", "rotate"]
SEVERITIES = [0.0, 0.25, 0.5, 1.0]
UNITS = {  # physical meaning of severity s (synth/noise.py)
    "blur": "Gaussian blur sigma = 3s px",
    "gaussian_noise": "pixel noise sigma = 25s (0-255 scale)",
    "jpeg": "JPEG quality = round(95 - 85s)",
    "lighting": "brightness drop up to 0.7s (gradient + shadow)",
    "rotate": "in-plane rotation = 15s degrees",
}
STAGES = ("vision", "perception", "reasoning")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e5e5"

ABLATIONS = {  # name -> (config, csv, parameter, value)
    "weighted_default": ("configs/ablations/weighted_default.json", "abl_weighted_default", "", ""),
    "calib_no_temperature": (
        "configs/ablations/calib_no_temperature.json",
        "abl_calib_no_temperature",
        "temperature",
        "off (T = 1)",
    ),
    "scale_10": ("configs/ablations/scale_10.json", "abl_scale_10", "SCALE", 10),
    "scale_100": ("configs/ablations/scale_100.json", "abl_scale_100", "SCALE", 100),
    "scale_10000": ("configs/ablations/scale_10000.json", "abl_scale_10000", "SCALE", 10000),
    "eps_1e-3": ("configs/ablations/eps_1e-3.json", "abl_eps_1e-3", "EPS", 1e-3),
    "eps_1e-9": ("configs/ablations/eps_1e-9.json", "abl_eps_1e-9", "EPS", 1e-9),
}

missing: list[str] = []


# ------------------------------------------------------------------ helpers


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        missing.append(str(path))
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_set(name: str) -> tuple[str, float]:
    if name == "clean":
        return "clean", 0.0
    kind, sev = name.rsplit("_", 1)
    return kind, float(sev)


def _num(rows, key):
    return [float(r[key]) for r in rows if r[key] not in ("", "nan")]


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    k = sum(r["correct"] == "True" for r in rows)
    lo, hi = wilson(k, n)
    ok = [r for r in rows if r["status"] != "error"]
    t = _num(ok, "solve_time_s")
    stages = defaultdict(int)
    for r in rows:
        stages[r["failure_stage"]] += 1
    return {
        "n": n,
        "solved": k,
        "solve_pct": 100 * k / n if n else float("nan"),
        "ci_lo_pct": 100 * lo,
        "ci_hi_pct": 100 * hi,
        "unsat_pct": 100 * sum(r["status"] == "unsat" for r in rows) / n if n else float("nan"),
        "error_pct": 100 * sum(r["status"] == "error" for r in rows) / n if n else float("nan"),
        "cell_acc_before_pct": 100 * float(np.mean(_num(ok, "cell_acc_before"))),
        "cell_acc_after_pct": 100 * float(np.mean(_num(ok, "cell_acc_after"))),
        "misread_cells": int(sum(_num(ok, "n_misreads"))),
        "misreads_corrected": int(sum(_num(ok, "misreads_corrected"))),
        "correct_overridden": int(sum(_num(ok, "correct_overridden"))),
        "images_with_misread": sum(float(r["n_misreads"]) > 0 for r in ok),
        "t_median_ms": 1000 * float(np.median(t)) if t else float("nan"),
        "t_p95_ms": 1000 * float(np.percentile(t, 95)) if t else float("nan"),
        "t_max_ms": 1000 * float(max(t)) if t else float("nan"),
        **{f"fail_{s}": stages[s] for s in STAGES},
    }


def fmt(v, nd=1):
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        return f"{v:.{nd}f}"
    return str(v)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def md_table(rows: list[dict], cols: list[tuple[str, str, int]]) -> str:
    """cols: (key, header, decimals)."""
    out = ["| " + " | ".join(h for _, h, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt(r[k], nd) for k, _, nd in cols) + " |")
    return "\n".join(out)


def style_axes(ax, ygrid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
    }
)


# ----------------------------------------------------------------- main e2e

CORE_COLS = [
    ("n", "n", 0),
    ("solve_pct", "solved %", 1),
    ("ci_lo_pct", "95% CI lo", 1),
    ("ci_hi_pct", "95% CI hi", 1),
    ("unsat_pct", "UNSAT %", 1),
    ("cell_acc_before_pct", "cell acc before %", 2),
    ("cell_acc_after_pct", "cell acc after %", 2),
    ("misread_cells", "misread cells", 0),
    ("misreads_corrected", "corrected", 0),
    ("correct_overridden", "correct overridden", 0),
    ("t_median_ms", "median ms", 0),
    ("t_p95_ms", "p95 ms", 0),
    ("fail_vision", "fail: vision", 0),
    ("fail_perception", "fail: perception", 0),
    ("fail_reasoning", "fail: reasoning", 0),
]


def e2e_tables(rows: list[dict]):
    by = defaultdict(list)
    for r in rows:
        by[(r["method"], r["dataset"])].append(r)
    methods = [m for m in METHODS if any(k[0] == m for k in by)]
    sets = sorted({k[1] for k in by}, key=lambda s: (parse_set(s)[0] != "clean", s))
    by_set = []
    for m in methods:
        for s in sets:
            kind, sev = parse_set(s)
            by_set.append({"method": m, "noise": kind, "severity": sev, **summarise(by[(m, s)])})
    groups = [("clean", lambda k, s: k == "clean")]
    groups += [(f"severity {s} (5 noise types)", lambda k, s, v=s: s == v) for s in SEVERITIES[1:]]
    groups += [(f"{n} (3 severities)", lambda k, s, v=n: k == v) for n in NOISE]
    groups += [("all 15 noisy sets", lambda k, s: k != "clean"), ("all 16 sets", lambda k, s: True)]
    pooled = []
    for m in methods:
        for g, pred in groups:
            rs = [r for (mm, s), v in by.items() if mm == m and pred(*parse_set(s)) for r in v]
            if rs:
                pooled.append({"method": m, "group": g, **summarise(rs)})
    return methods, by_set, pooled


def fig_headline(by_set, methods):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), sharex=True, sharey=True)
    clean = {r["method"]: r for r in by_set if r["noise"] == "clean"}
    for ax, kind in zip(axes.flat, NOISE, strict=False):
        for m in methods:
            pts = [clean[m]] + sorted(
                (r for r in by_set if r["method"] == m and r["noise"] == kind),
                key=lambda r: r["severity"],
            )
            x = [0.0] + [r["severity"] for r in pts[1:]]
            c, mk, ls = STYLE[m]
            ax.fill_between(
                x,
                [r["ci_lo_pct"] for r in pts],
                [r["ci_hi_pct"] for r in pts],
                color=c,
                alpha=0.12,
                lw=0,
            )
            ax.plot(
                x,
                [r["solve_pct"] for r in pts],
                color=c,
                marker=mk,
                ls=ls,
                lw=1.6,
                ms=5,
                mec="white",
                mew=0.6,
                label=label(m),
            )
        ax.set_title(kind.replace("_", " "), loc="left")
        ax.set_xticks(SEVERITIES)
        ax.set_xticklabels(["0", ".25", ".5", "1"])
        style_axes(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("puzzles solved correctly (%)")
    for ax in axes[1, :]:
        ax.set_xlabel("noise severity (0 = clean)")
    leg = axes.flat[5]
    leg.axis("off")
    h, lab = axes.flat[0].get_legend_handles_labels()
    leg.legend(h, lab, loc="center left", title="method", title_fontsize=8.5)
    n = by_set[0]["n"]
    fig.suptitle(
        f"End-to-end solve rate vs noise severity (n = {n} test images per point, "
        "bands = 95% Wilson CI)",
        x=0.01,
        ha="left",
        fontsize=9,
        color=INK2,
    )
    fig.tight_layout()
    save(fig, "fig1_headline_solve_rate")


def fig_corrections(pooled, methods):
    rows = [r for r in pooled if r["group"] == "all 16 sets"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(rows))
    w = 0.36
    a = ax.bar(
        x - w / 2 - 0.01,
        [r["misreads_corrected"] for r in rows],
        w,
        color="#4d4d4d",
        label="misread cells corrected",
    )
    b = ax.bar(
        x + w / 2 + 0.01,
        [r["correct_overridden"] for r in rows],
        w,
        color="white",
        edgecolor=INK,
        hatch="////",
        lw=0.8,
        label="correct reads wrongly overridden",
    )
    ax.bar_label(a, fontsize=7.5, padding=2, color=INK)
    ax.bar_label(b, fontsize=7.5, padding=2, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([r["method"].replace("_", "\n", 1) for r in rows], fontsize=8)
    ax.set_ylabel("cells (all 16 sets pooled)")
    total = rows[0]["misread_cells"]
    ax.set_title(
        f"Corrections vs wrong overrides ({total} misread cells in total, "
        f"n = {rows[0]['n']} images)",
        loc="left",
        color=INK2,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98))
    ax.set_ylim(0, 1.12 * max(r["misreads_corrected"] for r in rows) + 1)
    style_axes(ax)
    fig.tight_layout()
    save(fig, "fig2_corrections_vs_overrides")


def fig_failure_stages(by_set, methods):
    fig, axes = plt.subplots(1, len(methods), figsize=(7.4, 2.9), sharey=True)
    parts = [  # (key, label, face, hatch)
        ("solved", "solved", "#d9d9d9", ""),
        ("fail_perception", "failed: perception", "#4d4d4d", ""),
        ("fail_reasoning", "failed: reasoning", "white", "xxxx"),
        ("fail_vision", "failed: vision", "black", ""),
    ]
    for ax, m in zip(np.atleast_1d(axes), methods, strict=True):
        bottoms = np.zeros(len(SEVERITIES))
        for key, lab, face, hatch in parts:
            vals = []
            for s in SEVERITIES:
                rs = [r for r in by_set if r["method"] == m and r["severity"] == s]
                n = sum(r["n"] for r in rs)
                vals.append(100 * sum(r[key] for r in rs) / n if n else 0.0)
            ax.bar(
                range(len(SEVERITIES)),
                vals,
                0.7,
                bottom=bottoms,
                color=face,
                hatch=hatch,
                edgecolor=INK,
                lw=0.5,
                label=lab,
            )
            bottoms += vals
        ax.set_title(m.replace("_", " "), fontsize=8)
        ax.set_xticks(range(len(SEVERITIES)))
        ax.set_xticklabels(["0", ".25", ".5", "1"])
        ax.set_xlabel("severity")
        ax.set_ylim(0, 100)
        style_axes(ax, ygrid=False)
    np.atleast_1d(axes)[0].set_ylabel("% of images")
    h, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, "fig5_failure_stage_vs_severity")


def fig_solve_time(rows, methods):
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for m in methods:
        # Vision failures never reach the solver (time 0); excluded, as in the tables.
        t = np.sort(
            [
                1000 * float(r["solve_time_s"])
                for r in rows
                if r["method"] == m and r["status"] != "error"
            ]
        )
        c, mk, ls = STYLE[m]
        y = np.arange(1, len(t) + 1) / len(t)
        ax.plot(t, 100 * y, color=c, ls=ls, lw=1.6, label=label(m))
        i = int(0.95 * (len(t) - 1))
        ax.plot(t[i], 95, marker=mk, color=c, ms=6, mec="white", mew=0.6, ls="none")
    ax.set_xscale("log")
    ax.set_xlabel("solve time per image (ms, log scale)")
    ax.set_ylabel("cumulative % of images")
    ax.axhline(95, color=GRID, lw=0.8, zorder=0)
    ax.set_title(
        "Solve-time distribution, all 16 sets pooled (marker = p95)", loc="left", color=INK2
    )
    ax.legend(loc="lower right")
    style_axes(ax)
    fig.tight_layout()
    save(fig, "fig6_solve_time_distribution")


# ---------------------------------------------------------------- ablations


def ablation_tables(main_rows):
    out, per_type = [], []
    ref = [r for r in main_rows if r["method"] == "maxsat_weighted"]
    sources = {"main sweep, maxsat_weighted (SCALE 1000, EPS 1e-6, T on)": (CFG_MAIN, ref, "", "")}
    for name, (cfg, stem, param, value) in ABLATIONS.items():
        rows = read_rows(E2E / f"{stem}.csv")
        if rows:
            sources[name] = (cfg, rows, param, value)
    for name, (cfg, rows, param, value) in sources.items():
        for g, pred in (
            ("all 16 sets", lambda s: True),
            ("clean", lambda s: s == "clean"),
            ("all 15 noisy sets", lambda s: s != "clean"),
        ):
            rs = [r for r in rows if pred(r["dataset"])]
            if not rs:
                continue
            out.append(
                {
                    "variant": name,
                    "parameter": param,
                    "value": value,
                    "group": g,
                    "config": cfg,
                    **summarise(rs),
                }
            )
        for kind in ["clean", *NOISE]:
            rs = [r for r in rows if parse_set(r["dataset"])[0] == kind]
            if rs:
                per_type.append({"variant": name, "noise": kind, "config": cfg, **summarise(rs)})
    return out, per_type


def determinism_check(main_rows):
    """The default ablation run must reproduce the main sweep's maxsat_weighted rows."""
    other = read_rows(E2E / "abl_weighted_default.csv")
    if not other:
        return None
    key = lambda r: (r["dataset"], r["image"])  # noqa: E731
    a = {key(r): r for r in main_rows if r["method"] == "maxsat_weighted"}
    b = {key(r): r for r in other}
    common = set(a) & set(b)
    diff = [
        k
        for k in common
        if any(a[k][f] != b[k][f] for f in ("correct", "cell_acc_after", "n_misreads", "status"))
    ]
    return {
        "images_compared": len(common),
        "images_differing": len(diff),
        "examples": sorted(diff)[:5],
    }


def fig_calibration(per_type, cells):
    fig = plt.figure(figsize=(7.4, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.6])
    if cells is not None:
        edges = np.linspace(0, 1, N_BINS + 1)
        centres = (edges[:-1] + edges[1:]) / 2
        for i, (t, row) in enumerate(zip((1.0, cells["T"]), cells["rows"], strict=True)):
            ax = fig.add_subplot(gs[0, i])
            probs = softmax(cells["logits"], t)
            conf, ok = probs.max(1), probs.argmax(1) == cells["labels"]
            acc = []
            for lo, hi in zip(edges[:-1], edges[1:], strict=True):
                sel = (conf > lo) & (conf <= hi)
                acc.append(ok[sel].mean() if sel.any() else np.nan)
            ax.bar(centres, acc, 1 / N_BINS, color="#4d4d4d", edgecolor="white", lw=0.5)
            ax.plot([0, 1], [0, 1], ls="--", color=INK2, lw=0.8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("confidence")
            if i == 0:
                ax.set_ylabel("accuracy in bin")
            ax.set_title(f"T = {t:.3f}\nECE {row['ece']:.4f}, NLL {row['nll']:.4f}", fontsize=8)
            low = int((conf <= edges[-2]).sum())
            ax.text(
                0.03,
                0.97,
                f"{low} of {len(conf)} cells\nhave conf. ≤ {edges[-2]:.3f}",
                transform=ax.transAxes,
                va="top",
                fontsize=6.5,
                color=INK2,
            )
            style_axes(ax, ygrid=False)
    ax = fig.add_subplot(gs[0, 2])
    variants = [
        ("main sweep, maxsat_weighted (SCALE 1000, EPS 1e-6, T on)", "T on", "#4d4d4d", ""),
        ("calib_no_temperature", "T off (T = 1)", "white", "////"),
    ]
    kinds = ["clean", *NOISE]
    x = np.arange(len(kinds))
    w = 0.38
    for j, (v, lab, face, hatch) in enumerate(variants):
        rs = {r["noise"]: r for r in per_type if r["variant"] == v}
        if any(k not in rs for k in kinds):  # run incomplete
            continue
        y = [rs[k]["solve_pct"] for k in kinds]
        err = [
            [y[i] - rs[k]["ci_lo_pct"] for i, k in enumerate(kinds)],
            [rs[k]["ci_hi_pct"] - y[i] for i, k in enumerate(kinds)],
        ]
        ax.bar(
            x + (j - 0.5) * w,
            y,
            w * 0.95,
            color=face,
            hatch=hatch,
            edgecolor=INK,
            lw=0.6,
            yerr=err,
            error_kw={"lw": 0.7, "capsize": 1.5, "ecolor": INK2},
            label=lab,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [k.replace("gaussian_noise", "noise") for k in kinds], fontsize=7, rotation=30, ha="right"
    )
    ax.set_ylabel("maxsat_weighted solved (%)")
    ax.set_ylim(0, 118)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(loc="upper center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, 1.0))
    ax.set_title("solve rate with / without T (95% CI)", fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    save(fig, "fig3_calibration")


def fig_sensitivity(abl):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    ref = next(
        (r for r in abl if r["variant"].startswith("main") and r["group"] == "all 16 sets"), None
    )
    for ax, param, default in ((axes[0], "SCALE", 1000), (axes[1], "EPS", 1e-6)):
        pts = [
            (r["value"], r) for r in abl if r["parameter"] == param and r["group"] == "all 16 sets"
        ]
        if ref:
            pts.append((default, ref))
        pts.sort(key=lambda p: p[0])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        y = [p[1]["solve_pct"] for p in pts]
        lo = [p[1]["ci_lo_pct"] for p in pts]
        hi = [p[1]["ci_hi_pct"] for p in pts]
        ax.errorbar(
            xs,
            y,
            yerr=[np.subtract(y, lo), np.subtract(hi, y)],
            color=STYLE["maxsat_weighted"][0],
            marker="o",
            ms=5,
            lw=1.4,
            capsize=2.5,
            mec="white",
            mew=0.6,
        )
        ax.plot(
            [default],
            [ref["solve_pct"]] if ref else [],
            marker="o",
            ms=9,
            mfc="none",
            mec=INK,
            ls="none",
        )
        ax.set_xscale("log")
        ax.set_xlabel(f"{param} (log scale; ring = default {default:g})")
        style_axes(ax)
    axes[0].set_ylabel("maxsat_weighted solved (%)")
    n = ref["n"] if ref else ""
    fig.suptitle(
        f"Weight-encoding sensitivity, all 16 test sets pooled (n = {n}, 95% CI)",
        x=0.01,
        ha="left",
        fontsize=9,
        color=INK2,
    )
    fig.tight_layout()
    save(fig, "fig4_scale_eps_sensitivity")


def cell_calibration():
    if not LOGITS.exists():
        missing.append(str(LOGITS))
        return None
    d = np.load(LOGITS)
    logits, labels, T = d["logits"], d["labels"].astype(int), float(d["temperature"])
    rows = []
    for name, t in (("T = 1 (no scaling)", 1.0), (f"T = {T:.4f} (fitted on val)", T)):
        p = softmax(logits, t)
        rows.append(
            {
                "setting": name,
                "temperature": t,
                "cells": len(labels),
                "accuracy_pct": 100 * float((p.argmax(1) == labels).mean()),
                "mean_conf_pct": 100 * float(p.max(1).mean()),
                "ece": ece(p, labels),
                "nll": nll(logits, labels, t),
            }
        )
    bins = []
    edges = np.linspace(0, 1, N_BINS + 1)
    for name, t in (("T=1", 1.0), ("T=fitted", T)):
        p = softmax(logits, t)
        conf, ok = p.max(1), p.argmax(1) == labels
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            sel = (conf > lo) & (conf <= hi)
            bins.append(
                {
                    "setting": name,
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "cells": int(sel.sum()),
                    "accuracy": float(ok[sel].mean()) if sel.any() else float("nan"),
                    "mean_conf": float(conf[sel].mean()) if sel.any() else float("nan"),
                }
            )
    return {"logits": logits, "labels": labels, "T": T, "rows": rows, "bins": bins}


# ---------------------------------------------------------------- simulated


SIM_NAME = {
    "maxsat": "maxsat_weighted",
    "maxsat_unweighted": "maxsat_unweighted",
    "argmax": "argmax_then_sat",
    "threshold": "threshold_then_sat",
    "maxsat_unique": "maxsat_unique",
}


def sim_table():
    rows = read_rows(SIM_ROWS)
    if not rows:
        return []
    by = defaultdict(list)
    for r in rows:
        by[(SIM_NAME[r["method"]], float(r["error_rate"]))].append(r)
    out = []
    for (m, e), rs in sorted(by.items(), key=lambda kv: (METHODS.index(kv[0][0]), kv[0][1])):
        n = len(rs)
        k = sum(r["solved_correct"] == "True" for r in rs)
        lo, hi = wilson(k, n)
        t = [float(r["solve_time_s"]) for r in rs]
        out.append(
            {
                "method": m,
                "misread_rate_pct": 100 * e,
                "n": n,
                "solve_pct": 100 * k / n,
                "ci_lo_pct": 100 * lo,
                "ci_hi_pct": 100 * hi,
                "unsat_pct": 100 * sum(r["unsat"] == "True" for r in rs) / n,
                "cell_acc_before_pct": 100 * float(np.mean(_num(rs, "cell_acc_before"))),
                "cell_acc_after_pct": 100 * float(np.mean(_num(rs, "cell_acc_after"))),
                "misread_cells": int(sum(_num(rs, "n_misreads"))),
                "misreads_corrected": int(sum(_num(rs, "misreads_corrected"))),
                "correct_overridden": int(sum(_num(rs, "correct_overridden"))),
                "t_median_ms": 1000 * float(np.median(t)),
                "t_p95_ms": 1000 * float(np.percentile(t, 95)),
            }
        )
    return out


def fig_sim(sim):
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for m in METHODS:
        rs = [r for r in sim if r["method"] == m]
        if not rs:
            continue
        c, mk, ls = STYLE[m]
        x = [r["misread_rate_pct"] for r in rs]
        ax.fill_between(
            x, [r["ci_lo_pct"] for r in rs], [r["ci_hi_pct"] for r in rs], color=c, alpha=0.12, lw=0
        )
        ax.plot(
            x,
            [r["solve_pct"] for r in rs],
            color=c,
            marker=mk,
            ls=ls,
            lw=1.6,
            ms=5,
            mec="white",
            mew=0.6,
            label=f"{m} (tau {SIM_TAU})" if m == "threshold_then_sat" else m,
        )
    ax.set_xlabel("simulated misread rate (%)")
    ax.set_ylabel("puzzles solved correctly (%)")
    ax.set_ylim(-2, 102)
    ax.set_title(
        f"Reasoning in isolation (simulated CNN, n = {sim[0]['n']} per point, 95% CI)",
        loc="left",
        color=INK2,
    )
    ax.legend(loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    save(fig, "fig7_simulated_reasoning")


# ---------------------------------------------------------------- RESULTS.md


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    md = [
        "# Week 3 final results",
        "",
        "Generated by `python scripts/make_report.py` from the raw run outputs (`outputs/`, not "
        "committed). Every number below is copied by the script from the CSV named above its "
        "table; nothing is typed by hand. No interpretation here.",
        "",
        "Conventions: *solved %* = full grid equal to the true solution. 95% CI = Wilson score "
        "interval. Pooled rows treat images as independent, but the 16 test_final sets reuse the "
        "same 100 puzzles (paired design), so pooled intervals are optimistic. *cell acc* = mean "
        "over all 81 cells of images where vision worked, before (CNN argmax) and after (the "
        "method's reading). *corrected* = misread before and right after; *correct overridden* = "
        "right before, wrong after. Times are per image, reasoning only (`solve_time_s`). "
        "Failure stage = first stage that went wrong (`scripts/run_e2e.py`).",
        "",
        "Test images: `data/synthetic/test_final/` (TEST fonts, seed 40000, 100 images per set: "
        "clean + 5 noise types x severities 0.25 / 0.5 / 1.0). Severity units:",
        "",
        *[f"- {k}: {v}" for k, v in UNITS.items()],
        "",
    ]

    main_rows = read_rows(MAIN) + read_rows(UNIQUE)
    if main_rows:
        methods, by_set, pooled = e2e_tables(main_rows)
        write_csv(RES / "e2e_by_set.csv", by_set)
        write_csv(RES / "e2e_pooled.csv", pooled)
        src = f"Configs: `{CFG_MAIN}`" + (f", `{CFG_UNIQUE}`" if UNIQUE.exists() else "")
        md += ["## 1. Main sweep", "", src + ".", ""]
        md += ["### 1a. Headline: all 16 sets pooled, clean, all noisy (`e2e_pooled.csv`)", ""]
        head = [r for r in pooled if r["group"] in ("clean", "all 15 noisy sets", "all 16 sets")]
        md += [md_table(head, [("method", "method", 0), ("group", "set", 0), *CORE_COLS]), ""]
        md += ["### 1b. Per severity and per noise type, pooled (`e2e_pooled.csv`)", ""]
        rest = [r for r in pooled if r not in head]
        md += [md_table(rest, [("method", "method", 0), ("group", "set", 0), *CORE_COLS]), ""]
        md += ["### 1c. Every method x noise type x severity (`e2e_by_set.csv`)", ""]
        md += [
            md_table(
                by_set,
                [
                    ("method", "method", 0),
                    ("noise", "noise", 0),
                    ("severity", "severity", 2),
                    *CORE_COLS,
                ],
            ),
            "",
        ]
        md += [
            "Figures: `report/figures/fig1_headline_solve_rate`, `fig2_corrections_vs_overrides`, "
            "`fig5_failure_stage_vs_severity`, `fig6_solve_time_distribution` (.png, .pdf).",
            "",
        ]
        fig_headline(by_set, methods)
        fig_corrections(pooled, methods)
        fig_failure_stages(by_set, methods)
        fig_solve_time(main_rows, methods)
        if UNIQUE.exists():
            md += [
                "Note: the maxsat_unique run (`e2e_final_noise_cnn_unique.json`) ran at the "
                "same time as the ablation runs on this laptop; its times include that "
                "contention. The main sweep (4 methods) ran alone.",
                "",
            ]

    if TAU_JSON.exists():
        tj = json.loads(TAU_JSON.read_text("utf-8"))
        tau_rows = [{"tau": float(t), "val_solved_pct": v} for t, v in tj["solved_pct"].items()]
        write_csv(RES / "tau_selection_val_noise.csv", tau_rows)
        md += [
            "## 2. Ablations",
            "",
            "### 2d. threshold_then_sat: tau chosen on noisy VAL (`tau_selection_val_noise.csv`)",
            "",
            f'`scripts/tune_threshold.py --renders "data/synthetic/val_noise/*"` on {tj["n"]} '
            "VAL-font images (15 noise sets x 50, seed 50000). Rule fixed beforehand: most "
            f"solved, ties to the smaller tau. Chosen: **tau = {tj['best_tau']}**. Test results "
            "at this tau are the `threshold_then_sat` rows of section 1.",
            "",
            md_table(tau_rows, [("tau", "tau", 3), ("val_solved_pct", "VAL solved %", 1)]),
            "",
        ]
    else:
        missing.append(str(TAU_JSON))
        md += ["## 2. Ablations", ""]

    if main_rows:
        abl, per_type = ablation_tables(main_rows)
        write_csv(RES / "ablations.csv", abl)
        write_csv(RES / "ablations_by_noise_type.csv", per_type)
        cols = [
            ("variant", "variant", 0),
            ("group", "set", 0),
            ("config", "config", 0),
            *CORE_COLS[:4],
            *CORE_COLS[7:12],
        ]
        md += [
            "### 2a-c. Calibration, SCALE, EPS (maxsat_weighted only) (`ablations.csv`)",
            "",
            "Timing note: the ablation runs ran 2-5 at a time on the same laptop, so their times "
            "are inflated and not comparable with the main sweep's (`weighted_default` repeats "
            "the main sweep's exact computation). Compare times only within this table, with care.",
            "",
            md_table(abl, cols),
            "",
        ]
        md += [
            "### 2a. Calibration by noise type (`ablations_by_noise_type.csv`)",
            "",
            md_table(
                [r for r in per_type if r["variant"].startswith(("main", "calib"))],
                [
                    ("variant", "variant", 0),
                    ("noise", "noise", 0),
                    *CORE_COLS[:4],
                    *CORE_COLS[7:10],
                ],
            ),
            "",
        ]
        det = determinism_check(main_rows)
        if det:
            md += [
                "Reproducibility check: `weighted_default.json` (SCALE 1000, EPS 1e-6 passed "
                "explicitly) vs the main sweep's maxsat_weighted rows: "
                f"{det['images_differing']} of {det['images_compared']} images differ "
                f"(compared: status, correct, cell_acc_after, n_misreads)"
                + (f"; e.g. {det['examples']}" if det["images_differing"] else "")
                + ".",
                "",
            ]
        cells = cell_calibration()
        if cells:
            write_csv(RES / "calibration_cells.csv", cells["rows"])
            write_csv(RES / "calibration_reliability_bins.csv", cells["bins"])
            md += [
                "### 2a. Cell-level calibration on test_final (`calibration_cells.csv`, "
                "`calibration_reliability_bins.csv`)",
                "",
                "Logits from `scripts/dump_cell_logits.py` (real detection + extraction) on all "
                "16 test_final sets. ECE uses the top-class confidence, "
                f"{N_BINS} equal-width bins.",
                "",
                md_table(
                    cells["rows"],
                    [
                        ("setting", "setting", 0),
                        ("cells", "cells", 0),
                        ("accuracy_pct", "accuracy %", 3),
                        ("mean_conf_pct", "mean confidence %", 3),
                        ("ece", "ECE", 5),
                        ("nll", "NLL", 5),
                    ],
                ),
                "",
            ]
        fig_calibration(per_type, cells)
        fig_sensitivity(abl)
        md += ["Figures: `fig3_calibration`, `fig4_scale_eps_sensitivity`.", ""]

    sim = sim_table()
    if sim:
        write_csv(RES / "simulated_sweep.csv", sim)
        md += [
            "### 2e. Reasoning in isolation: simulated CNN (`simulated_sweep.csv`)",
            "",
            "Config `configs/ablations/sweep_simulated_n300.json` "
            "(`python scripts/sweep_simulated.py --config ...`): generated puzzles 0-299, no "
            "images; misread rate e, missed e/4, hallucinated e/20; threshold tau 0.8 "
            "(the Week 1 value, set from the simulator's design, not tuned on these puzzles).",
            "",
            md_table(
                sim,
                [
                    ("method", "method", 0),
                    ("misread_rate_pct", "misread rate %", 1),
                    ("n", "n", 0),
                    ("solve_pct", "solved %", 1),
                    ("ci_lo_pct", "95% CI lo", 1),
                    ("ci_hi_pct", "95% CI hi", 1),
                    ("unsat_pct", "UNSAT %", 1),
                    ("cell_acc_before_pct", "cell acc before %", 2),
                    ("cell_acc_after_pct", "cell acc after %", 2),
                    ("misread_cells", "misread cells", 0),
                    ("misreads_corrected", "corrected", 0),
                    ("correct_overridden", "correct overridden", 0),
                    ("t_median_ms", "median ms", 0),
                    ("t_p95_ms", "p95 ms", 0),
                ],
            ),
            "",
            "Figure: `fig7_simulated_reasoning`.",
            "",
        ]
        fig_sim(sim)

    if missing:
        md += ["## Missing inputs (skipped)", "", *[f"- `{m}`" for m in missing], ""]
    (RES / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {RES}/RESULTS.md, CSVs and {FIG}/")
    if missing:
        print("missing:", *missing, sep="\n  ")


if __name__ == "__main__":
    main()
