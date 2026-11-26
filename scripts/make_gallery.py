"""Failure and success gallery: real examples from the evaluation runs, one figure each.

    python scripts/make_gallery.py

Every example is a REAL image from an existing run; nothing is hand-made:
  outputs/e2e/e2e_final_noise_cnn.csv          synthetic test_final (seed 40000), 4 methods
  outputs/e2e/e2e_final_noise_cnn_unique.csv   maxsat_unique on the same images
  outputs/e2e/e2e_real_wichtounet_cnn.csv      200 real photos (public wichtounet v2), 5 methods

How an example is chosen (all rules are in CATEGORIES below, applied in a fixed order):
1. The CSVs give candidate images for a category (e.g. "maxsat_weighted correct AND
   argmax_then_sat UNSAT").
2. Each candidate is RE-RUN exactly the way scripts/run_e2e.py ran it: same image loader
   (load_render: raw pixels, no EXIF rotation), same CNN checkpoint and temperature, same
   tau and solver options from the run's config. The CSVs do not store override
   explanations, so this re-run is where the explanation text comes from.
3. The re-run must REPRODUCE the CSV row (status, correct, n_misreads, misreads_corrected,
   correct_overridden) for every method shown; otherwise the candidate is skipped and the
   skip is logged in gallery.md.
4. The overrides are then classified against the ground truth (e.g. "missed digit
   recovered" = true digit, CNN read empty, solver put the true digit back) and the
   candidate is kept only if it shows what the category is about.

Writes report/figures/gallery/<name>.{png,pdf} (300 dpi, same style as make_report.py)
and report/results/gallery.md. A category with no example is reported as such.
"""

from __future__ import annotations

import csv
import json
import math
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from sudoku_ns.eval.metrics import evaluate  # noqa: E402
from sudoku_ns.interfaces import SolveResult, SolveStatus  # noqa: E402
from sudoku_ns.pipeline import make_method, perceive  # noqa: E402
from sudoku_ns.reasoning.encoding import weight  # noqa: E402
from sudoku_ns.reasoning.explain import FALLBACK_REASON, describe_override  # noqa: E402
from sudoku_ns.synth.render import load_render  # noqa: E402
from sudoku_ns.vision.cell_extract import overlay  # noqa: E402
from sudoku_ns.vision.grid_detect import GridNotFoundError  # noqa: E402

E2E = Path("outputs/e2e")
FIG = Path("report/figures/gallery")
OUT_MD = Path("report/results/gallery.md")
SYN_ROOT = Path("data/synthetic/test_final")
REAL_ROOT = Path("data/real")
RUNS = {  # csv -> config it was produced with
    E2E / "e2e_final_noise_cnn.csv": Path("configs/e2e_final_noise_cnn.json"),
    E2E / "e2e_final_noise_cnn_unique.csv": Path("configs/e2e_final_noise_cnn_unique.json"),
    E2E / "e2e_real_wichtounet_cnn.csv": Path("configs/e2e_real_wichtounet_cnn.json"),
}
CHECK = ("status", "correct", "n_misreads", "misreads_corrected", "correct_overridden")
MAX_TRIES = 40  # candidates re-run per category before giving up

# Style shared with scripts/make_report.py (same ink colours, method colours, fonts).
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e5e5"
BLUE, GREEN, RED = "#2a78d6", "#1baf7a", "#d03b3b"
CONF_CMAP = LinearSegmentedColormap.from_list("conf", ["#eda100", "#fbe7b5", "#ffffff"])
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


# ------------------------------------------------------------------ data
def load_runs() -> tuple[dict, dict]:
    """rows[(dataset, image)][method] = csv row; cfg_of[(dataset, image)] = config dict."""
    rows: dict = {}
    cfg_of: dict = {}
    for path, cfg_path in RUNS.items():
        cfg = json.loads(cfg_path.read_text("utf-8"))
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (r["dataset"], r["image"])
                rows.setdefault(key, {})[r["method"]] = r
                cfg_of.setdefault(key, {"tau": cfg["tau"], "method_kwargs": {}})
                cfg_of[key]["method_kwargs"] |= cfg.get("method_kwargs", {})
    return rows, cfg_of


def is_real(dataset: str) -> bool:
    return dataset.startswith("wichtounet")


def json_path(dataset: str, image: str) -> Path:
    return (REAL_ROOT if is_real(dataset) else SYN_ROOT) / dataset / f"{image}.json"


@dataclass
class Case:
    category: str
    dataset: str
    image: str
    noise: str
    image_rgb: np.ndarray
    givens: np.ndarray
    solution: np.ndarray
    gt_corners: np.ndarray
    corners: np.ndarray | None
    prob: np.ndarray | None
    vision_msg: str
    results: dict[str, SolveResult]
    csv_rows: dict[str, dict]
    show: str  # method whose final grid is drawn
    notes: list[str] = field(default_factory=list)


class Mismatch(Exception):
    pass


LOGITS = E2E / "cell_logits_test_final.npz"  # synthetic, scripts/dump_cell_logits.py
REAL_PROBS = E2E / "e2e_real_wichtounet_cnn_perception"  # real, run_e2e.py save_perception


def census(rows) -> dict:
    """(n_missed, n_hallucinated) CNN-argmax cells per image, from SAVED perception outputs.

    missed = true digit read as empty; hallucinated = empty cell read as a digit.
    Synthetic: the per-cell logit dump (81 rows per image, in sorted order, images whose
    grid was not found skipped); checked cell by cell against the JSON labels. Real: the
    ProbGrids saved by the run. Used to search categories (b) and (c) exhaustively
    instead of re-running every image.
    """
    out = {}
    z = np.load(LOGITS)
    arg, lab, ds = z["logits"].argmax(1), z["labels"], z["dataset"]
    for d in sorted(set(ds)):
        keys = sorted(k for k in rows if k[0] == d and rows[k][W]["status"] != "error")
        a, t = arg[ds == d].reshape(-1, 9, 9), lab[ds == d].reshape(-1, 9, 9)
        if len(keys) != len(a):
            raise SystemExit(f"{LOGITS}: {d} has {len(a)} images, CSV has {len(keys)} with a grid")
        for k, ai, ti in zip(keys, a, t, strict=True):
            g = load_render(json_path(*k)).givens
            if not np.array_equal(g, ti):
                raise SystemExit(f"{LOGITS}: labels do not match {k}; order assumption broken")
            out[k] = (int(((ti > 0) & (ai == 0)).sum()), int(((ti == 0) & (ai > 0)).sum()))
    for p in sorted(REAL_PROBS.glob("*/*.npz")):
        k = (p.parent.name, p.stem)
        prob = np.load(p)["prob"]
        if np.isnan(prob).all():
            continue
        a, g = prob.argmax(2), load_render(json_path(*k)).givens
        out[k] = (int(((g > 0) & (a == 0)).sum()), int(((g == 0) & (a > 0)).sum()))
    return out


CENSUS: dict = {}


def rerun(key, methods, rows, cfg_of, show, category) -> Case:
    """Re-run one image like run_e2e.py did and check it reproduces the CSV."""
    ds, img = key
    r = load_render(json_path(ds, img))
    cfg = cfg_of[key]
    try:
        corners, _, prob = perceive(r.image, cnn())
        vision_msg = ""
    except GridNotFoundError as e:
        corners, prob, vision_msg = None, None, f"vision: {e}"
    results = {}
    notes = []

    def solve_and_check(m, kwargs):
        res = make_method(m, cfg["tau"], **kwargs)(prob)
        ev = evaluate(res, prob, r.givens, r.solution)
        got = {k: str(ev[k]) for k in CHECK if k != "correct"}
        got["correct"] = str(ev["solved_correct"])
        return res, got

    for m in methods:
        row = rows[key][m]
        if prob is None:
            res = SolveResult(SolveStatus.ERROR, message=vision_msg)
            got = {"status": "error", "correct": "False"}
            want = {k: row[k] for k in got}
        else:
            kwargs = cfg["method_kwargs"].get(m, {})
            res, got = solve_and_check(m, kwargs)
            want = {k: row[k] for k in CHECK}
            if got != want and "time_limit" in kwargs and not row["message"]:
                # maxsat_unique's time limit is wall-clock, its rounds are deterministic: if
                # the original run finished without timing out, the same rounds without the
                # limit reproduce it even when this machine is slower now.
                res, got = solve_and_check(
                    m, {k: v for k, v in kwargs.items() if k != "time_limit"}
                )
                if got == want:
                    notes.append(
                        f"{m} re-run without its {kwargs['time_limit']} s time limit: the original "
                        f"run finished in {float(row['solve_time_s']):.1f} s without timing out, "
                        "this re-run is slower, so with the limit it would have timed out."
                    )
        if got != want:
            raise Mismatch(f"{ds}/{img} {m}: re-run {got} != csv {want}")
        if prob is None and row["message"] != vision_msg:
            raise Mismatch(f"{ds}/{img}: vision message differs: {vision_msg!r}")
        results[m] = res
    noise = rows[key][methods[0]]["noise"]
    return Case(
        category,
        ds,
        img,
        "none (real photo)" if is_real(ds) else noise,
        r.image,
        r.givens,
        r.solution,
        r.corners,
        corners,
        prob,
        vision_msg,
        results,
        {m: rows[key][m] for m in methods},
        show,
        notes,
    )


_CNN = {}


def cnn():
    if "m" not in _CNN:
        from sudoku_ns.perception.perceiver import CNNPerceiver

        spec = json.loads(next(iter(RUNS.values())).read_text("utf-8"))["perceiver"]
        _CNN["m"] = CNNPerceiver(spec["checkpoint"], spec.get("use_temperature", True))
    return _CNN["m"]


# --------------------------------------------------------- classification
def override_kinds(case: Case, method: str) -> list[tuple[str, object]]:
    """(kind, Override) for each override of `method`, judged against ground truth."""
    out = []
    g = case.givens
    for o in case.results[method].overrides:
        t, a, f = int(g[o.row, o.col]), o.cnn_label, o.final_label
        if a == t:
            kind = "correct read overridden"
        elif f == t and t > 0 and a > 0:
            kind = "misread corrected"
        elif f == t and t > 0 and a == 0:
            kind = "missed digit recovered"
        elif f == t == 0:
            kind = "hallucination removed"
        else:
            kind = "misread changed to another wrong value"
        out.append((kind, o))
    return out


def has_kind(case, method, kind, core=None) -> bool:
    """core: None = any reason; 'direct' = 1-given unsat core; 'indirect' = larger core."""
    for k, o in override_kinds(case, method):
        if k != kind:
            continue
        if core is None:
            return True
        if o.reason == FALLBACK_REASON:
            continue
        direct = o.reason.startswith("conflicts with given")
        if (core == "direct") == direct:
            return True
    return False


def reading_cost(prob: np.ndarray, reading: np.ndarray, cells) -> list[tuple]:
    """Per-cell MaxSAT weights (encoding.weight) of a reading, for the listed cells."""
    return [
        (
            r,
            c,
            int(reading[r, c]),
            float(prob[r, c, reading[r, c]]),
            weight(float(prob[r, c, reading[r, c]])),
        )
        for r, c in cells
    ]


def cost_explanation(case: Case, method: str) -> list[str]:
    """Why the solver's reading beat the TRUE reading: compare their soft-clause costs.

    The true givens are a consistent reading (they have the true solution), so the
    solver only picks something else if it is no more expensive. Only cells where
    the two readings differ contribute to the difference (per-cell shifts cancel).
    """
    res = case.results[method]
    diff = list(zip(*np.nonzero(res.read_givens != case.givens), strict=True))
    chosen = reading_cost(case.prob, res.read_givens, diff)
    true = reading_cost(case.prob, case.givens, diff)
    lines = [
        "cells where the solver's reading differs from the true givens "
        "(weight = round(1000 * -ln p), reasoning/encoding.py):"
    ]
    for (r, c, kc, pc, wc), (_, _, kt, pt, wt) in zip(chosen, true, strict=True):
        lines.append(
            f"  ({r}, {c}): solver {kc or 'empty'} p={pc:.4f} weight {wc}  |  "
            f"truth {kt or 'empty'} p={pt:.4f} weight {wt}"
        )
    sc, st = sum(x[4] for x in chosen), sum(x[4] for x in true)
    lines.append(
        f"  total over these cells: solver's reading {sc}, true reading {st} "
        f"(difference {st - sc} in the solver's favour)"
    )
    return lines


# -------------------------------------------------------------- selection
def csv_true(row, k):
    return row.get(k) == "True"


def num(row, k):
    v = row.get(k, "")
    return float(v) if v not in ("", "nan") else math.nan


@dataclass
class Category:
    key: str
    title: str
    want: str  # what the category shows, for gallery.md
    real: bool | None  # None = search synthetic and real
    methods: tuple[str, ...]
    show: str
    n: int
    filt: object  # (key, rows_by_method) -> bool
    order: object  # (key, rows_by_method) -> sort key
    accept: object  # Case -> bool
    fixed: tuple = ()  # (dataset, image) chosen by the user, not searched


W, A, U = "maxsat_weighted", "argmax_then_sat", "maxsat_unique"
SEV = {"0.25": 0, "0.5": 1, "1.0": 2}


def by_simplicity(k, m):  # fewest misreads first, then set name, then image id
    return (num(m[W], "n_misreads"), k[0], k[1])


CATEGORIES = [
    Category(
        "a",
        "MaxSAT corrects a misread where argmax_then_sat is UNSAT",
        "maxsat_weighted solved correctly; argmax_then_sat UNSAT on the same reading; the "
        "fixed cell was a wrong digit (not an empty / extra one) and its explanation is an "
        "unsat core. One example with a direct (1-given) core, one with a larger core.",
        False,
        (W, A),
        W,
        2,
        lambda k, m: (
            csv_true(m[W], "correct")
            and m[A]["status"] == "unsat"
            and num(m[W], "misreads_corrected") >= 1
        ),
        by_simplicity,
        None,  # set below: two sub-rules (direct, indirect)
    ),
    Category(
        "b",
        "MaxSAT recovers a missed digit (given read as empty)",
        "the solver put back a true digit the CNN read as EMPTY. With plain MAP "
        "(maxsat_weighted) this cannot happen: an empty reading adds no constraint, so "
        "keeping it is never more expensive (reasoning/explain.py). It can only happen with "
        "maxsat_unique, which rejects readings that leave the puzzle ambiguous. Searched "
        "exhaustively: every image (synthetic and real) whose CNN argmax missed a digit and "
        "where maxsat_weighted or maxsat_unique corrected at least one misread.",
        None,
        (W, U),
        U,
        1,
        lambda k, m: (
            U in m
            and CENSUS.get(k, (0, 0))[0] >= 1
            and max(num(m[W], "misreads_corrected"), num(m[U], "misreads_corrected")) >= 1
        ),
        lambda k, m: (num(m[U], "solve_time_s"), k),
        lambda c: (
            has_kind(c, U, "missed digit recovered") or has_kind(c, W, "missed digit recovered")
        ),
    ),
    Category(
        "c",
        "MaxSAT removes a hallucinated digit",
        "the CNN read a digit in an EMPTY cell; maxsat_weighted read the cell as empty and "
        "solved correctly. Searched over every image (synthetic and real) whose CNN argmax "
        "has at least one such cell; v2 test split first, then fewest misreads.",
        None,
        (W, A),
        W,
        1,
        lambda k, m: CENSUS.get(k, (0, 0))[1] >= 1 and csv_true(m[W], "correct"),
        lambda k, m: (k[0] != "wichtounet_v2_test", num(m[W], "n_misreads"), k),
        lambda c: has_kind(c, W, "hallucination removed"),
    ),
    Category(
        "d",
        "MaxSAT wrongly overrides a correct read",
        "maxsat_weighted changed a cell the CNN had read CORRECTLY. The explanation below "
        "is the system's override reason, plus the soft-clause cost of the solver's reading "
        "vs the true reading (computed by this script from the CNN probabilities with "
        "reasoning/encoding.weight).",
        False,
        (W, A),
        W,
        1,
        lambda k, m: num(m[W], "correct_overridden") >= 1,
        lambda k, m: (num(m[W], "correct_overridden") + num(m[W], "n_misreads"), k),
        lambda c: has_kind(c, W, "correct read overridden"),
    ),
    Category(
        "e",
        "Ambiguous reading gives a non-unique solution",
        "argmax_then_sat found a solution, but the reading it solved has more than one "
        "solution (unique=False), so the solution it returned is one of several.",
        False,
        (A, W, U),
        A,
        1,
        lambda k, m: m[A]["status"] == "solved" and m[A]["unique"] == "False",
        lambda k, m: (-SEV.get(k[0].rsplit("_", 1)[-1], -1), k),  # most severe first
        lambda c: c.results[A].unique is False,
    ),
    Category(
        "f",
        "Vision failure (fixed examples)",
        "the two synthetic images named in the brief: rotate_1.0/00080 (grid not found) "
        "and rotate_1.0/00021 (grid found with a large corner error).",
        False,
        (W, A, U),
        W,
        2,
        None,
        None,
        lambda c: True,
        fixed=(("rotate_1.0", "00080"), ("rotate_1.0", "00021")),
    ),
    Category(
        "g",
        "CNN errors too dense for reasoning to recover",
        "vision fine (corner error < 5 px), but maxsat_weighted fails because too many "
        "cells were misread. Taken from blur severity 1.0, most misreads first.",
        False,
        (W, A),
        W,
        1,
        lambda k, m: k[0] == "blur_1.0" and m[W]["failure_stage"] == "perception",
        lambda k, m: (-num(m[W], "n_misreads"), k),
        lambda c: True,
    ),
    Category(
        "h",
        "maxsat_unique fixes what maxsat_weighted got wrong",
        "same CNN probabilities; maxsat_weighted returned a wrong solution, maxsat_unique "
        "(next-cheapest reading until the puzzle is unique) returned the right one.",
        False,
        (W, U),
        U,
        1,
        lambda k, m: (
            U in m
            and csv_true(m[U], "correct")
            and not csv_true(m[W], "correct")
            and m[W]["failure_stage"] != "vision"
        ),
        lambda k, m: (num(m[U], "solve_time_s"), k),
        lambda c: True,
    ),
    Category(
        "r1",
        "Real photo: success with a MaxSAT correction",
        "real photo; maxsat_weighted solved correctly after overriding the CNN, while "
        "argmax_then_sat was UNSAT. v2 test split preferred.",
        True,
        (W, A),
        W,
        1,
        lambda k, m: csv_true(m[W], "correct") and m[A]["status"] == "unsat",
        lambda k, m: (k[0] != "wichtounet_v2_test", num(m[W], "n_misreads"), k),
        lambda c: any(kd != "correct read overridden" for kd, _ in override_kinds(c, W)),
    ),
    Category(
        "r2",
        "Real photo: failure",
        "real photo where maxsat_weighted failed (vision or dense CNN errors). v2 test split "
        "preferred, then vision failures.",
        True,
        (W, A, U),
        W,
        1,
        lambda k, m: not csv_true(m[W], "correct"),
        lambda k, m: (k[0] != "wichtounet_v2_test", m[W]["failure_stage"] != "vision", k),
        lambda c: True,
    ),
]


def select(cat: Category, rows, cfg_of, log: list[str], taken: set) -> list[Case]:
    """taken: images already shown in an earlier category (each image appears once)."""
    if cat.fixed:
        keys = list(cat.fixed)
    else:
        keys = [
            k
            for k, m in rows.items()
            if cat.real in (None, is_real(k[0]))
            and all(x in m for x in cat.methods)
            and cat.filt(k, m)
        ]
        keys.sort(key=lambda k: cat.order(k, rows[k]))
        keys = [k for k in keys if k not in taken]
    log.append(f"({cat.key}) {len(keys)} CSV candidate(s)")
    accepts = [cat.accept] * cat.n
    if cat.key == "a":
        accepts = [
            lambda c: has_kind(c, W, "misread corrected", core="direct"),
            lambda c: has_kind(c, W, "misread corrected", core="indirect"),
        ]
    picked: list[Case] = []
    used: set = set()
    for acc in accepts:
        for k in [k for k in keys if k not in used][:MAX_TRIES]:
            try:
                case = rerun(k, list(cat.methods), rows, cfg_of, cat.show, cat.key)
            except Mismatch as e:
                log.append(f"({cat.key}) skipped, not reproduced: {e}")
                continue
            if acc(case):
                picked.append(case)
                taken.add(k)
                used.add(k)
                break
            used.add(k)  # re-run but not showing the category; do not try it again
            if len(keys) <= 5:  # few candidates: say why each one was rejected
                kinds = {m: [kd for kd, _ in override_kinds(case, m)] for m in case.results}
                log.append(f"({cat.key}) {k[0]}/{k[1]} reproduced but rejected; overrides {kinds}")
        else:
            n = min(len(keys), MAX_TRIES)
            log.append(f"({cat.key}) no candidate passed ({n} of {len(keys)} re-run)")
    return picked


# ----------------------------------------------------------------- figure
def board(ax, title):
    ax.set_xlim(0, 9)
    ax.set_ylim(9, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(10):
        lw = 1.6 if i % 3 == 0 else 0.4
        ax.axhline(i, color=INK, lw=lw)
        ax.axvline(i, color=INK, lw=lw)
    ax.set_title(title, loc="left")


def draw_cnn(ax, case: Case):
    board(ax, "CNN reads (argmax) and confidence")
    arg, conf = case.prob.argmax(2), case.prob.max(2)
    im = ax.imshow(conf, cmap=CONF_CMAP, vmin=0.3, vmax=1.0, extent=(0, 9, 9, 0), zorder=0)
    for r in range(9):
        for c in range(9):
            d = "" if arg[r, c] == 0 else str(arg[r, c])
            ax.text(c + 0.5, r + 0.42, d, ha="center", va="center", fontsize=11, color=INK)
            if conf[r, c] < 0.995:
                ax.text(
                    c + 0.5,
                    r + 0.84,
                    f"{conf[r, c]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color=INK2,
                )
            if arg[r, c] != case.givens[r, c]:
                ax.add_patch(
                    Rectangle(
                        (c + 0.06, r + 0.06),
                        0.88,
                        0.88,
                        fill=False,
                        ec=RED,
                        lw=1.3,
                        ls="--",
                        zorder=3,
                    )
                )
    return im


def draw_final(ax, case: Case, method: str):
    res = case.results[method]
    if res.solution is None:
        board(ax, f"{method}: {res.status.value.upper()}")
        ax.text(
            4.5,
            4.5,
            "\n".join(textwrap.wrap(res.message or "", 30)),
            ha="center",
            va="center",
            fontsize=8,
            color=INK,
            bbox={"fc": "white", "ec": INK2, "lw": 0.6},
        )
        return
    ok = np.array_equal(res.solution, case.solution)
    board(ax, f"{method}: {'correct' if ok else 'WRONG'} (unique={res.unique})")
    kinds = {(o.row, o.col): k for k, o in override_kinds(case, method)}
    for r in range(9):
        for c in range(9):
            given = res.read_givens[r, c] != 0
            if (r, c) in kinds:
                good = (
                    kinds[(r, c)] != "correct read overridden"
                    and res.read_givens[r, c] == case.givens[r, c]
                )
                ax.add_patch(
                    Rectangle(
                        (c, r), 1, 1, color=GREEN if good else RED, alpha=0.28, lw=0, zorder=0
                    )
                )
            wrong = res.solution[r, c] != case.solution[r, c]
            colour = RED if wrong else (INK if given else BLUE)
            ax.text(
                c + 0.5,
                r + 0.5,
                str(res.solution[r, c]),
                ha="center",
                va="center",
                fontsize=11,
                color=colour,
                fontweight="bold" if given else "normal",
            )


def outcome_line(case: Case) -> str:
    parts = []
    for m, res in case.results.items():
        if res.status is SolveStatus.SOLVED:
            ok = np.array_equal(res.solution, case.solution)
            s = f"{m}: {'correct' if ok else 'wrong'}, unique={res.unique}"
            if res.message:
                s += " (timed out)"
        else:
            s = f"{m}: {res.status.value.upper()}"
        parts.append(s)
    return "   |   ".join(parts)


def figure(case: Case, name: str, title: str) -> None:
    fig = plt.figure(figsize=(11.5, 5.4))
    gs = fig.add_gridspec(
        2, 4, width_ratios=[1.15, 1, 1, 0.04], height_ratios=[1, 0.30], wspace=0.12, hspace=0.08
    )
    ax0 = fig.add_subplot(gs[0, 0])
    if case.corners is not None:
        gt = None if np.isnan(case.gt_corners).all() else case.gt_corners
        ax0.imshow(overlay(case.image_rgb, case.corners, gt))
        err = (
            ""
            if gt is None
            else f", corner error {np.linalg.norm(case.corners - gt, axis=1).mean():.1f} px"
        )
        ax0.set_title(f"input + detected grid{err}", loc="left")
    else:
        ax0.imshow(case.image_rgb)
        ax0.set_title("input: grid NOT found", loc="left")
    ax0.axis("off")
    ax1, ax2 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    if case.prob is not None:
        im = draw_cnn(ax1, case)
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 3]))
        cb.set_label("CNN top-class probability", fontsize=7.5)
        cb.ax.tick_params(labelsize=7, colors=INK2)
        draw_final(ax2, case, case.show)
    else:
        for ax in (ax1, ax2):
            ax.axis("off")
        ax1.text(
            0,
            0.5,
            "\n".join(textwrap.wrap(case.vision_msg, 48)),
            fontsize=9,
            color=INK,
            va="center",
            transform=ax1.transAxes,
        )
    axt = fig.add_subplot(gs[1, :])
    axt.axis("off")
    lines = [outcome_line(case)]
    res = case.results[case.show]
    for _, o in override_kinds(case, case.show)[:4] if case.prob is not None else []:
        lines += textwrap.wrap(describe_override(o), 170, subsequent_indent="      ")
    if len(res.overrides) > 4:
        lines.append(f"(+{len(res.overrides) - 4} more overrides: see gallery.md)")
    legend = (
        "CNN panel: red dashed box = CNN argmax differs from ground truth.   Final panel: "
        "black bold = read givens, blue = filled by solver, red digit = wrong vs truth, "
        "green cell = override that fixed a misread, red cell = harmful override."
    )
    axt.text(
        0,
        1,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=7.3,
        color=INK,
        family="DejaVu Sans Mono",
        transform=axt.transAxes,
    )
    axt.text(
        0,
        0.95 - 0.105 * len(lines),  # just below the last text line (~0.105 axes units per line)
        legend,
        va="top",
        ha="left",
        fontsize=6.8,
        color=INK2,
        transform=axt.transAxes,
        wrap=True,
    )
    src = "real photo" if is_real(case.dataset) else "synthetic"
    fig.suptitle(
        f"{title}  -  {case.dataset}/{case.image} ({src})",
        x=0.01,
        ha="left",
        fontsize=10,
        color=INK,
    )
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- markdown
def system_text(case: Case, m: str) -> str:
    """The system's own result text: describe_override lines + status/message fields."""
    res = case.results[m]
    out = [f"status={res.status.value} unique={res.unique} cost={res.cost}"]
    if res.message:
        out.append(f"message: {res.message}")
    if res.overrides:
        out.append(f"{len(res.overrides)} override(s):")
        out += ["  " + describe_override(o) for o in res.overrides]
    elif res.status is SolveStatus.SOLVED:
        out.append("no overrides: the CNN argmax was already consistent")
    return "\n".join(out)


def what_happened(case: Case) -> list[str]:
    out = []
    if case.prob is None:
        return [f"Vision failed: `{case.vision_msg}`. No CNN reads, no reasoning."]
    a = case.prob.argmax(2)
    t = case.givens
    out.append(
        f"CNN argmax misread {int((a != t).sum())} of 81 cells "
        f"(wrong digit {int(((a != t) & (a > 0) & (t > 0)).sum())}, "
        f"missed digit {int(((a == 0) & (t > 0)).sum())}, "
        f"extra digit {int(((a > 0) & (t == 0)).sum())})."
    )
    if not np.isnan(case.gt_corners).all():
        err = np.linalg.norm(case.corners - case.gt_corners, axis=1).mean()
        out.append(f"Grid found, mean corner error {err:.2f} px (vision check: < 5 px).")
    for m in case.results:
        kinds = [k for k, _ in override_kinds(case, m)]
        if kinds:
            counts = {k: kinds.count(k) for k in dict.fromkeys(kinds)}
            out.append(f"{m} overrides: " + ", ".join(f"{v} x {k}" for k, v in counts.items()))
    return out


def csv_table(case: Case) -> str:
    cols = [
        "method",
        "status",
        "correct",
        "failure_stage",
        "n_misreads",
        "misreads_corrected",
        "correct_overridden",
        "unique",
        "solve_time_s",
    ]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for m, r in case.csv_rows.items():
        vals = [m] + [r[c] for c in cols[1:]]
        if vals[-1] not in ("", "nan"):
            vals[-1] = f"{float(vals[-1]):.3f}"
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def write_md(found: dict, log: list[str], rows_all: dict) -> None:
    md = [
        "# Gallery: real successes and failures",
        "",
        "Generated by `python scripts/make_gallery.py`. Every example is a real image from an "
        "existing evaluation run (synthetic `test_final`, seed 40000; or the public wichtounet "
        "v2 real photos). Nothing was hand-made or edited. Each image was chosen by a fixed rule "
        "(below and in the script), **re-run** exactly as `scripts/run_e2e.py` ran it, and kept "
        "only if the re-run reproduced the CSV row (status, correct, n_misreads, "
        "misreads_corrected, correct_overridden) for every method shown. The CSVs do not store "
        "override explanations; the explanation text below is the system's output from that "
        "re-run (`reasoning/explain.py`, `describe_override`), copied verbatim. Solve times in "
        "the CSV tables are from the original run.",
        "",
        "Figures: `report/figures/gallery/<name>.{png,pdf}` (300 dpi). Panels: input with the "
        "detected grid (synthetic: green = true outline) | CNN argmax with its top-class "
        "probability (red dashed box = differs from ground truth) | final grid of the method "
        "named in the panel title (green cell = override that fixed a misread, red cell = "
        "harmful override, red digit = wrong vs the true solution).",
        "",
        "## Summary",
        "",
        "| cat | what | example(s) | dataset | noise | outcome | figure |",
        "|---|---|---|---|---|---|---|",
    ]
    for cat in CATEGORIES:
        cases = found[cat.key]
        if not cases:
            md.append(f"| {cat.key} | {cat.title} | **none found** | | | | |")
        for i, (case, name) in enumerate(cases):
            src = "real" if is_real(case.dataset) else "synthetic"
            md.append(
                f"| {cat.key} | {cat.title if i == 0 else ''} | {case.dataset}/{case.image} | "
                f"{src} | `{case.noise}` | {outcome_line(case).replace('|', ';')} | "
                f"[{name}](../figures/gallery/{name}.png) |"
            )
    md += [
        "",
        "## Census: missed and hallucinated digits (CNN argmax, every image with a grid)",
        "",
        "From the saved perception outputs (`"
        + str(LOGITS).replace("\\", "/")
        + "`, `"
        + str(REAL_PROBS).replace("\\", "/")
        + "/`), checked against the labels. "
        "Missed = true digit read as empty; hallucinated = empty cell read as a digit.",
        "",
        "| data | images | images with a missed digit | missed cells | "
        "images with a hallucinated digit | hallucinated cells |",
        "|---|---|---|---|---|---|",
    ]
    for name, real in (("synthetic test_final", False), ("real wichtounet v2", True)):
        v = [x for k, x in CENSUS.items() if is_real(k[0]) == real]
        md.append(
            f"| {name} | {len(v)} | {sum(x[0] > 0 for x in v)} | {sum(x[0] for x in v)} | "
            f"{sum(x[1] > 0 for x in v)} | {sum(x[1] for x in v)} |"
        )
    for cat in CATEGORIES:
        md += ["", f"## ({cat.key}) {cat.title}", "", f"Rule: {cat.want}", ""]
        if not found[cat.key]:
            md.append(
                "**No example found.** No candidate in the CSVs passed the rule and the "
                "re-run check; none was made up."
            )
        if cat.key == "b":
            md += [
                "",
                "Every image whose CNN argmax missed a digit, and what each method did (CSV):",
                "",
                "| image | missed cells | method | correct | failure_stage | misreads_corrected |",
                "|---|---|---|---|---|---|",
            ]
            for k in sorted(k for k, x in CENSUS.items() if x[0] > 0):
                for m in (W, U):
                    r = rows_all[k].get(m)
                    if r:
                        md.append(
                            f"| {k[0]}/{k[1]} | {CENSUS[k][0]} | {m} | {r['correct']} | "
                            f"{r['failure_stage'] or '-'} | {r['misreads_corrected']} |"
                        )
        for case, name in found[cat.key]:
            src = "real photo" if is_real(case.dataset) else "synthetic"
            md += [
                f"### {case.dataset}/{case.image}",
                "",
                f"- Dataset: {src}; image `{json_path(case.dataset, case.image).as_posix()}`"
                f"; noise `{case.noise}`",
                f"- Figure: [{name}.png](../figures/gallery/{name}.png), "
                f"[{name}.pdf](../figures/gallery/{name}.pdf)",
            ]
            md += [f"- {x}" for x in what_happened(case)]
            md += case.notes and [f"- {x}" for x in case.notes] or []
            md += ["", "CSV rows (original run):", "", csv_table(case), ""]
            for m in case.results:
                md += [
                    f"System output, `{m}` (re-run):",
                    "",
                    "```",
                    system_text(case, m),
                    "```",
                    "",
                ]
            if cat.key == "d":
                md += [
                    "Why the cost favoured the override (computed by make_gallery.py):",
                    "",
                    "```",
                    *cost_explanation(case, W),
                    "```",
                    "",
                ]
    md += ["", "## Selection log", "", *[f"- {x}" for x in log], ""]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    rows, cfg_of = load_runs()
    CENSUS.update(census(rows))
    found, log, taken = {}, [], set()
    for cat in CATEGORIES:
        cases = select(cat, rows, cfg_of, log, taken)
        found[cat.key] = []
        for i, case in enumerate(cases, 1):
            name = f"{cat.key}{i if cat.n > 1 else ''}_{case.dataset}_{case.image}"
            if cat.key == "f" and case.prob is not None:
                res = case.results[W]
                case.notes.append(
                    f"Despite the corner error, maxsat_weighted {res.status.value} "
                    f"({'correct' if np.array_equal(res.solution, case.solution) else 'wrong'}); "
                    "see the other methods below."
                )
            figure(case, name, f"({cat.key}) {cat.title}")
            found[cat.key].append((case, name))
            print(f"({cat.key}) {case.dataset}/{case.image} -> {name}")
        if not cases:
            print(f"({cat.key}) none found")
    write_md(found, log, rows)
    print(f"wrote {OUT_MD} and {FIG}/")


if __name__ == "__main__":
    main()
