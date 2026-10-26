"""MaxSAT (plain and require_unique) vs argmax/threshold baselines on simulated CNN output.

    python scripts/sweep_simulated.py --n 30 --seed 0

For error rate e, each given is misread with prob e, missed (read empty) with
prob e/4, and each empty cell hallucinates a digit with prob e/20. Tying all
three to one knob keeps the x-axis simple; the exact split is recorded in the
CSV. Writes outputs/sweep_simulated.csv and outputs/sweep_simulated.png.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tqdm import tqdm  # noqa: E402

from sudoku_ns.eval.metrics import aggregate, evaluate  # noqa: E402
from sudoku_ns.eval.simulate import simulate_prob_grid  # noqa: E402
from sudoku_ns.reasoning.baselines import argmax_then_sat, threshold_then_sat  # noqa: E402
from sudoku_ns.reasoning.generator import generate  # noqa: E402
from sudoku_ns.reasoning.solver import solve  # noqa: E402

ERROR_RATES = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
# Fixed categorical order (blue, orange, aqua) with distinct markers so series are
# not distinguished by colour alone.
STYLE = {
    "maxsat": ("#2a78d6", "o"),
    "argmax": ("#eb6834", "s"),
    "threshold": ("#1baf7a", "^"),
    "maxsat_unique": ("#eda100", "D"),
}


def run_sweep(n: int, seed: int, tau: float) -> list[dict]:
    puzzles = [generate(seed + i) for i in tqdm(range(n), desc="generating")]
    methods = {
        "maxsat": lambda p: solve(p, check_unique=False),
        "argmax": lambda p: argmax_then_sat(p, check_unique=False),
        "threshold": lambda p: threshold_then_sat(p, tau, check_unique=False),
        "maxsat_unique": lambda p: solve(p, require_unique=True),
    }
    rows = []
    for ei, e in enumerate(tqdm(ERROR_RATES, desc="error rates")):
        per_method = {m: [] for m in methods}
        for i, (givens, solution) in enumerate(puzzles):
            prob, _ = simulate_prob_grid(
                givens,
                error_rate=e,
                empty_miss_rate=e / 4,
                hallucination_rate=e / 20,
                seed=10_000 * (ei + 1) + seed + i,
            )
            for m, fn in methods.items():
                per_method[m].append(evaluate(fn(prob), prob, givens, solution))
        for m, results in per_method.items():
            rows.append(
                {"method": m, "error_rate": e, "miss_rate": e / 4, "halluc_rate": e / 20}
                | aggregate(results)
            )
    return rows


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return [{k: v if k == "method" else float(v) for k, v in row.items()} for row in rows]


def plot(rows: list[dict], path: Path, tau: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True)
    panels = [("solved_correct", "Puzzles solved correctly"), ("unsat", "UNSAT rate")]
    for ax, (key, title) in zip(axes, panels, strict=True):
        for m, (color, marker) in STYLE.items():
            pts = [(r["error_rate"] * 100, r[key] * 100) for r in rows if r["method"] == m]
            xs, ys = zip(*pts, strict=True)
            ax.plot(xs, ys, color=color, marker=marker, lw=2, ms=7, label=m)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("Simulated misread rate (%)")
        ax.set_ylabel("% of puzzles")
        ax.set_ylim(-3, 103)
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    # One legend row above both panels, so it never sits on top of the data.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.93)
    )
    n = int(rows[0]["n"])
    fig.suptitle(
        f"Simulated perception, n={n} puzzles per point, tau={tau}",
        x=0.01,
        ha="left",
        fontsize=10,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=150)


def print_table(rows: list[dict]) -> None:
    print(
        f"{'method':<14} {'err%':>5} {'solved%':>8} {'unsat%':>7} {'acc_before':>10} "
        f"{'acc_after':>9} {'misreads':>8} {'fixed':>6} {'broken':>6} {'ms':>7}"
    )
    for r in rows:
        print(
            f"{r['method']:<14} {r['error_rate'] * 100:>5.1f} {r['solved_correct'] * 100:>8.1f} "
            f"{r['unsat'] * 100:>7.1f} {r['cell_acc_before']:>10.4f} {r['cell_acc_after']:>9.4f} "
            f"{int(r['n_misreads']):>8} {int(r['misreads_corrected']):>6} "
            f"{int(r['correct_overridden']):>6} {r['solve_time_s'] * 1000:>7.1f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=30, help="puzzles per error rate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.8, help="threshold baseline cutoff")
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--from-csv", action="store_true", help="only redraw plot/table from the CSV")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "sweep_simulated.csv"
    png_path = args.out / "sweep_simulated.png"

    if args.from_csv:
        rows = read_csv(csv_path)
    else:
        rows = run_sweep(args.n, args.seed, args.tau)
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    plot(rows, png_path, args.tau)
    print(f"wrote {csv_path} and {png_path}\n")
    print_table(rows)


if __name__ == "__main__":
    main()
