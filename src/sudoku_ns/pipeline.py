"""End to end: image file -> grid corners -> cell images -> probabilities -> solved grid.

    python -m sudoku_ns.pipeline photo.jpg --checkpoint outputs/models/cnn_v1/model.pt
    python -m sudoku_ns.pipeline photo.jpg --checkpoint ... --method maxsat_unique --out vis.png

The perceiver is any callable CellImages -> ProbGrid. Passing it in (rather
than importing a model here) keeps vision and reasoning testable without the
CNN, and lets us swap the CNN, a calibrated CNN or a simulator without
touching this file.

Failures never crash the pipeline: a grid that cannot be found comes back as
SolveResult(status=ERROR, message="vision: <reason>"), so batch evaluation
can count failures per stage instead of stopping.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from sudoku_ns.eval.simulate import CorruptionRecord, simulate_prob_grid
from sudoku_ns.interfaces import (
    CellImages,
    ProbGrid,
    SolveResult,
    SolveStatus,
    validate_cell_images,
    validate_prob_grid,
)
from sudoku_ns.reasoning.baselines import argmax_then_sat, threshold_then_sat, unweighted_maxsat
from sudoku_ns.reasoning.solver import solve
from sudoku_ns.vision.cell_extract import extract_cells
from sudoku_ns.vision.grid_detect import GridNotFoundError, find_grid_corners

Perceiver = Callable[[CellImages], ProbGrid]
DEFAULT_TAU = 0.8  # threshold_then_sat; re-chosen on the val split for the CNN (tune_threshold.py)


def make_method(name: str, tau: float = DEFAULT_TAU) -> Callable[[ProbGrid], SolveResult]:
    """Reasoning method by name (the names used in configs and CSVs)."""
    methods = {
        "maxsat_weighted": solve,
        "maxsat_unique": partial(solve, require_unique=True),
        "maxsat_unweighted": unweighted_maxsat,
        "argmax_then_sat": argmax_then_sat,
        "threshold_then_sat": partial(threshold_then_sat, tau=tau),
    }
    if name not in methods:
        raise ValueError(f"unknown method {name!r}; choose from {sorted(methods)}")
    return methods[name]


METHODS = (
    "maxsat_weighted",
    "maxsat_unique",
    "maxsat_unweighted",
    "argmax_then_sat",
    "threshold_then_sat",
)


@dataclass
class PipelineOutput:
    corners: np.ndarray | None  # (4, 2) tl, tr, br, bl; None if vision failed
    cells: CellImages | None
    prob: ProbGrid | None
    result: SolveResult


class OraclePerceiver:
    """Stand-in for the CNN: ignores the pixels and returns a simulated
    ProbGrid built from the ground-truth givens (see eval/simulate.py).

    Lets the full pipeline run with a dialled-in perception error, and gives
    an upper bound for the CNN in end-to-end evaluation (error_rate=0 means
    "perfect perception"). `last_record` keeps what was corrupted.
    """

    def __init__(self, givens: np.ndarray, seed: int | None = None, **simulate_kwargs):
        self.givens = givens
        self.seed = seed
        self.simulate_kwargs = {"error_rate": 0.0} | simulate_kwargs
        self.last_record: CorruptionRecord | None = None

    def __call__(self, cells: CellImages) -> ProbGrid:
        validate_cell_images(cells)
        prob, self.last_record = simulate_prob_grid(
            self.givens, seed=self.seed, **self.simulate_kwargs
        )
        return prob


def perceive(image: np.ndarray, perceiver: Perceiver):
    """Vision + perception: (corners, cells, prob), or GridNotFoundError."""
    corners = find_grid_corners(image)
    cells = extract_cells(image, corners)
    return corners, cells, validate_prob_grid(perceiver(cells))


def run(
    image_path: Path | str,
    perceiver: Perceiver,
    method: str = "maxsat_weighted",
    require_unique: bool = False,
    tau: float = DEFAULT_TAU,
) -> PipelineOutput:
    """Run the whole pipeline on one image. Vision failures return status ERROR.

    require_unique=True is kept as a shortcut for method="maxsat_unique".
    """
    if require_unique and method == "maxsat_weighted":
        method = "maxsat_unique"
    solver = make_method(method, tau)
    # exif_transpose: phone photos are often stored sideways plus a rotation tag.
    image = np.asarray(ImageOps.exif_transpose(Image.open(image_path)).convert("RGB"))
    try:
        corners, cells, prob = perceive(image, perceiver)
    except GridNotFoundError as e:
        return PipelineOutput(
            None, None, None, SolveResult(SolveStatus.ERROR, message=f"vision: {e}")
        )
    return PipelineOutput(corners, cells, prob, solver(prob))


# ------------------------------------------------------------- visualisation
def visualize(image: np.ndarray, out: PipelineOutput, path: Path | str, title: str = "") -> None:
    """Three panels: detected grid on the photo | CNN reads with confidence | final solution.

    Solution colours: black = read givens kept as the CNN read them, red =
    givens the solver overrode, blue = digits filled in by solving.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sudoku_ns.vision.cell_extract import overlay

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    ax = axes[0]
    ax.imshow(overlay(image, out.corners) if out.corners is not None else image)
    ax.set_title("detected grid" if out.corners is not None else "grid NOT found")
    ax.axis("off")

    def board(ax, heading):
        ax.set_xlim(0, 9)
        ax.set_ylim(9, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for i in range(10):
            lw = 2 if i % 3 == 0 else 0.5
            ax.axhline(i, color="k", lw=lw)
            ax.axvline(i, color="k", lw=lw)
        ax.set_title(heading)

    res = out.result
    if out.prob is not None:
        board(axes[1], "CNN reading (top class, probability)")
        arg, conf = out.prob.argmax(2), out.prob.max(2)
        for r in range(9):
            for c in range(9):
                d = "" if arg[r, c] == 0 else str(arg[r, c])
                colour = "k" if conf[r, c] >= 0.9 else ("#d97706" if conf[r, c] >= 0.6 else "r")
                axes[1].text(
                    c + 0.5, r + 0.45, d, ha="center", va="center", fontsize=15, color=colour
                )
                axes[1].text(
                    c + 0.5,
                    r + 0.85,
                    f"{conf[r, c]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=colour,
                )
    else:
        axes[1].axis("off")
    if res.solution is not None:
        board(axes[2], f"solution ({res.status.value}, unique={res.unique})")
        over = {(o.row, o.col) for o in res.overrides}
        for r in range(9):
            for c in range(9):
                given = res.read_givens is not None and res.read_givens[r, c] != 0
                colour = "r" if (r, c) in over else ("k" if given else "#2563eb")
                weight = "bold" if given or (r, c) in over else "normal"
                axes[2].text(
                    c + 0.5,
                    r + 0.5,
                    str(res.solution[r, c]),
                    ha="center",
                    va="center",
                    fontsize=15,
                    color=colour,
                    fontweight=weight,
                )
    else:
        axes[2].axis("off")
        axes[2].text(
            0.5,
            0.5,
            f"{res.status.value}\n{res.message or ''}",
            ha="center",
            va="center",
            wrap=True,
        )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    from sudoku_ns.perception.perceiver import CNNPerceiver
    from sudoku_ns.reasoning.explain import describe_result, format_grid

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--method", choices=METHODS, default="maxsat_weighted")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU, help="threshold_then_sat only")
    ap.add_argument("--no-temperature", action="store_true", help="uncalibrated probabilities")
    ap.add_argument("--out", type=Path, default=None, help="visualisation PNG")
    args = ap.parse_args()

    perceiver = CNNPerceiver(args.checkpoint, use_temperature=not args.no_temperature)
    out = run(args.image, perceiver, method=args.method, tau=args.tau)
    res = out.result
    if out.prob is None:
        print(f"{res.status.value}: {res.message}")
    else:
        print("CNN reading (argmax):")
        print(format_grid(out.prob.argmax(2), color=False))
        if res.read_givens is not None:
            print(f"\n{args.method} reading (overridden cells marked):")
            print(format_grid(res.read_givens, {(o.row, o.col) for o in res.overrides}, False))
        print()
        print(describe_result(res))
        if res.solution is not None:
            print("\nsolution:")
            print(format_grid(res.solution, color=False))
        elif res.message:
            print(res.message)
    path = args.out or args.image.with_name(f"{args.image.stem}_solved.png")
    image = np.asarray(Image.open(args.image).convert("RGB"))
    visualize(image, out, path, f"{args.image.name} - {args.method}")
    print(f"\nvisualisation: {path}")


if __name__ == "__main__":
    main()
