"""Human-readable text for grids and overrides (used by the demo and the report)."""

from __future__ import annotations

import numpy as np

from sudoku_ns.interfaces import Override, SolveResult

# ANSI colours. Kept optional (color=False) so output can be pasted into reports.
_RED, _GREEN, _RESET = "\033[31m", "\033[32m", "\033[0m"


def format_grid(
    grid: np.ndarray, highlight: set[tuple[int, int]] | None = None, color: bool = True
) -> str:
    """Grid as text with box separators; highlighted cells are marked [d] (and coloured)."""
    highlight = highlight or set()
    lines = []
    for r in range(9):
        if r in (3, 6):
            lines.append("---------+---------+---------")
        cells = []
        for c in range(9):
            v = int(grid[r, c])
            s = "." if v == 0 else str(v)
            if (r, c) in highlight:
                s = f"[{s}]" if not color else f"{_RED}[{s}]{_RESET}"
            else:
                s = f" {s} "
            cells.append(s)
            if c in (2, 5):
                cells.append("|")
        lines.append("".join(cells))
    return "\n".join(lines)


def describe_override(o: Override) -> str:
    cnn = "empty" if o.cnn_label == 0 else str(o.cnn_label)
    fin = "empty" if o.final_label == 0 else str(o.final_label)
    return (
        f"({o.row}, {o.col}): CNN read {cnn} (p={o.cnn_prob:.2f}) -> solver reads {fin} "
        f"(p={o.final_prob:.2f}); {o.reason}"
    )


def describe_result(result: SolveResult) -> str:
    lines = [
        f"status={result.status.value} unique={result.unique} cost={result.cost} "
        f"time={result.solve_time_s * 1000:.1f} ms"
    ]
    if result.overrides:
        lines.append(f"{len(result.overrides)} override(s):")
        lines += ["  " + describe_override(o) for o in result.overrides]
    else:
        lines.append("no overrides: the CNN argmax was already consistent")
    return "\n".join(lines)
