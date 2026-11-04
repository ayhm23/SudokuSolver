"""Why the solver overrode the CNN, plus human-readable text for grids and results.

Override reasons come from UNSAT CORES, not from a hand-written "look for the
same digit in the row" check. For an overridden cell (r, c) where the CNN read
digit a, ask a SAT solver: "is there a valid Sudoku with a at (r, c) AND every
other digit the solver finally kept?" (the kept digits are passed as
assumptions over the Sudoku rules).

- If that is UNSAT, the solver returns a core: a subset of the assumptions that
  is already contradictory. We shrink it to a MINIMAL core (deletion: drop one
  assumption, re-check, keep the drop if still UNSAT), so every given listed is
  needed for the contradiction. A core of size 1 is a direct clash ("8 twice in
  row 2"); larger cores are indirect ("with these 3 givens, no cell in box 4
  can take a 7") which a row/column/box scan cannot find at all. That is the
  point of using the solver for explanations: it covers both cases with one
  method and the explanation is a proof, not a heuristic.
- If it is SAT, keeping the CNN's digit was possible; the solver dropped it
  because the overall reading was cheaper (or, with require_unique, because
  keeping it left the puzzle ambiguous). We say "lower total cost".

Cells the CNN read as EMPTY never conflict with anything (an empty reading
adds no constraint), so they always get the fallback reason. With plain MAP
they are never overridden at all; they only appear with require_unique.

Coordinates are 0-based (row, col) everywhere, matching the arrays.
"""

from __future__ import annotations

import numpy as np
from pysat.solvers import Solver

from sudoku_ns.interfaces import Override, ProbGrid, SolveResult
from sudoku_ns.reasoning.encoding import decode_var, sudoku_cnf, x_var

SAT_SOLVER = "g4"  # Glucose 4: supports solving under assumptions and returns cores
FALLBACK_REASON = "lower total cost (no direct rule conflict)"

# ANSI colours. Kept optional (color=False) so output can be pasted into reports.
_RED, _GREEN, _RESET = "\033[31m", "\033[32m", "\033[0m"


# ------------------------------------------------------------ unsat cores
def _relation(r: int, c: int, rr: int, cc: int) -> str | None:
    """Which unit two cells share, or None."""
    if r == rr:
        return f"row {r}"
    if c == cc:
        return f"column {c}"
    if (r // 3, c // 3) == (rr // 3, cc // 3):
        return f"box {(r // 3) * 3 + c // 3}"
    return None


def conflict_core(
    solver: Solver, read_givens: np.ndarray, r: int, c: int, d: int
) -> list[tuple[int, int, int]] | None:
    """Minimal set of other givens that, with digit d at (r, c), admit no solution.

    `solver` must hold only the Sudoku rules (sudoku_cnf()); everything else is
    passed as assumptions so one solver can be reused for every override.
    Returns None if d at (r, c) is consistent with all other read givens.
    """
    me = x_var(r, c, d)
    others = [
        x_var(rr, cc, int(read_givens[rr, cc]))
        for rr, cc in zip(*np.nonzero(read_givens), strict=True)
        if (rr, cc) != (r, c)
    ]
    if solver.solve(assumptions=[me, *others]):
        return None
    # The other givens alone are consistent (they come from a solved reading),
    # so `me` is in every core; only the others need minimising.
    core = [lit for lit in solver.get_core() if lit != me]
    i = 0
    while i < len(core):
        trial = core[:i] + core[i + 1 :]
        if not solver.solve(assumptions=[me, *trial]):
            # Still UNSAT without core[i]: drop it, and take the solver's
            # (possibly even smaller) new core.
            new = set(solver.get_core())
            core = [lit for lit in trial if lit in new]
        else:
            i += 1
    out = []
    for lit in core:
        _, rr, cc, dd = decode_var(lit)
        out.append((rr, cc, dd))
    return sorted(out)


def conflict_reason(core: list[tuple[int, int, int]], r: int, c: int, d: int) -> str:
    if len(core) == 1:
        (rr, cc, dd) = core[0]
        where = _relation(r, c, rr, cc) or "the grid"
        return f"conflicts with given {dd} at ({rr}, {cc}) in {where}"
    listed = ", ".join(f"{dd} at ({rr}, {cc})" for rr, cc, dd in core)
    return f"no valid Sudoku has {d} here together with givens {listed}"


def find_overrides(prob: ProbGrid, read_givens: np.ndarray) -> list[Override]:
    """Cells where the solver's reading differs from the CNN argmax, each with a reason.

    The conflict is checked against the other FINAL read givens: those are the
    digits the solver kept, so a contradiction with them is the concrete reason
    the CNN's digit could not stay.
    """
    argmax = prob.argmax(axis=2)
    cells = list(zip(*np.nonzero(argmax != read_givens), strict=True))
    if not cells:
        return []
    out = []
    with Solver(name=SAT_SOLVER, bootstrap_with=sudoku_cnf()) as solver:
        for r, c in cells:
            r, c = int(r), int(c)
            a, f = int(argmax[r, c]), int(read_givens[r, c])
            core = conflict_core(solver, read_givens, r, c, a) if a >= 1 else None
            out.append(
                Override(
                    row=r,
                    col=c,
                    cnn_label=a,
                    cnn_prob=float(prob[r, c, a]),
                    final_label=f,
                    final_prob=float(prob[r, c, f]),
                    reason=conflict_reason(core, r, c, a) if core else FALLBACK_REASON,
                )
            )
    return out


# ------------------------------------------------------------------- text
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
