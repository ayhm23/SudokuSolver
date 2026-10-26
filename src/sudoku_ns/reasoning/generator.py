"""Seeded generation of solutions and unique-solution puzzles.

Full solutions come from randomised backtracking. A common shortcut is to
shuffle one fixed grid (permute digits, rows within bands, ...), but every
grid made that way is equivalent to the same seed grid, so the test set would
be far less varied than real puzzles. Backtracking from an empty grid with a
random digit order reaches many structurally different grids and only takes
milliseconds.

Puzzles are made by removing givens one at a time in random order, keeping a
removal only if the puzzle stays unique (checked with our SAT encoding), and
stopping at `target_givens`. Stopping early on purpose keeps the puzzle
NON-minimal: some givens are redundant, so a CNN that misses one of them still
leaves a uniquely solvable puzzle. A minimal puzzle (every given essential)
would make every "missed digit" experiment ambiguous by construction.
"""

from __future__ import annotations

import numpy as np

from sudoku_ns.reasoning.solver import is_unique


def _candidates(grid: np.ndarray, r: int, c: int) -> set[int]:
    br, bc = 3 * (r // 3), 3 * (c // 3)
    used = set(grid[r, :]) | set(grid[:, c]) | set(grid[br : br + 3, bc : bc + 3].ravel())
    return set(range(1, 10)) - used


def random_solution(rng: np.random.Generator) -> np.ndarray:
    """A random completed Sudoku grid (int8, 9x9)."""
    grid = np.zeros((9, 9), np.int8)

    def fill() -> bool:
        empties = list(zip(*np.nonzero(grid == 0), strict=True))
        if not empties:
            return True
        # Fill the most constrained cell first (fewest candidates): standard
        # heuristic that keeps backtracking shallow.
        best, cands = None, None
        for r, c in empties:
            cs = _candidates(grid, r, c)
            if cands is None or len(cs) < len(cands):
                best, cands = (r, c), cs
                if len(cs) <= 1:
                    break
        r, c = best
        for d in rng.permutation(sorted(cands)):
            grid[r, c] = d
            if fill():
                return True
        grid[r, c] = 0
        return False

    if not fill():  # pragma: no cover - an empty grid is always completable
        raise RuntimeError("backtracking failed on an empty grid")
    return grid


def make_puzzle(
    solution: np.ndarray, rng: np.random.Generator, target_givens: int = 30
) -> np.ndarray:
    """Remove givens from `solution` until `target_givens` remain, keeping uniqueness.

    If the random removal order gets stuck above the target (every remaining
    given is essential) the puzzle is returned with more givens than asked;
    that is rare for targets >= 25.
    """
    puzzle = solution.copy()
    n = 81
    for idx in rng.permutation(81):
        if n <= target_givens:
            break
        r, c = divmod(int(idx), 9)
        keep = puzzle[r, c]
        puzzle[r, c] = 0
        if is_unique(puzzle):
            n -= 1
        else:
            puzzle[r, c] = keep
    return puzzle


def redundant_givens(puzzle: np.ndarray) -> list[tuple[int, int]]:
    """Givens whose removal keeps the puzzle unique (empty list = minimal puzzle)."""
    out = []
    for r, c in zip(*np.nonzero(puzzle), strict=True):
        p = puzzle.copy()
        p[r, c] = 0
        if is_unique(p):
            out.append((int(r), int(c)))
    return out


def generate(seed: int, target_givens: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """(puzzle, solution) for a seed. Same seed -> same pair."""
    rng = np.random.default_rng(seed)
    solution = random_solution(rng)
    return make_puzzle(solution, rng, target_givens), solution
