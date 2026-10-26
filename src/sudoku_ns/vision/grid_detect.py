"""Find the Sudoku grid's 4 outer corners in a photo.

Classical pipeline, chosen because it needs no training data and every step
can be inspected:

1. grayscale            colour carries no information about grid lines
2. Gaussian blur 5x5    suppresses paper texture/sensor noise before thresholding
3. adaptive threshold   a local (11x11 block) threshold copes with shadows and
                        uneven lighting, where one global threshold would fail;
                        inverted so ink is white (what findContours expects)
4. all contours         RETR_LIST, not RETR_EXTERNAL: the edge of the paper
                        sheet is also a closed shape that ENCLOSES the grid, so
                        "outermost contour" finds the paper, not the grid
                        (that was the first version's main failure: ~70% wrong)
5. approxPolyDP         simplify each large contour; keep those that reduce to
                        exactly 4 convex corners (quadrilaterals)
6. grid check           warp each candidate to a square and measure how much ink
                        lies on the 10 expected row/column line positions versus
                        between them. A real Sudoku grid scores high; the paper
                        sheet, a photo frame or a table edge do not. Pick the
                        best-scoring candidate.
7. order corners        tl, tr, br, bl, so the perspective warp is consistent

Known limitation: the border line is several pixels wide and a contour
follows one of its edges, while ground truth is the line's centre. The
offset is ~half the line width (1-3 px), inside our 5 px tolerance.
"""

from __future__ import annotations

import cv2
import numpy as np

MIN_AREA_FRAC = 0.05  # grid must cover >= 5% of the image; smaller = probably a cell or noise
APPROX_EPS_FRAC = 0.02  # approxPolyDP tolerance as a fraction of the contour perimeter


class GridNotFoundError(RuntimeError):
    """Raised when no plausible 4-corner grid outline is found; message says why."""


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as tl, tr, br, bl.

    tl has the smallest x + y and br the largest; tr has the smallest y - x
    and bl the largest. Works for any moderately tilted quadrilateral.
    """
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32
    )


def preprocess(image: np.ndarray) -> np.ndarray:
    """Binary image with ink = 255 (steps 1-3 above)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )


def grid_score(binary: np.ndarray, corners: np.ndarray, size: int = 180) -> float:
    """How much a quadrilateral looks like a 9x9 grid, roughly in [-1, 1].

    Warp the binary image (ink = 255) to size x size and compare the ink
    fraction on the 10 expected horizontal/vertical line positions with the
    ink fraction between them. Digits add some ink between lines, but much
    less than full-length grid lines add on them.
    """
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32)
    H = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), dst)
    w = cv2.warpPerspective(binary, H, (size, size)).astype(np.float32) / 255
    rows, cols = w.mean(axis=1), w.mean(axis=0)
    on = np.zeros(size, bool)
    for i in range(10):
        p = round(i * (size - 1) / 9)
        on[max(p - 2, 0) : p + 3] = True
    return float(
        (rows[on].mean() + cols[on].mean()) / 2 - (rows[~on].mean() + cols[~on].mean()) / 2
    )


MIN_GRID_SCORE = 0.15  # below this the best quadrilateral does not have grid lines where expected


def find_grid_corners(image: np.ndarray) -> np.ndarray:
    """(4, 2) float32 corners tl, tr, br, bl in image pixels (x, y).

    Raises GridNotFoundError with the reason when nothing grid-like is found.
    """
    if image is None or image.size == 0:
        raise GridNotFoundError("empty image")
    binary = preprocess(image)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise GridNotFoundError("no contours after thresholding")

    min_area = MIN_AREA_FRAC * image.shape[0] * image.shape[1]
    large = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not large:
        biggest = max(cv2.contourArea(c) for c in contours)
        raise GridNotFoundError(
            f"largest contour covers {biggest / (image.shape[0] * image.shape[1]):.1%} "
            f"of the image (< {MIN_AREA_FRAC:.0%})"
        )
    quads = []
    for cnt in large:
        approx = cv2.approxPolyDP(cnt, APPROX_EPS_FRAC * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append(order_corners(approx))
    if not quads:
        raise GridNotFoundError(
            f"{len(large)} large contour(s), none simplifies to a convex quadrilateral"
        )
    scores = [grid_score(binary, q) for q in quads]
    best = int(np.argmax(scores))
    if scores[best] < MIN_GRID_SCORE:
        raise GridNotFoundError(
            f"{len(quads)} quadrilateral(s) found but none has grid lines "
            f"(best score {scores[best]:.2f} < {MIN_GRID_SCORE})"
        )
    return quads[best]
