"""Find the Sudoku grid's 4 outer corners in a photo.

Classical pipeline, chosen because it needs no training data and every step
can be inspected. It is "propose many, verify each":

PROPOSE candidate quadrilaterals from three binarisations of the image, because
no single one works for every photo (measured on 50 renders x 6 noise types,
see docs/vision.md):

  plain     5x5 Gaussian blur + adaptive threshold. Sharpest edges; the Week 1
            detector, 100% on clean renders.
  denoised  stronger blur (sigma ~ 1/250 of the image side) + adaptive threshold
            + morphological opening (removes isolated speckles) and closing
            (reconnects grid lines that noise or blur broke into dashes). Fixes
            sensor noise, where the plain binary is mostly speckle (Week 1:
            22% success at noise severity 1.0).
  clahe     contrast-limited histogram equalisation first, then as "denoised".
            Lifts faint ink in dark or washed-out regions (uneven lighting).

Adaptive thresholding (each pixel against its own neighbourhood) is used in
all three because a single global threshold fails as soon as one side of the
grid is in shadow.

From each binary: ALL contours (RETR_LIST, not just outermost: the paper
sheet's edge, a photo frame or a book page ENCLOSE the grid, so "outermost" or
"largest" finds them instead). Each large contour's convex hull is simplified
to 4 corners; the hull makes this robust to a ragged, noisy outline.

RANK candidates by shape before the (more expensive) verification: area
(bigger is better, but only as a weak tie-breaker), convexity (contour area /
hull area: a grid outline is solid), squareness (shortest / longest side, and
how close the corner angles are to 90 degrees after allowing for perspective).
Only the best MAX_VERIFY are verified.

VERIFY each by its internal line structure: warp the candidate to a square and
compare the ink on the 10 expected row/column line positions with the ink
between them (projection profiles, `grid_score`). This is what separates the
real grid from the paper, a picture frame or a table edge. Scoring always uses
the DENOISED binary, so speckle between lines does not hide a real grid.

REFINE the winner (`refine_corners`): contours follow the OUTER edge of the
border line, but the cell geometry is defined by its centre. Each side is
moved to the ink-weighted centre of the border line found in a strip of the
warped image, then corners are re-intersected. We chose this over
cv2.cornerSubPix, which converges to an intensity corner (the outer corner of
the dark line) and so keeps exactly the half-line-width bias we want to remove.

Failures raise GridNotFoundError with a specific reason.
"""

from __future__ import annotations

import cv2
import numpy as np

MIN_AREA_FRAC = 0.05  # grid must cover >= 5% of the image; smaller = probably a cell or noise
APPROX_EPS_FRACS = (0.02, 0.04)  # approxPolyDP tolerances, fraction of the hull perimeter
MAX_VERIFY = 40  # candidates verified per image after shape ranking (each costs one small warp)
MIN_GRID_SCORE = 0.15  # below this the best quadrilateral does not have grid lines where expected
MIN_SQUARENESS = 0.4  # side ratio below this is not a (perspective-distorted) square


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


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image


def _adaptive(gray: np.ndarray, block: int) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 2
    )


def preprocess(image: np.ndarray) -> np.ndarray:
    """The 'plain' binary: ink = 255 (kept for callers and tests from Week 1)."""
    return _adaptive(cv2.GaussianBlur(_gray(image), (5, 5), 0), 11)


def binarize(image: np.ndarray) -> dict[str, np.ndarray]:
    """The three binaries described in the module docstring, ink = 255."""
    gray = _gray(image)
    side = min(gray.shape[:2])
    sigma = max(1.0, side / 250)
    block = 2 * round(side / 60) + 1  # ~1/30 of the side: larger than a line, smaller than a cell
    kernel = np.ones((3, 3), np.uint8)

    def cleaned(g: np.ndarray) -> np.ndarray:
        b = _adaptive(cv2.GaussianBlur(g, (0, 0), sigma), block)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return {"plain": preprocess(image), "denoised": cleaned(gray), "clahe": cleaned(clahe)}


def grid_score(binary: np.ndarray, corners: np.ndarray, size: int = 180) -> float:
    """How much a quadrilateral looks like a 9x9 grid, roughly in [-1, 1].

    Warp the binary image (ink = 255) to size x size and compare the ink
    fraction on the 10 expected horizontal/vertical line positions with the
    ink fraction between them (row and column projection profiles). Digits
    add some ink between lines, but much less than full-length grid lines add
    on them.
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


def shape_score(quad: np.ndarray, contour: np.ndarray, image_area: float) -> float:
    """Prior plausibility of a quadrilateral as a grid outline, in [0, 1]."""
    sides = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
    squareness = sides.min() / max(sides.max(), 1e-6)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    convexity = cv2.contourArea(contour) / max(hull_area, 1e-6)
    # Interior angles: perspective bends them, so only penalise far-off ones.
    v1 = np.roll(quad, 1, axis=0) - quad
    v2 = np.roll(quad, -1, axis=0) - quad
    norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    cos = np.abs((v1 * v2).sum(1)) / np.maximum(norms, 1e-6)
    rightness = 1 - cos.max()
    area = cv2.contourArea(quad) / image_area
    return float(squareness * convexity * rightness * (0.5 + 0.5 * min(area * 2, 1)))


def _quads(binary: np.ndarray, min_area: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """(ordered quad, source contour) for every large contour that fits 4 corners."""
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        hull = cv2.convexHull(cnt)
        for eps in APPROX_EPS_FRACS:
            approx = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                out.append((order_corners(approx), cnt))
                break
    return out


def refine_corners(gray: np.ndarray, corners: np.ndarray, size: int = 450) -> np.ndarray:
    """Move each side to the centre of the border line (see module docstring).

    Warp with a margin so the border line sits inside the image, take the ink
    profile across each side over its middle 80%, and replace the side's
    position by the ink-weighted centroid within +-search px. Falls back to the
    input if a side has no clear ink.
    """
    pad = size // 15
    s = size - 2 * pad
    dst = np.array([[pad, pad], [pad + s, pad], [pad + s, pad + s], [pad, pad + s]], np.float32)
    H = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), dst)
    w = 255 - cv2.warpPerspective(gray, H, (size, size), borderMode=cv2.BORDER_REPLICATE)
    w = w.astype(np.float32)
    lo, hi = pad + s // 10, pad + s - s // 10
    search = max(3, s // 60)

    def centre(profile: np.ndarray, at: int) -> float:
        idx = np.arange(at - search, at + search + 1)
        p = profile[idx] - np.median(profile[lo:hi])  # ink above the typical background
        p = np.clip(p, 0, None)
        if p.sum() <= 1e-3:
            return float(at)
        return float((idx * p).sum() / p.sum())

    top = centre(w[:, lo:hi].mean(axis=1), pad)
    bottom = centre(w[:, lo:hi].mean(axis=1), pad + s)
    left = centre(w[lo:hi, :].mean(axis=0), pad)
    right = centre(w[lo:hi, :].mean(axis=0), pad + s)
    new = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], np.float32)
    return cv2.perspectiveTransform(new[None], np.linalg.inv(H))[0].astype(np.float32)


def find_grid_corners(image: np.ndarray, refine: bool = True) -> np.ndarray:
    """(4, 2) float32 corners tl, tr, br, bl in image pixels (x, y).

    Raises GridNotFoundError with the reason when nothing grid-like is found.
    """
    if image is None or image.size == 0:
        raise GridNotFoundError("empty image")
    h, w = image.shape[:2]
    binaries = binarize(image)
    min_area = MIN_AREA_FRAC * h * w

    candidates = []
    for b in binaries.values():
        for quad, cnt in _quads(b, min_area):
            sq = shape_score(quad, cnt, h * w)
            if sq > 0 and np.linalg.norm(quad[0] - quad[1]) > 0:
                sides = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
                if sides.min() / sides.max() >= MIN_SQUARENESS:
                    candidates.append((sq, quad))
    if not candidates:
        contours, _ = cv2.findContours(binaries["plain"], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise GridNotFoundError("no contours after thresholding")
        biggest = max(cv2.contourArea(c) for c in contours) / (h * w)
        if biggest < MIN_AREA_FRAC:
            raise GridNotFoundError(
                f"largest contour covers {biggest:.1%} of the image (< {MIN_AREA_FRAC:.0%})"
            )
        raise GridNotFoundError(
            "large contour(s) found, none simplifies to a square-ish convex quadrilateral "
            "(grid cut off by the image edge or heavily occluded?)"
        )

    candidates.sort(key=lambda t: -t[0])
    scored = [(grid_score(binaries["denoised"], q), q) for _, q in candidates[:MAX_VERIFY]]
    best_score, best = max(scored, key=lambda t: t[0])
    if best_score < MIN_GRID_SCORE:
        raise GridNotFoundError(
            f"{len(candidates)} quadrilateral(s) found but none has grid lines "
            f"(best score {best_score:.2f} < {MIN_GRID_SCORE})"
        )
    if refine:
        refined = refine_corners(_gray(image), best)
        # Refinement moves each side by at most `search` px in warped space.
        # Keep it if the result still verifies as a grid. (Not "if the score
        # does not drop": grid_score's +-2 px line windows slightly favour the
        # outer-edge outline, so the score dips by ~0.02 even when the corner
        # error halves; measured on 50 clean renders, 2.6 -> 1.2 px.)
        if grid_score(binaries["denoised"], refined) >= MIN_GRID_SCORE:
            best = order_corners(refined)
    return best
