"""Cut a detected grid into 81 normalised 32x32 cell images.

1. Perspective-warp the grid to a 288 x 288 square (32 px per cell), so every
   cell is axis-aligned and the same size regardless of camera angle.
2. Split into a 9x9 array of cells and crop a small inner margin (8% per side)
   so most of the cell's own grid lines fall outside the crop.
3. Resize to 32x32.
4. Clean (`clean_cell`): erase grid-line remnants and centre the digit.
5. Scale to [0, 1] with ink dark (paper ~1, ink ~0).

THIS IS THE ONLY CROP CODE PATH. The digit dataset (perception/dataset.py) calls
`extract_cells` on detected corners exactly as the pipeline does, so the CNN
is trained on the same distribution it sees at inference: same margin, same
leftover line fragments, same centring. Any change here (margin, cleaning
rules) changes that distribution, so the dataset must be regenerated and the
CNN retrained after it.

Why clean at all, if the CNN could learn to ignore remnants? Because the
remnants depend on corner error, which differs between synthetic renders and
real photos; removing them makes crops less sensitive to how good detection
was. Why not decide "empty" here (e.g. "no blob found -> empty")? Because that
is a hard decision without a confidence, exactly what the neurosymbolic design
avoids: the CNN must output a probability for "empty" that MaxSAT can weigh.
So cleaning only ERASES ink touching the crop border and only SHIFTS what is
inside; whatever is in the middle reaches the CNN untouched.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sudoku_ns.interfaces import CELL_SIZE, CellImages, validate_cell_images

WARP_SIZE = 9 * CELL_SIZE  # 288
DEFAULT_MARGIN = 0.08

# clean_cell thresholds, as fractions of the 32 px side (or of its area).
INK_DROP_MIN = 40  # a pixel is ink if >= this many grey levels darker than the paper ...
INK_DROP_FRAC = 0.5  # ... or this fraction of the paper-to-darkest range, whichever is larger
EDGE_BAND = 0.3  # a border-touching blob centred farther out than this is a line remnant
DIGIT_MIN_AREA = 0.02  # smallest blob that can be (part of) a digit
DIGIT_MIN_HEIGHT = 0.2  # printed digits are tall; specks and dashes are not
MAX_SHIFT = 0.25  # never move content more than this; a bigger shift means a bad blob choice


def warp_grid(image: np.ndarray, corners: np.ndarray, size: int = WARP_SIZE) -> np.ndarray:
    """Top-down grayscale view of the grid, size x size pixels."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32)
    H = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), dst)
    return cv2.warpPerspective(gray, H, (size, size), flags=cv2.INTER_AREA)


def crop_cell(warped: np.ndarray, r: int, c: int, margin: float = DEFAULT_MARGIN) -> np.ndarray:
    """One 32x32 uint8 cell from a warped grid, inner margin removed (not cleaned)."""
    step = warped.shape[0] / 9
    m = margin * step
    y0, y1 = round(r * step + m), round((r + 1) * step - m)
    x0, x1 = round(c * step + m), round((c + 1) * step - m)
    return cv2.resize(warped[y0:y1, x0:x1], (CELL_SIZE, CELL_SIZE), interpolation=cv2.INTER_AREA)


def clean_cell(cell: np.ndarray) -> np.ndarray:
    """Erase grid-line remnants at the crop border and centre the digit (uint8 in and out).

    1. Ink mask: smooth lightly (so sensor noise does not split blobs), then
       mark pixels clearly darker than the paper level (90th percentile).
    2. Connected components (8-connected). A blob is a grid-line remnant if it
       TOUCHES the crop border and its centroid lies in the outer band
       (offset from the centre > EDGE_BAND of the side on either axis). Those
       pixels are painted with the paper level. A digit that touches the
       border still has a central centroid and is kept.
    3. Digit blob: of the remaining blobs that are big and tall enough, the one
       whose centroid is nearest the centre, plus any blob overlapping its
       bounding box (a blurred or noisy digit can break into pieces).
    4. Scale brightness so the paper level becomes white (see below).
    5. Shift so that bounding box is centred. Translation only: digit size
       says something about the font and is not normalised away.
    No digit blob found -> no shift. Nothing in the middle is ever erased.
    """
    side = cell.shape[0]
    paper = float(np.percentile(cell, 90))
    smooth = cv2.GaussianBlur(cell, (3, 3), 0).astype(np.float32)
    drop = max(INK_DROP_MIN, INK_DROP_FRAC * (paper - float(smooth.min())))
    ink = (smooth < paper - drop).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(ink, connectivity=8)
    out = cell.copy()
    centre = (side - 1) / 2
    keep = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches = x == 0 or y == 0 or x + w == side or y + h == side
        off = np.abs(cents[i] - centre).max() / side
        if touches and off > EDGE_BAND:
            mask = cv2.dilate((labels == i).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            out[mask] = round(paper)
        elif area >= DIGIT_MIN_AREA * side * side and h >= DIGIT_MIN_HEIGHT * side:
            keep.append(i)
    # Brightness: map the paper level to white. Shadows and white balance
    # change paper brightness from cell to cell; that is not information about
    # the digit. (Only a gain, so dark ink stays dark and contrast is kept.)
    out = np.clip(out.astype(np.float32) * (255.0 / max(paper, 1.0)), 0, 255).astype(np.uint8)
    if not keep:
        return out
    main = min(keep, key=lambda i: np.linalg.norm(cents[i] - centre))
    x0, y0, w0, h0 = stats[main][:4]
    x1, y1 = x0 + w0, y0 + h0
    for i in keep:
        x, y, w, h = stats[i][:4]
        if i != main and x < x1 and x + w > x0 and y < y1 and y + h > y0:
            x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x + w), max(y1, y + h)
    dx = float(np.clip(side / 2 - (x0 + x1) / 2, -MAX_SHIFT * side, MAX_SHIFT * side))
    dy = float(np.clip(side / 2 - (y0 + y1) / 2, -MAX_SHIFT * side, MAX_SHIFT * side))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    # Replicate the edge into the vacated strip: a constant fill leaves a visible
    # patch whenever the paper is shaded across the cell.
    return cv2.warpAffine(
        out, M, (side, side), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def overlay(image: np.ndarray, corners: np.ndarray, gt: np.ndarray | None = None) -> np.ndarray:
    """RGB debug image: detected outline + the 9x9 cell grid it implies (red),
    ground-truth outline if given (green). Interior lines are drawn through
    the inverse warp, so a wrong corner shows up as cells not matching the print.
    """
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    H = cv2.getPerspectiveTransform(
        np.array([[0, 0], [9, 0], [9, 9], [0, 9]], np.float32), np.asarray(corners, np.float32)
    )
    for i in range(10):
        seg = np.array([[[0, i], [9, i]], [[i, 0], [i, 9]]], np.float32)
        for a, b in cv2.perspectiveTransform(seg.reshape(1, -1, 2), H)[0].reshape(2, 2, 2):
            cv2.line(
                out, tuple(map(int, a)), tuple(map(int, b)), (255, 0, 0), 2 if i % 3 == 0 else 1
            )
    if gt is not None:
        cv2.polylines(out, [np.asarray(gt, np.int32)], True, (0, 200, 0), 2)
    for x, y in np.asarray(corners):
        cv2.circle(out, (int(x), int(y)), 5, (255, 0, 0), -1)
    return out


def mosaic(cells: np.ndarray, gap: int = 2) -> np.ndarray:
    """81 uint8 crops as one 9x9 image with grey separators."""
    s = cells.shape[1]
    out = np.full((9 * (s + gap) - gap,) * 2, 180, np.uint8)
    for i, x in enumerate(cells):
        r, c = divmod(i, 9)
        out[r * (s + gap) : r * (s + gap) + s, c * (s + gap) : c * (s + gap) + s] = x
    return out


def save_debug(image: np.ndarray, corners: np.ndarray, cells: np.ndarray, path) -> None:
    """Overlay of corners and cell grid (left) next to the 81 crops (right)."""
    left = overlay(image, corners)
    right = cv2.cvtColor(mosaic(cells), cv2.COLOR_GRAY2RGB)
    right = cv2.resize(right, (left.shape[0], left.shape[0]), interpolation=cv2.INTER_NEAREST)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.hstack([left, right]), cv2.COLOR_RGB2BGR))


def extract_cells(
    image: np.ndarray,
    corners: np.ndarray,
    margin: float = DEFAULT_MARGIN,
    clean: bool = True,
    debug_path: str | Path | None = None,
) -> CellImages:
    """81 cells, row-major, float32 in [0, 1], validated against interfaces.

    debug_path: also save a PNG with the detected corners and the implied cell
    grid drawn on the image, next to the 81 crops as a 9x9 mosaic.
    """
    warped = warp_grid(image, corners)
    cells = [crop_cell(warped, r, c, margin) for r in range(9) for c in range(9)]
    if clean:
        cells = [clean_cell(x) for x in cells]
    cells = np.stack(cells)
    if debug_path is not None:
        save_debug(image, corners, cells, debug_path)
    return validate_cell_images(cells.astype(np.float32) / 255.0)
