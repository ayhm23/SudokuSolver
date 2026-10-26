"""Cut a detected grid into 81 normalised 32x32 cell images.

1. Perspective-warp the grid to a 288 x 288 square (32 px per cell), so every
   cell is axis-aligned and the same size regardless of camera angle.
2. Split into a 9x9 array of 32x32 cells.
3. Crop an inner margin (default 12% per side) to drop most grid-line pixels;
   lines are the most common thing a digit classifier confuses with a "1".
   Some remnants remain on purpose: the CNN must learn to ignore them, because
   real photos are never perfectly aligned.
4. Resize back to 32x32 and scale to [0, 1] with ink dark (paper ~1, ink ~0),
   matching how the digit dataset is built.
"""

from __future__ import annotations

import cv2
import numpy as np

from sudoku_ns.interfaces import CELL_SIZE, CellImages, validate_cell_images

WARP_SIZE = 9 * CELL_SIZE  # 288
DEFAULT_MARGIN = 0.12


def warp_grid(image: np.ndarray, corners: np.ndarray, size: int = WARP_SIZE) -> np.ndarray:
    """Top-down grayscale view of the grid, size x size pixels."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32)
    H = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), dst)
    return cv2.warpPerspective(gray, H, (size, size), flags=cv2.INTER_AREA)


def crop_cell(warped: np.ndarray, r: int, c: int, margin: float = DEFAULT_MARGIN) -> np.ndarray:
    """One 32x32 uint8 cell from a warped grid, inner margin removed."""
    step = warped.shape[0] / 9
    m = margin * step
    y0, y1 = round(r * step + m), round((r + 1) * step - m)
    x0, x1 = round(c * step + m), round((c + 1) * step - m)
    return cv2.resize(warped[y0:y1, x0:x1], (CELL_SIZE, CELL_SIZE), interpolation=cv2.INTER_AREA)


def extract_cells(
    image: np.ndarray, corners: np.ndarray, margin: float = DEFAULT_MARGIN
) -> CellImages:
    """81 cells, row-major, float32 in [0, 1], validated against interfaces."""
    warped = warp_grid(image, corners)
    cells = np.stack([crop_cell(warped, r, c, margin) for r in range(9) for c in range(9)])
    return validate_cell_images(cells.astype(np.float32) / 255.0)
