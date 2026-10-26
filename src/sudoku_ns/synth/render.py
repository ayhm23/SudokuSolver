"""Render a puzzle as a synthetic "photo" with exact ground truth.

Pipeline of one render:
1. Draw a flat, top-down sheet: off-white paper, thin cell lines, thick box
   lines, givens in one random font with small size/offset jitter.
2. Paste the sheet onto a random background (solid, gradient or noise) through
   a random perspective warp, as if photographed at a slight angle.
3. Push the 4 grid corners through the same warp, so ground truth is exact.

Why synthetic: we need thousands of images with known corners and digits to
test grid detection and to train the CNN. Hand-labelling photos cannot give
corner coordinates to the pixel. Real photos are still collected (Week 2) to
check the synthetic data is not too easy.

Ground-truth corners are the CENTRES of the outer border line, ordered
tl, tr, br, bl; that is the geometric grid boundary the cell extractor wants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sudoku_ns.interfaces import validate_grid


@dataclass
class Render:
    image: np.ndarray  # uint8 RGB (H, W, 3)
    givens: np.ndarray  # int8 (9, 9)
    solution: np.ndarray  # int8 (9, 9)
    corners: np.ndarray  # float32 (4, 2) x, y in image pixels: tl, tr, br, bl
    font: str  # font file name


def list_fonts(font_dir: Path | str = "data/fonts") -> list[Path]:
    fonts = sorted(Path(font_dir).glob("*.ttf"))
    if not fonts:
        raise FileNotFoundError(f"no .ttf in {font_dir}; run scripts/download_fonts.py")
    return fonts


def _draw_sheet(
    givens: np.ndarray, font_path: Path, rng: np.random.Generator
) -> tuple[Image.Image, np.ndarray]:
    """Flat top-down sheet and the grid corners in sheet coordinates."""
    cell = int(rng.integers(40, 64))
    margin = int(rng.integers(cell // 2, cell * 2))
    side = 9 * cell + 2 * margin
    paper = tuple(int(v) for v in rng.integers(232, 256, size=3))
    ink = tuple([int(rng.integers(0, 60))] * 3)
    thin = int(rng.integers(1, 3))
    thick = int(rng.integers(3, 6))

    img = Image.new("RGB", (side, side), paper)
    d = ImageDraw.Draw(img)
    for i in range(10):
        w = thick if i % 3 == 0 else thin
        p = margin + i * cell
        d.line([(margin, p), (margin + 9 * cell, p)], fill=ink, width=w)
        d.line([(p, margin), (p, margin + 9 * cell)], fill=ink, width=w)

    size = int(cell * rng.uniform(0.55, 0.75))
    font = ImageFont.truetype(str(font_path), size)
    for r, c in zip(*np.nonzero(givens), strict=True):
        jx, jy = rng.uniform(-0.06, 0.06, size=2) * cell
        cx = margin + (c + 0.5) * cell + jx
        cy = margin + (r + 0.5) * cell + jy
        d.text((cx, cy), str(int(givens[r, c])), font=font, fill=ink, anchor="mm")

    # PIL draws a width-w line centred on the coordinate (rounded), so the
    # border's centre line is at margin and margin + 9 * cell.
    a, b = margin, margin + 9 * cell
    corners = np.array([[a, a], [b, a], [b, b], [a, b]], np.float32)
    return img, corners


def _background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    kind = rng.choice(["solid", "gradient", "noise"])
    base = rng.integers(30, 220, size=3).astype(np.float32)
    if kind == "solid":
        bg = np.broadcast_to(base, (h, w, 3)).copy()
    elif kind == "gradient":
        other = rng.integers(30, 220, size=3).astype(np.float32)
        t = np.linspace(0, 1, w if rng.random() < 0.5 else h, dtype=np.float32)
        t = t[None, :, None] if t.size == w else t[:, None, None]
        bg = base * (1 - t) + other * t
        bg = np.broadcast_to(bg, (h, w, 3)).copy()
    else:
        # Smooth blotchy texture (wood/fabric-like): upsampled random field.
        small = rng.normal(0, 40, size=(h // 16 + 1, w // 16 + 1, 3)).astype(np.float32)
        tex = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        fine = rng.normal(0, 8, size=(h, w, 3)).astype(np.float32)
        bg = base + tex + fine
    return np.clip(bg, 0, 255).astype(np.uint8)


def render_puzzle(
    givens: np.ndarray,
    solution: np.ndarray,
    rng: np.random.Generator,
    fonts: list[Path],
    canvas: tuple[int, int] = (640, 640),
    max_tilt: float = 0.08,
) -> Render:
    """Render `givens` on a random background with a random perspective warp.

    max_tilt: how far each sheet corner may move, as a fraction of the canvas.
    0.08 gives a visibly angled but fully visible grid, like a casual phone photo.
    """
    validate_grid(givens)
    font_path = fonts[int(rng.integers(len(fonts)))]
    sheet, corners = _draw_sheet(givens, font_path, rng)
    sheet = np.asarray(sheet)
    s = sheet.shape[0]
    h, w = canvas

    # Place the sheet to fill 70-90% of the canvas, then jitter each corner.
    scale = rng.uniform(0.7, 0.9) * min(h, w)
    ox, oy = (w - scale) / 2, (h - scale) / 2
    dst = np.array([[ox, oy], [ox + scale, oy], [ox + scale, oy + scale], [ox, oy + scale]])
    dst += rng.uniform(-max_tilt, max_tilt, size=(4, 2)) * min(h, w)
    src = np.array([[0, 0], [s, 0], [s, s], [0, s]], np.float32)
    H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))

    warped = cv2.warpPerspective(sheet, H, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((s, s), 255, np.uint8), H, (w, h))
    bg = _background(h, w, rng)
    alpha = (mask.astype(np.float32) / 255)[..., None]
    image = (warped * alpha + bg * (1 - alpha)).astype(np.uint8)

    img_corners = cv2.perspectiveTransform(corners[None], H)[0].astype(np.float32)
    return Render(image, givens.copy(), solution.copy(), img_corners, font_path.name)


def save_render(r: Render, out_dir: Path, stem: str) -> None:
    """Write <stem>.png and <stem>.json (givens, solution, corners, font)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(r.image).save(out_dir / f"{stem}.png")
    meta = {
        "image": f"{stem}.png",
        "givens": r.givens.tolist(),
        "solution": r.solution.tolist(),
        "corners": r.corners.round(2).tolist(),
        "corner_order": "tl, tr, br, bl (x, y)",
        "font": r.font,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(meta))


def load_render(json_path: Path) -> Render:
    meta = json.loads(Path(json_path).read_text())
    image = np.asarray(Image.open(Path(json_path).parent / meta["image"]).convert("RGB"))
    return Render(
        image,
        np.array(meta["givens"], np.int8),
        np.array(meta["solution"], np.int8),
        np.array(meta["corners"], np.float32),
        meta["font"],
    )
