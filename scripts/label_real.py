"""Label real Sudoku photos: confirm the grid corners and the 81 printed givens.

    python scripts/label_real.py data/real/news_03_tilt20-lamp.jpg \
        --checkpoint outputs/models/cnn_v1/model.pt
    python scripts/label_real.py data/real/        # every photo without a .json yet
    python scripts/label_real.py data/real/ --no-window   # preview saved as PNG instead of a window

Protocol and naming: data/real/README.md. For each photo:
1. grid detection runs; a window shows the photo with the detected grid
   (left) and the straightened grid with the CNN's reading on it (right);
2. terminal: confirm the corners, or click the 4 corners (tl, tr, br, bl) in
   the window;
3. terminal: the reading is printed as 9 rows ("." = empty). Enter accepts
   it, a row number lets you retype that row, "all" retypes all 9 rows;
4. the givens are checked (no repeated digit in a row/column/box, has a
   solution, unique) and saved to <photo>.json next to the photo.

The JSON has the same keys as a synthetic render (image, givens, solution,
corners, font), so scripts/run_e2e.py runs on data/real unchanged.

Why a matplotlib window and not cv2.imshow: the project depends on
opencv-python-headless (no GUI code, works on CI and servers), and installing
the GUI build next to it is a known conflict. matplotlib is already a
dependency and its Tk window supports clicking (ginput).
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from sudoku_ns.reasoning.solver import count_solutions, sat_solve
from sudoku_ns.vision.cell_extract import extract_cells, overlay, warp_grid
from sudoku_ns.vision.grid_detect import GridNotFoundError, find_grid_corners, order_corners

PHOTO_EXT = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
NAME_RE = re.compile(r"^(news|book|screen|print)_(\d{2})_([\w-]+)$")


def load_photo(path: Path) -> np.ndarray:
    # exif_transpose: phones store portrait photos as landscape + a rotation tag.
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


def grid_text(g: np.ndarray) -> str:
    return "\n".join(
        f"  row {r + 1}: " + "".join("." if v == 0 else str(v) for v in g[r]) for r in range(9)
    )


def parse_row(text: str) -> list[int] | None:
    s = text.strip().replace(" ", "").replace("0", ".")
    if len(s) != 9 or any(ch not in ".123456789" for ch in s):
        return None
    return [0 if ch == "." else int(ch) for ch in s]


def duplicates(g: np.ndarray) -> list[str]:
    out = []
    units = (
        [("row", r + 1, g[r, :]) for r in range(9)]
        + [("column", c + 1, g[:, c]) for c in range(9)]
        + [("box", b + 1, g[3 * (b // 3) : 3 * (b // 3) + 3, 3 * (b % 3) : 3 * (b % 3) + 3])
           for b in range(9)]
    )  # fmt: skip
    for kind, i, vals in units:
        v = vals.ravel()
        v = v[v > 0]
        rep = sorted({int(d) for d in v if (v == d).sum() > 1})
        if rep:
            out.append(f"{kind} {i} repeats {rep}")
    return out


class Viewer:
    """Window (or PNG file with --no-window) showing the photo and the warped grid."""

    def __init__(self, window: bool, preview_path: Path):
        import matplotlib

        if not window:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.plt, self.window, self.preview = plt, window, preview_path
        self.fig, self.axes = plt.subplots(1, 2, figsize=(13, 6.5))

    def show(self, image, corners, reading, conf):
        ax0, ax1 = self.axes
        ax0.clear()
        ax1.clear()
        ax0.imshow(overlay(image, corners) if corners is not None else image)
        ax0.set_title("photo + detected grid" if corners is not None else "no grid: click corners")
        ax0.axis("off")
        if corners is not None:
            ax1.imshow(warp_grid(image, corners, 450), cmap="gray")
            step = 450 / 9
            for r in range(9):
                for c in range(9):
                    if reading is not None and reading[r, c]:
                        colour = "lime" if conf is None or conf[r, c] >= 0.9 else "orange"
                        ax1.text(
                            (c + 0.82) * step,
                            (r + 0.3) * step,
                            str(reading[r, c]),
                            color=colour,
                            fontsize=11,
                            fontweight="bold",
                        )
            ax1.set_title("straightened grid; small coloured digit = current reading")
        ax1.axis("off")
        self.fig.tight_layout()
        if self.window:
            self.plt.show(block=False)
            self.plt.pause(0.3)
        else:
            self.fig.savefig(self.preview, dpi=90)
            print(f"  preview: {self.preview}")

    def click_corners(self) -> np.ndarray:
        if not self.window:
            raise SystemExit("clicking corners needs the window (run without --no-window)")
        print("  click the 4 outer grid corners in the LEFT image: tl, tr, br, bl")
        pts = self.plt.ginput(4, timeout=0)
        return order_corners(np.array(pts, np.float32))


def label_one(path: Path, perceiver, viewer: Viewer) -> bool:
    print(f"\n=== {path.name} ===")
    m = NAME_RE.match(path.stem)
    if not m:
        print("  warning: name does not follow <source>_<nn>_<condition> (see data/real/README.md)")
    image = load_photo(path)
    try:
        corners, how = find_grid_corners(image), "detected"
    except GridNotFoundError as e:
        print(f"  grid not found: {e}")
        corners, how = None, "clicked"
    reading = conf = None

    while True:
        if corners is not None and perceiver is not None:
            prob = perceiver(extract_cells(image, corners))
            reading, conf = prob.argmax(2).astype(np.int8), prob.max(2)
        viewer.show(image, corners, reading, conf)
        if corners is None:
            corners = viewer.click_corners()
            continue
        ans = input("  corners right? [Y]es / [n]o, click them / [s]kip photo: ").strip().lower()
        if ans == "s":
            return False
        if ans == "n":
            corners, how = viewer.click_corners(), "clicked"
            continue
        break

    givens = reading.copy() if reading is not None else np.zeros((9, 9), np.int8)
    if reading is None:
        print("  no CNN checkpoint given: type all 9 rows")
        ans = "all"
    else:
        ans = None
    while True:
        if ans is None:
            print("  reading (. = empty), CHECK EVERY ROW AGAINST THE PHOTO:")
            print(grid_text(givens))
            ans = input("  Enter = accept, 1-9 = retype that row, all = retype all, s = skip: ")
            ans = ans.strip().lower()
        if ans == "s":
            return False
        if ans not in ("", "all", *"123456789"):
            print("  answer with Enter, a row number 1-9, all, or s")
            ans = None
            continue
        rows = range(9) if ans == "all" else ([int(ans) - 1] if ans else [])
        for r in rows:
            while (row := parse_row(input(f"    row {r + 1} (9 chars, . or 0 = empty): "))) is None:
                print("    need exactly 9 characters from .123456789")
            givens[r] = row
        if rows:
            viewer.show(image, corners, givens, None)
            ans = None
            continue
        problems = duplicates(givens)
        n_sol = count_solutions(givens, limit=2) if not problems else 0
        if problems or n_sol == 0:
            print("  NOT a valid puzzle: " + ("; ".join(problems) or "no solution") + ". Fix rows.")
            ans = None
            continue
        if (
            n_sol > 1
            and input("  more than one solution; misread digit? save anyway [y/N]: ") != "y"
        ):
            ans = None
            continue
        break

    meta = {
        "image": path.name,
        "givens": givens.tolist(),
        "solution": sat_solve(givens).tolist(),
        "unique": n_sol == 1,
        "corners": np.asarray(corners).round(2).tolist(),
        "corner_order": "tl, tr, br, bl (x, y), in EXIF-rotated image pixels",
        "corners_source": how,
        "font": "real",
        "split": "real",
        "source": m.group(1) if m else "",
        "conditions": m.group(3).split("-") if m else [],
        "notes": input("  notes (optional, e.g. pencil marks): ").strip(),
    }
    out = path.with_suffix(".json")
    out.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"  saved {out} ({int((givens > 0).sum())} givens)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("path", type=Path, help="a photo, or a folder of photos")
    ap.add_argument("--checkpoint", type=Path, default=None, help="CNN to pre-fill the reading")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--relabel", action="store_true", help="also photos that have a .json")
    args = ap.parse_args()

    photos = (
        sorted(p for p in args.path.iterdir() if p.suffix.lower() in PHOTO_EXT)
        if args.path.is_dir()
        else [args.path]
    )
    if not args.relabel:
        photos = [p for p in photos if not p.with_suffix(".json").exists()]
    if not photos:
        print("nothing to label")
        return
    perceiver = None
    if args.checkpoint:
        from sudoku_ns.perception.perceiver import CNNPerceiver

        perceiver = CNNPerceiver(args.checkpoint)
    viewer = Viewer(not args.no_window, Path("outputs/label_preview.png"))
    Path("outputs").mkdir(exist_ok=True)
    done = sum(label_one(p, perceiver, viewer) for p in photos)
    print(f"\nlabelled {done} of {len(photos)} photo(s)")


if __name__ == "__main__":
    main()
