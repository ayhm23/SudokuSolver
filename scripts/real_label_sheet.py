"""Contact sheets for checking the imported real-photo labels BY EYE.

    python scripts/real_label_sheet.py              # 20 random photos, seed 0
    python scripts/real_label_sheet.py --n 20 --seed 0

Writes to outputs/real_label_check/:
  contact_sheet.png   the 20 photos, each cropped around the grid, with the imported givens
                      drawn in magenta in the TOP-LEFT corner of each cell (the printed digit
                      stays visible next to it) and the grid outline in cyan
  <image>.png         one page per photo: the same overlay, full size, plus the givens as a
                      plain 9x9 grid (use this if the overlay looks misplaced)
  chosen.txt          which photos were drawn

The digits are placed using the THIRD-PARTY outline (outline_corners, unverified), not our
detector, so a mismatch can mean a wrong label OR a wrong outline: the plain grid panel tells
them apart. This script only draws; it never changes a label.
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

DATA = Path("data/real")
OUT = Path("outputs/real_label_check")
UNIT = np.float32([[0, 0], [9, 0], [9, 9], [0, 9]])  # tl, tr, br, bl in cell units
HALO = [pe.withStroke(linewidth=2.5, foreground="white")]


def load(jp: Path):
    meta = json.loads(jp.read_text(encoding="utf-8"))
    image = np.asarray(Image.open(jp.parent / meta["image"]).convert("RGB"))  # raw pixels
    return meta, image


def draw_overlay(ax, meta: dict, image: np.ndarray, crop: bool, fontsize: float) -> None:
    ax.imshow(image)
    ax.axis("off")
    g = np.array(meta["givens"])
    if meta["outline_corners"] is None:
        ax.set_title(f"{meta['image']}: no outline, use the grid panel", fontsize=8)
        return
    q = np.float32(meta["outline_corners"])
    h = cv2.getPerspectiveTransform(UNIT, q)
    ax.add_patch(plt.Polygon(q, fill=False, edgecolor="cyan", lw=1.2))
    for r in range(9):
        for c in range(9):
            if g[r, c]:
                x, y = cv2.perspectiveTransform(np.float32([[[c + 0.2, r + 0.27]]]), h)[0, 0]
                ax.text(x, y, str(g[r, c]), color="magenta", fontsize=fontsize,
                        fontweight="bold", ha="center", va="center", path_effects=HALO)  # fmt: skip
    if crop:
        pad = 0.06 * np.ptp(q, axis=0).max()
        x0, y0 = np.maximum(q.min(0) - pad, 0)
        x1, y1 = np.minimum(q.max(0) + pad, [image.shape[1], image.shape[0]])
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)


def draw_grid(ax, g: np.ndarray) -> None:
    ax.set_xlim(0, 9)
    ax.set_ylim(9, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        ax.axhline(i, color="k", lw=lw)
        ax.axvline(i, color="k", lw=lw)
    for r in range(9):
        for c in range(9):
            if g[r, c]:
                ax.text(c + 0.5, r + 0.5, str(g[r, c]), ha="center", va="center", fontsize=14)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    jsons = sorted(DATA.glob("wichtounet_v2_*/*.json"))
    if not jsons:
        raise SystemExit("no imported photos; run scripts/import_wichtounet.py first")
    chosen = sorted(random.Random(args.seed).sample(jsons, args.n), key=lambda p: p.stem)
    OUT.mkdir(parents=True, exist_ok=True)

    cols = 4
    rows = -(-len(chosen) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    for ax, jp in zip(axes.ravel(), chosen, strict=False):
        meta, image = load(jp)
        draw_overlay(ax, meta, image, crop=True, fontsize=7)
        ax.set_title(f"{meta['image']} ({meta['wichtounet_split']}, {meta['phone']})", fontsize=8)
    for ax in axes.ravel()[len(chosen) :]:
        ax.axis("off")
    fig.suptitle(
        "Imported givens (magenta, top-left of each cell) on the photo; cyan = third-party outline",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT / "contact_sheet.png", dpi=110)
    plt.close(fig)

    for jp in chosen:
        meta, image = load(jp)
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 7), width_ratios=[1.6, 1])
        draw_overlay(a0, meta, image, crop=False, fontsize=10)
        a0.set_title(f"{meta['image']}: photo + imported givens (magenta)")
        draw_grid(a1, np.array(meta["givens"]))
        a1.set_title("imported givens (.dat)")
        fig.tight_layout()
        fig.savefig(OUT / f"{jp.stem}.png", dpi=90)
        plt.close(fig)
    (OUT / "chosen.txt").write_text(
        "".join(f"{jp.parent.name}/{jp.stem}\n" for jp in chosen), encoding="utf-8"
    )
    print(f"{len(chosen)} photos -> {OUT}/contact_sheet.png (+ one page per photo)")


if __name__ == "__main__":
    main()
