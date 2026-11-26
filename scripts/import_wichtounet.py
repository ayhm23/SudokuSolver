"""Import the public wichtounet Sudoku photo dataset (v2) as a real-photo test set.

    git clone https://github.com/wichtounet/sudoku_dataset data/external/wichtounet
    python scripts/import_wichtounet.py            # -> data/real/wichtounet_v2_{train,test}/

Source and licence: data/real/README.md. We use ONLY the 200 v2 images (the files listed in
datasets/v2_train.desc and datasets/v2_test.desc). mixed/, mixed_incomplete/, mixed_natural/ and v1
are ignored: the mixed puzzles were completed artificially. The CNN never saw these photos, and
nothing may be tuned on them.

Each photo is copied next to a JSON in the format scripts/label_real.py writes, so
scripts/run_e2e.py runs on the folders unchanged. Differences from a hand-labelled photo:
- "corners" is null: there are no verified grid corners. run_e2e.py then judges the vision stage
  from the CNN reads instead (see "vision check" in run_e2e.py).
- "outline_corners": the third-party outline from outlines_sorted.csv (contributed to the dataset,
  not checked by us), converted to the same pixel frame as "givens" (see EXIF below). Kept for a
  cross-check only, never used as ground truth.
- the original split ("wichtounet_split": train/test), phone model, declared resolution (line 2 of
  the .dat) and actual pixel size are stored. 44 photos declared as 1600x1200 or 1632x1224 are
  stored at 640x480, so use "size_actual".
- "puzzle_id": the same puzzle is often photographed several times (200 photos show only 76
  distinct puzzles, and some test photos repeat a train puzzle), so results must not treat the
  photos as independent. puzzle_id = first 10 hex digits of sha1 over the 81 givens.
- the solution comes from our SAT solver. Labels whose puzzle is unsolvable or has several
  solutions are kept and flagged ("label_flags"), never changed.

.dat format: line 1 phone model, line 2 resolution ("640x480:24 JPG"), then 9 lines of 9 digits
(0 = empty).

EXIF: 9 v2 photos carry an orientation tag (6 = rotate 90, 3 = rotate 180). Checked by eye: the
.dat digits match the RAW pixel orientation, and applying the tag turns the grid sideways or upside
down. So we use raw pixels (load_render does not apply EXIF) and store
"exif_orientation_ignored". The outlines were drawn on the EXIF-rotated image, so they are rotated
back into the raw frame here.
"""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from sudoku_ns.reasoning.solver import count_solutions, sat_solve
from sudoku_ns.vision.grid_detect import order_corners

SRC = Path("data/external/wichtounet")
OUT = Path("data/real")
SPLITS = ("train", "test")


def parse_dat(path: Path) -> tuple[str, str, np.ndarray]:
    """(phone model, declared resolution, 9x9 givens) from a wichtounet .dat file."""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) != 11:
        raise ValueError(f"{path}: expected 11 non-empty lines, got {len(lines)}")
    rows = [[int(v) for v in ln.split()] for ln in lines[2:]]
    if any(len(r) != 9 for r in rows):
        raise ValueError(f"{path}: every grid line needs 9 digits")
    givens = np.array(rows, np.int8)
    if givens.min() < 0 or givens.max() > 9:
        raise ValueError(f"{path}: digits must be 0-9")
    resolution = lines[1].split(":")[0].strip()
    return lines[0], resolution, givens


def unit_repeats(g: np.ndarray) -> list[str]:
    out = []
    for i in range(9):
        b = g[3 * (i // 3) : 3 * (i // 3) + 3, 3 * (i % 3) : 3 * (i % 3) + 3]
        for kind, vals in (("row", g[i]), ("column", g[:, i]), ("box", b)):
            v = vals.ravel()
            v = v[v > 0]
            if len(v) != len(set(v.tolist())):
                out.append(f"{kind} {i + 1}")
    return out


def exif_to_raw(pts: np.ndarray, orientation: int, raw_w: int, raw_h: int) -> np.ndarray:
    """Points in the EXIF-rotated (displayed) frame -> raw pixel frame."""
    x, y = pts[:, 0], pts[:, 1]
    if orientation == 1:
        return pts
    if orientation == 3:  # displayed = raw rotated 180
        return np.stack([raw_w - 1 - x, raw_h - 1 - y], 1)
    if orientation == 6:  # displayed = raw rotated 90 clockwise
        return np.stack([y, raw_h - 1 - x], 1)
    if orientation == 8:  # displayed = raw rotated 90 counter-clockwise
        return np.stack([raw_w - 1 - y, x], 1)
    raise ValueError(f"EXIF orientation {orientation} not handled")


def load_outlines(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8") as f:
        return {
            Path(r["filepath"]).name: np.array(
                [[float(r[f"p{i}_x"]), float(r[f"p{i}_y"])] for i in range(1, 5)], np.float32
            )
            for r in csv.DictReader(f)
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    commit = subprocess.run(
        ["git", "-C", str(args.src), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    outlines = load_outlines(args.src / "outlines_sorted.csv")
    flagged, counts, puzzles = [], Counter(), {}
    for split in SPLITS:
        names = (args.src / "datasets" / f"v2_{split}.desc").read_text(encoding="utf-8").split()
        dest = args.out / f"wichtounet_v2_{split}"
        dest.mkdir(parents=True, exist_ok=True)
        for rel in names:
            photo = args.src / "images" / Path(rel).name
            phone, declared, givens = parse_dat(photo.with_suffix(".dat"))
            with Image.open(photo) as im:
                w, h = im.size
                orientation = int(im.getexif().get(274, 1))
            repeats = unit_repeats(givens)
            n_sol = 0 if repeats else count_solutions(givens, limit=2)
            solution = sat_solve(givens) if n_sol else None
            label_flags = []
            if repeats:
                label_flags.append("repeated digit in " + ", ".join(repeats))
            if n_sol == 0:
                label_flags.append("unsolvable")
            elif n_sol > 1:
                label_flags.append("multiple solutions")
            outline = outlines.get(photo.name)
            if outline is not None:
                outline = order_corners(exif_to_raw(outline, orientation, w, h))
            shutil.copy2(photo, dest / photo.name)
            meta = {
                "image": photo.name,
                "puzzle_id": hashlib.sha1(givens.tobytes()).hexdigest()[:10],
                "givens": givens.tolist(),
                "solution": solution.tolist() if solution is not None else None,
                "unique": n_sol == 1,
                "n_solutions_capped_at_2": n_sol,
                "label_flags": label_flags,
                "corners": None,
                "corners_source": "none (no verified corners; see outline_corners)",
                "outline_corners": outline.round(1).tolist() if outline is not None else None,
                "outline_source": "outlines_sorted.csv (third-party, unverified), raw pixel frame",
                "corner_order": "tl, tr, br, bl (x, y), in raw (non-EXIF-rotated) image pixels",
                "font": "real",
                "split": "real",
                "source": "wichtounet_v2",
                "source_commit": commit,
                "source_path": f"images/{photo.name}",
                "wichtounet_split": split,
                "phone": phone,
                "resolution_declared": declared,
                "size_actual": [w, h],
                "exif_orientation_ignored": orientation,
                "conditions": [],
                "notes": "",
            }
            (dest / f"{photo.stem}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
            counts[split] += 1
            counts["givens"] += int((givens > 0).sum())
            puzzles.setdefault(meta["puzzle_id"], set()).add(split)
            counts["no_outline"] += outline is None
            if label_flags:
                flagged.append((split, photo.name, "; ".join(label_flags)))

    n = counts["train"] + counts["test"]
    print(f"source commit {commit}")
    print(f"imported {n} photos: train {counts['train']}, test {counts['test']} -> {args.out}")
    print(f"mean givens per puzzle: {counts['givens'] / n:.1f}")
    print(f"distinct puzzles: {len(puzzles)} (in both train and test: "
          f"{sum(len(v) == 2 for v in puzzles.values())})")  # fmt: skip
    print(f"photos without a third-party outline: {counts['no_outline']}")
    print(f"flagged labels (kept unchanged): {len(flagged)}")
    for split, name, why in flagged:
        print(f"  {split:5s} {name}: {why}")


if __name__ == "__main__":
    main()
