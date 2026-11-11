"""Evaluate grid-corner detection on rendered images, clean and degraded.

    python scripts/eval_grid_detect.py --renders data/synthetic/clean --n 50

Each of the first --n renders is degraded with every noise type in synth/noise.py
(rotate, blur, gaussian_noise, jpeg, lighting, clutter) at each --severities
value, plus once clean. Ground-truth corners are moved with the image for
`rotate` (the only degradation that changes geometry).

A detection counts as a success when it returns corners AND their mean
distance to ground truth is below --tol pixels (default 5). The table reports
success rate and mean corner error over the detected images. Up to 5 failure
overlays per noise type go to outputs/vision_failures/<noise>/ (ground truth in
green, detected outline and implied cell grid in red). All rows are also written
to outputs/vision_failures/grid_detect.csv.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from sudoku_ns.synth import noise
from sudoku_ns.synth.render import load_render
from sudoku_ns.vision.cell_extract import overlay
from sudoku_ns.vision.grid_detect import GridNotFoundError, find_grid_corners

NOISES = ("rotate", "blur", "gaussian_noise", "jpeg", "lighting", "clutter")
MAX_SAVED = 5


def degrade(image, corners, name, s, seed):
    """Degraded image and its ground-truth corners."""
    if name == "clean":
        return image, corners
    if name == "clutter":
        return noise.clutter(image, corners, s, seed), corners
    if name == "rotate":
        M = noise.rotation_matrix(image.shape, s)
        return noise.rotate(image, s), cv2.transform(corners[None], M)[0]
    if name in ("gaussian_noise", "lighting"):
        return getattr(noise, name)(image, s, seed), corners
    return getattr(noise, name)(image, s), corners


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--renders", type=Path, default=Path("data/synthetic/clean"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--severities", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=Path("outputs/vision_failures"))
    args = ap.parse_args()

    renders = [load_render(jp) for jp in sorted(args.renders.glob("*.json"))[: args.n]]
    conditions = [("clean", 0.0)] + [(nm, s) for nm in NOISES for s in args.severities]
    rows, saved = [], {}
    for name, s in tqdm(conditions, desc="conditions"):
        for i, r in enumerate(renders):
            img, gt = degrade(r.image, r.corners, name, s, seed=i)
            try:
                det, why = find_grid_corners(img), ""
                err = float(np.linalg.norm(det - gt, axis=1).mean())
            except GridNotFoundError as e:
                det, why, err = None, str(e), float("nan")
            ok = det is not None and err < args.tol
            rows.append(
                {
                    "noise": name,
                    "severity": s,
                    "render": i,
                    "found": det is not None,
                    "error_px": err,
                    "success": ok,
                    "reason": why,
                }
            )
            if not ok and saved.get(name, 0) < MAX_SAVED:
                saved[name] = saved.get(name, 0) + 1
                d = args.out / name
                d.mkdir(parents=True, exist_ok=True)
                vis = overlay(img, det, gt) if det is not None else overlay(img, gt, gt)
                label = f"err {err:.1f}px" if det is not None else f"NOT FOUND: {why}"
                cv2.putText(vis, label[:70], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.imwrite(str(d / f"s{s:.2f}_{i:03d}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "grid_detect.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    n = len(renders)
    print(f"\n{n} renders per cell; success = found and mean corner error < {args.tol:g} px")
    print("cells: success % / mean error px over detected (not found count)\n")
    header = f"{'noise':<15}" + "".join(f"{f's={s:g}':>20}" for s in args.severities)
    clean = [r for r in rows if r["noise"] == "clean"]
    print(f"{'clean':<15}{_cell(clean):>20}\n")
    print(header)
    for nm in NOISES:
        line = f"{nm:<15}"
        for s in args.severities:
            line += f"{_cell([r for r in rows if r['noise'] == nm and r['severity'] == s]):>20}"
        print(line)
    print(f"\nfailure overlays: {args.out}/<noise>/  ({sum(saved.values())} saved)")


def _cell(rows) -> str:
    succ = np.mean([r["success"] for r in rows]) * 100
    errs = [r["error_px"] for r in rows if r["found"]]
    nf = sum(not r["found"] for r in rows)
    e = f"{np.mean(errs):.2f}" if errs else "-"
    return f"{succ:.0f}% / {e}" + (f" ({nf})" if nf else "")


if __name__ == "__main__":
    main()
