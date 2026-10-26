"""Evaluate grid-corner detection on rendered images.

    python scripts/eval_grid_detect.py --renders data/synthetic/clean

A render counts as a success when detection returns corners AND their mean
distance to ground truth is below --tol pixels (default 5). Up to 3 failures
are saved to outputs/grid_detect_failures/ with ground truth in green and the
detection in red.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from sudoku_ns.synth.render import load_render
from sudoku_ns.vision.grid_detect import GridNotFoundError, find_grid_corners


def draw(image: np.ndarray, gt: np.ndarray, det: np.ndarray | None) -> np.ndarray:
    out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.polylines(out, [gt.astype(np.int32)], True, (0, 200, 0), 2)
    if det is not None:
        cv2.polylines(out, [det.astype(np.int32)], True, (0, 0, 255), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--renders", type=Path, default=Path("data/synthetic/clean"))
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=Path("outputs/grid_detect_failures"))
    args = ap.parse_args()

    jsons = sorted(args.renders.glob("*.json"))
    errors, failures = [], []
    for jp in jsons:
        r = load_render(jp)
        try:
            det = find_grid_corners(r.image)
        except GridNotFoundError as e:
            failures.append((jp.stem, r, None, f"not found: {e}"))
            continue
        err = float(np.linalg.norm(det - r.corners, axis=1).mean())
        errors.append(err)
        if err >= args.tol:
            failures.append((jp.stem, r, det, f"corner error {err:.1f} px"))

    n = len(jsons)
    print(f"renders:            {n}")
    print(f"detected:           {len(errors)}/{n}")
    print(f"success (< {args.tol:g} px):  {n - len(failures)}/{n} = {(n - len(failures)) / n:.1%}")
    if errors:
        e = np.array(errors)
        print(
            f"corner error (px):  mean {e.mean():.2f}  median {np.median(e):.2f}  "
            f"95th pct {np.percentile(e, 95):.2f}  max {e.max():.2f}  (over detected)"
        )
    if failures:
        args.out.mkdir(parents=True, exist_ok=True)
        print(f"\n{len(failures)} failure(s); saving up to 3 to {args.out}:")
        for stem, r, det, why in failures[:3]:
            path = args.out / f"{stem}.png"
            cv2.imwrite(str(path), draw(r.image, r.corners, det))
            print(f"  {stem}: {why}  (font {r.font}) -> {path}")


if __name__ == "__main__":
    main()
