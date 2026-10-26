# Vision: finding the grid and cutting out cells

Code: [`src/sudoku_ns/vision/`](../src/sudoku_ns/vision/). Tests: [`tests/test_vision.py`](../tests/test_vision.py).
Evaluation: `python scripts/eval_grid_detect.py`.

## Step 1: find the 4 grid corners (`grid_detect.py`)

```
photo ─► grayscale ─► 5x5 Gaussian blur ─► adaptive threshold (ink = white)
      ─► ALL contours (RETR_LIST) ─► keep large ones (>= 5% of image)
      ─► approxPolyDP: keep those that simplify to 4 convex corners
      ─► grid_score each candidate ─► best one ─► order as tl, tr, br, bl
```

| Step | Why |
|---|---|
| adaptive threshold | each pixel is compared with its 11x11 neighbourhood, so shadows and uneven light do not break it |
| RETR_LIST, not RETR_EXTERNAL | the paper sheet's edge is also a closed shape that *encloses* the grid. Our first version used "outermost contour" and found the paper ~70% of the time (corner errors 40-116 px) |
| `grid_score` | warps a candidate to a square and checks for ink on the 10 expected line positions. The real grid scores high; paper, photo frames and table edges score low |
| `order_corners` | smallest x+y = top-left, largest = bottom-right, smallest y−x = top-right, largest = bottom-left |

Failure is explicit: `GridNotFoundError` says why ("no contours", "largest contour covers 2% of the image",
"none simplifies to a convex quadrilateral", "none has grid lines").

## Step 2: cut 81 cells (`cell_extract.py`)

```
corners ─► perspective warp to 288x288 (32 px per cell) ─► 9x9 split
        ─► crop 12% inner margin (drops most grid-line pixels) ─► resize to 32x32 ─► /255
        ─► CellImages float32 (81, 32, 32), validated
```

Paper is ~1.0 and ink ~0.0. The digit dataset is built with these same functions, so the CNN trains on exactly
what it will see.

## Results (Week 1, real numbers)

`scripts/eval_grid_detect.py` on the 200 clean renders:

| metric | value |
|---|---|
| detected | 200/200 |
| success (mean corner error < 5 px) | 200/200 = 100% |
| corner error, mean / median / 95th pct / max | 1.74 / 1.66 / 2.62 / 3.27 px |

No failures, so no failure images. Clean renders are the easy case. A quick check on 50 renders with blur,
Gaussian noise and JPEG at equal severity (not part of the script yet):

| severity | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|
| success | 50/50 | 50/50 | 50/50 | 30/50 (19 not found) |

The remaining ~1.7 px error is mostly systematic: contours follow the edge of the 3-5 px border line, while
ground truth is its centre.
