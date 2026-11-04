# Vision: finding the grid and cutting out cells

Code: [`src/sudoku_ns/vision/`](../src/sudoku_ns/vision/). Tests: [`tests/test_vision.py`](../tests/test_vision.py).
Evaluation: `python scripts/eval_grid_detect.py --renders data/synthetic/test_clean --n 50`.

## Step 1: find the 4 grid corners (`grid_detect.py`)

"Propose many, verify each":

```
photo ─► 3 binarisations (plain / denoised / CLAHE), all with an adaptive threshold
      ─► ALL contours of each (RETR_LIST) ─► keep large ones (>= 5% of image)
      ─► convex hull ─► approxPolyDP to 4 corners  (= candidate quadrilaterals)
      ─► rank by shape: squareness × convexity × right angles × size
      ─► verify the best 40: warp to a square, check ink on the 10 expected line positions (grid_score)
      ─► best verified candidate ─► refine each side to the centre of the border line
      ─► corners tl, tr, br, bl
```

| Step | Why |
|---|---|
| adaptive threshold | each pixel is compared with its own neighbourhood, so a shadow across the grid does not break it |
| 3 binarisations | **plain** (5x5 blur) has the sharpest edges. **denoised** (stronger blur + morphological open/close) survives sensor noise, which turned the plain binary into speckle. **CLAHE** (local contrast equalisation) lifts faint ink in dark regions |
| RETR_LIST, not "outermost" or "largest" | the paper edge, a book page or a photo frame *encloses* the grid. Week 1's first version picked the paper about 70% of the time |
| convex hull before approxPolyDP | a noisy outline is ragged; its hull is still a clean quadrilateral |
| shape ranking | cheap prior so only plausible candidates are verified |
| `grid_score` verification | row and column projection profiles of the warped candidate: ink on the 10 line positions minus ink between them. The grid scores about 0.4; paper, frames and table edges score below 0.15. It is always computed on the *denoised* binary |
| `refine_corners` | contours follow the *outer* edge of the 3-5 px border line, but cell geometry is defined by its *centre*. Each side moves to the ink-weighted centre of the line. This halves the corner error (2.6 → 1.1 px on clean renders). We did not use `cv2.cornerSubPix`, which converges to an intensity corner, i.e. back to the outer edge |

Failure is explicit: `GridNotFoundError` says why ("no contours", "largest contour covers 2% of the image",
"none simplifies to a square-ish convex quadrilateral (grid cut off by the image edge?)", "none has grid lines").
The pipeline turns it into `SolveResult(status=ERROR, message="vision: ...")`.

## Step 2: cut 81 cells (`cell_extract.py`)

```
corners ─► perspective warp to 288x288 (32 px per cell) ─► 9x9 split
        ─► crop 8% inner margin ─► resize to 32x32
        ─► clean_cell: erase line remnants at the border, normalise brightness, centre the digit
        ─► /255 ─► CellImages float32 (81, 32, 32), validated
```

`clean_cell`, in detail:

1. Ink mask: pixels clearly darker than the paper level (the cell's 90th percentile).
2. Connected components. A blob that **touches the crop border and is centred in the outer band** is a grid-line
   remnant and is painted over with paper. A digit touching the border keeps a central centroid, so it survives.
3. Brightness gain so the paper becomes white (shadows change paper brightness from cell to cell).
4. The digit blob is the big, tall blob nearest the centre, plus any blob overlapping its box. The cell is
   **shifted** so that box is centred. Shift only, no scaling: digit size is real information.

**What it does NOT do: decide empty vs digit.** Nothing in the middle of a cell is ever erased, and a cell with no
digit-like blob is simply not shifted. Deciding "empty" is the CNN's job, because the CNN gives a *probability* that
MaxSAT can weigh. A hard "no blob → empty" rule here would bring back exactly the brittleness the project argues
against.

**The crop/dataset coupling.** The digit dataset is made by running `find_grid_corners` + `extract_cells` on
rendered images ([synthetic-data.md](synthetic-data.md#step-4-digit-dataset)), so the CNN trains on exactly what the
pipeline gives it. **Change anything here (margin, cleaning) → rebuild the dataset and retrain the CNN.**
`tests/test_dataset.py::test_crops_are_exactly_the_pipeline_crops` checks the two paths agree.

Debug: `extract_cells(image, corners, debug_path="x.png")` saves the photo with the detected grid drawn on it next
to the 81 crops as a 9x9 mosaic.

## Results (Week 2, real numbers)

`scripts/eval_grid_detect.py`: success = grid found **and** mean corner error < 5 px. Each cell shows success % /
mean corner error in px (number not found, if any). There are 50 renders per cell; noise types and severities are
defined in [synthetic-data.md](synthetic-data.md#step-3-noise-synthnoisepy).

**On held-out renders** (`test_clean`, first 50; the detector was not tuned on these):

| noise | s = 0.25 | s = 0.5 | s = 0.75 | s = 1.0 |
|---|---|---|---|---|
| clean | 100% / 1.09 | | | |
| rotate | 100% / 0.93 | 100% / 0.73 | 100% / 0.58 | 100% / 0.72 |
| blur | 100% / 1.06 | 100% / 1.07 | 100% / 1.20 | 100% / 1.05 |
| gaussian_noise | 100% / 1.09 | 100% / 1.12 | 100% / 1.06 | 100% / 1.04 |
| jpeg | 100% / 1.08 | 100% / 1.06 | 100% / 1.09 | 100% / 1.09 |
| lighting | 100% / 1.04 | 100% / 1.02 | 100% / 1.05 | 100% / 1.05 |
| clutter | 100% / 1.08 | 100% / 1.08 | 100% / 1.10 | 100% / 1.17 |

**Week 1 detector on the same table** (Week 1 `clean` renders 0-49), for comparison:

| noise | s = 0.25 | s = 0.5 | s = 0.75 | s = 1.0 |
|---|---|---|---|---|
| clean | 100% / 1.71 | | | |
| rotate | 100% / 1.72 | 100% / 1.69 | 100% / 1.66 | 98% / 1.65 (1) |
| blur | 100% / 1.74 | 100% / 1.91 | 100% / 2.12 | 100% / 2.40 |
| gaussian_noise | 100% / 1.67 | 94% / 1.69 (3) | **44%** / 1.76 (28) | **22%** / 1.86 (39) |
| jpeg | 100% / 1.72 | 100% / 1.71 | 100% / 1.74 | 100% / 1.70 |
| lighting | 100% / 1.71 | 100% / 1.69 | 100% / 1.69 | 100% / 1.68 |
| clutter | 100% / 1.70 | 100% / 1.67 | 100% / 1.64 | 100% / 1.62 |

What changed and why:

- **Gaussian noise** was the real weakness. The grid *was* found as a quadrilateral, but speckle between the lines
  pushed `grid_score` under its 0.15 threshold. Scoring on the denoised binary fixed it.
- **Clutter** with enclosing frames: in one image the real grid ranked below 12 perfect rectangles and was never
  verified. Verifying up to 40 candidates fixed it, at one 180x180 warp each.
- **Corner error** 1.7 → about 1.1 px from `refine_corners`.
- The single Week 1 rotation failure was a grid partly **outside the image**. That is not recoverable and is
  reported as "grid cut off by the image edge?".

Caveats: these are synthetic renders. Uneven lighting and clutter are our own simulations (`noise.lighting`,
`noise.clutter`); real photos (Week 3, [../data/real/README.md](../data/real/README.md)) are the real test. The
failure images that guided the fixes came from the Week 1 renders, which is why the table above uses `test_clean`.

Speed: about 15 ms for detection and 15 ms for cell extraction per 640x640 image on this laptop.
