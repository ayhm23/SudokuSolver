# Viva demo script (3 minutes)

Three images, chosen beforehand, each checked in the app and on the command line on 2026-11-26. Every number below
is what the system printed in those checks. In the app each image takes about 1 s to show (vision + CNN +
reasoning; the solve itself is 40-100 ms). Do **not** tick `maxsat_unique` for image 3: it hits its 10 s limit there.

| # | what it shows | input in the app | same image as a file |
|---|---|---|---|
| 1 | clean success | test set *synthetic*, image `00000`, noise *none* | `data/synthetic/test_final/clean/00000.png` |
| 2 | MaxSAT corrects, argmax fails | test set *synthetic*, image `00018`, noise *blur*, severity *0.25* | `data/synthetic/test_final/blur_0.25/00018.png` |
| 3 | honest failure | test set *real photo*, image `image1009`, noise *none* | `data/real/wichtounet_v2_test/image1009.jpg` |

Image 2 in the app (clean render + blur 0.25, seed 0) is pixel-identical to the `test_final` file, so the app, the
CLI and gallery figure a1 all show the same case.

## Before the viva

```powershell
.venv\Scripts\activate
streamlit run demo/app.py        # open http://localhost:8501
```

Click through image 1 once before you start: the first run loads the CNN (a few seconds). Keep a second terminal
open with the venv active, for the fallback below.

## Script

**0:00-0:40 · Image 1: clean success.** Say what the pipeline does: find the grid, cut 81 cells, the CNN gives a
probability for each digit (and "empty") per cell, and weighted MaxSAT picks the most probable reading that is a
valid Sudoku. On screen:
- detected grid sits on the printed grid; the heatmap is uniformly confident;
- "CNN argmax cells wrong vs ground truth: 0 / 81";
- maxsat_weighted: solved, correct, unique yes, no overrides (cost 0). argmax_then_sat: solved, correct too.

Point: when perception is right, reasoning stays out of the way.

**0:40-1:50 · Image 2: blur 0.25. MaxSAT corrects, the baseline fails.** Switch the noise to *blur*, severity 0.25.
- Heatmap: one cell stands out, (3, 4) (row 3, column 4, counting from 0), where the CNN reads 6 with p = 0.69.
- "CNN argmax cells wrong vs ground truth: 1 / 81".
- argmax_then_sat: **UNSAT** (a 6 at (3, 4) clashes with the 6 at (6, 4)), so the baseline returns nothing.
- maxsat_weighted: **solved, correct**, 1 override, explained as
  `(3, 4): CNN read 6 (p=0.69) -> solver reads 5 (p=0.31); conflicts with given 6 at (6, 4) in column 4`.

Point: the CNN's second choice had 31 % probability. The solver pays that cost because the rules forbid the first
choice. The explanation is an unsat core, a proof, not a heuristic.

**1:50-2:45 · Image 3: a real photo where we fail.** Switch to *real photo*, `image1009`.
- The overlay shows the grid detector locked onto **one 3x3 box** (bottom right), not the whole grid.
- "CNN argmax cells wrong vs ground truth: 29 / 81". The reads are of the wrong cells.
- argmax_then_sat: UNSAT. maxsat_weighted: **solved, WRONG**, and **unique solution: NO**.

Point: reasoning cannot repair a vision failure; it makes the reads consistent, not true. The "unique: NO" flag
is the warning sign; a real system should refuse to answer here. This is 1 of the 3 vision failures among the
200 real photos (`report/results/real_photos.md`).

**2:45-3:00 · Close.** On the 200 real photos: maxsat_weighted solves 195 (97.5 %), argmax_then_sat 176 (88.0 %),
same CNN, same probabilities (`report/results/real_photos.md`). More cases, including the solver overriding a
correct read: `report/results/gallery.md`.

## Fallback if Streamlit misbehaves

Same three images on the command line (PowerShell, from the repo root, venv active). Each prints the CNN reading,
the solver's reading with every override explained, and the solution, and saves a picture (detected grid | CNN reads
with confidences | solution). About 4 s each, mostly loading torch.

```powershell
# 1. clean success
python -m sudoku_ns.pipeline data/synthetic/test_final/clean/00000.png --checkpoint outputs/models/cnn_v1/model.pt --out outputs/demo_1.png
start outputs/demo_1.png

# 2. MaxSAT corrects ...
python -m sudoku_ns.pipeline data/synthetic/test_final/blur_0.25/00018.png --checkpoint outputs/models/cnn_v1/model.pt --out outputs/demo_2.png
start outputs/demo_2.png
#    ... and the baseline is UNSAT on the same image
python -m sudoku_ns.pipeline data/synthetic/test_final/blur_0.25/00018.png --checkpoint outputs/models/cnn_v1/model.pt --method argmax_then_sat --out outputs/demo_2_argmax.png

# 3. honest failure
python -m sudoku_ns.pipeline data/real/wichtounet_v2_test/image1009.jpg --checkpoint outputs/models/cnn_v1/model.pt --out outputs/demo_3.png
start outputs/demo_3.png
```

What they print (checked): (1) `status=solved unique=True cost=0`, no overrides; (2) `status=solved unique=True
cost=792` with the (3, 4) override above, and the argmax_then_sat run prints `status=unsat` and
`reasoning: the read givens break a Sudoku rule or have no solution`; (3) `status=solved unique=False cost=577`
with one override, `(7, 7): CNN read 4 (p=0.49) -> solver reads empty (p=0.27); conflicts with given 4 at (6, 7)
in column 7`.

If nothing runs at all, show the pre-made figures: `report/figures/gallery/a1_blur_0.25_00018.png` (image 2) and
`report/figures/gallery/r2_wichtounet_v2_test_image1009.png` (image 3).
