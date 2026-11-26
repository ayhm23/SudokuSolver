# Real-photo test set

## What we use: the public wichtounet Sudoku dataset (v2)

Our real-photo test set is a public dataset of phone photos of newspaper Sudokus, not our own photos. The CNN never
trained on these images. They are a **test set only**: never train, tune, calibrate or choose thresholds on them.

| | |
|---|---|
| Source | https://github.com/wichtounet/sudoku_dataset |
| Commit used | `0db6de036b80b4e8e4574abe6e15026331bd5c2c` (2019-08-26, "Update README.rst") |
| Images used | the 200 **v2** photos listed in `datasets/v2_train.desc` (160) and `datasets/v2_test.desc` (40), files in `images/` with one `.dat` label each |
| Not used | `mixed/`, `mixed_incomplete/`, `mixed_natural/` (puzzles completed artificially) and v1 (superseded by v2) |
| Licence | the repository has **no LICENSE file**. Its README states: "The dataset and the images are released under the CC-BY-4.0 License." We use it for academic evaluation only and cite it as asked (below). |
| Citation | Wicht, B., Hennebert, J. "Camera-based Sudoku recognition with deep belief network." SoCPaR 2014, pp. 83-88. |

### Getting it

```
git clone https://github.com/wichtounet/sudoku_dataset data/external/wichtounet
git -C data/external/wichtounet checkout 0db6de036b80b4e8e4574abe6e15026331bd5c2c
python scripts/import_wichtounet.py      # -> data/real/wichtounet_v2_train/, data/real/wichtounet_v2_test/
```

`data/external/` and the imported folders are gitignored. The importer copies each photo and writes a JSON in the
same format as `scripts/label_real.py` (so `scripts/run_e2e.py` runs unchanged), plus: original split, phone model,
declared and actual resolution, and solvability flags. The solution comes from our SAT solver.

### What the import found

- **All 200 labels are valid puzzles with exactly one solution**; none had to be flagged. Mean 29.1 givens per puzzle.
- **Repeated puzzles.** The 200 photos show only **76 distinct puzzles**: 22 puzzles were photographed 2-18 times
  (146 photos), and 26 of the 40 test photos show a puzzle that also appears in a train photo (15 puzzles span both
  splits). The photos are therefore not independent samples. The importer stores `puzzle_id` (sha1 of the givens),
  and results report a cluster bootstrap over puzzles next to per-photo intervals.
- **No verified grid corners.** The dataset has a third-party file `outlines_sorted.csv` (4-point outlines,
  contributed to the dataset by another user) covering 199 of the 200 v2 photos (`image103.jpg` has none). We store it
  as `outline_corners` for a cross-check only, never as ground truth; it was not checked by us.
- **Declared vs actual resolution.** 44 photos say 1600x1200 or 1632x1224 in their `.dat` but are stored at
  640x480. Use `size_actual` (the real pixel size), not `resolution_declared`.
- **EXIF orientation.** 9 photos carry an EXIF orientation tag (8 x "rotate 90", 1 x "rotate 180"). Checked by eye:
  the `.dat` digits match the **raw** pixels, and applying the tag turns the grid sideways or upside down. So the
  pipeline uses raw pixels for these files (`load_render` does not apply EXIF), and the tag is kept as
  `exif_orientation_ignored`. The third-party outlines were drawn on the rotated image and are rotated back into
  the raw frame by the importer.
- Phones: Sony Ericsson w880i 50, w660i 46, t660i 35, s500i 5, w810i 1; Samsung Galaxy S4 23; iPhone 5s 20;
  iPhone 3GS 10; Nokia e65 7; Google Nexus 4 2; BlackBerry 1.

### Label spot-check

`python scripts/real_label_sheet.py` writes contact sheets of 20 random photos with the imported givens drawn on
them to `outputs/real_label_check/`, for a person to check by eye. Labels are never edited by us.

---

## Original plan: our own photos (not used for the results)

The protocol below was written for photographing our own puzzles. It is kept for reference; `label_real.py` still
works, but the reported real-photo results use the public dataset above.

This folder would hold the real Sudoku photos we use to check that the synthetic renders are not easier than reality.
The photos themselves are **not committed** (only this README is). Share them through the team drive, and copy
them here locally.

Target: **40 photos**, labelled with `scripts/label_real.py`. They are a **test set only**: never train, tune or
calibrate on them.

## What to photograph

Spread the 40 photos over these sources (the counts are targets, not hard rules):

| Source | Count | Notes |
|---|---|---|
| Newspaper | 10 | thin paper, grey print, text and other puzzles around the grid (clutter) |
| Puzzle book | 10 | clean print, may be bold or very thin grid lines |
| Screen (phone, laptop, tablet) | 10 | moiré, glare, backlight; photograph the screen, don't screenshot it |
| Printout / handwritten copy | 10 | our own A4 printouts of generated puzzles; handwritten givens are optional |

Only photograph **unsolved** puzzles (printed givens only). Pencil marks in a few photos are fine; note them in the
label file (`"notes"`).

## How to vary the photos

Across each source, mix these on purpose:

- **Angle**: straight on, about 20° tilted, about 40° tilted (strong perspective).
- **Rotation**: upright, and rotated 5-15° in the frame.
- **Lighting**: daylight, warm indoor lamp, a shadow across part of the grid (hand or phone), dim light.
- **Distance**: the grid filling about 80% of the frame, and about 40% (more background).
- **Focus**: mostly sharp; 3-4 slightly blurred photos.

The whole grid, including all four outer corners, must be inside the frame.

## Phone settings

- default camera app, no filters, no "document scan" mode (it straightens the grid, which hides what we test)
- JPEG, full resolution; `label_real.py` resizes as needed
- no flash, except in 2-3 photos to get glare

## File naming

```
<source>_<nn>_<condition>.jpg
```

- `source`: `news`, `book`, `screen` or `print`
- `nn`: two-digit counter per source, `01`-`10`
- `condition`: short tags joined by `-`, e.g. `tilt40-shadow`, `straight-daylight`, `rot10-dim-blur`

Example: `news_03_tilt20-lamp.jpg`, `screen_07_straight-glare.jpg`.

## Labelling

```
python scripts/label_real.py data/real/news_03_tilt20-lamp.jpg --checkpoint outputs/models/cnn_v1/model.pt
python scripts/label_real.py data/real/                   # every unlabelled photo in the folder
```

For each photo, the tool:

1. runs grid detection and shows the warped (straightened) grid in an OpenCV window, with the CNN's current reading
   drawn on top;
2. asks in the terminal whether the 4 corners are right. If they are not, you click the 4 corners
   (top-left, top-right, bottom-right, bottom-left) in the window;
3. shows the CNN's reading as 9 rows of digits, `.` = empty. Press Enter to accept it, or type the corrected row
   numbers;
4. saves `<photo name>.json` next to the photo: `givens` (9x9, 0 = empty), `corners` (tl, tr, br, bl, in image
   pixels), `source`, `conditions`, `notes`, and whether the corners were detected or clicked.

The ground truth is what is **printed**, read by a human, never the CNN's guess. Check every row against the
photo before accepting it.

## Checklist before handing the set over

- [ ] 40 photos, about 10 per source
- [ ] every photo has a `.json` label
- [ ] `python scripts/run_e2e.py` with a config pointing at `data/real` runs without crashing
