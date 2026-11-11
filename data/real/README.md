# Real-photo test set: capture protocol

This folder holds the real Sudoku photos we use to check that the synthetic renders are not easier than reality.
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
