# Synthetic data: fonts, rendered puzzles, noise

Code: [`src/sudoku_ns/synth/`](../src/sudoku_ns/synth/), scripts `download_fonts.py`, `render_synthetic.py`.
Everything produced here lives in `data/` (gitignored). Re-create it with the commands below.

## Why synthetic images at all?

- **Grid detection** needs the exact pixel position of the 4 grid corners to measure error. Nobody can click
  corners on 200 photos to the pixel; a renderer knows them exactly.
- **The CNN** (Week 2) needs thousands of labelled cell crops. The renderer gives the digit of every cell for free.
- Real photos still matter: Week 2 collects a small real test set to check that the synthetic data is not easier
  than reality.

## Step 1: fonts (`scripts/download_fonts.py`)

```
python scripts/download_fonts.py      # -> data/fonts/*.ttf, fonts.json, <family>_OFL.txt
```

15 OFL-licensed fonts from the `google/fonts` GitHub repo: serif (Merriweather, Lora, Libre Baskerville,
PT Serif), sans (Lato, Fira Sans, Open Sans), mono (Space Mono, IBM Plex Mono, Source Code Pro), condensed
(Barlow Condensed, Oswald) and bold (Fira Sans Bold, Lato Black, PT Serif Bold).

Why many fonts: the CNN must learn "what a 7 looks like", not "what a 7 looks like in Lato". The digit dataset
holds out whole fonts for the test split, which only means something if there are enough fonts to hold out.

Tests do not need the download: they use DejaVu Sans, which ships with matplotlib (`tests/conftest.py`).

## Step 2: render (`synth/render.py`, `scripts/render_synthetic.py`)

Week 2 data. Each split uses **only its own fonts** and **its own seeds**:

```
python scripts/render_synthetic.py --split train --n 2000 --seed 10000 --noise-frac 0.4 --out data/synthetic/train
python scripts/render_synthetic.py --split val   --n 300  --seed 20000 --noise-frac 0.4 --out data/synthetic/val
python scripts/render_synthetic.py --split test  --n 200  --seed 30000 --out data/synthetic/test_clean
# noisy test sets: same seeds as test_clean, so image i = test_clean image i + noise (a paired comparison)
python scripts/render_synthetic.py --split test --n 50 --seed 30000 --noise blur=0.5 --out data/synthetic/test_noise/blur_0.5
```

(The noisy sets are rotate, blur, gaussian_noise, jpeg and lighting, each at severity 0.25, 0.5 and 1.0: 15 sets
of 50.) `--noise-frac 0.4`: each train/val image gets random noise with probability 0.4, using 1-2 noise types at
severity 0.1-1 (`noise.random_spec`), so the CNN sees degraded crops. Rendering uses 10 worker processes; all
3,250 images took a few minutes.

What one render does:

```
generate(seed) ─► givens, solution
      │
      ▼
 _draw_sheet: off-white paper, thin cell lines (1-2 px), thick box lines (3-5 px),
              digits in one random font of the split, size 55-75% of a cell, offset jitter ±6%
      │
      ▼
 random perspective warp (each corner moves up to 8% of the canvas)
      │
      ▼
 pasted on a 640x640 background: solid colour, gradient, or blotchy noise texture
      │
      ▼
 optional noise (apply_noise); corners rotated with the image if the noise includes rotation
      │
      ▼
 Render(image, givens, solution, corners, font) ─► PNG + JSON
```

The JSON holds `givens`, `solution`, `corners` (x, y in pixels, order tl, tr, br, bl), `font`, `split`, `seed` and
`noise` (the spec applied, `{}` if clean). Corners are the **centre of the outer border line**, pushed through the
same warp as the image, so they are exact. The noise uses its own random stream, so a noisy render is exactly the
clean render plus noise.

## Step 3: noise (`synth/noise.py`)

Every function takes a severity `s` in [0, 1] mapped to one physical parameter, so results can be reported in
real units:

| function | parameter | s = 1 |
|---|---|---|
| `blur` | Gaussian sigma = 3.0·s px | strong defocus |
| `rotate` | angle = 15·s degrees | noticeably crooked photo |
| `jpeg` | quality = 95 − 85·s | quality 10, blocky |
| `gaussian_noise` | sigma = 25·s grey levels | dim-light sensor noise |
| `lighting` | darkest point = 1 − 0.7·s of the original brightness (a linear fall-off plus a soft shadow blob) | strong side light and a shadow |
| `clutter` | 12·s distractor objects around (never on) the grid: enclosing frames, boxes, lines, digit text, a small fake grid | busy desk |

`apply_noise(img, {"blur": 0.3, "jpeg": 0.5})` applies them in camera order: lighting → rotate → blur → noise →
JPEG. `noisy_corners(shape, corners, spec)` moves ground-truth corners the same way (only rotation moves them).
`clutter(img, corners, s)` needs the corners so it stays off the grid; covering the grid would be occlusion, a
different problem. It is used by the grid-detection evaluation, not by training (it never reaches the cell crops).

Tests (`tests/test_noise.py`): s = 0 is the identity, damage grows with s, shape and dtype are kept, a severity
outside [0, 1] is rejected, and clutter leaves the grid pixels untouched.

## Step 4: digit dataset

Code: `perception/dataset.py`, `scripts/make_digit_dataset.py`.

```
python scripts/make_digit_dataset.py --train data/synthetic/train --val data/synthetic/val     --test data/synthetic/test_clean          # -> data/digits/digits.npz
```

```
render (PNG + JSON)
   │  vision.grid_detect.find_grid_corners      ◄── DETECTED corners, as in the pipeline
   │  vision.cell_extract.extract_cells         ◄── the SAME crop + clean code the pipeline uses
   ▼
81 uint8 32x32 crops per image, label = given (0 = empty)
   │  skipped if the grid is not found or the corners are > 8 px off (labels would not match the cells)
   ▼
digits.npz: train_x, train_y, train_font, train_image, val_x, ..., test_image
```

**Why detected corners:** the CNN must see exactly what the pipeline will hand it, including small corner errors
and the cleaning/centring of `clean_cell`. `test_crops_are_exactly_the_pipeline_crops` checks that a dataset crop
is identical to a pipeline crop. Consequence: **if `cell_extract.py` or `grid_detect.py` changes, rebuild this
dataset and retrain the CNN.** It was rebuilt after the Week 2 vision changes.

Augmentation is **online** in `torch_dataset(npz, "train", augment_seed=...)`: a new random perturbation every
epoch (rotation ±5°, scale 0.92-1.08, shift ±1.5 px, blur, contrast/brightness, pixel noise). Val and test are
not augmented.

**Split by font family**, chosen by hand so each split has a mix of styles:

| split | families | used for |
|---|---|---|
| test | Lora (serif), Open Sans (sans), Space Mono (mono) | reporting only; never seen in training |
| val | Merriweather (serif), Oswald (condensed) | early stopping, temperature scaling, choosing tau |
| train | the other 7 families (10 font files) | training |

Families, not files: Fira Sans Regular and Fira Sans Bold have the same digit shapes, so splitting them would leak.
`build_split(dir, split)` refuses a render whose font is not in that split, so a mixed-up directory cannot leak test
fonts into training.

Class counts (Week 2, real output; 0 of 2,500 images skipped):

| split | images | crops | empty | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | empty % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 2000 | 162000 | 102000 | 6738 | 6622 | 6674 | 6633 | 6667 | 6810 | 6576 | 6655 | 6625 | 63.0 |
| val | 300 | 24300 | 15300 | 1000 | 979 | 948 | 1022 | 1012 | 1005 | 1021 | 1010 | 1003 | 63.0 |
| test | 200 | 16200 | 10200 | 680 | 659 | 654 | 648 | 677 | 644 | 679 | 665 | 694 | 63.0 |

Exactly 63.0% empty, because every puzzle has 30 givens (51 of 81 cells empty). **We do not rebalance**
(no class weights, no balanced sampling); see [perception.md](perception.md#class-imbalance) for why.

The Week 1 dataset (200 renders, ground-truth corners, 12% margin, 2 stored augmented copies) is replaced.

`torch_dataset` imports torch only inside the function (install with `pip install -e ".[ml]"`).
