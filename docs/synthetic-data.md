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

```
python scripts/render_synthetic.py --n 200 --seed 0    # -> data/synthetic/clean/00000.png + 00000.json ...
```

What one render does:

```
generate(seed) ─► givens, solution
      │
      ▼
 _draw_sheet: off-white paper, thin cell lines (1-2 px), thick box lines (3-5 px),
              digits in one random font, size 55-75% of a cell, offset jitter ±6%
      │
      ▼
 random perspective warp (each corner moves up to 8% of the canvas)
      │
      ▼
 pasted on a 640x640 background: solid colour, gradient, or blotchy noise texture
      │
      ▼
 Render(image, givens, solution, corners, font) ─► PNG + JSON
```

The JSON holds `givens`, `solution`, `corners` (x, y in pixels, order tl, tr, br, bl) and `font`.
Corners are the **centre of the outer border line**, pushed through the same warp as the image, so they are exact.

## Step 3: noise (`synth/noise.py`)

Every function takes a severity `s` in [0, 1] mapped to one physical parameter, so results can be reported in
real units:

| function | parameter | s = 1 |
|---|---|---|
| `blur` | Gaussian sigma = 3.0·s px | strong defocus |
| `rotate` | angle = 15·s degrees | noticeably crooked photo |
| `jpeg` | quality = 95 − 85·s | quality 10, blocky |
| `gaussian_noise` | sigma = 25·s grey levels | dim-light sensor noise |

`apply_noise(img, {"blur": 0.3, "jpeg": 0.5})` applies them in camera order: rotate → blur → noise → JPEG.
`rotation_matrix(shape, s)` gives the matrix to rotate ground-truth corners the same way.

Tests (`tests/test_noise.py`): s = 0 is the identity, damage grows with s, shape and dtype are kept, a severity
outside [0, 1] is rejected.
