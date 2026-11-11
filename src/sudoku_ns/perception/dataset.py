"""32x32 digit/empty crops for training the cell classifier.

THE CROPS COME FROM THE PIPELINE'S OWN CODE PATH. For every rendered image we
run grid detection (vision.grid_detect.find_grid_corners) and then
vision.cell_extract.extract_cells on the DETECTED corners, exactly as
pipeline.run does on a photo. So training crops contain what pipeline crops
contain: the same small corner errors, the same leftover line fragments, the
same cleaning and centring. A classifier trained on ground-truth-corner crops
(Week 1) or on clean MNIST-style digits would see a slightly different
distribution at test time. Consequence: whenever cell_extract or grid_detect
changes, this dataset must be rebuilt and the CNN retrained.

Images where detection fails are skipped and counted (the pipeline would not
produce crops for them either); detections farther than MAX_CORNER_ERROR px
from ground truth are skipped too, because their labels would no longer match
the cells (a crop shifted by half a cell shows the neighbour's digit).

Split by font FAMILY, not by image or crop: every crop of one puzzle shares
its font, and Regular/Bold of one family share digit shapes. Splitting by
crop would put near-identical digits in train and test and overstate accuracy.
Each split is rendered into its own directory from its own fonts and seeds
(scripts/render_synthetic.py --split ...), and build_split checks that every
render's font really belongs to the split it is used for.

Augmentation is ONLINE (a fresh random perturbation each epoch, train split
only, in the torch Dataset) rather than stored copies: more variety for the
same disk space. It models what the renders do not: small shifts/scale/rotation
(residual corner error), blur (focus), contrast changes (lighting) and pixel
noise (sensor). Val and test stay unaugmented so they measure the pipeline
as it is.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from sudoku_ns.interfaces import CELL_SIZE
from sudoku_ns.synth.render import Render, load_render
from sudoku_ns.vision.cell_extract import extract_cells
from sudoku_ns.vision.grid_detect import GridNotFoundError, find_grid_corners

# Held-out families, chosen by hand so each split spans serif / sans / mono:
# test = serif + sans + mono, val = serif + condensed.
TEST_FAMILIES = ("lora", "opensans", "spacemono")
VAL_FAMILIES = ("merriweather", "oswald")
MAX_CORNER_ERROR = 8.0  # px; ~1/6 of the smallest rendered cell (40 px)


def font_family(font_file: str, manifest: dict[str, str] | None = None) -> str:
    """Family for a font file: from fonts.json if available, else the file-name stem."""
    if manifest and font_file in manifest:
        return manifest[font_file]
    stem = font_file.split("[")[0].split("-")[0].split(".")[0]
    return stem.replace("_", "").lower()


def split_of(family: str) -> str:
    if family in TEST_FAMILIES:
        return "test"
    if family in VAL_FAMILIES:
        return "val"
    return "train"


def load_manifest(path: Path | str | None = "data/fonts/fonts.json") -> dict[str, str] | None:
    """{font file: family} from fonts.json, or None if it does not exist."""
    if path and Path(path).exists():
        return {m["file"]: m["family"] for m in json.loads(Path(path).read_text("utf-8"))}
    return None


def fonts_for_split(fonts: list[Path], split: str, manifest: dict | None = None) -> list[Path]:
    """The font files whose family belongs to `split` ("train", "val" or "test")."""
    out = [f for f in fonts if split_of(font_family(f.name, manifest)) == split]
    if not out:
        raise ValueError(f"no fonts for split {split!r}")
    return out


def crops_from_render(r: Render) -> tuple[np.ndarray, np.ndarray, float]:
    """81 uint8 crops (row-major), their labels (0 = empty), and the corner error in px.

    Uses detected corners (see module docstring). Raises GridNotFoundError.
    """
    corners = find_grid_corners(r.image)
    err = float(np.linalg.norm(corners - r.corners, axis=1).mean())
    cells = extract_cells(r.image, corners)
    crops = np.round(cells * 255).astype(np.uint8)
    return crops, r.givens.reshape(-1).astype(np.int8), err


def _process(args: tuple) -> tuple[str, dict | str]:
    json_path, manifest, split = args
    r = load_render(json_path)
    if split is not None and split_of(font_family(r.font, manifest)) != split:
        raise ValueError(f"{json_path}: font {r.font} does not belong to split {split!r}")
    try:
        crops, labels, err = crops_from_render(r)
    except GridNotFoundError as e:
        return "not_found", str(e)
    if err > MAX_CORNER_ERROR:
        return "bad_corners", f"{err:.1f} px"
    return "ok", {"x": crops, "y": labels, "font": r.font, "image": Path(json_path).stem}


def build_split(
    render_dir: Path | str,
    split: str | None = None,
    font_manifest: Path | str | None = "data/fonts/fonts.json",
    workers: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """({"x": uint8 (N,32,32), "y": int8 (N,), "font": str (N,), "image": str (N,)}, skip counts).

    split: if given, every render's font must belong to it (guards against
    mixing up directories, which would leak test fonts into training).
    """
    manifest = load_manifest(font_manifest)
    jsons = sorted(Path(render_dir).glob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"no renders in {render_dir}; run scripts/render_synthetic.py")
    jobs = [(jp, manifest, split) for jp in jsons]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(_process, jobs, chunksize=8))
    else:
        results = [_process(j) for j in jobs]
    counts = {"images": len(jsons), "not_found": 0, "bad_corners": 0}
    xs, ys, fonts, images = [], [], [], []
    for status, item in results:
        if status != "ok":
            counts[status] += 1
            continue
        xs.append(item["x"])
        ys.append(item["y"])
        fonts.append(np.full(81, item["font"]))
        images.append(np.full(81, item["image"]))
    data = {
        "x": np.concatenate(xs) if xs else np.zeros((0, CELL_SIZE, CELL_SIZE), np.uint8),
        "y": np.concatenate(ys) if ys else np.zeros(0, np.int8),
        "font": np.concatenate(fonts) if fonts else np.zeros(0, str),
        "image": np.concatenate(images) if images else np.zeros(0, str),
    }
    return data, counts


def augment(crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One randomly perturbed copy of a uint8 32x32 crop (light: crops are already realistic)."""
    c = CELL_SIZE / 2
    angle = rng.uniform(-5, 5)  # degrees
    scale = rng.uniform(0.92, 1.08)
    M = cv2.getRotationMatrix2D((c, c), angle, scale)
    M[:, 2] += rng.uniform(-1.5, 1.5, size=2)  # pixel shift, like residual corner error
    out = cv2.warpAffine(
        crop, M, (CELL_SIZE, CELL_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    ).astype(np.float32)
    sigma = rng.uniform(0, 1.0)
    if sigma > 0.3:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    gain, bias = rng.uniform(0.7, 1.1), rng.uniform(-20, 20)
    out = (out - 128) * gain + 128 + bias
    out += rng.normal(0, rng.uniform(0, 8), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_npz(ds: dict, path: Path | str) -> None:
    """One .npz with keys like train_x, train_y, train_font, val_x, ..."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"{s}_{k}": v for s, d in ds.items() for k, v in d.items()})


def class_counts(y: np.ndarray) -> np.ndarray:
    return np.bincount(y.astype(np.int64), minlength=10)


def torch_dataset(npz_path: Path | str, split: str, augment_seed: int | None = None):
    """A torch Dataset over one split; imports torch only when called (needs the `ml` extra).

    Items are (float tensor (1, 32, 32) in [0, 1], int label), the same scaling
    as CellImages from vision.cell_extract. augment_seed: if set, every
    __getitem__ returns a fresh `augment`ed copy (use for train only).
    """
    import torch
    from torch.utils.data import Dataset

    with np.load(npz_path) as data:
        x = data[f"{split}_x"]
        y = torch.from_numpy(data[f"{split}_y"].astype(np.int64))
    rng = np.random.default_rng(augment_seed) if augment_seed is not None else None

    class DigitDataset(Dataset):
        def __len__(self) -> int:
            return len(y)

        def __getitem__(self, i: int):
            img = augment(x[i], rng) if rng is not None else x[i]
            t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
            return t, y[i]

    return DigitDataset()
