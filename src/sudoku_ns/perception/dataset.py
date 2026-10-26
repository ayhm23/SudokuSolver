"""32x32 digit/empty crops for training the cell classifier.

Crops are cut from rendered puzzles with the SAME code the pipeline uses at
inference time (vision.cell_extract.warp_grid + crop_cell), only with
ground-truth corners instead of detected ones. So training crops contain what
real pipeline crops contain: grid-line remnants at the edges, off-centre
digits, empty cells with paper texture. A classifier trained on clean,
centred MNIST-style digits would see a different distribution at test time.

Split by font FAMILY, not by image or crop: every crop of one puzzle shares
its font, and Regular/Bold of one family share digit shapes. Splitting by
crop would put near-identical digits in train and test and overstate accuracy.

Augmentation (train split only) models what grid detection and the camera do
that clean renders do not: small shifts/scale/rotation (corner error),
blur (focus), contrast changes (lighting) and pixel noise (sensor).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from sudoku_ns.interfaces import CELL_SIZE
from sudoku_ns.synth.render import Render, load_render
from sudoku_ns.vision.cell_extract import DEFAULT_MARGIN, crop_cell, warp_grid

# Held-out families, chosen by hand so each split spans serif / sans / mono:
# test = serif + sans + mono, val = serif + condensed.
TEST_FAMILIES = ("lora", "opensans", "spacemono")
VAL_FAMILIES = ("merriweather", "oswald")


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


def crops_from_render(r: Render, margin: float = DEFAULT_MARGIN) -> tuple[np.ndarray, np.ndarray]:
    """81 uint8 crops (row-major) and their labels (0 = empty) from one render."""
    warped = warp_grid(r.image, r.corners)
    crops = np.stack([crop_cell(warped, i, j, margin) for i in range(9) for j in range(9)])
    return crops, r.givens.reshape(-1).astype(np.int8)


def augment(crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One randomly perturbed copy of a uint8 32x32 crop."""
    c = CELL_SIZE / 2
    angle = rng.uniform(-6, 6)  # degrees
    scale = rng.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((c, c), angle, scale)
    M[:, 2] += rng.uniform(-2, 2, size=2)  # pixel shift, like a ~2 px corner error
    out = cv2.warpAffine(
        crop, M, (CELL_SIZE, CELL_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    ).astype(np.float32)
    sigma = rng.uniform(0, 1.0)
    if sigma > 0.2:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    gain, bias = rng.uniform(0.6, 1.2), rng.uniform(-25, 25)
    out = (out - 128) * gain + 128 + bias
    out += rng.normal(0, rng.uniform(0, 10), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def build_dataset(
    render_dir: Path | str,
    n_aug: int = 2,
    seed: int = 0,
    font_manifest: Path | str | None = "data/fonts/fonts.json",
) -> dict[str, dict[str, np.ndarray]]:
    """{split: {"x": uint8 (N, 32, 32), "y": int8 (N,), "font": str (N,)}}.

    Train gets the original crops plus n_aug augmented copies each; val/test
    are left clean so they measure the pipeline as it is.
    """
    manifest = None
    if font_manifest and Path(font_manifest).exists():
        manifest = {
            m["file"]: m["family"] for m in json.loads(Path(font_manifest).read_text("utf-8"))
        }
    rng = np.random.default_rng(seed)
    parts: dict[str, dict[str, list]] = {
        s: {"x": [], "y": [], "font": []} for s in ("train", "val", "test")
    }
    jsons = sorted(Path(render_dir).glob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"no renders in {render_dir}; run scripts/render_synthetic.py")
    for jp in jsons:
        r = load_render(jp)
        split = split_of(font_family(r.font, manifest))
        crops, labels = crops_from_render(r)
        copies = [crops]
        if split == "train":
            copies += [np.stack([augment(c, rng) for c in crops]) for _ in range(n_aug)]
        for x in copies:
            parts[split]["x"].append(x)
            parts[split]["y"].append(labels)
            parts[split]["font"].append(np.full(len(labels), r.font))
    return {
        s: {
            "x": np.concatenate(d["x"])
            if d["x"]
            else np.zeros((0, CELL_SIZE, CELL_SIZE), np.uint8),
            "y": np.concatenate(d["y"]) if d["y"] else np.zeros(0, np.int8),
            "font": np.concatenate(d["font"]) if d["font"] else np.zeros(0, str),
        }
        for s, d in parts.items()
    }


def save_npz(ds: dict, path: Path | str) -> None:
    """One .npz with keys like train_x, train_y, train_font, val_x, ..."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"{s}_{k}": v for s, d in ds.items() for k, v in d.items()})


def class_counts(y: np.ndarray) -> np.ndarray:
    return np.bincount(y.astype(np.int64), minlength=10)


def torch_dataset(npz_path: Path | str, split: str):
    """A torch Dataset over one split; imports torch only when called (needs the `ml` extra).

    Items are (float tensor (1, 32, 32) in [0, 1], int label), the same scaling
    as CellImages from vision.cell_extract.
    """
    import torch
    from torch.utils.data import Dataset

    data = np.load(npz_path)
    x = torch.from_numpy(data[f"{split}_x"].astype(np.float32) / 255.0).unsqueeze(1)
    y = torch.from_numpy(data[f"{split}_y"].astype(np.int64))

    class DigitDataset(Dataset):
        def __len__(self) -> int:
            return len(y)

        def __getitem__(self, i: int):
            return x[i], y[i]

    return DigitDataset()
