import numpy as np
import pytest

from sudoku_ns.perception.dataset import (
    augment,
    build_dataset,
    class_counts,
    crops_from_render,
    font_family,
    save_npz,
    split_of,
    torch_dataset,
)
from sudoku_ns.synth.render import save_render


def test_font_family_groups_weights_together():
    assert font_family("FiraSans-Bold.ttf") == font_family("FiraSans-Regular.ttf") == "firasans"
    assert font_family("Lora[wght].ttf") == "lora"
    assert font_family("PT_Serif-Web-Bold.ttf") == "ptserif"
    assert font_family("x.ttf", {"x.ttf": "fam"}) == "fam"
    assert (
        split_of("lora") == "test" and split_of("oswald") == "val" and split_of("lato") == "train"
    )


def test_crops_match_labels(renders):
    crops, labels = crops_from_render(renders[0])
    assert crops.shape == (81, 32, 32) and crops.dtype == np.uint8
    np.testing.assert_array_equal(labels, renders[0].givens.ravel())
    # Digit crops contain dark ink; empty crops are mostly paper.
    ink = (crops < 100).mean(axis=(1, 2))
    assert ink[labels > 0].mean() > 3 * ink[labels == 0].mean()


def test_augment_changes_but_keeps_format(renders):
    crops, _ = crops_from_render(renders[0])
    out = augment(crops[0], np.random.default_rng(0))
    assert out.shape == (32, 32) and out.dtype == np.uint8
    assert not np.array_equal(out, crops[0])


def test_build_dataset_splits_and_saves(renders, tmp_path):
    for i, r in enumerate(renders[:2]):
        save_render(r, tmp_path / "renders", f"{i:05d}")
    ds = build_dataset(tmp_path / "renders", n_aug=1, font_manifest=None)
    # DejaVu Sans is not held out, so everything is train: 2 renders x 81 x (1 + 1 aug).
    assert len(ds["train"]["y"]) == 2 * 81 * 2 and len(ds["test"]["y"]) == 0
    assert class_counts(ds["train"]["y"]).sum() == len(ds["train"]["y"])
    save_npz(ds, tmp_path / "d.npz")
    with np.load(tmp_path / "d.npz") as z:
        assert z["train_x"].shape == (324, 32, 32)


def test_torch_wrapper(renders, tmp_path):
    pytest.importorskip("torch")
    save_render(renders[0], tmp_path / "r", "00000")
    save_npz(build_dataset(tmp_path / "r", n_aug=0, font_manifest=None), tmp_path / "d.npz")
    ds = torch_dataset(tmp_path / "d.npz", "train")
    x, y = ds[0]
    assert len(ds) == 81 and tuple(x.shape) == (1, 32, 32) and 0 <= float(x.max()) <= 1
