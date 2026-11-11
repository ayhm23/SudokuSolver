import numpy as np
import pytest

from sudoku_ns.perception.dataset import (
    augment,
    build_split,
    class_counts,
    crops_from_render,
    font_family,
    save_npz,
    split_of,
    torch_dataset,
)
from sudoku_ns.synth.render import save_render
from sudoku_ns.vision.cell_extract import extract_cells
from sudoku_ns.vision.grid_detect import find_grid_corners


def test_font_family_groups_weights_together():
    assert font_family("FiraSans-Bold.ttf") == font_family("FiraSans-Regular.ttf") == "firasans"
    assert font_family("Lora[wght].ttf") == "lora"
    assert font_family("PT_Serif-Web-Bold.ttf") == "ptserif"
    assert font_family("x.ttf", {"x.ttf": "fam"}) == "fam"
    assert (
        split_of("lora") == "test" and split_of("oswald") == "val" and split_of("lato") == "train"
    )


def test_crops_are_exactly_the_pipeline_crops(renders):
    # The coupling the CNN relies on: dataset crops == what the pipeline feeds it.
    r = renders[0]
    crops, labels, err = crops_from_render(r)
    pipeline = extract_cells(r.image, find_grid_corners(r.image))
    np.testing.assert_array_equal(crops, np.round(pipeline * 255).astype(np.uint8))
    np.testing.assert_array_equal(labels, r.givens.ravel())
    assert err < 5.0
    ink = (crops < 100).mean(axis=(1, 2))
    assert ink[labels > 0].mean() > 3 * ink[labels == 0].mean()


def test_augment_changes_but_keeps_format(renders):
    crops, _, _ = crops_from_render(renders[0])
    out = augment(crops[0], np.random.default_rng(0))
    assert out.shape == (32, 32) and out.dtype == np.uint8
    assert not np.array_equal(out, crops[0])


def test_build_split_checks_fonts_and_saves(renders, tmp_path):
    for i, r in enumerate(renders[:2]):
        save_render(r, tmp_path / "renders", f"{i:05d}")
    # DejaVu Sans is not held out, so it belongs to train ...
    data, counts = build_split(tmp_path / "renders", "train", font_manifest=None)
    assert len(data["y"]) == 2 * 81 and counts["not_found"] == 0
    assert class_counts(data["y"]).sum() == len(data["y"])
    save_npz({"train": data}, tmp_path / "d.npz")
    with np.load(tmp_path / "d.npz") as z:
        assert z["train_x"].shape == (162, 32, 32)
    # ... and using it as a test directory is refused (would leak fonts).
    with pytest.raises(ValueError, match="does not belong"):
        build_split(tmp_path / "renders", "test", font_manifest=None)


def test_torch_wrapper(renders, tmp_path):
    pytest.importorskip("torch")
    save_render(renders[0], tmp_path / "r", "00000")
    data, _ = build_split(tmp_path / "r", font_manifest=None)
    save_npz({"train": data}, tmp_path / "d.npz")
    ds = torch_dataset(tmp_path / "d.npz", "train")
    x, y = ds[0]
    assert len(ds) == 81 and tuple(x.shape) == (1, 32, 32) and 0 <= float(x.max()) <= 1
    aug = torch_dataset(tmp_path / "d.npz", "train", augment_seed=0)
    assert not np.array_equal(aug[0][0].numpy(), x.numpy())
