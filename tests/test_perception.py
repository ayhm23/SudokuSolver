"""CNN, training, calibration and perceiver tests (skipped if torch is not installed)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sudoku_ns.interfaces import validate_prob_grid  # noqa: E402
from sudoku_ns.perception.calibrate import ece, fit_temperature, nll, softmax  # noqa: E402
from sudoku_ns.perception.dataset import build_split, save_npz  # noqa: E402
from sudoku_ns.perception.model import DigitCNN  # noqa: E402
from sudoku_ns.perception.perceiver import CNNPerceiver  # noqa: E402
from sudoku_ns.perception.train import TrainConfig, save_checkpoint, train  # noqa: E402
from sudoku_ns.synth.render import save_render  # noqa: E402
from sudoku_ns.vision.cell_extract import extract_cells  # noqa: E402


def test_model_forward_shape():
    model = DigitCNN().eval()
    out = model(torch.rand(5, 1, 32, 32))
    assert tuple(out.shape) == (5, 10)
    # Logits, not probabilities: rows need not sum to 1.
    assert not torch.allclose(out.sum(1), torch.ones(5))


def test_perceiver_output_is_a_valid_prob_grid(renders, tmp_path):
    torch.manual_seed(0)
    save_checkpoint(tmp_path / "m.pt", DigitCNN(), TrainConfig(), temperature=1.7)
    r = renders[0]
    cells = extract_cells(r.image, r.corners)
    prob = CNNPerceiver(tmp_path / "m.pt")(cells)
    validate_prob_grid(prob)
    assert prob.min() >= 1e-6 * 0.99
    raw = CNNPerceiver(tmp_path / "m.pt", use_temperature=False)
    assert raw.temperature == 1.0


def test_perceiver_requires_fitted_temperature(tmp_path):
    save_checkpoint(tmp_path / "m.pt", DigitCNN(), TrainConfig(), temperature=None)
    with pytest.raises(ValueError, match="temperature"):
        CNNPerceiver(tmp_path / "m.pt")


def test_temperature_fixes_an_overconfident_model():
    # True model: logits with moderate margins, labels sampled from its softmax,
    # so it is calibrated at T = 1. Multiplying the logits by 4 makes it
    # overconfident; fitting T should recover ~4 and lower NLL and ECE.
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 1.5, size=(4000, 10))
    p = softmax(logits)
    labels = np.array([rng.choice(10, p=row) for row in p])
    over = logits * 4
    t = fit_temperature(over, labels)
    assert 3.0 < t < 5.0
    assert nll(over, labels, t) < nll(over, labels)
    assert ece(softmax(over, t), labels) < ece(softmax(over), labels)
    # Accuracy cannot change: dividing by T > 0 keeps the argmax.
    np.testing.assert_array_equal(softmax(over, t).argmax(1), over.argmax(1))


def test_training_smoke(renders, tmp_path):
    for i, r in enumerate(renders[:4]):
        save_render(r, tmp_path / "renders", f"{i:05d}")
    data, _ = build_split(tmp_path / "renders", font_manifest=None)
    # 324 crops; reuse them as val/test (this only checks the plumbing).
    save_npz({"train": data, "val": data, "test": data}, tmp_path / "d.npz")
    cfg = TrainConfig(
        data=str(tmp_path / "d.npz"),
        run="smoke",
        out_root=str(tmp_path),
        epochs=1,
        max_train=300,
        max_eval=200,
        batch_size=64,
    )
    run_dir = train(cfg)
    for name in ("model.pt", "config.json", "history.json", "metrics.json"):
        assert (run_dir / name).exists()
