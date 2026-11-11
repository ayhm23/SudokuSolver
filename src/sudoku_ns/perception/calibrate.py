"""Temperature scaling: make the CNN's probabilities mean what they say.

    python -m sudoku_ns.perception.calibrate --run outputs/models/cnn_v1

WHY: MaxSAT turns each probability into a weight -log p (reasoning/encoding.py),
so the solver trades one cell's confidence against another's. That trade is
only meaningful if p is calibrated: among cells the CNN calls 0.9, about 90%
should be right. Networks trained with cross-entropy are typically
overconfident (Guo et al., 2017, "On Calibration of Modern Neural Networks"):
a wrong reading at p = 0.999 would cost -log(0.001) ~ 6.9 nats to override,
so the solver could not fix it even when the rules clearly say it is wrong.

WHAT: divide all logits by one scalar T > 0 before the softmax. T > 1 softens
(less confident), T < 1 sharpens. One parameter cannot change which class is
the argmax, so ACCURACY IS UNCHANGED; only the confidence moves. That is why it
is safe to fit after training and why it cannot overfit much.

HOW: choose T minimising the negative log-likelihood (NLL) of the VALIDATION
set, with LBFGS on log T (so T stays positive). NLL is the right objective
because it is exactly the quantity MaxSAT sums. The test set is never used to
fit or choose anything; it is only reported.

Reported before and after: NLL, accuracy, and ECE (expected calibration error,
15 equal-width confidence bins: the average |accuracy - confidence| weighted by
bin size), plus a reliability diagram (accuracy vs confidence per bin; a
calibrated model lies on the diagonal).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

N_BINS = 15


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    t = torch.as_tensor(logits, dtype=torch.float64) / temperature
    return float(torch.nn.functional.cross_entropy(t, torch.as_tensor(labels, dtype=torch.long)))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error of the top-class confidence, equal-width bins."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 100) -> float:
    """T > 0 minimising validation NLL (LBFGS on log T, starting at T = 1)."""
    x = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(x / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def summary(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> dict:
    probs = softmax(logits, temperature)
    return {
        "nll": nll(logits, labels, temperature),
        "accuracy": float((probs.argmax(1) == labels).mean()),
        "ece": ece(probs, labels),
        "mean_confidence": float(probs.max(1).mean()),
    }


def reliability_diagram(
    logits: np.ndarray, labels: np.ndarray, temperature: float, path: Path, title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    edges = np.linspace(0, 1, N_BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    for ax, t, name in ((axes[0], 1.0, "before (T = 1)"), (axes[1], temperature, None)):
        probs = softmax(logits, t)
        conf, correct = probs.max(1), probs.argmax(1) == labels
        acc = [
            correct[(conf > lo) & (conf <= hi)].mean() if ((conf > lo) & (conf <= hi)).any() else 0
            for lo, hi in zip(edges[:-1], edges[1:], strict=True)
        ]
        ax.bar(centres, acc, width=1 / N_BINS, edgecolor="#333", color="#4c78a8", alpha=0.85)
        ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
        ax.set_xlabel("confidence (top-class probability)")
        ax.set_title(f"{name or f'after (T = {t:.3f})'}\nECE {ece(probs, labels):.4f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy in bin")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def calibrate_run(run_dir: Path | str) -> dict:
    """Fit T on val, report val and test before/after, store T in model.pt and metrics.json."""
    from sudoku_ns.perception.dataset import torch_dataset
    from sudoku_ns.perception.train import load_model, predict_logits

    run_dir = Path(run_dir)
    model, ckpt = load_model(run_dir / "model.pt")
    data = ckpt["config"]["data"]
    out = {}
    logits = {}
    for split in ("val", "test"):
        ds = torch_dataset(data, split)
        logits[split] = predict_logits(model, DataLoader(ds, 1024), "cpu")
    t = fit_temperature(*logits["val"])
    out["temperature"] = t
    for split, (lg, y) in logits.items():
        out[split] = {"before": summary(lg, y), "after": summary(lg, y, t)}
        reliability_diagram(
            lg, y, t, run_dir / f"reliability_{split}.png", f"{split} split, {len(y)} crops"
        )
    ckpt["temperature"] = t
    torch.save(ckpt, run_dir / "model.pt")
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text("utf-8")) if metrics_path.exists() else {}
    metrics["calibration"] = out
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="outputs/models/<run>")
    args = ap.parse_args()
    out = calibrate_run(args.run)
    print(f"temperature T = {out['temperature']:.4f} (fitted on val)\n")
    print(f"{'split':<6} {'':<7} {'NLL':>8} {'acc':>8} {'ECE':>8} {'mean conf':>10}")
    for split in ("val", "test"):
        for when in ("before", "after"):
            m = out[split][when]
            print(
                f"{split:<6} {when:<7} {m['nll']:>8.4f} {m['accuracy']:>8.4f} "
                f"{m['ece']:>8.4f} {m['mean_confidence']:>10.4f}"
            )
    print(f"\nreliability diagrams: {args.run}/reliability_{{val,test}}.png")


if __name__ == "__main__":
    main()
