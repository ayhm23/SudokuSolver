"""Train the cell CNN on the digit dataset.

    python -m sudoku_ns.perception.train --data data/digits/digits.npz --run cnn_v1

Choices, and why:
- AdamW (lr 2e-3, weight decay 1e-4): Adam converges fast on small CNNs
  without LR tuning; decoupled weight decay is the standard mild regulariser.
- Cosine LR schedule over the maximum number of epochs, stepped per batch:
  large steps early, small steps at the end, no schedule hyper-parameters.
- Early stopping on VALIDATION LOSS (patience 3), keeping the best weights.
  Loss rather than accuracy because we care about the probabilities (they
  become MaxSAT weights); accuracy can stay flat while the model grows
  overconfident.
- Plain cross-entropy with NO class weights or balanced sampling, although
  ~63% of cells are empty. MaxSAT reads the softmax as P(class | crop).
  Reweighting would shift every prediction towards the digit classes by a
  fixed factor (a prior shift), and temperature scaling, a single scalar,
  cannot undo that; the result would be digits "seen" in empty cells. A
  63%-vs-4%-per-digit imbalance is mild, and per-class accuracy is reported
  so a weak digit class would be visible.
- Fixed seeds (python, numpy, torch, augmentation stream, data order).
- Light online augmentation on train only (dataset.augment).

The test split (held-out font families) is evaluated once at the end and
REPORTED ONLY: nothing (epochs, LR, temperature) is chosen on it.

Writes outputs/models/<run>/: model.pt (weights + config + temperature slot),
config.json, history.json, metrics.json.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from sudoku_ns.perception.dataset import torch_dataset
from sudoku_ns.perception.model import NUM_CLASSES, DigitCNN, pick_device


@dataclass
class TrainConfig:
    data: str = "data/digits/digits.npz"
    run: str = "cnn_v1"
    out_root: str = "outputs/models"
    epochs: int = 20
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 3
    dropout: float = 0.2
    seed: int = 0
    max_train: int | None = None  # subsample train (smoke tests only)
    max_eval: int | None = None  # subsample val/test (smoke tests only)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _subset(ds, n: int | None, seed: int):
    if n is None or n >= len(ds):
        return ds
    idx = np.random.default_rng(seed).choice(len(ds), n, replace=False)
    return Subset(ds, sorted(idx.tolist()))


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    """(logits (N, 10), labels (N,)) for a whole loader, model in eval mode."""
    model.eval()
    out, ys = [], []
    for x, y in loader:
        out.append(model(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


def classification_report(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Accuracy, per-class accuracy (recall) and confusion matrix (rows = true class)."""
    pred = logits.argmax(axis=1)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), int)
    np.add.at(cm, (labels, pred), 1)
    support = cm.sum(axis=1)
    per_class = np.where(support > 0, np.diag(cm) / np.maximum(support, 1), np.nan)
    return {
        "n": int(len(labels)),
        "accuracy": float((pred == labels).mean()),
        "per_class_accuracy": [None if np.isnan(a) else float(a) for a in per_class],
        "support": support.tolist(),
        "confusion": cm.tolist(),
    }


def train(cfg: TrainConfig) -> Path:
    """Train, save the best checkpoint, report val/test metrics. Returns the run directory."""
    seed_everything(cfg.seed)
    device = pick_device()
    run_dir = Path(cfg.out_root) / cfg.run
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = _subset(torch_dataset(cfg.data, "train", augment_seed=cfg.seed), cfg.max_train, 1)
    val_ds = _subset(torch_dataset(cfg.data, "val"), cfg.max_eval, 2)
    gen = torch.Generator().manual_seed(cfg.seed)
    train_dl = DataLoader(train_ds, cfg.batch_size, shuffle=True, generator=gen)
    val_dl = DataLoader(val_ds, 1024)

    model = DigitCNN(dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * len(train_dl))
    loss_fn = nn.CrossEntropyLoss()
    print(
        f"device {device}; train {len(train_ds)}, val {len(val_ds)} crops; "
        f"{sum(p.numel() for p in model.parameters())} parameters"
    )

    best_loss, best_state, best_epoch, bad, history = float("inf"), None, 0, 0, []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(y)
            n += len(y)
        logits, labels = predict_logits(model, val_dl, device)
        val_loss = float(loss_fn(torch.from_numpy(logits), torch.from_numpy(labels)))
        val_acc = float((logits.argmax(1) == labels).mean())
        history.append(
            {"epoch": epoch, "train_loss": total / n, "val_loss": val_loss, "val_acc": val_acc}
        )
        print(
            f"epoch {epoch:>2}  train loss {total / n:.4f}  val loss {val_loss:.4f}  "
            f"val acc {val_acc:.4f}  ({time.perf_counter() - t0:.0f} s)"
        )
        if val_loss < best_loss:
            best_loss, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"early stop: val loss has not improved for {cfg.patience} epochs")
                break

    model.load_state_dict(best_state)
    save_checkpoint(run_dir / "model.pt", model, cfg, temperature=None)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    metrics = {"best_epoch": best_epoch, "best_val_loss": best_loss}
    for split in ("val", "test"):
        ds = _subset(torch_dataset(cfg.data, split), cfg.max_eval, 2)
        if len(ds) == 0:
            continue
        logits, labels = predict_logits(model, DataLoader(ds, 1024), device)
        metrics[split] = classification_report(logits, labels)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print_report(metrics)
    return run_dir


def save_checkpoint(path: Path, model: nn.Module, cfg: TrainConfig, temperature: float | None):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(cfg),
            "temperature": temperature,  # filled in by calibrate.py
        },
        path,
    )


def load_model(path: Path | str, device=None) -> tuple[DigitCNN, dict]:
    """(model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model = DigitCNN(dropout=ckpt["config"].get("dropout", 0.2))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device or "cpu").eval(), ckpt


def print_report(metrics: dict) -> None:
    names = ["empty", *map(str, range(1, 10))]
    for split in ("val", "test"):
        if split not in metrics:
            continue
        m = metrics[split]
        print(f"\n{split}: accuracy {m['accuracy']:.4f} on {m['n']} crops")
        print(
            "  per class: "
            + "  ".join(
                f"{k}={a:.3f}" if a is not None else f"{k}=-"
                for k, a in zip(names, m["per_class_accuracy"], strict=True)
            )
        )
        print("  confusion (rows = true, cols = predicted):")
        print("        " + "".join(f"{k:>6}" for k in names))
        for k, row in zip(names, m["confusion"], strict=True):
            print(f"  {k:>5} " + "".join(f"{v:>6}" for v in row))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    for name, val in asdict(TrainConfig()).items():
        typ = type(val) if val is not None else int
        ap.add_argument(f"--{name.replace('_', '-')}", type=typ, default=val)
    cfg = TrainConfig(**vars(ap.parse_args()))
    train(cfg)


if __name__ == "__main__":
    main()
