"""Small CNN that classifies one 32x32 grayscale cell crop into {empty, 1..9}.

Architecture: 3 conv blocks (32 -> 64 -> 128 channels), each two 3x3 convs
with BatchNorm + ReLU, then 2x2 max-pooling and dropout; global average
pooling; one linear layer to 10 logits. About 290k parameters.

Why this size: printed digits in 10 classes are an easy visual problem (no
handwriting variation), so capacity is not the bottleneck; generalising to
unseen FONTS is. A small model trains on a laptop CPU in minutes and has less
room to memorise font-specific details. Global average pooling (instead of a
large flattened dense layer) keeps the parameter count low and makes the
model tolerant of small shifts.

The model returns LOGITS, never probabilities: the loss (cross-entropy) wants
logits, and temperature scaling (calibrate.py) divides logits by T before the
softmax. A softmax inside the model would hide both from us.
"""

from __future__ import annotations

import torch
from torch import nn

NUM_CLASSES = 10


def _block(c_in: int, c_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout(dropout),
    )


class DigitCNN(nn.Module):
    """(N, 1, 32, 32) float in [0, 1] -> (N, 10) logits; class 0 = empty."""

    def __init__(self, channels: tuple[int, int, int] = (32, 64, 128), dropout: float = 0.2):
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            _block(1, c1, dropout), _block(c1, c2, dropout), _block(c2, c3, dropout)
        )
        self.head = nn.Linear(c3, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Centre inputs around 0 (paper ~1 -> ~0.5): plain affine, kept in the
        # model so callers always pass the CellImages scaling [0, 1].
        x = self.features(x - 0.5)
        return self.head(x.mean(dim=(2, 3)))


def pick_device() -> torch.device:
    """GPU if available, else CPU (this laptop: CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
