"""The trained CNN as a pipeline perceiver: CellImages -> ProbGrid.

Any callable CellImages -> ProbGrid can be passed to pipeline.run; this is the
real one (OraclePerceiver in pipeline.py is the simulated stand-in).

Output details, and why:
- all 81 cells in one batch (one forward pass, ~ms on CPU);
- logits divided by the fitted temperature T (calibrate.py) unless
  use_temperature=False, which exists for the calibration on/off ablation;
- probabilities clipped below at EPS and renormalised. EPS is the SAME floor
  the MaxSAT encoding uses (reasoning.encoding.EPS), so the clip changes
  nothing the solver would not do anyway, but it guarantees no exact zeros
  reach the solver and makes the output deterministic to compare;
- validated with interfaces.validate_prob_grid before it is returned.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sudoku_ns.interfaces import CellImages, ProbGrid, validate_cell_images, validate_prob_grid
from sudoku_ns.perception.train import load_model
from sudoku_ns.reasoning.encoding import EPS


class CNNPerceiver:
    def __init__(
        self, checkpoint_path: Path | str, use_temperature: bool = True, device: str | None = None
    ):
        self.model, ckpt = load_model(checkpoint_path, device)
        self.device = device or "cpu"
        t = ckpt.get("temperature")
        if use_temperature and t is None:
            raise ValueError(
                f"{checkpoint_path} has no fitted temperature; run "
                "python -m sudoku_ns.perception.calibrate --run <run dir>, "
                "or pass use_temperature=False"
            )
        self.temperature = float(t) if use_temperature else 1.0

    @torch.no_grad()
    def logits(self, cells: CellImages) -> np.ndarray:
        """(81, 10) raw logits, before temperature."""
        x = torch.from_numpy(validate_cell_images(cells)).unsqueeze(1).to(self.device)
        return self.model(x).cpu().numpy()

    def __call__(self, cells: CellImages) -> ProbGrid:
        z = torch.from_numpy(self.logits(cells)).double() / self.temperature
        p = torch.softmax(z, dim=1).numpy()
        p = np.maximum(p, EPS)
        p /= p.sum(axis=1, keepdims=True)
        return validate_prob_grid(p.reshape(9, 9, 10).astype(np.float32))
