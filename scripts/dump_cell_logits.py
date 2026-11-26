"""Save the CNN's raw per-cell logits on rendered sets, for calibration analysis.

    python scripts/dump_cell_logits.py --renders "data/synthetic/test_final/*" \
        --out outputs/e2e/cell_logits_test_final.npz

Runs the real pipeline front end (grid detection -> cell extraction -> CNN) on
every image and stores, per cell: raw logits (before temperature), the true
class (0 = empty, 1-9), and which dataset it came from. Temperature is applied
later (scripts/make_report.py), so one dump serves both T = 1 and the fitted T.
Images where the grid is not found are skipped and counted.
"""

import argparse
import glob
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sudoku_ns.perception.perceiver import CNNPerceiver
from sudoku_ns.pipeline import perceive
from sudoku_ns.synth.render import load_render
from sudoku_ns.vision.grid_detect import GridNotFoundError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=Path("outputs/models/cnn_v1/model.pt"))
    ap.add_argument("--renders", nargs="+", required=True, help="dirs or globs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dirs = sorted({Path(p) for pat in args.renders for p in glob.glob(pat) if Path(p).is_dir()})
    perceiver = CNNPerceiver(args.checkpoint)
    logits, labels, dataset, skipped = [], [], [], 0
    for d in dirs:
        for jp in tqdm(sorted(d.glob("*.json")), desc=d.name):
            r = load_render(jp)
            try:
                _, cells, _ = perceive(r.image, perceiver)
            except GridNotFoundError:
                skipped += 1
                continue
            logits.append(perceiver.logits(cells))
            labels.append(r.givens.reshape(-1))
            dataset += [d.name] * 81
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        logits=np.concatenate(logits).astype(np.float32),
        labels=np.concatenate(labels).astype(np.int8),
        dataset=np.array(dataset),
        temperature=perceiver.temperature,
    )
    print(f"{len(labels)} images ({skipped} skipped: grid not found) -> {args.out}")


if __name__ == "__main__":
    main()
