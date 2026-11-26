#!/usr/bin/env bash
# Week 3 final-experiment renders (see docs/evaluation.md, "Week 3").
#   test_final/: TEST fonts, NEW seed 40000 (never used before), 100 images per set.
#                Every set uses the same seed, so image i is the same puzzle in all
#                16 sets (clean = severity 0), i.e. the sets are paired.
#   val_noise/:  VAL fonts, seed 50000, 50 images per set; used ONLY to choose tau
#                for threshold_then_sat (scripts/tune_threshold.py). Never test.
set -e
V=${PYTHON:-.venv/Scripts/python.exe}
export PYTHONUTF8=1
R="$V scripts/render_synthetic.py --workers 10"
$R --split test --n 100 --seed 40000 --out data/synthetic/test_final/clean
for t in rotate blur gaussian_noise jpeg lighting; do for s in 0.25 0.5 1.0; do
  $R --split test --n 100 --seed 40000 --noise $t=$s --out data/synthetic/test_final/${t}_$s
  $R --split val --n 50 --seed 50000 --noise $t=$s --out data/synthetic/val_noise/${t}_$s
done; done
echo ALL RENDERS DONE
