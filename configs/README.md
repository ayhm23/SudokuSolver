# Experiment configs

JSON configs for `scripts/run_e2e.py` (keys are documented at the top of that script).

| Config | What it measures |
|---|---|
| `e2e_test_clean_cnn.json` | the real pipeline (detection + trained CNN + each method) on 200 clean renders in held-out test fonts |
| `e2e_test_noise_cnn.json` | the same on the small noisy test sets (50 images per noise type and severity), preliminary |
| `e2e_test_clean_oracle.json` | real vision, but the simulated CNN at the Week 1 error model (5% misreads): links the end-to-end numbers to the Week 1 sweep |

`tau` (threshold baseline) in the CNN configs is chosen on the VAL split with `scripts/tune_threshold.py`,
never on test.

## Week 3 final experiments

Test images: `data/synthetic/test_final/` (TEST fonts, seed 40000, never used before; 100 images in each of
16 sets: `clean` plus 5 noise types x 3 severities, all paired on the same 100 puzzles). Rendered by
`scripts/render_final_sets.sh`, which also renders `data/synthetic/val_noise/` (VAL fonts) for choosing tau.

| Config | What it measures |
|---|---|
| `e2e_final_noise_cnn.json` | main sweep: maxsat_weighted, maxsat_unweighted, argmax_then_sat, threshold_then_sat |
| `e2e_final_noise_cnn_unique.json` | maxsat_unique on the same images (separate run: long runtime tail) |
| `ablations/weighted_default.json` | maxsat_weighted, SCALE 1000, EPS 1e-6, T on: the reference point for the ablations |
| `ablations/calib_no_temperature.json` | maxsat_weighted with T = 1 (no temperature scaling) |
| `ablations/scale_{10,100,10000}.json` | weight scale SCALE (default 1000 = `weighted_default`) |
| `ablations/eps_{1e-3,1e-9}.json` | probability floor EPS (default 1e-6 = `weighted_default`) |
| `ablations/sweep_simulated_n300.json` | reasoning in isolation: `python scripts/sweep_simulated.py --config ...` |

tau for threshold_then_sat: `python scripts/tune_threshold.py --checkpoint outputs/models/cnn_v1/model.pt
--renders "data/synthetic/val_noise/*" --out outputs/tune_threshold_val_noise.json` (VAL only) chose 0.5.
