# Experiment configs

JSON configs for `scripts/run_e2e.py` (keys are documented at the top of that script).

| Config | What it measures |
|---|---|
| `e2e_test_clean_cnn.json` | the real pipeline (detection + trained CNN + each method) on 200 clean renders in held-out test fonts |
| `e2e_test_noise_cnn.json` | the same on the small noisy test sets (50 images per noise type and severity), preliminary |
| `e2e_test_clean_oracle.json` | real vision, but the simulated CNN at the Week 1 error model (5% misreads): links the end-to-end numbers to the Week 1 sweep |

`tau` (threshold baseline) in the CNN configs is chosen on the VAL split with `scripts/tune_threshold.py`,
never on test.
