# Perception: the cell CNN and its calibration

Code: [`src/sudoku_ns/perception/`](../src/sudoku_ns/perception/). Tests:
[`tests/test_perception.py`](../tests/test_perception.py), [`tests/test_dataset.py`](../tests/test_dataset.py).
Data: [synthetic-data.md](synthetic-data.md#step-4-digit-dataset).

```
python -m sudoku_ns.perception.train --run cnn_v1                       # -> outputs/models/cnn_v1/
python -m sudoku_ns.perception.calibrate --run outputs/models/cnn_v1     # T fitted on val, stored in model.pt
```

## Model (`model.py`)

```
(N, 1, 32, 32) in [0, 1] ─► minus 0.5
  ─► [conv3x3 32 ─ BN ─ ReLU ─ conv3x3 32 ─ BN ─ ReLU ─ maxpool 2 ─ dropout 0.2]   32x32 → 16x16
  ─► [same with 64 channels]                                                       16x16 → 8x8
  ─► [same with 128 channels]                                                       8x8 → 4x4
  ─► global average pool ─► linear 128 → 10 ─► LOGITS  (class 0 = empty, 1-9 = digits)
```

288,170 parameters. Small on purpose: printed digits are an easy visual problem, so capacity is not the
bottleneck; generalising to **unseen fonts** is. Global average pooling keeps the parameter count low and tolerates
small shifts. The model returns logits, not probabilities, because cross-entropy and temperature scaling both need
logits.

## Training (`train.py`)

| Choice | Why |
|---|---|
| AdamW, lr 2e-3, weight decay 1e-4, batch 256 | fast convergence on small CNNs without LR tuning |
| cosine LR schedule over 20 epochs, stepped per batch | large steps early, fine steps late, no extra hyper-parameters |
| early stopping on **val loss**, patience 3, keep the best weights | we care about the probabilities (MaxSAT weights), and loss sees overconfidence that accuracy does not |
| light online augmentation (train only) | rotation ±5°, scale ±8%, shift ±1.5 px, blur, contrast, pixel noise |
| fixed seeds | python, numpy, torch, augmentation stream, data order |
| CPU (GPU used automatically if present) | about 5.7 min per epoch on this laptop |

### Class imbalance

63% of cells are empty. **We do not rebalance** (no class weights, no balanced sampling). MaxSAT reads the softmax
as P(class | crop). Reweighting the loss or the sampling is a *prior shift*: it pushes every prediction towards the
digit classes by a fixed factor. Temperature scaling (one scalar) cannot undo that, so the solver would see digits
in empty cells. The imbalance is mild (63% vs about 4% per digit), and empty cells are visually easy. The result
confirms it: empty-cell accuracy is 100% on val and test, with no empty↔digit confusions at all.

### Results: run `cnn_v1` (real output)

Training stopped early after epoch 11; the best val loss was at **epoch 8** (0.0007). Total time 63 min.

| split | fonts | crops | accuracy |
|---|---|---|---|
| val | Merriweather, Oswald | 24,300 | **100.00%** (1 error: an 8 read as 1) |
| test | Lora, Open Sans, Space Mono (never seen) | 16,200 | **99.26%** (120 errors) |

Per-class accuracy on **test**: empty, 1, 2, 3, 4, 6, 7, 8 = 1.000; **5 = 0.830**; 9 = 0.993.

Test confusion, rows = true and columns = predicted, showing only the non-zero off-diagonal cells:

| true | predicted | count |
|---|---|---|
| 5 | 6 | **115** of 677 |
| 9 | 8 | 5 of 694 |

**All 120 errors come from one font, Space Mono** (97.8% accuracy on it; Lora and Open Sans are 100%). Space Mono's
5 has a large, almost closed lower bowl that looks like a 6, and no training font draws a 5 like that. This is a
real font-generalisation gap, the kind of failure the font-family split exists to expose. We have **not** changed
the model because of it, because that would be tuning on the test set. The error affects 51 of the 200 test
images (at least one misread each).

On the wrong crops, the CNN's confidence has median 0.87 (13 of 120 above 0.99), and the true digit keeps a median
probability of 0.10. So the truth is usually the runner-up, which is what MaxSAT needs to correct it.

## Calibration (`calibrate.py`)

Temperature scaling: divide the logits by one scalar T before the softmax, with T fitted to minimise the **val**
NLL (LBFGS on log T). It cannot change the argmax, so accuracy is unchanged; only the confidence moves. Why it
matters: the MaxSAT weight of a reading is −log p. An overconfident wrong reading (p = 0.999) costs about 6.9 nats
to override, and the rules may not be able to pay that.

Result (real output): **T = 1.1027**.

| split | | NLL | accuracy | ECE (15 bins) | mean confidence |
|---|---|---|---|---|---|
| val | before | 0.0007 | 1.0000 | 0.0001 | 0.9999 |
| val | after | 0.0006 | 1.0000 | 0.0001 | 0.9999 |
| test | before | 0.0231 | 0.9926 | 0.0044 | 0.9968 |
| test | after | 0.0217 | 0.9926 | 0.0038 | 0.9964 |

Reliability diagrams: `outputs/models/cnn_v1/reliability_{val,test}.png`.

**Honest reading of these numbers:**

- Calibration helps only a little on test (NLL −6%, ECE 0.0044 → 0.0038). The reason is the **val set is too
  easy**: one error in 24,300 crops gives LBFGS almost no wrong predictions to learn "how overconfident" from, so T
  stays close to 1.
- The test reliability diagram shows the remaining problem. After scaling, the 217 test crops with confidence
  between 0.47 and 0.93 are right only 36-74% of the time per bin; for example, 83 crops at a mean confidence of
  0.905 are right 73.5% of the time. **The model is still overconfident on the unseen font.** ECE looks tiny only
  because 15,983 of the 16,200 crops sit in the top bin (confidence about 0.999, accuracy 99.7%).
- Fix for Week 3, chosen without looking at test numbers again: make val harder, so it contains the kind of
  errors calibration has to learn from. Options: a third val font family (taken from train), and/or calibrating on
  a **noisier** val set (val renders with `--noise-frac 1.0`).

## Perceiver (`perceiver.py`)

`CNNPerceiver(checkpoint_path, use_temperature=True)`: all 81 crops in one batch → logits / T → softmax → clip at
`EPS = 1e-6` (the same floor as the MaxSAT encoding) → renormalise → `validate_prob_grid`. It refuses to run with
`use_temperature=True` on a checkpoint without a fitted T, so the uncalibrated model cannot be used by accident.
`use_temperature=False` exists for the calibration on/off ablation.

## Tests

`test_model_forward_shape`, `test_perceiver_output_is_a_valid_prob_grid`,
`test_perceiver_requires_fitted_temperature`, `test_temperature_fixes_an_overconfident_model` (a model made 4x
overconfident on purpose: the fitted T comes out between 3 and 5, and NLL and ECE drop), `test_training_smoke`
(300 crops, 1 epoch, all output files written), and in `test_dataset.py`,
`test_crops_are_exactly_the_pipeline_crops`.
