# TorNet Project: Summary

## 1. Dataset (TorNet v1.1)

| Item | Detail |
|---|---|
| Size | 203,133 samples, ~6.8% confirmed tornadoes |
| Storage | ~150 GB across 10 yearly files on Zenodo (2013 ≈ 3 GB; other years ≈ 12–19 GB) |
| Sample shape | `[T=4, S=2, R=240, A=120]`: 4 frames (t, t‑5, t‑10, t‑15 min) × 2 sweeps (0.5°, 0.9°) |
| Variables | DBZ, VEL, WIDTH, ZDR, RHOHV, KDP |
| Labels | One label per frame |
| v1.1 changes | Fixed mislabeled frames and event/episode IDs; added tornado start/end times |

## 2. Baseline CNN

- **Input:** the last frame only. That's 13 channels: 6 variables × 2 sweeps, plus a range-folded mask.
- **Architecture:** VGG-style CoordConv blocks → 1×1 convs → sigmoid → GlobalMaxPool.
- **Training:**
  - Label smoothing of 10–20%.
  - Sample weights: EF2+ = 2.0, warnings = 0.5, everything else = 1.0.
  - 10 epochs.
- **Compute:** about 2 hours per model on a V100.
- **Pretrained weights:** `tornet-ml/tornado_detector_baseline_v1` on Hugging Face. These were likely trained on v1 data, not v1.1.

## 3. Evaluation Protocol

- **Train/test split:** made by storm-episode day of year, not by year.
  - Episodes where day-of-year mod 20 is under 17 go to train; 17 or higher go to test.
  - That gives 84.5% train and 15.5% test, with every year represented in both.
- **5-fold CV:** used only within the training set, to tune hyperparameters. The folds are year pairs: 2013–14, 2015–16, 2017–18, 2019–20, 2021–22.
- **Final results:** retrain on all the training data, evaluate on the test set, and average over 5 seeds.

**CNN cross-validation AUC by fold**

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|
| 0.8950 | 0.8744 | 0.8720 | 0.8602 | 0.8708 | 0.8745 |

**CNN test-set results (confirmed tornadoes vs. all non-tornadic samples)**

| AUC | AUC-PD | CSI | ACC |
|---|---|---|---|
| 0.874 | 0.529 | 0.338 | 0.950 |

> Compare models on AUC, AUC-PD, and CSI. Accuracy is misleading here: always predicting "no tornado" already scores 93.7%.

## 4. Project Plan

- **Architecture:** a CNN + transformer hybrid in the style of CvT. Their CNN blocks act as the conv stem and tokenizer, and transformer layers replace GlobalMaxPool.
- **Phase 1:** single frame only, for a direct comparison with their baseline.
- **Phase 2 (optional):** add the time steps, with light temporal mixing since there are only 4 frames.
- **Watch for:** class imbalance, and overfitting as the model gets larger.

## 5. Workflow

| Stage | Where | What |
|---|---|---|
| Development | Local laptop (5070 Ti, 16 GB VRAM) | Fork `mit-ll/tornet`, work on a branch, iterate on the 2013–2016 subset (~50 GB) |
| Full runs | RunPod | Clone your fork, pull data from Zenodo, run the full protocol |

- **The subset works with their real protocol.** Every year has both train and test samples, so you can apply their split directly to 2013–2016. Check `catalog.csv` and the repo loaders for the split assignment.
- **Tuning:** 2013–2016 covers CV Folds 1 and 2, so you can tune on those.
- **Expectations:** subset scores will come in below the paper's numbers because there's less training data. Compare your hybrid against their CNN on the same subset, not against the published results.
- **Compute:** full training is feasible on the 5070 Ti. RunPod mostly buys you speed and parallel seeds.

## 6. Final Run Checklist

1. Reproduce their CNN first and confirm it reaches about 0.874 test AUC.
2. Run your hybrid through the identical pipeline: same split, 5 seeds, all 3 test views.
3. Train on v1.1 yourself instead of comparing against the v1 Hugging Face weights.
4. Checkpoint your runs, and do a single-seed dress rehearsal first.

**Sources:** [TorNet paper](https://arxiv.org/html/2401.16437) · [mit-ll/tornet](https://github.com/mit-ll/tornet) · [Hugging Face model](https://huggingface.co/tornet-ml/tornado_detector_baseline_v1)
