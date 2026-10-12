# TorNet Hybrid: Project Notes

Working notes for a CNN–transformer hybrid built on the TorNet CNN baseline: the dataset, the
evaluation protocol, the baseline as it actually ships, the hybrid design, the local and cloud
workflow, and the papers behind it.

Numbers below come from the **journal version** of the TorNet paper (AIES, January 2025, in
`references/`). The January 2024 arXiv preprint reports a smaller model (32 starting filters) with
lower scores; see section 3 for the differences. Where the paper and the code disagree, the
**pretrained weights** are the ground truth, since they are what any comparison is made against.

---

## 1. Dataset (TorNet v1.1)

| Item | Detail |
|---|---|
| Size | 203,133 samples: 171,666 train (84.5%), 31,467 test (15.5%); 6.3% of test files are confirmed tornadoes |
| Storage | ~150 GB across 10 yearly archives on Zenodo (2013 ≈ 3 GB; other years 12–19 GB) |
| Sample shape | `[T=4, A=120, R=240, S=2]`: 4 frames (t, t−5, t−10, t−15 min) × 120 azimuths × 240 range gates × 2 sweeps (0.5°, 0.9°) |
| Variables | DBZ, VEL, KDP, RHOHV, ZDR, WIDTH, plus a range-folded mask |
| Labels | One label per frame; the last frame's label is forced to `ef_number >= 0` by the loader |
| Categories | TOR (confirmed), NUL (random null), WRN (warned but no tornado) |
| v1.1 changes | Fixed mislabeled frames and event/episode IDs; added tornado start/end times |

Every yearly archive contains both `train/YYYY/` and `test/YYYY/`, so any subset of years carries its
own held-out test set.

---

## 2. Evaluation protocol

**Train/test split** is by storm episode, within each year, not by year. Take the Julian day of
the episode's start time modulo 20: below 17 is train, 17 and above is test. Any training sample
within 30 minutes and 0.25° of a test sample was removed. The split is fixed in `catalog.csv`
(`type` column); the code never recomputes it.

**Five-fold cross-validation** is used only inside the training partition, to pick hyperparameters
and the epoch count. Folds are year pairs: 2013–14, 2015–16, 2017–18, 2019–20, 2021–22. The
selection criterion is the fold-averaged validation AUC (the preprint used validation loss).

**Final model**: retrain on the full training partition with the chosen settings, evaluate once on
the test partition, report the mean over five seeds.

**Three test views** (paper Table 2): NC, all nulls vs confirmed; RC, random nulls vs confirmed;
WC, warnings vs confirmed. Warnings are the hard case.

**Journal CNN results**

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std |
|---|---|---|---|---|---|---|
| 0.8893 | 0.8903 | 0.8693 | 0.8704 | 0.8884 | 0.8815 | 0.0096 |

| View | ACC | AUC | AUC-PD | CSI |
|---|---|---|---|---|
| NC | 0.9505 | 0.8760 | 0.5294 | 0.3487 |
| RC | 0.9471 | 0.9245 | 0.7436 | 0.5163 |
| WC | 0.8797 | 0.7910 | 0.5698 | 0.3617 |

> Compare on AUC, AUC-PD and CSI. Accuracy is misleading: always predicting "no tornado" scores
> 93.7% on the NC view.

**Our tooling mirrors this.** `tornet-train` records the best validation AUC epoch per fold,
`tornet-results --group` averages folds that share an experiment name, and `tornet-score` reports
all three views with bootstrap confidence intervals and a paired comparison between models.

---

## 3. The baseline as it actually ships

Verified against `src/tornet/models/cnn_baseline.py`, `configs/tornado_baseline.json`, and the
layers of the Hugging Face weights `tornet-ml/tornado_detector_baseline_v1`.

```
INPUTS (each 120 azimuth × 240 range × 2 sweeps)
  6 radar variables ─► Normalize_<var> ─┐
                                        ├─► Concatenate ─► FillNaNs(-3) ─► + range_folded_mask  = 14 channels
  coordinates (r, 1/r) ─────────────────┼──────────────────────────────────────────────┐
                                        ▼                                              ▼
┌─────────────────────────── KEEP: feature extractor ─────────────────────────────────┐
│ Block 1: 2× CoordConv 3×3,  48 filters, ReLU → MaxPool → Dropout(0.1)   → 60×120×48  │
│ Block 2: 2× CoordConv 3×3,  96 filters          → MaxPool → Dropout      → 30×60×96   │
│ Block 3: 3× CoordConv 3×3, 192 filters          → MaxPool → Dropout      → 15×30×192  │
│ Block 4: 3× CoordConv 3×3, 384 filters          → MaxPool → Dropout      → 8×15×384   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        ▼  feature map: 8 × 15 × 384 (max-pools use "same" padding: 120→60→30→15→8)
┌──────────────────────────── CUT: head='maxpool' ────────────────────────────────────┐
│ 1×1 Conv, 512, ReLU                                                                 │
│ 1×1 Conv, 256, ReLU                                                                 │
│ 1×1 Conv, 1  ("heatmap", linear)   → per-location tornado logits                    │
│ GlobalMaxPooling2D                  → single logit (sigmoid applied in the loss)     │
└─────────────────────────────────────────────────────────────────────────────────────┘
Total: 4.67M parameters. No batch norm; regularization is L2 (1e-5) + dropout (0.1).
```

### Paper vs code vs weights

| | arXiv v1 (Jan 2024) | Journal (Jan 2025) | Code default | Shipped config | Pretrained weights |
|---|---|---|---|---|---|
| Convs per block | 2, 3, 3, 3 | 2, 3, 3, 3 | **2, 2, 3, 3** | — | **2, 2, 3, 3** |
| Starting filters | 32 | 48 | 64 (always overridden) | 48 | 48 |
| Input channels | 13 | — | 14 | — | 14 |
| Epochs | 10, min val loss | 9, max val AUC | 10 | 15, checkpoint each | — |
| Label smoothing | 10–20% effective | 10–20% effective | 0 | 0 | — |
| Output | sigmoid → max pool | sigmoid → max pool | max pool → sigmoid in loss | — | same as code |

- **Convolutions per block is a genuine mismatch.** Both paper versions say 2, 3, 3, 3; the code and
  the weights have 2, 2, 3, 3 (ten CoordConv layers). Follow the weights.
- **Starting filters is a version mix, not a mismatch.** Everything that matters uses 48.
- **Sigmoid placement is not a mismatch.** The sigmoid is monotonic, so max(σ(x)) = σ(max(x)).
- **Label smoothing** is on in the paper's text and off in the shipped config. Our fold configs use
  0.1; the authors' config is kept as-is as a reference artifact.

### Tap point

Block 4's dropout layer (`dropout_11` in the pretrained model), output `8 × 15 × 384` →
**120 tokens of dimension 384**.

### Where the pretrained model is weak

Scored on the 2013–2014 test partition (`results/score_pretrained_2013_2014.json`): NC AUC 0.904,
WC AUC 0.840. Recall at the CSI-maximising threshold is 23% for EF0, 61% for EF1, and 82–88% for
EF2 and above. Weak tornadoes are the target.

---

## 4. Hybrid design

```
◄ PURE CONVOLUTION ─────────────────────────────────────────── PURE ATTENTION ►

      CNN               ViT-hybrid              CvT                  ViT
       ●─────────────────────◉─────────────────────●────────────────────●
  TorNet baseline      THIS DESIGN         conv inside           patches →
                   CNN features →         transformer          transformer
                    transformer
```

| Architecture | How it works | Strength | Weakness |
|---|---|---|---|
| CNN | Stacked convolutions and pooling | Data-efficient, strong locality bias | Limited global context |
| ViT-hybrid ⭐ | CNN backbone, then a transformer over its feature map | Reuses a pretrained CNN, adds global attention | Two parts; positional encoding choice matters |
| CvT | Transformer with conv token embeddings and conv Q/K/V | Hierarchical; no positional embeddings needed | More to build from scratch |
| ViT | Patches straight into a transformer | Global attention from layer one | Data-hungry, weak inductive bias |

### The head

```
Block 4 output (8×15×384)
  → reshape to 120 tokens × 384
  → (optional) Linear 384→256 to shrink
  → positional encoding (relative or convolutional; see note on full-scan inference)
  → Transformer encoder × 2–4 layers
  → CLS token or mean pool
  → Dense(1) → logit
```

Implemented as `head='transformer'` in `build_model`, with `head='token_mlp'` as the matched
control and `head='maxpool'` the authors' head; see `tornet.models.heads`. The three heads share
every line of input and backbone code.

### Phases

1. **Single frame**, direct comparison with the baseline. Frozen backbone with cached features
   first (120 × 384 floats ≈ 90 KB per sample as float16, ≈ 4.3 GB for all of 2013–2015), then fine-tune,
   then from scratch at full scale.
2. **Temporal**, optional: add the earlier frames with light mixing, since there are only four.

### Experiment 1 result: heads on frozen pretrained features (fold B, 2026-10-10)

Seven heads trained on cached block-4 features (`results/head_sweep_fold_b_*`). On the 2013–2015
test partition every head lands within 0.005 AUC of the pretrained reference (0.8946), and the
no-attention control matches or beats every transformer variant. Max-pooled transformers overfit
early (best epoch 5–9). The only faint positive signal is the class-token transformer: best EF0 and
EF1 recall (0.40 / 0.63 vs 0.35 / 0.58) and best warnings-view AUC (0.8315 vs 0.8270), within noise.

Interpretation: the backbone was trained jointly with the max-pool head, so its features are already
local and "strongest spot wins"; a head on frozen features cannot build relational context. The head
shape alone is not the lever. Next: change the training signal (auxiliary targets), then let the head
train with the backbone.

### Experiment 1b result: training-signal changes on frozen features (fold B, 2026-10-10)

Auxiliary targets (`proximity`, `frame_fraction`) on the class-token transformer and the matched
control, and the metric-learning head with label-based and proximity-based contrastive supervision
(`results/head_aux_embed_fold_b_scores.json`). None moved test AUC beyond noise: every run is within
0.005 of the pretrained reference, and the class-token transformer's earlier EF0/EF1 bump did not
survive the auxiliary targets, so it was noise. The embedding runs peaked early (epoch 12) with the
highest AUCPR of any head (0.69) but slightly lower AUC.

The embedding geometry (`tornet-embeddings`) is informative anyway: warned nulls are not a separate
cluster. Their centroid is nearly identical to the random-null centroid (cosine 0.995) and 54–57% of
warned nulls fall nearest the random-null centroid, 14–16% nearest tornadoes, only 28–32% nearest
their own. In this backbone's feature space, a warned storm that produced no tornado looks like a
null, not like a near-miss. Context (why a forecaster warned) is not in the chip features.

Conclusion of experiment 1: on frozen features, neither head shape, auxiliary targets, nor
contrastive geometry changes detection. The lever has to be the backbone. Next: end-to-end
training (fine-tune from pretrained, then from scratch), where these heads and losses get to shape
the features.

### Experiment 2 result: end-to-end fine-tuning from the pretrained backbone (fold B, 2026-10-10)

`configs/finetune/fold_b_{maxpool,transformer_cls}.json`: backbone initialised from the pretrained
weights (`init_from: pretrained`), head fresh, backbone learning rate at 0.1× the head's
(`backbone_lr_scale`), batch 64, 10 epochs, checkpoint by validation AUC on 2014. Scores on the
2013–2015 test partition (`results/finetune_fold_b_scores.json`, histories in
`results/finetune_*_history.csv`):

| Run | best epoch | val AUC | NC AUC | NC AUCPR | RC AUC | WC AUC | EF0 / EF1 recall |
|---|---|---|---|---|---|---|---|
| pretrained reference | | | 0.8946 | 0.696 | 0.9422 | 0.8270 | 0.35 / 0.58 |
| fine-tune, maxpool head (control) | 3 | 0.9127 | 0.9086 | 0.703 | 0.9529 | 0.8458 | 0.31 / 0.55 |
| fine-tune, class-token transformer | 8 | 0.9113 | 0.9065 | 0.679 | 0.9528 | 0.8408 | 0.38 / 0.60 |

Paired bootstrap ΔAUC (NC): control vs pretrained +0.014 [+0.007, +0.022]; transformer vs
pretrained +0.012 [+0.006, +0.020]; **transformer vs control −0.002 [−0.007, +0.002]**, and
−0.025 AUCPR. The two end-to-end runs are indistinguishable in AUC; the transformer ranks the
positives slightly worse.

Two things worth separating:

- **Fine-tuning helps, but it is partly year adaptation.** Both runs beat the pretrained weights,
  with the gain concentrated in 2014 (0.895 → 0.906/0.909) and 2015 (0.884 → 0.906/0.897) while
  2013 is flat. The pretrained model was fit across 2013–2020; specialising to the early years
  helps on the early years. This is not evidence about the full dataset.
- **Attention still buys nothing.** Letting the backbone move at 0.1× did not give the transformer
  head anything the maxpool head lacks. The control also overfit faster (train AUC 0.965 vs val
  0.906 by epoch 10; the transformer reached train 0.952), so the transformer's only measurable
  effect is slower fitting of the training set. The EF0/EF1 recall bump (0.38/0.60 vs 0.31/0.55)
  is at a lower CSI-maximising threshold (0.53 vs 0.66) and comes with lower precision; AUCPR says
  the ranking is not better.

Conclusion so far: on this subset, the detection ceiling is set by the data and the backbone, not by
how the 8 × 15 feature map is aggregated. What has not been tested: training from scratch with the
hybrid (the backbone could be shaped by attention from the start rather than nudged), temporal
attention over the four frames, and any of this at full scale, where two years of training data is
no longer the limiting factor.

### Experiment 3 result: from scratch on fold B, single frame vs four frames (2026-10-11)

`configs/fold_b.json` (the authors' architecture, batch 64), `configs/temporal/fold_b_attention.json`
and `configs/temporal/fold_b_concat.json` (four frames through the shared backbone, batch 16, the
two heads parameter-matched at 5.26M vs 4.67M for the baseline). All three from random
initialisation, 15 epochs, label smoothing 0.1, checkpoint by validation AUC on 2014, `decay_steps`
equal to one epoch so the schedule is the same per epoch. Scores on the 2013–2015 test partition
(`results/scratch_fold_b_scores.{json,txt}`, `results/scratch_fold_b_results.csv`, histories in
`results/{baseline,temporal_attention,temporal_concat}_fold_b_history.csv`):

| Run | best epoch | val AUC | NC AUC | NC AUCPR | RC AUC | WC AUC | AUC 2013 / 2014 / 2015 |
|---|---|---|---|---|---|---|---|
| pretrained reference (8 years) | | | 0.8946 | 0.696 | 0.9422 | 0.8270 | 0.962 / 0.895 / 0.884 |
| baseline, 1 frame | 11 | 0.8372 | 0.8792 | 0.573 | 0.9385 | 0.7951 | 0.932 / 0.866 / 0.880 |
| temporal attention, 4 frames | 14 | 0.8264 | 0.8740 | 0.584 | 0.9235 | 0.8039 | 0.923 / 0.873 / 0.867 |
| temporal concat, 4 frames | 8 | 0.8290 | 0.8734 | 0.625 | 0.9168 | 0.8118 | 0.914 / 0.868 / 0.871 |

Paired bootstrap ΔAUC against the single-frame baseline:

| | NC | RC | WC | NC ΔAUCPR |
|---|---|---|---|---|
| attention − baseline | −0.005 [−0.014, +0.004] | −0.015 [−0.023, −0.008] | +0.009 [−0.002, +0.021] | +0.011 |
| concat − baseline | −0.006 [−0.016, +0.004] | −0.022 [−0.032, −0.012] | +0.017 [+0.004, +0.031] | +0.053 |
| attention − concat | +0.001 [−0.010, +0.010] | +0.007 [−0.004, +0.019] | −0.008 [−0.022, +0.004] | −0.042 |

What this says:

- **Two years from scratch costs about 0.015–0.02 AUC** against the pretrained weights, for every
  architecture. Test AUC is above validation AUC for all three because 2013 and 2015 are training
  years whose test partitions are in the test set; 2014 is the honest held-out column, and there
  the three models sit within 0.007 of each other.
- **Four frames do not raise overall AUC.** Both temporal models are 0.005–0.006 below the single
  frame on NC, inside the interval. The split by view is the interesting part and it is consistent
  across both heads: temporal context **hurts on random negatives** (RC −0.015/−0.022, both
  significant) and **helps on warned negatives** (WC +0.009/+0.017, the concat one significant),
  with higher AUCPR. Reading: the earlier frames help separate a tornado from a storm that drew a
  warning but produced none, and cost something on the easy clear-air cases, where a single scan is
  enough and three more are noise to overfit.
- **Attention vs concat is a tie on AUC**, and concat has the better AUCPR. The curves differ:
  concat overfits by epoch 8 (train AUC 0.94 at epoch 15 with validation falling to 0.80) while
  attention is still improving at epoch 15 (train 0.90). Attention acts as a regulariser here, not
  as a better aggregator. It may pull ahead with more epochs or more data; there is no evidence yet.
- **Caveats.** One seed each, so differences under 0.01 are within run-to-run noise. The temporal
  runs took four times as many optimiser steps (batch 16 vs 64) at the same per-epoch decay. The
  baseline here is well below the paper's reported numbers because it saw two years, not eight.

Decision for the full-scale run: the single-frame baseline and the two temporal variants are the
trio to carry to RunPod, since the WC gain is the one signal worth testing on eight years, and the
single-frame hybrid head is dropped (Experiments 1 and 2 found nothing for it). Local training cost
for reference, 12 GB laptop GPU: baseline 3.3 min/epoch, four-frame models 9.2–9.8 min/epoch.

Engineering notes from these runs: the authors' batch 128 does not fit in 12 GB with real training
(fold configs are now batch 64 with `decay_steps` doubled); Keras on torch leaks collectable
reference cycles holding GPU tensors every step, in training and in prediction, so both
`tornet-train` and `tornet-test` collect every 50 batches; `tornet-test` divides its default batch
of 64 by the model's frame count.

### Experiment 4 result: temporal heads on frozen pretrained features (fold B, 2026-10-12)

The cheap version of the temporal question: the published backbone, frozen, run once over the last
four scans of every sample (`tornet-cache-features --frames 4`; the last frame reproduces the
pretrained model's logits to 0.001), then heads trained on the cache for 30 epochs
(`configs/frozen/fold_b_*.json`). The control is the authors' maxpool head retrained on the last
frame only, from the same cache. Scores on the 2013–2015 test partition
(`results/frozen_fold_b_scores.{json,txt}`, `results/frozen_fold_b_results.csv`, histories in
`results/frozen_*_fold_b_history.csv`):

| Run | params | best epoch | val AUC | NC AUC | NC AUCPR | RC AUC | WC AUC |
|---|---|---|---|---|---|---|---|
| pretrained reference | | | | 0.8946 | 0.696 | 0.9422 | 0.8270 |
| frozen, maxpool head, last frame (control) | 0.33M | 22 | 0.9049 | 0.8973 | 0.685 | 0.9465 | 0.8276 |
| frozen, temporal attention, 4 frames | 0.92M | 10 | 0.9051 | 0.9069 | 0.702 | 0.9546 | 0.8392 |
| frozen, temporal concat, 4 frames | 0.92M | 19 | 0.9064 | 0.9031 | 0.707 | 0.9489 | 0.8382 |

Paired bootstrap ΔAUC:

| | NC | RC | WC | NC ΔAUCPR |
|---|---|---|---|---|
| attention − control | **+0.0096 [+0.0046, +0.0143]** | +0.0081 [+0.0043, +0.0126] | +0.0116 [+0.0053, +0.0180] | +0.017 |
| concat − control | +0.0058 [+0.0001, +0.0114] | +0.0023 [−0.0026, +0.0071] | +0.0106 [+0.0032, +0.0178] | +0.022 |
| attention − concat | +0.0038 [−0.0008, +0.0089] | +0.0058 [+0.0015, +0.0101] | +0.0010 [−0.0039, +0.0051] | −0.005 |

What this says:

- **The first significant win for a variant in this project.** Temporal attention over the four
  frames beats the single-frame control by 0.010 NC AUC with the interval clear of zero, on every
  view, with the same frozen features and the same training procedure. The gain is on warned
  negatives (WC +0.012) as much as on random ones (RC +0.008), so this is not only the
  "earlier frames help with warned storms" effect seen from scratch (Experiment 3).
- **Most of it is the frames, some of it may be the attention.** Concat gets +0.006 overall and
  the same WC gain; attention minus concat is +0.004 with the interval touching zero, significant
  only on RC. Attention has the better AUC, concat the better AUCPR.
- **Validation does not see any of this.** The three validation AUCs on 2014 are within 0.002
  and the curves are flat after epoch 5; best epochs (22, 10, 19) are noise. The test differences
  are real by the paired bootstrap, but a single held-out year at this size cannot resolve 0.01,
  which is an argument for the full data.
- **Frozen features already contain temporal signal.** The single-frame backbone was never
  trained to compare scans, yet a head that compares its per-frame features gains 0.01. A backbone
  that trains with temporal context (Plan A fine-tuning, Plan B from scratch) is the natural next
  step if the full-data result holds.
- **Caveats.** One seed each; the cache is fixed so seeds only vary head initialisation and batch
  order, and they are cheap (16–30 s per epoch locally). 2013 and 2015 training-year test
  partitions are in the test set, as in every fold-B number here.

Decision: run this exact protocol at full scale on RunPod (`scripts/runpod/frozen_heads.sh`):
cache 2013–2022 four-frame features once (about 75 GB on the volume), train the three heads on
2013–2020 with validation on the 2021–2022 train partition, score on the 2021–2022 test partition,
15 epochs. Seeds are cheap on the cache and should follow if the first pass is within noise.

### Experiment 5 result: frozen temporal heads at full scale (RunPod, 2026-10-12)

The Experiment 4 protocol on all ten years (`scripts/runpod/frozen_heads.sh`, one RTX PRO 6000 pod
in US-NE-1, 1.5 h, about $4): the published backbone's four-frame features cached for every sample
(171,666 train and 31,467 test files, 75 GB on the volume at `/workspace/tornet/features_4f`),
heads trained for 15 epochs on 2013–2020 (130,479 samples), checkpoint by validation AUC on the
2021–2022 train partition (41,187), scored on the 2021–2022 test partition (7,045 samples, 272
confirmed tornadoes). The published model was scored on the same test files for the reference row.
Artifacts in `results/full_frozen/` (per-run `results.json`, `history.csv`, `predictions.csv`;
`scores_all.{json,txt}` is the four-way paired score; `pod.log`).

| Run | best epoch | val AUC | NC AUC | NC AUCPR | RC AUC | WC AUC | AUC 2021 / 2022 |
|---|---|---|---|---|---|---|---|
| published model | | | 0.8786 | 0.341 | 0.9293 | 0.7835 | 0.865 / 0.900 |
| frozen, maxpool head, last frame (control) | 5 | 0.9482 | 0.8915 | 0.342 | 0.9420 | 0.7966 | 0.886 / 0.898 |
| frozen, temporal attention, 4 frames | 10 | 0.9488 | 0.8908 | 0.310 | 0.9430 | 0.7928 | 0.880 / 0.905 |
| frozen, temporal concat, 4 frames | 9 | 0.9474 | 0.8885 | 0.323 | 0.9413 | 0.7893 | 0.885 / 0.891 |

Paired bootstrap ΔAUC:

| | NC | RC | WC | NC ΔAUCPR |
|---|---|---|---|---|
| control − published | +0.0128 [+0.0053, +0.0210] | +0.0127 [+0.0061, +0.0203] | +0.0131 [+0.0054, +0.0210] | +0.001 |
| attention − control | −0.0006 [−0.0076, +0.0083] | +0.0011 [−0.0071, +0.0096] | −0.0038 [−0.0130, +0.0072] | −0.032 |
| concat − control | −0.0030 [−0.0088, +0.0061] | −0.0007 [−0.0080, +0.0077] | −0.0074 [−0.0146, +0.0008] | −0.019 |

What this says:

- **The fold B temporal gain did not replicate.** On eight training years and a held-out pair of
  test years, both temporal heads tie the single-frame control on AUC and are worse on AUCPR. The
  Experiment 4 result (+0.010 with an interval clear of zero) was either specific to 2013–2015 or a
  fluke of that test set; either way it is not a property of the frozen features in general.
  Validation on 41k samples agrees: the three heads are within 0.0014.
- **Retraining the authors' head on their own frozen features beats the shipped weights by
  0.013 AUC**, significant on every view, with equal AUCPR. Same backbone, same head architecture,
  same validation years as the authors used; the differences are label smoothing 0.1, our
  learning-rate schedule, and checkpoint selection over 15 epochs. This is the only robust gain
  found so far and it is a training-recipe effect, not an architecture one. Worth a seed check
  before claiming it.
- **Statistical power is thinner than fold B.** 272 confirmed tornadoes in the 2021–2022 test
  partition against 770 on fold B; the NC AUC interval is about ±0.02 and the paired interval
  about ±0.008. A 0.005 effect would not be visible here. Seeds on the cache cost about $1.50 per
  head per three seeds plus 14 minutes of feature loading per run from the volume.

Where this leaves the hybrid question: four experiments on heads over frozen or fine-tuned
single-frame features, plus one from scratch on a subset, have not produced an architectural gain
that survives a change of years. Remaining untested: a backbone trained with temporal context from
the start at full scale (Plan B), which Experiment 3's warned-negative split hinted at but is a
multi-hour GPU run per variant; and the training-recipe finding above, which is cheap to firm up.

### Training-signal ideas (agreed order: A then B)

- **A. Continuous auxiliary targets, not EF-based** (EF ratings are survey-based, inconsistent at the
  low end). `proximity`: minutes to the tornado interval from the v1.1 start/end times, clipped to a
  window; `frame_fraction`: share of the four frames that are tornadic. Trained jointly with the
  binary loss on cached features. Implemented in `tornet-train-head`.
- **B. Metric learning**: an embedding head with a contrastive loss (similarity from time-to-event),
  classification by distance to learned tornado/null prototypes, with inspection tooling to see
  whether warned nulls form their own cluster. Motivation: label-noise robustness. Not built yet.
- Later, end to end: rotation-strength auxiliary (azimuthal shear computed from VEL) only makes sense
  when the backbone trains; temporal attention over the four frames after the full run.

### Things to control for

- **Capacity.** The baseline head is ~330k parameters; a two-layer transformer at width 384 is
  ~3.5M, nearly doubling the model. The existing `head='mlp'` is not a usable control (it flattens
  the feature map into a 4096-unit dense layer, ~165M parameters). **Decision:** build a
  parameter-matched no-attention head alongside the transformer head: the same per-token MLP width
  and depth and the same pooling, minus attention, so a win is attributable to attention.
- **Full-scan inference.** Paper section 4 runs the CNN fully convolutionally on whole radar scans,
  which works because max-pool is size-agnostic. Learned 2D positional embeddings fixed to 8 × 15
  break that. **Decision:** convolutional position encoding (a depthwise 3 × 3 conv over the token
  grid added to the tokens), which is size-agnostic.
- **Class imbalance and overfitting** as the model grows; the EF breakdown from `tornet-score` is
  the diagnostic.

---

## 5. Workflow

| Stage | Where | What |
|---|---|---|
| Development | Laptop, RTX 5070 Ti (12 GB) | Branch `jake/dev`, torch backend, 2013–2015 subset at `~/data/tornet` (40,666 train / 7,021 test files) |
| Full runs | RunPod, US-NE-1 | Network volume `xqc3pvmwyr` with all ten years; GPU ladder from PRO 6000 MIG 24 GB (0.69/hr) to H100 |

**Backend: torch everywhere.** PyPI tensorflow wheels carry no Blackwell kernels, so tensorflow runs
on CPU on the laptop and on the PRO 6000 slices. The tensorflow-only code paths (tf.data loader, TFDS
builder, TensorBoard callbacks) were removed from the fork; git history has them. The model is pure
Keras ops, so the architecture is identical on either backend and the weights load on both.

**Local loop**

```
export TORNET_ROOT=~/data/tornet
uv run tornet-test --years 2013 2014 2015 --output ... --predictions ...   # pretrained reference (done)
uv run tornet-train configs/fold_a.json       # hold out 2013
uv run tornet-train configs/fold_b.json       # hold out 2014
uv run tornet-train configs/fold_c.json       # hold out 2015
uv run tornet-results --group                 # fold-averaged AUC per experiment
uv run tornet-train configs/final.json        # retrain on all three, no validation
uv run tornet-test --model_path experiments/latest/checkpoints/last.keras --years 2013 2014 2015 --predictions preds.csv
uv run tornet-score results/predictions_pretrained_2013_2015.csv preds.csv
```

**Head experiments on cached features** (experiment 1):

```
uv run tornet-cache-features --years 2013 2014 2015 --verify results/predictions_pretrained_2013_2015.csv
for c in configs/heads/*.json; do uv run tornet-train-head $c; done
uv run tornet-results --group
uv run tornet-score results/predictions_pretrained_2013_2015.csv experiments/<run>/predictions.csv
```

**End-to-end fine-tuning** (experiment 2, ~35 min per run on the laptop):

```
uv run tornet-train configs/finetune/fold_b_maxpool.json
uv run tornet-train configs/finetune/fold_b_transformer_cls.json
uv run tornet-test --model_path experiments/<run>/checkpoints/best.keras --years 2013 2014 2015 --predictions experiments/<run>/predictions.csv
uv run tornet-score results/predictions_pretrained_2013_2015.csv experiments/<control>/predictions.csv experiments/<hybrid>/predictions.csv
```

**From scratch, single vs four frames** (experiment 3; baseline ~50 min, four-frame runs ~2.5 h each
on the laptop):

```
uv run tornet-train configs/fold_b.json
uv run tornet-train configs/temporal/fold_b_attention.json
uv run tornet-train configs/temporal/fold_b_concat.json
uv run tornet-test --model_path experiments/<run>/checkpoints/best.keras --years 2013 2014 2015 --predictions experiments/<run>/predictions.csv
uv run tornet-score experiments/<baseline>/predictions.csv experiments/<attention>/predictions.csv experiments/<concat>/predictions.csv
```

**Frozen backbone, four-frame feature cache** (experiment 4; the cache is one forward pass, then
heads train in minutes and seeds are free):

```
uv run tornet-cache-features --years 2013 2014 2015 --frames 4 --out $TORNET_ROOT/features_4f --verify results/predictions_pretrained_2013_2015.csv
for c in maxpool temporal_attention temporal_concat; do uv run tornet-train-head configs/frozen/fold_b_$c.json; done
uv run tornet-score experiments/<maxpool>/predictions.csv experiments/<attention>/predictions.csv experiments/<concat>/predictions.csv
```

At full scale the same steps run from `scripts/runpod/frozen_heads.sh` on a pod mounting the
network volume (loader benchmark, cache of all years, the three `configs/frozen/full_*.json` runs,
paired scores).

A variant is a new `exp_name` prefix on the same configs. Subset scores sit below the paper's because
of less training data and a higher tornado base rate (13% vs 6%); compare the hybrid against the
baseline on the same subset, never against the published numbers. The bootstrap CI on the subset
NC AUC is about ±0.016, which is the noise floor a variant has to clear.

**Config notes.** `decay_steps` in each config equals that config's steps per epoch, matching the
paper's one-epoch decay period. The local configs use batch 64: the paper's batch 128 runs out of
memory on the 12 GB laptop GPU (confirmed 2026-10-11, out of memory in epoch 2). At full scale on a
larger GPU, restore batch 128 and halve `decay_steps`.

**Loader throughput** (measured 2026-10-10, 2014 train files, batch 128): single-threaded 58 files/s;
threaded `workers > 1` **segfaults** (NetCDF/HDF5 is not thread-safe); multiprocessing 4 workers
156 files/s, 8 workers 248 files/s, more workers no better. Configs use 8 worker processes, giving
roughly 3 minutes per fold-A epoch and about 2.5 hours for the whole three-fold-plus-final protocol.

---

## 6. References

> ViT and ViT-hybrid come from the same paper. There is no single "CNN paper"; cite the specific
> ideas the model uses.

| Topic | Paper | Venue | Link |
|---|---|---|---|
| **TorNet + CNN baseline** ⭐ | Veillette et al., *A Benchmark Dataset for Tornado Detection and Prediction using Full-Resolution Polarimetric Weather Radar Data* | AMS *AIES* 4(1), 2025 | [Journal](https://journals.ametsoc.org/view/journals/aies/4/1/AIES-D-24-0006.1.xml) · [arXiv 2401.16437](https://arxiv.org/abs/2401.16437) (preprint, differs; see §3) |
| CNN foundations | LeCun et al., *Gradient-Based Learning Applied to Document Recognition* | Proc. IEEE, 1998 | classic citation |
| VGG | Simonyan & Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition* | ICLR 2015 | [1409.1556](https://arxiv.org/abs/1409.1556) |
| CoordConv | Liu et al., *An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution* | NeurIPS 2018 | [1807.03247](https://arxiv.org/abs/1807.03247) |
| Transformer | Vaswani et al., *Attention Is All You Need* | NeurIPS 2017 | [1706.03762](https://arxiv.org/abs/1706.03762) |
| **ViT + ViT-hybrid** ⭐ | Dosovitskiy et al., *An Image is Worth 16x16 Words* | ICLR 2021 | [2010.11929](https://arxiv.org/abs/2010.11929) |
| CvT | Wu et al., *CvT: Introducing Convolutions to Vision Transformers* | ICCV 2021 | [2103.15808](https://arxiv.org/abs/2103.15808) |

**Related work**

| Topic | Paper | Link |
|---|---|---|
| Other TorNet models | *KANs Layer Integration: Benchmarking Deep Learning Architectures for Tornado Prediction* (Penn State, Dec 2025). CNN, VGG19 and Xception with KAN heads; closest analog to swapping the head. | [Penn State](https://pure.psu.edu/en/publications/kans-layer-integration-benchmarking-deep-learning-architectures-f/) |
| Prior ML tornado detection | Sandmæl et al., *The Tornado Probability Algorithm*, Weather and Forecasting, 2023. Random forest on polarimetric radar; cited in the TorNet paper. | via TorNet references |

**Optional, depending on scope**

| If you… | Cite |
|---|---|
| Discuss the wider landscape | Swin ([2103.14030](https://arxiv.org/abs/2103.14030)) · CoAtNet ([2106.04803](https://arxiv.org/abs/2106.04803)) · ConvNeXt ([2201.03545](https://arxiv.org/abs/2201.03545)) |
| Use residual connections | He et al., *Deep Residual Learning* ([1512.03385](https://arxiv.org/abs/1512.03385)) |
| Try pretrained ViT weights on radar features | Lu et al., *Pretrained Transformers as Universal Computation Engines* ([2103.05247](https://arxiv.org/abs/2103.05247)) |

**Reading order:** TorNet journal paper → ViT paper, hybrid section → CvT → KAN TorNet paper.

**Citation tips.** Cite the journal version of TorNet, not the preprint; their numbers differ. Pull
BibTeX from the publisher page. Call the model a "CNN–transformer hybrid"; reserve "CvT" for designs
with convolutional token embeddings and projections. Check "Cited by" on the TorNet paper before
claiming novelty.

---

## 7. Final run checklist

1. Reproduce the CNN baseline on the full protocol first and confirm it lands near the journal's
   0.876 NC test AUC. Train it yourself on v1.1 rather than relying on the Hugging Face weights,
   whose training data version is undocumented.
2. Run the hybrid and the MLP control through the identical pipeline: same split, five folds for
   epoch selection, all-years retrain, five seeds, all three test views via `tornet-score`.
3. Dress-rehearse with a single seed before the full sweep. Checkpoints land on the network volume.

---

*Consolidated October 2026 from the earlier summary notes and reading list. Verify venues and years
against publisher pages before final citation.*
