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

### Experiment 6 result: prediction from earlier scans (lead-time study, 2026-10-12)

TorNet's label follows the storm report's start and end times, and most tornado samples sit at
onset: in the 2021–2022 test partition the tornado is present in only the newest scan for 125 of
272 tornado samples, in the newest two for 83, three for 37, all four for 27, and for 67 the newest
scan is itself up to 5 min before onset (still labelled 1). So hiding the newer scans turns
detection into short-lead prediction with the same labels. Heads on the frozen four-frame cache
were given only the older scans (`frames` key in `configs/lead/*.json`): the t−15 scan alone
(≈15 min lead), t−10 alone or t−15..t−10 (≈10 min), t−5 alone or t−15..t−5 (≈5 min). The control
is the all-scans maxpool head from Experiments 4/5. Fold B at 30 epochs
(`results/lead_fold_b_scores.{json,txt}`, `results/lead_fold_b/`), full data at 15 epochs on one
pod for 1.2 h, about $3 (`results/full_frozen/lead_*`, `scripts/runpod/lead_heads.sh`).

NC AUC (AUCPR) and WC AUC by lead, single scan vs the two temporal heads over the scans up to
that lead:

| lead | input | fold B NC | fold B WC | full NC | full WC |
|---|---|---|---|---|---|
| 15 min | t−15 alone | 0.838 (0.53) | 0.735 | 0.795 (0.13) | 0.643 |
| 10 min | t−10 alone | 0.876 (0.60) | 0.790 | 0.821 (0.15) | 0.679 |
| 10 min | concat t−15, t−10 | 0.874 (0.60) | 0.788 | **0.833** (0.16) | **0.695** |
| 10 min | attention t−15, t−10 | 0.873 (0.59) | 0.785 | **0.831** (0.16) | **0.690** |
| 5 min | t−5 alone | 0.893 (0.65) | 0.817 | 0.855 (0.24) | 0.739 |
| 5 min | concat t−15..t−5 | 0.893 (0.67) | 0.820 | 0.858 (0.22) | 0.737 |
| 5 min | attention t−15..t−5 | 0.896 (0.65) | 0.819 | 0.859 (0.23) | 0.739 |
| 0 | t alone (control) | 0.897 (0.68) | 0.828 | 0.892 (0.34) | 0.797 |

Paired ΔAUC of the temporal heads against the single scan at the same lead:

| | fold B NC | full NC | full WC |
|---|---|---|---|
| 10 min, concat − single | −0.002 [−0.006, +0.002] | **+0.012 [+0.006, +0.017]** | **+0.017 [+0.008, +0.025]** |
| 10 min, attention − single | −0.003 [−0.007, +0.000] | **+0.010 [+0.002, +0.017]** | +0.011 [+0.001, +0.023] |
| 5 min, concat − single | +0.000 [−0.004, +0.004] | +0.004 [−0.004, +0.009] | −0.002 [−0.013, +0.008] |
| 5 min, attention − single | +0.003 [−0.001, +0.008] | +0.005 [−0.002, +0.012] | −0.000 [−0.010, +0.009] |
| 0, either − single (Exp 4/5) | +0.006 to +0.010 | −0.001 to −0.003 | −0.004 to −0.007 |

What this says:

- **The frozen detection features predict, with a steep decay.** On 2021–2022 a single scan
  gives 0.855 at 5 min, 0.821 at 10, 0.795 at 15, against 0.892 at onset; AUCPR falls from 0.34
  to 0.13. The warned-versus-confirmed view loses about twice as much as the overall one
  (0.797 → 0.643), so what disappears with lead is mostly the ability to tell a tornadic storm
  from a warned one; separating storms from clear air survives. Fold B decays more gently
  (0.897 → 0.838) because its test set includes training-year storms.
- **Older scans help exactly where they should, and only there.** At 10 min lead on the full
  data, adding the t−15 scan is worth +0.012 (concat) or +0.010 (attention), significant on every
  view and largest on warned negatives (+0.017). At 5 min the gain is +0.004 and not significant;
  at onset it is nil (Experiment 5). Fold B shows nothing at any lead. The consistent reading is
  that history matters when the newest scan is pre-tornado, that the effect is small (about a
  sixth of the lead penalty recovered), and that two years of training data cannot learn it.
- **Attention is still not better than concat.** At every lead the two are within 0.003 of each
  other; concat is slightly ahead at 10 min on the full data. Nothing in six experiments
  separates them.
- **Caveats.** One seed; 272 confirmed tornadoes in the test partition; the frozen backbone was
  trained for detection and only incidentally encodes precursors, so these curves are a lower
  bound on predictability; lead is approximate (scans are 4–6 min apart and onset times are
  report-based).

Decision point. The prediction framing is the first place temporal context has shown a
significant, interpretable gain, and it is the natural one for an operational use (a warning
needs lead). Two ways forward, in order of cost: (a) fine-tune the backbone on the prediction
target with the t−15..t−10 window as input, so it can learn precursor features rather than reuse
detection ones (Plan A machinery, hours per run on the pod); (b) longer sequences via
TorNet-Temporal (8–22 scans of the same events), where leads of 30–60 min become testable and a
3D CNN or attention over time is the natural model.

### Experiment 7 plan: exhaust the cache, then the backbone as a feature extractor (2026-10-13)

Agreed direction: keep the transfer-learning framing (the pretrained backbone is never retrained)
and switch the question from detection to early prediction, "will this storm have a tornado in
the next 15 minutes", scored at the 10-minute lead where Experiment 6 found the only temporal
signal. Three parts, all on the existing four-frame cache, run unattended by
`scripts/runpod/cache_round2.sh`:

- **A. Last round of heads** (`configs/cache2/`), each at lead 10 (frames t−15, t−10) and at
  onset (all four): `spacetime_transformer` (attention over every cell of every frame, the first
  head that can relate a feature to another location in another scan; both current heads pool
  per location first), `temporal_diff_concat` (the concat head handed frame differences, i.e.
  change between scans as an explicit input), and the concat head with a `frame_labels` auxiliary
  target (asked for the label of every cached frame, including the ones it cannot see, so the
  gradient carries "when does it start" rather than only "is it there at the end"). Paired against
  Experiment 6's lead-10 single-scan and concat heads.
- **B. Meta model, radar only** (`configs/meta/*_lead10_{emb,radar,emb_radar}.json`): one row per
  sample from the pooled backbone embeddings (max and mean over the 8 × 15 map per scan, 768
  values), sample metadata (range to radar, hour, day of year), and rotation metrics computed from
  the raw velocity field (rotational velocity proxy, azimuthal shear, peak velocity and
  reflectivity per frame and sweep, with frame-to-frame trends). LightGBM with the authors' sample
  weights. The `radar`-only config is the "what a forecaster's rotation metrics alone give"
  floor; `emb` tests whether the backbone's pooled features beat a head on the full map.
- **C. Meta model with environment** (`configs/meta/*_lead10_{env,all}.json`): HRRR fields at
  the storm location from the run initialised two hours before the scan (CAPE, CIN, 0–1/0–3 km
  helicity, 0–1/0–6 km shear, LCL height, 2 m temperature and dewpoint, each as point value and
  27 km box max and mean), plus derived shear magnitudes and a significant-tornado-parameter
  composite. This is the one input that addresses what Experiment 6 showed is lost with lead:
  telling a tornadic storm from a warned one. `env`-only is the environment floor; `all` is the
  model. Caveats: HRRR exists from 2014-07-30, so 2013 and early-2014 rows carry NaN fields and an
  availability flag; the honest view remains WC, because nulls drawn at random sit in benign air
  and inflate NC. Lead 0 runs of `emb` and `all` are included to check the same inputs at onset.

What decides the next step: if C beats B on WC at lead 10, the early-prediction meta model is the
project and longer leads (TorNet-Temporal) are the extension; if nothing in A–C moves, the frozen
features are the ceiling and fine-tuning the backbone on the prediction target is the next spend.

Dry run on fold B locally before the pod (heads, radar metrics, environment fetch for 2014–2015,
meta models); the pod run adds the full years and stops itself when finished.

### Experiment 7 result: cache round 2 and the meta model (RunPod, 2026-10-13)

Everything on the frozen four-frame cache of the published backbone, full data (train 2013–2020,
val and test 2021–2022, 7,045 test samples, 272 confirmed), one seed, pod `39f2naos69etsv`
for 3.4 h (~$8.50) running `scripts/runpod/cache_round2.sh` unattended. Paired bootstraps
against the Experiment 6 single-scan head at the same lead; run directories and scores under
`results/full_frozen/` (`cache2_lead10_full_scores.*`, `cache2_lead10_vs_concat_scores.json`,
`cache2_lead10_env_effect_scores.json`, `cache2_lead0_full_scores.*`).

**Lead 10 min (heads and meta models see scans t−15 and t−10; label at t).**

| model | NC AUC | NC AUCPR | WC AUC | ΔNC vs single scan | ΔWC vs single scan |
|---|---|---|---|---|---|
| single scan t−10 (Exp 6) | 0.821 | 0.154 | 0.679 | — | — |
| concat head (Exp 6) | 0.833 | 0.164 | 0.695 | +0.012 [+0.006, +0.017] | +0.017 [+0.008, +0.025] |
| space-time transformer head | 0.827 | 0.166 | 0.687 | +0.006 n.s. | +0.008 n.s. |
| diff-concat head | 0.832 | 0.164 | 0.694 | +0.011 [+0.004, +0.018] | +0.015 [+0.006, +0.026] |
| onset (frame-label aux) head | 0.832 | 0.167 | 0.692 | +0.011 [+0.004, +0.017] | +0.013 [+0.003, +0.024] |
| meta: embeddings + metadata | 0.825 | 0.168 | 0.681 | +0.004 n.s. | +0.002 n.s. |
| meta: radar metrics + metadata | 0.763 | 0.114 | 0.638 | −0.058 | −0.041 |
| meta: environment + metadata | 0.830 | 0.149 | 0.689 | +0.008 n.s. | +0.010 n.s. |
| meta: embeddings + radar + metadata | 0.836 | 0.189 | 0.698 | +0.015 [+0.004, +0.028] | +0.020 [+0.005, +0.032] |
| **meta: everything (+ environment)** | **0.870** | **0.224** | **0.733** | **+0.049 [+0.030, +0.065]** | **+0.054 [+0.032, +0.075]** |

Same-lead pairings that isolate the pieces: everything minus concat head +0.037 NC [+0.020,
+0.054], +0.038 WC [+0.014, +0.060]; everything minus (embeddings + radar), i.e. the
environment's own contribution on top of radar, +0.034 NC [+0.019, +0.046], +0.035 WC [+0.017,
+0.050]. LightGBM gain shares in the everything model: embeddings 0.64, environment 0.24, time
and range 0.08, radar metrics 0.04.

**Lead 0 (detection, all four scans; control = frozen maxpool head, 0.8915 NC).** Space-time
transformer +0.004 n.s., diff-concat −0.004, onset −0.000, meta embeddings −0.008 n.s.; meta
everything **0.9125 NC, +0.021 [+0.009, +0.034], +0.022 WC [+0.009, +0.035]**.

Reading:

- **The three richer heads add nothing over concat.** Attention over all cells and frames,
  explicit frame differences and the onset trajectory target all land within 0.002 of the plain
  concat head at lead 10 and within noise of the control at lead 0. Together with Experiments 4–6
  this closes the question: no classifier design extracts more early-warning skill from the
  frozen features. The cache is exhausted on the head side.
- **Gradient boosting on pooled features matches the heads, no more.** Embeddings alone tie the
  single-scan head; with the hand-computed rotation metrics they edge past concat by a
  non-significant +0.003 (but +0.025 AUCPR). Radar metrics alone are far below the backbone,
  so the backbone's features carry much more than peak rotation and shear.
- **The environment is the result.** Adding HRRR fields issued two hours before the scan lifts
  the warned-versus-confirmed view by +0.035 on top of radar, and the whole model sits +0.054
  WC above the single-scan head, roughly three times the best any head achieved. Environment
  alone is no better than a single scan, so it is complementary: radar says a storm is
  organising, environment says whether this storm can produce. This is exactly the gap
  Experiment 6 identified (what lead erases is the ability to separate tornadic from warned
  storms), and it is the first input in the project to close part of it. The gain also appears
  at onset (+0.022 WC), where the detector already sees the tornado, so environment acts as a
  prior as well as a precursor.
- **Caveats.** One seed. HRRR exists from 2014-07-30, so 1,969 of 18,230 storm hours (all of
  2013, early 2014) have NaN environment; LightGBM handles the gaps, but the training signal
  for environment comes from 2014-08 onward. Fields come from the fxx=2 forecast of the run
  initialised two hours before the scan, so the model had not seen the storm, but a real-time
  system would have slightly staler data at some hours. Null storms sit in benign environments,
  which inflates NC and RC (RC 0.94); WC is the number to quote. Metadata (hour, season, range)
  carries 8% of the gain, which is climatology and legitimate but worth remembering.

Decision: the early-prediction meta model is the main line. The forecaster (Step 2, below)
remains worth running because it is the only path that changes the frozen features, and a
better encoder feeds this same table; a pod probe of it started the same night.

### Step 2 design: forecast the radar, predict and detect on the forecast (agreed 2026-10-13; built; probed and trimmed 2026-10-14)

Decided after Experiment 7 was launched; runs only after its results are in and the user says go.

**Idea.** Train a forecaster that predicts the future radar chip (all channels plus the validity
mask, in polar coordinates) at +5, +10 and +15 min from the frames observed so far, and score
each forecast with the *frozen* pretrained detector. The detector's numbers stay comparable with
everything measured so far, the forecasts are inspectable as radar fields, and the experiment has
its floor and ceiling already: the detector on the last observed frame (persistence) is the
Experiment 6 single-scan lead curve (0.795 / 0.821 / 0.855 at 15 / 10 / 5 min on 2021-2022),
and the detector on the real onset frame is the ceiling (0.892). The question is whether forecast
frames push the detector above persistence toward the ceiling, horizon by horizon.

**Training pairs** from every sample's four frames: input t-15 -> targets t-10, t-5, t; input
t-15..t-10 -> targets t-5, t; input t-15..t-5 -> target t. Real scan timestamps (from
`tornet-radar-metrics`), not nominal spacing. Loss masked to valid gates; the forecaster also
predicts the validity mask.

**The known failure and the fix.** An MAE/Huber forecaster predicts the conditional mean, which
blurs exactly the few-pixel velocity couplet the detector needs, so a plain regression forecaster
is expected to sit near the persistence floor. Two runs with the same forecaster:

1. control: masked MAE on the future fields;
2. variant: masked MAE plus the frozen detector's tornado loss on the forecast (the detector as a
   differentiable feature-space loss, the same device as a perceptual loss). This is the
   differentiable form of "reward forecasts the detector recognises"; reinforcement learning is
   not needed because the reward is differentiable and the model's output does not change what it
   observes next.

If the variant still blurs velocity, the next lever is an adversarial critic (DGMR-style), and
after that a diffusion forecaster with ensembles; neither is planned yet.

**Guards.** WC view and false-alarm rate per horizon (the variant can learn to paint couplets
wherever the environment looks favourable; the forecast fields are viewable, so a hallucinated
couplet is visible); storms in one partition only (check by event id in the catalog); Grad-CAM
on the detector's 8 x 15 heat map per horizon.

**Task definition (agreed 2026-10-13).**

- *Unit.* One sample's four frames, oldest first, real timestamps. A prefix of frames goes in,
  the rest come out, all horizons at once (never autoregressive). Horizon = minutes from the
  last input frame to the target frame.

  | frames in | frames out | horizons |
  |---|---|---|
  | 0 | 1, 2, 3 | +5, +10, +15 min |
  | 0, 1 | 2, 3 | +5, +10 |
  | 0, 1, 2 | 3 | +5 |

  First slice: the middle row only. Full version: one model trained on all three prefixes.
- *Target.* The full radar chip at each horizon, twelve channels, in the detector's normalised
  representation with empty gates at the background value (−3), so the target is defined
  everywhere and no mask enters the loss. The range-folding flag is carried forward from the
  last input frame, not predicted.
- *Training set.* Every sample, unrestricted: tornadic in every label pattern, warned, random
  nulls. The forecaster learns storm dynamics from all of them.
- *Evaluation set, scoped.* A tornadic sample is a prediction case for a given prefix only if
  **no input frame has the tornado and the last target frame does**: the tornado appears after
  everything the forecaster saw. For two frames in, that is the per-frame label patterns
  `0 0 0 1` and `0 0 1 1` (208 of the 272 tornadic samples in the 2021–2022 test partition;
  `0 1 1 1` is excluded because frame 1 already shows it and persistence would fire).
  Negatives: all warned storms (the WC view), plus random nulls for NC. Ongoing-tornado cases
  (tornado already in an input frame) are reported separately as a secondary line.
- *Scores, per horizon.* Primary: the frozen detector's skill on the forecast, the label being
  the real frame's label at that horizon, WC first, NC second. Summary: "tornado by the end of
  the window" from the final-horizon forecast. Secondary: MAE of the forecast against the real
  frame. Floor: detector on the last input frame (persistence). Ceiling: detector on the real
  frame. Paired bootstraps against the floor.
- *Baselines.* Persistence and the ceiling in the first slice; a classical advection forecast
  (last frame moved along the estimated storm motion) in the full version, since a learned
  forecaster that cannot beat advection through the detector has learned nothing beyond motion.
- *Loss focus (agreed 2026-10-13).* The field loss centres on **velocity and reflectivity**:
  the pre-onset signature is the velocity couplet, with the reflectivity hook and inflow notch
  as its structural context; the dual-pol products mostly flag debris that is already lofted.
  This is taken as the starting point without a channel-ablation study (the cache cannot give
  channel importance, since its features come after the backbone has mixed all twelve
  channels; an ablation of the full detector is ~20 min locally and remains available if a
  result makes the actual ratios matter). Details of the weighting and the detector-loss
  weight are still under discussion.

**Modeling (agreed and built 2026-10-13, trimmed 2026-10-14; `tornet-forecast`, one script,
`src/tornet/scripts/forecast_smp.py`).** Multi-task in PyTorch with Segmentation Models PyTorch
(SMP): one network that forecasts radar products at each horizon *and* predicts a tornado
probability per horizon from its bottleneck, with the frozen TorNet detector kept as the
external yardstick.

```
observed scans (n_in x 14 channels, polar)       all 12 products + range-folded in
        |
encoder: the TorNet CNN ported to torch (tornet.models.torch_backbone), initialised from the
         published weights and fine-tuned at a tenth of the learning rate; run per scan,
         features concatenated across scans at every scale (pre-pool maps at strides 1-8 as
         skips, bottleneck at 16, the SMP feature convention)
        |
bottleneck: identity (A) or a small transformer over the 8x15 tokens (B)
        |
SMP U-Net decoder (256/128/64/32) -> regression head: the target products per horizon, 120x240
bottleneck features --------------> classification head: tornado logit per horizon
```

*Names, fixed.* Two copies of the published TorNet. The **fine-tuned encoder** is its backbone
only, weights updated in training, shared by both heads. The **frozen detector** is the complete
published model (backbone and max-pool head), weights never updated; it scores the floor (last
observed scan), the ceiling (real future scan) and the forecast, and in the detector-loss configs
it also sits inside the loss as a critic. Two routes to "tornado at +h": **direct**, the
classification head's logit, no frozen detector involved; **indirect**, the forecast chip read
by the frozen detector.

*Classification head.* The authors' design on the fine-tuned bottleneck: two 1x1 convolutions,
one logit per cell per horizon, max over cells, so the paper's spatial likelihood map comes with
it (saved per horizon in `maps_h<k>.npz` together with the frozen detector's maps on
persistence, forecast and truth). It reads 768 stacked channels, so its weights start fresh.
(An SMP pool-then-linear alternative existed until 2026-10-14 and was removed unused.)

*Why TorNet as the encoder, and only TorNet (decided 2026-10-14).* It is the only pretrained
network trained on these storms, these channels and this polar geometry, and the project's
thesis is reuse of it. An ImageNet encoder swap (ResNet34 via SMP) was considered and dropped:
it would mix architecture, initialisation and capacity in one number, it answers a different
project's question, and the probe pointed at the loss rather than capacity as the lever. The
option is gone from the script. If capacity ever becomes the question, a wider or deeper
TorNet-style stack from the published weights is the cleaner test.

*Channels.* All twelve products go in (the encoder was trained on them). `target_products`
sets what the decoder draws: VEL and DBZ (4 channels, the primary configs) or all six products
(12 channels). Whatever is not drawn, and the range-folded flags, is persisted from the last
observed scan when a full chip is assembled for the frozen detector. The truth chip keeps its
own range-folded channels: they change between scans (mean absolute change 0.08 over 15 min)
and move the detector's logit by up to 9, so persisting them onto the truth would corrupt the
ceiling.

*Losses, one total.* Each term is attached to an output and updates the parts that produced
it: the decoder only from the field terms, the classification head only from its cross-entropy,
the shared encoder from everything.

- L1 on the drawn products in the detector's normalised space, weight 1 on the
  `focus_products` (VEL, DBZ) and `anchor_weight` (0.1) on any other drawn product, optionally
  multiplied by `1 + heatmap_weight x` the frozen detector's min-max-normalised heat map on the
  real target scan (`heatmap_weight` 2.0 in the configs).
- Weighted cross-entropy on the classification head's per-horizon logits, the authors' sample
  weights.
- Optionally the frozen detector's cross-entropy on the forecast chip (`detector_weight` 0.5).
  Gradients pass back through the frozen detector into the decoder and encoder; its weights do
  not move. It can only reach the drawn channels, so with 4 drawn channels its whole message is
  "move velocity and reflectivity so that I see a tornado".

*The 12-channel variant and the anchor (agreed 2026-10-14).* Drawing all twelve channels lets
the detector loss weight every channel by how much the published model actually uses it, per
pixel and per case, instead of our hand-picked list. With the detector loss as the only teacher
of the eight secondary channels nothing would hold them to reality, and the decoder could paint
whatever texture makes the critic fire (an adversarial-example risk, scored as a gain). The 0.1
anchor keeps them near real values while the gradient budget stays on velocity and reflectivity.
The script refuses a config that draws products outside the focus without the detector loss,
since the anchor alone would never teach them. The gaming check is per-product MAE
(`mae_by_product` in `summary.json`): if the indirect AUC rises while the secondary products'
MAE drifts above persistence, the critic is being fooled and the anchor needs to be heavier.
Expected effect on the headline number is small (the persisted channels cost about 0.08 logit);
the value is removing a choice of ours from the pipeline and letting the result say whether the
velocity-and-reflectivity focus was right.

*Port verification.* `TorNetDetector` (encoder + max-pool head) with weights copied from the
published Keras model reproduces its logits to 0.002 on real 2014 test files and matches the
cached `results/predictions_pretrained_2013_2015.csv`.

*Evaluation.* Test years, per horizon: the classification head's probability and the frozen
detector on persistence / forecast / truth, written as tornet-score CSVs with the label pattern
(e.g. `0011`) and `pre_onset` / `ongoing` flags; `summary.json` has AUC/AUCPR per view for all
cases and for the pre-onset subset, MAE against truth for forecast and persistence (headline over
the focus products, and per product), and paired bootstraps against persistence.

*Configs (`configs/forecast/`, fold B, ~2 h each locally, ~45 min on the pod).* Trimmed on
2026-10-14 after the probe; see "Trim" below for what was dropped and why.

| config | varies | what it answers |
|---|---|---|
| `fold_b_control` | plain L1 on VEL + DBZ, cross-entropy on the head | the floor for the loss comparison |
| `fold_b_detector` | + heat-map weighting + detector loss (the probe variant) | does the forecast clear the persistence floor at full fold B |
| `fold_b_detector_12ch` | as above, all twelve channels drawn, 0.1 anchor | was the velocity-and-reflectivity focus right |
| `fold_b_transformer` (held) | bottleneck B with the same losses as `fold_b_detector` | does attention at the bottleneck add anything; run only if A clears the floor |
| `smoke_detector` | `fold_b_detector` on 6,000 files, 4 epochs | a fresh pod or laptop works before a long run |

`scripts/runpod/forecast_runs.sh [configs...]` runs them back to back on a pod (default: the
three fold B configs) and stops the pod. The Keras U-Net built on 2026-10-13 was removed on
2026-10-14 (last at commit `7c026ec`); `tornet-forecast` is the only forecaster.

**Probe result (2026-10-14; capped fold B: 6,000 train files from 2013+2015, 2,000 val files
from 2014, 4 epochs; scored on the full 2014 test partition, 2,546 samples, 220 pre-onset
tornado cases, 133 ongoing).** Control on the pod (`fc_probe_a_control_fold_b`, ~5 min);
the heat-map + detector-loss variant on the laptop after RunPod reclaimed the pod's GPU
(`fc_probe_a_heatmap_detector_fold_b`, ~15 min). Pre-onset cases vs warned storms (WC AUC):

| horizon | persistence (floor) | judge on forecast: control | variant | trained head: control | variant | truth (ceiling) |
|---|---|---|---|---|---|---|
| +5 min | 0.753 | 0.599 | 0.708 | 0.747 | 0.753 | 0.771 |
| +10 min | 0.732 | 0.593 | 0.732 | 0.749 | 0.752 | 0.798 |

Pixel MAE (VEL+DBZ, normalised units) vs persistence: control 0.387 / 0.492 vs 0.404 / 0.533
(better at both horizons); variant 0.415 / 0.510 (slightly worse than persistence at +5).

- **The blur is real and the detector loss fixes most of it.** A plain-L1 forecast is closer to
  the real scan in pixels yet carries far less tornado signature than the last observed scan
  (judge −0.15 WC vs persistence, significant). Weighting the L1 by the judge's heat map and
  adding the judge's loss on the forecast lifts the judge's reading from 0.59 to 0.71 / 0.73,
  i.e. back to the persistence floor at +10 min (−0.001 n.s.) and within 0.044 of it at +5, at
  the cost of a small pixel-error increase. On 6,000 files and 4 epochs the drawn forecast does
  not yet beat persistence through the judge; whether more data and epochs push it over is the
  open question for the full fold B runs.
- **The trained head is the stronger predictor already.** At +10 min it is above the floor in
  both runs (control +0.017 [−0.002, +0.038]; variant +0.020 [+0.001, +0.040]) and at +5 it sits
  on it, with the ceiling 0.045 higher. This is the fine-tuned backbone learning the prediction
  target, which Experiments 4–7 showed the frozen features cannot do.
- Mechanics all hold: loading, mixed precision, the frozen judge in the graph, maps saved,
  pre-onset scoping. Runs: `results/forecast_probe/`.
- Reading the two routes side by side: the loss change repaired the one output that was broken
  (the drawn forecast, whose blur is exactly what the frozen detector keys on) and that output
  happens to be the indirect route's input; the direct head had nothing broken and only felt the
  change through the shared encoder (+0.003 to +0.006). "Better" for the head means better than
  the published TorNet on the latest real scan, never better than TorNet given the true future
  scan, which stays 0.045 higher at +10 min.

**Trim (decided 2026-10-14).** The probe settled enough to cut the run list from five to three
and to remove code no longer on the table.

- Dropped the heat-map-only run: it asks how much of the fix was the heat map versus the
  detector term, a nice-to-know rather than a decision driver; an ablation remains possible if
  the full run clears the floor.
- Held the transformer bottleneck: the decisive question is whether any forecast clears the
  persistence floor, and if the loss cannot get there a different bottleneck will not. Its config
  now carries the same losses as `fold_b_detector`, so bottleneck is the only variable when it
  does run (as written before the probe it had no detector loss and would have confounded B
  against A).
- Added the 12-channel anchored variant (above) as the one open design question.
- Removed the Keras U-Net forecaster, the ImageNet encoder option and the SMP classification
  head; replaced the two probe configs with `smoke_detector`. Evaluation (floor / forecast /
  truth / head, pre-onset scoping, paired bootstraps, likelihood maps) is untouched.

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

**Cache round 2 and the meta model** (experiment 7; everything reads the four-frame cache):

```
uv run tornet-radar-metrics --years 2013 2014 2015 --workers 8          # $TORNET_ROOT/radar_metrics
uv run tornet-pool-features --features_dir features_4f --out pooled_4f  # $TORNET_ROOT/pooled_4f
uv run tornet-env-fetch --years 2014 2015 --workers 8                   # $TORNET_ROOT/env_hrrr (resumable)
uv run tornet-train-head configs/cache2/fold_b_lead10_*.json configs/cache2/fold_b_lead0_*.json
uv run tornet-meta-model configs/meta/fold_b_lead10_*.json configs/meta/fold_b_lead0_*.json
uv run tornet-score experiments/<lead10_maxpool>/predictions.csv experiments/<lead10_concat>/predictions.csv experiments/<new heads and meta models>/predictions.csv
```

At full scale `scripts/runpod/cache_round2.sh` runs the same steps unattended (environment fetch
in the background while the heads use the GPU) and stops the pod when done through `runpodctl`,
which needs the laptop's `~/.runpod/config.toml` copied to the pod at launch.

**Radar forecaster** (step 2; reads the raw files, no cache):

```
uv run tornet-forecast configs/forecast/smoke_detector.json      # ~15 min on the laptop, ~5 min on the pod
uv run tornet-forecast configs/forecast/fold_b_detector.json     # ~2 h locally, ~45 min on the pod
# on a pod (fresh pod, repo rsynced to /workspace/tornet-repo, runpodctl + config copied):
cd /workspace/tornet-repo && UV_PROJECT_ENVIRONMENT=/root/venv setsid nohup \
  bash scripts/runpod/forecast_runs.sh > /workspace/forecast_runs.log 2>&1 < /dev/null &
```

Runs land under `$EXP_DIR` (default `experiments/`); configs must not set `exp_dir`. Each run
writes `forecast_scores/summary.json` (pre-onset WC AUC for persistence / forecast / truth /
head per horizon, MAE per product, paired bootstraps) and the likelihood maps.

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
