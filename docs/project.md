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

A variant is a new `exp_name` prefix on the same configs. Subset scores sit below the paper's because
of less training data and a higher tornado base rate (13% vs 6%); compare the hybrid against the
baseline on the same subset, never against the published numbers. The bootstrap CI on the subset
NC AUC is about ±0.016, which is the noise floor a variant has to clear.

**Config notes.** `decay_steps` in each config equals that config's steps per epoch at batch 128,
matching the paper's one-epoch decay period. Batch 128 may not fit in 12 GB; if so, drop to 64 and
double `decay_steps`.

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
