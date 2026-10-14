# Changes from upstream

This is a fork of [mit-ll/tornet](https://github.com/mit-ll/tornet) at commit `92d9d97` (upstream
`main`, July 2024). It keeps the authors' data loaders, CNN baseline, metrics and display code, and
builds on them toward a CNN–transformer hybrid (see `docs/project.md`). This file is the high-level
record of how the fork differs from upstream. It is kept current as the work progresses.

## Modeling and data processing

The data loaders, preprocessing, CNN backbone, layers, losses and metrics are the authors' code,
byte-for-byte apart from formatting and type annotations. Changes so far are confined to the
classification head and to tooling around cached features:

- `tornet.models.heads` (new): heads as functions from a feature map to a logit, attachable to the
  backbone or to cached features. `maxpool` is the authors' head factored out unchanged.
  `transformer` tokenises the feature map, adds a convolutional position encoding (depthwise 3x3
  over the grid, so it stays size-agnostic), runs pre-norm encoder layers, and pools by max over
  per-token logits, mean, or a class token. `token_mlp` is the no-attention control: the same
  feed-forward blocks and pooling with no attention and no position encoding, its width chosen so
  the parameter count matches the transformer at the same depth (within 0.2% at width 384).
- `build_model(head=..., head_kwargs=...)` dispatches to those heads. The upstream `head='mlp'`
  option is removed; it flattened the feature map into a 4096-unit dense layer (~165M parameters)
  and was not a usable control.
- `tornet-cache-features` (new): cuts a model at the end of its backbone and writes float16
  feature maps (8 x 15 x 384 for the baseline) plus metadata per year and partition. `--verify`
  applies the model's own head to the cache and compares with saved full-model logits; on the
  pretrained weights the difference is at most 4e-4 over 7,021 test samples.
- `tornet-train-head` (new): trains any head on cached features with the same loss, sample weights,
  metrics, checkpointing and `results.json` layout as `tornet-train`, then scores the selected
  checkpoint on cached test features into `predictions.csv` and `scores.json`.
- `configs/heads/*.json`: a head sweep on fold B (train 2013+2015, validate 2014).
- Auxiliary regression targets: the loader now carries all four frame labels, and prediction and
  cache metadata include `minutes_to_start`, `minutes_since_end` (from the v1.1 tornado times) and
  `frame_fraction`. Heads accept `outputs=(...)` to emit named extra scalar outputs from the shared
  representation. `tornet-train-head` can train a second output jointly with the binary logit:
  `aux_target` `proximity` (distance in minutes to the tornado interval, clipped to a window, so
  nulls sit at the far edge) or `frame_fraction`, with a Huber loss and a weight. The binary logit
  remains the scored output, so every comparison is unchanged.
- Metric-learning head (`head='embedding'`): transformer tokens pooled to an L2-normalised
  embedding, with the logit computed from two learned prototypes (cosine margin over a
  temperature) so it trains, thresholds and scores like any other logit. Trained jointly with a
  soft supervised contrastive loss whose positive weights decay with the distance between a scalar
  target per sample (`contrastive_target`: the label, or `proximity`). `tornet-embeddings` (new)
  inspects a run's test embeddings: PCA scatters by category, EF and year, centroid cosines,
  nearest-centroid rates, and a k-means cross-tab against the categories.
- End-to-end fine-tuning: `build_model` names the backbone's final dropout `features`, so the
  backbone can be located in any model (upstream checkpoints fall back to the last Dropout).
  `tornet-train` gains `head_kwargs`, `init_from` (a `.keras` path or `pretrained` for the authors'
  Hugging Face weights, whose backbone is copied layer by layer while the head starts fresh) and
  `backbone_lr_scale` (a learning-rate multiplier for the backbone, implemented as a per-variable
  scaled Adam since Adam is invariant to gradient scale; 0 freezes the backbone).
  `configs/finetune/*.json` fine-tune on fold B from the pretrained backbone with a fresh maxpool
  head and with a class-token transformer head, at batch 64 (batch 128 needs 10.8 GB on the laptop
  GPU) with `decay_steps` doubled so the per-epoch decay is unchanged.
- Temporal variants (`frames` config key, default 1): the loader can return the last N scans
  stacked behind the batch axis, and `build_model(frames=N)` runs the unchanged backbone on every
  frame with shared weights before a head combines the per-frame feature maps.
  `temporal_attention` applies self-attention over the frames at each feature-map location (with
  a learned per-frame position) and feeds the last frame's attended token to the authors' maxpool
  head; `temporal_concat` is the no-attention control, concatenating frames along channels, and
  is parameter-matched to one attention layer. Both keep the per-location logit and global max.
  `tornet-test` infers the frame count from a checkpoint. `configs/temporal/*.json` run both on
  fold B at batch 16.
- Frozen-backbone temporal heads: `tornet-cache-features --frames N` runs the pretrained backbone
  on the last N scans with shared weights and writes `(N, T, H, W, C)` caches (`--verify` checks
  the last frame against single-frame predictions). `tornet-train-head` builds the temporal heads
  on such a cache, and gives single-frame heads the last frame so the maxpool control trains from
  the same cache. `features_dir` may be relative to `TORNET_ROOT`. `configs/frozen/` holds fold B
  and full-data configs for the control and both temporal heads.
- Lead-time study: `tornet-train-head` takes a `frames` key naming which scans of a multi-frame
  cache the head sees (oldest first; one index gives a single-frame head that scan, several give a
  temporal head that window), so the detection label is predicted from scans 5, 10 or 15 minutes
  earlier. Several configs in one invocation share the loaded features. `configs/lead/` and
  `scripts/runpod/lead_heads.sh`.
- Cache round 2 (richer heads on the same cache): `spacetime_transformer` attends over every cell
  of every frame (time × H × W tokens, convolutional position encoding plus a learned per-frame
  embedding); `temporal_diff_concat` hands the concat head the frame-to-frame feature differences
  as well; every head accepts `output_units` so an extra output can be a vector, and
  `tornet-train-head` has the auxiliary target `frame_labels` (one cross-entropy logit per cached
  frame, so a head that sees only older scans is also asked for the label trajectory). The
  per-frame labels come from `meta_extra`, a CSV or directory of CSVs joined onto the cache metadata
  by filename. `configs/cache2/`.
- Meta model (the frozen backbone as a feature extractor rather than a thing to put heads on):
  `tornet-pool-features` reduces a feature cache to one max+mean vector per scan (768 values at
  width 384; the whole dataset fits in ~1.2 GB); `tornet-radar-metrics` reads the raw chips once
  for per-frame labels and times, storm location (site and chip centre) and hand-computed
  rotation metrics per frame and sweep (rotational velocity proxy, azimuthal shear, peak velocity
  and reflectivity, valid fraction); `tornet-env-fetch` joins near-storm HRRR environment (CAPE,
  CIN, 0–1 and 0–3 km helicity, 0–1 and 0–6 km shear, LCL height, 2 m temperature and dewpoint;
  point plus a 27 km box max and mean) from the public AWS archive by HTTP byte range through
  Herbie, from the run initialised two hours before the scan so the forecast predates the storm;
  `tornet-meta-model` trains LightGBM on any combination of those groups (`features`) with the
  authors' sample weights, early stopping on validation AUC, and writes the usual run directory
  plus gain-based feature importance by group. `configs/meta/`. HRRR exists from 2014-07-30, so
  earlier samples carry an availability flag and NaN fields; LightGBM handles the gaps natively.

Known paper/code discrepancy, not altered: the paper describes 2, 3, 3, 3 convolutions per block,
while the code and the pretrained weights have 2, 2, 3, 3. See `docs/project.md` section 3.

## Packaging and environment

- `pyproject.toml` with a lockfile (`uv.lock`) replaces `setup.py`, `pytest.ini` and the five
  `requirements/*.txt` files. Hatchling build, Python 3.11+, numpy 2.
- Package moved from `tornet/` to `src/tornet/`.
- Dependencies are split into extras: core (numpy, pandas, xarray, netCDF4, matplotlib, tqdm),
  `torch`, `keras`, `hf`, `notebooks`. The `torch` and `hf` groups install by default. Torch is the
  only backend supported, because PyPI tensorflow wheels have no Blackwell GPU kernels.
- Dropped the unused `glob2` dependency and the conda-specific `fix_cuda_paths.sh`.

## Layout

- Package moved to `src/tornet/` and flattened now that Keras is the only framework:
  `data/keras/loader.py` → `data/keras_loader.py`, `metrics/keras/metrics.py` → `metrics.py`,
  `models/keras/*` → `models/*`. Models saved by this fork reference `tornet.models.layers`; the
  published weights still load because the custom layers are found by their registered names.
- `get_dataloader` takes no `dataloader` argument and always returns the Keras loader. The
  `dataloader` config key is gone.

## Removed

All of this is in git history (last present at commit `34754a8`).

- The TensorFlow-only paths: the `tf.data` loader (`data/tf/`), the TFDS dataset builder
  (`data/tfds/`), `display/tboard.py`, and the TensorBoard image callback in `models/keras/callbacks.py`.
  The TFDS builder was the authors' fast-loading option for full-scale training; if the basic loader
  proves too slow on the full dataset, a numpy or zarr shard cache is the planned replacement.
- The standalone PyTorch Lightning port of the model (`models/torch/`) and its torch `Dataset`
  loader (`data/torch/`). Nothing in the pipeline used them.
- The three `ModelTraining_*` notebooks; `tornet-train` covers them. `DataLoaders` and
  `VisualizeSamples` remain, trimmed of the removed loaders and of `sys.path` hacks.
- `models/README.md` (folded into the README) and the `tensorflow` dependency extra.
- torchvision, torchmetrics and lightning from the `torch` extra; only the Lightning port used them.

## Commands

The two scripts under `scripts/tornado_detection/` are now console commands in `tornet.scripts`:

- `tornet-train <config.json>`: unchanged training loop, plus the behavior changes below.
- `tornet-test`: unchanged metrics, plus `--years` to evaluate a subset of test years,
  `--batch_size`, `--output` for a JSON of scores, and `--predictions` for a per-sample CSV of
  logits and metadata. The `--dataloader` option is gone with the other loaders.
- `tornet-score` (new): scores prediction CSVs offline. Reports the paper's three test views,
  AUC with bootstrap confidence intervals, CSI at the best threshold, recall by EF rating, AUC by
  year, a reliability table, and a paired bootstrap of the AUC difference between models.
- `tornet-results` (new): tabulates `results.json` files under `experiments/`, optionally averaging
  folds that share an experiment name.
- `tornet-cache-features`, `tornet-train-head` and `tornet-embeddings` (new): see Modeling above.
- `tornet-pool-features`, `tornet-radar-metrics`, `tornet-env-fetch` and `tornet-meta-model`
  (new): the meta-model pipeline, see Modeling above. `scripts/runpod/cache_round2.sh` runs all of
  it unattended on a pod and stops the pod when done.
- `tornet-train-forecaster` and `tornet-score-forecast` (new): the Keras radar forecaster, kept
  as a reference. `tornet-forecast` (new): the SMP multi-task forecaster, see Modeling above.

Both original commands default `KERAS_BACKEND` to `torch` when it is unset, and read `TORNET_ROOT`
at run time rather than at import.

## Training behavior

- Seeded: a `seed` config key (default 1234) seeds Python, numpy and the backend, and is recorded.
- Checkpoints: by default only `checkpoints/best.keras` (highest validation AUC) and
  `checkpoints/last.keras` are kept. `save_all_checkpoints: true` restores one file per epoch.
- Run directories are `experiments/<exp_name>_<yymmdd_HHMMSS>` (SLURM ids appended only when set)
  and contain a `results.json` with the best epoch, AUC, AUCPR, years, seed, git commit and dirty
  flag, alongside the upstream `history.csv`, `params.json` and `data.json`.
- An empty `val_years` list trains without a validation set, for the final all-years retrain.
- The config is deep-copied so repeated runs in one process do not share loader state.

- A garbage-collection callback in both trainers, every 50 training batches and at epoch end. Keras
  on the torch backend leaves reference cycles holding GPU tensors after every step; the four-frame
  models grew from 5 GB to an out-of-memory error inside one 1,452-step epoch, and multi-output
  heads grew about 0.15 GB per epoch. Collecting keeps memory flat. The trainers and `tornet-test`
  also default `PYTORCH_CUDA_ALLOC_CONF` to expandable segments, since the smaller last batch of an
  epoch otherwise fragments the allocator.

## Evaluation behavior

- `tornet-test` runs the model once, keeps every logit, and derives the same six metrics from
  them. Verified identical to `model.evaluate` on the pretrained weights.
- Evaluation and caching iterate the loader through Keras' own adapter, so `--workers` processes
  apply there too (about 4x faster than the single-process path).
- `tornet-test` collects garbage every 50 batches (the same per-call leak as in training) and its
  default batch is 64 scans divided by the model's frame count, so four-frame checkpoints test at
  16 samples per batch.
- The custom Keras layers are imported before loading a model. Upstream's test script could not load
  the published weights under current Keras without this.

## Configs

- `configs/tornado_baseline.json` is the authors' config, moved from
  `scripts/tornado_detection/config/params.json`; its hyperparameters are unchanged (only the
  obsolete `dataloader` key was dropped).
- `configs/fold_a.json`, `fold_b.json`, `fold_c.json`, `final.json` (new): a three-fold local
  cross-validation on 2013–2015, each fold holding out one year, plus the all-years retrain. They
  differ from the baseline config in year lists, batch 64 (128 runs out of memory on a 12 GB GPU)
  with `decay_steps` set to each run's steps per epoch, `label_smooth` 0.1 (the paper reports 10–20% helped; the shipped config has 0), eight loader
  worker processes (threaded workers segfault in NetCDF/HDF5; processes give ~4x the throughput of a
  single thread), and output directory.
- `configs/cache2/` and `configs/meta/`: the cache-round-2 heads and the meta-model ablations
  (embeddings, radar metrics, environment, and their combinations) at the 10-minute lead and at
  onset, for fold B and the full data.

## Tests

- Upstream's dataset tests skip instead of crashing when `TORNET_ROOT` is unset, and discover
  which years are present rather than hardcoding four.
- New data-free tests: a synthetic TorNet directory built in `tests/conftest.py` exercises the file
  reader, catalog query, loaders, every preprocessing function, both commands end to end, the
  offline scorer, and the results table. 37 tests run without the dataset; 4 more run against real
  data when `TORNET_ROOT` points at it.

## Code quality

- Formatted with ruff; lint rules E, F, W, I, UP, B. `ty` runs as an advisory type checker.
- Mechanical fixes only: implicit `Optional` annotations made explicit, mutable argument defaults
  replaced, lambdas and percent formatting converted, a bare `except` narrowed, dead variables and
  duplicate imports removed, license and description strings merged into one docstring per file.
- One upstream test bug fixed: the dataset training test built an optimizer and never passed it to
  `compile`.

## Data and infrastructure

- `scripts/runpod/download_tornet.sh` (new): downloads and extracts all ten years from Zenodo with
  `aria2c`, verifying the published MD5 checksums. Idempotent and resumable. `zenodo_get`, which the
  upstream README recommends, managed ~100 KB/s from a datacenter; `aria2c` with 16 connections
  reached 10–16 MB/s.
- Reference artifacts in `results/`: the pretrained model's per-sample predictions and score report
  on the 2013–2014 test partition.

- Optional extras `env` (Herbie, scipy) and `meta` (LightGBM) in `pyproject.toml`, both in the
  default `uv sync` groups.
- Full-data derived tables from the 2026-10-13 pod run live on the volume and locally under
  `~/data/tornet/full/`: `radar_metrics/` (per-frame labels, storm locations, rotation metrics
  for all 203k samples), `env_hrrr/env_hrrr.csv` (HRRR environment for 18,230 storm hours,
  1,969 without data before 2014-07-30) and `pooled_4f/` (max+mean pooled backbone features,
  1.2 GB). `configs/cache2` and `configs/meta` no longer set `exp_dir`, so pod runs land under
  `$EXP_DIR`.

## Documentation

- README rewritten for the fork: attribution and citation, setup, data download, commands, the
  subset workflow, layout. The upstream README is archived verbatim at
  `references/upstream-README.md`.
- `docs/project.md`: project notes, verified baseline architecture, hybrid design, workflow,
  references. `references/`: the journal version of the TorNet paper.
- `.gitignore`: notebooks under `notebooks/` are tracked again; `.venv`, tool caches and build output
  are ignored.
- `tornet_image.png` moved to `docs/`.

## Not changed

- `display/display.py`, the radar plotting code.
- The `LICENSE` file and the DFARS distribution notice at the top of every upstream-derived file.
  New files written for the fork do not carry the notice.
