# Changes from upstream

This is a fork of [mit-ll/tornet](https://github.com/mit-ll/tornet) at commit `92d9d97` (upstream
`main`, July 2024). It keeps the authors' data loaders, CNN baseline, metrics and display code, and
builds on them toward a CNN–transformer hybrid (see `docs/project.md`). This file is the high-level
record of how the fork differs from upstream. It is kept current as the work progresses.

## Modeling and data processing

No changes yet. The model builder, layers, losses, metrics, loaders and preprocessing are the
authors' code, byte-for-byte apart from formatting and type annotations. Any change to model or
data logic will be recorded here first.

Known paper/code discrepancy, not altered: the paper describes 2, 3, 3, 3 convolutions per block,
while the code and the pretrained weights have 2, 2, 3, 3. See `docs/project.md` §3.

## Packaging and environment

- `pyproject.toml` with a lockfile (`uv.lock`) replaces `setup.py`, `pytest.ini` and the five
  `requirements/*.txt` files. Hatchling build, Python 3.11+, numpy 2.
- Package moved from `tornet/` to `src/tornet/`.
- Dependencies are split into extras: core (numpy, pandas, xarray, netCDF4, matplotlib, tqdm),
  `torch`, `tensorflow`, `keras`, `hf`, `notebooks`. The `torch` and `hf` groups install by default;
  `tensorflow` is opt-in because PyPI wheels have no Blackwell GPU kernels.
- Dropped the unused `glob2` dependency and the conda-specific `fix_cuda_paths.sh`.

## Commands

The two scripts under `scripts/tornado_detection/` are now console commands in `tornet.scripts`:

- `tornet-train <config.json>`: unchanged training loop, plus the behavior changes below.
- `tornet-test`: unchanged metrics, plus `--years` to evaluate a subset of test years,
  `--batch_size`, `--output` for a JSON of scores, and `--predictions` for a per-sample CSV of
  logits and metadata (keras loader only).
- `tornet-score` (new): scores prediction CSVs offline. Reports the paper's three test views,
  AUC with bootstrap confidence intervals, CSI at the best threshold, recall by EF rating, AUC by
  year, a reliability table, and a paired bootstrap of the AUC difference between models.
- `tornet-results` (new): tabulates `results.json` files under `experiments/`, optionally averaging
  folds that share an experiment name.

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

## Evaluation behavior

- With the keras loader, `tornet-test` runs the model once, keeps every logit, and derives the same
  six metrics from them. Verified identical to `model.evaluate` on the pretrained weights.
- The custom Keras layers are imported before loading a model. Upstream's test script could not load
  the published weights under current Keras without this.

## Configs

- `configs/tornado_baseline.json` is the authors' config, moved from `scripts/tornado_detection/config/params.json` and unchanged.
- `configs/fold_a.json`, `fold_b.json`, `final.json` (new): a two-fold local cross-validation on
  2013–2014 mirroring the paper's year-pair folds, plus the all-years retrain. They differ from the
  baseline config in year lists, `decay_steps` scaled to the subset, `label_smooth` 0.1 (the paper
  reports 10–20% helped; the shipped config has 0), loader workers, and output directory.

## Tests

- Upstream's dataset tests skip instead of crashing when `TORNET_ROOT` is unset, and discover
  which years are present rather than hardcoding four.
- New data-free tests: a synthetic TorNet directory built in `tests/conftest.py` exercises the file
  reader, catalog query, loaders, every preprocessing function, both commands end to end, the
  offline scorer, and the results table. 37 tests run without the dataset; 6 more run against real data when `TORNET_ROOT` points at it.

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

## Documentation

- README: setup section rewritten for `uv`, a "local iteration on a subset" section, and the new
  commands. The rest is upstream's.
- `docs/project.md`: project notes, verified baseline architecture, hybrid design, workflow,
  references. `references/`: the journal version of the TorNet paper.
- `.gitignore`: notebooks under `notebooks/` are tracked again; `.venv`, tool caches and build output
  are ignored.

## Not changed

- The five notebooks.
- The TFDS dataset builder and the tf and torch data loaders.
- The standalone PyTorch Lightning port of the model in `tornet.models.torch`.
- The license and the DFARS notices in every file.
