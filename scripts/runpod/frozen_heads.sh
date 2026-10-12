#!/usr/bin/env bash
# Frozen-backbone temporal heads at full scale, on a RunPod pod with the TorNet network volume
# mounted at /workspace (data at /workspace/tornet, this repo at /workspace/tornet-repo).
#
#   1. benchmark the loader from the network volume (how many files/s eight workers read)
#   2. cache the pretrained backbone's features for the last four scans of every sample,
#      all years and both partitions, into $TORNET_ROOT/features_4f (float16, ~370 KB/sample)
#   3. train the three heads in configs/frozen/full_*.json on the cache and score each on the
#      2021-2022 test partition; paired bootstraps against the maxpool control
#
# Idempotent: steps whose outputs exist are skipped, so re-run after any failure.
#
# Usage:
#   cd /workspace/tornet-repo && nohup bash scripts/runpod/frozen_heads.sh > /workspace/frozen_heads.log 2>&1 &
set -uo pipefail

export TORNET_ROOT="${TORNET_ROOT:-/workspace/tornet}"
export KERAS_BACKEND=torch
export EXP_DIR="${EXP_DIR:-/workspace/experiments}"
REPO="${REPO:-/workspace/tornet-repo}"
YEARS="${YEARS:-2013 2014 2015 2016 2017 2018 2019 2020 2021 2022}"
WORKERS="${WORKERS:-8}"
cd "$REPO"
mkdir -p "$EXP_DIR"

log() { echo "=== $(date -Is) $*"; }

if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
log "syncing environment"
uv sync --quiet || { log "uv sync failed"; exit 1; }
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

log "loader benchmark (2014 train, 200 batches of 64, $WORKERS workers)"
uv run python - <<PY
import os, time
from tornet.data.loader import get_dataloader
from tornet.data.keras_loader import iterate_batches
ds = get_dataloader(os.environ["TORNET_ROOT"], [2014], "train", 64, workers=$WORKERS, use_multiprocessing=True, n_frames=4)
t0 = time.time(); n = 0
for i, (x, y) in enumerate(iterate_batches(ds)):
    n += len(y)
    if i + 1 == 200: break
dt = time.time() - t0
print(f"loader: {n} files in {dt:.0f}s = {n/dt:.0f} files/s")
PY

FEATURES="$TORNET_ROOT/features_4f"
mkdir -p "$FEATURES"
for part in train test; do
  for y in $YEARS; do
    if [ -f "$FEATURES/${part}_${y}_features.npy" ]; then continue; fi
    log "cache $part $y"
    uv run tornet-cache-features --years "$y" --partitions "$part" --frames 4 --workers "$WORKERS" \
        --out "$FEATURES" || { log "CACHE FAILED $part $y"; exit 1; }
  done
done
du -sh "$FEATURES"

for cfg in full_maxpool full_temporal_attention full_temporal_concat; do
  if ls "$EXP_DIR"/frozen_${cfg#full_}_full_*/results.json >/dev/null 2>&1; then continue; fi
  log "train configs/frozen/$cfg.json"
  uv run tornet-train-head "configs/frozen/$cfg.json" || log "TRAIN FAILED $cfg"
done

preds() { ls -d "$EXP_DIR"/frozen_$1_full_* | tail -1; }
log "paired scores"
uv run tornet-score "$(preds maxpool)/predictions.csv" "$(preds temporal_attention)/predictions.csv" \
    "$(preds temporal_concat)/predictions.csv" --output "$EXP_DIR/frozen_full_scores.json" \
    | tee "$EXP_DIR/frozen_full_scores.txt"
uv run tornet-results "$EXP_DIR" --csv "$EXP_DIR/frozen_full_results.csv"
log "done"
