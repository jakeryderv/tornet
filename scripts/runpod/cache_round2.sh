#!/usr/bin/env bash
# Cache round 2 and the meta model, unattended, on a pod with the TorNet volume at /workspace
# (data and the four-frame feature cache at /workspace/tornet, this repo at
# /workspace/tornet-repo). Every step writes to the volume and is skipped when its output
# exists, so the script can be re-run after a failure. The last step stops the pod.
#
#   1. radar metrics: per-frame labels, storm locations and rotation metrics from the raw chips
#      (CPU, ~10 min with 16 workers)                        -> $TORNET_ROOT/radar_metrics
#   2. pooled embeddings of the four-frame cache (~5 min)   -> $TORNET_ROOT/pooled_4f
#   3. heads: configs/cache2/full_*.json in one process     -> $EXP_DIR/lead{10,0}_{...}_full_*
#      (GPU, ~30 min to load features then 10-15 min each)
#   4. HRRR environment for every storm hour from 2014-07-30 (network, hours) -> $TORNET_ROOT/env_hrrr
#   5. meta models: configs/meta/full_*.json (CPU, minutes) -> $EXP_DIR/meta_*_full_*
#   6. paired scores against the Experiment 6 lead-10 heads  -> $EXP_DIR/cache2_full_scores.*
#      (configs must not set exp_dir, or runs land under the repo instead of $EXP_DIR; the
#      2026-10-13 run did, so its runs were pulled from /workspace/tornet-repo/experiments)
#   7. stop the pod (runpodctl with ~/.runpod/config.toml copied from the laptop, or
#      RUNPOD_API_KEY in the pod environment for the REST fallback)
#
# Steps 3 and 4 do not depend on each other; 4 runs in the background while 3 uses the GPU.
#
# Usage (from a login shell on the pod; setsid so the ssh call returns):
#   cd /workspace/tornet-repo && UV_PROJECT_ENVIRONMENT=/root/venv setsid nohup \
#     bash scripts/runpod/cache_round2.sh > /workspace/cache_round2.log 2>&1 < /dev/null &
set -uo pipefail

export TORNET_ROOT="${TORNET_ROOT:-/workspace/tornet}"
export KERAS_BACKEND=torch
export EXP_DIR="${EXP_DIR:-/workspace/experiments}"
REPO="${REPO:-/workspace/tornet-repo}"
WORKERS="${WORKERS:-16}"
ENV_WORKERS="${ENV_WORKERS:-12}"
STOP_POD="${STOP_POD:-1}"
cd "$REPO"
mkdir -p "$EXP_DIR"
log() { echo "=== $(date -Is) $*"; }

stop_pod() {
  [ "$STOP_POD" = "1" ] || { log "STOP_POD=0, leaving the pod running"; return; }
  # Variables set at pod creation reach PID 1 but not SSH shells, so read the pod id from there.
  # runpodctl reads its key from ~/.runpod/config.toml (copied from the laptop at launch);
  # RUNPOD_API_KEY is the fallback for the REST call.
  [ -n "${RUNPOD_POD_ID:-}" ] || RUNPOD_POD_ID=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p')
  [ -n "${RUNPOD_API_KEY:-}" ] || RUNPOD_API_KEY=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_API_KEY=//p')
  if [ -z "$RUNPOD_POD_ID" ]; then
    log "RUNPOD_POD_ID missing; cannot stop the pod"
    return
  fi
  log "stopping pod $RUNPOD_POD_ID"
  sync
  # new CLI: `runpodctl pod stop`; the one baked into the pod images: `runpodctl stop pod`
  if command -v runpodctl >/dev/null 2>&1 && { runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"; }; then
    log "stop requested via runpodctl"
  elif [ -n "$RUNPOD_API_KEY" ] && curl -sS -f -X POST "https://rest.runpod.io/v2/pods/$RUNPOD_POD_ID/action" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
      -d '{"action":"stop"}' > /dev/null; then
    log "stop requested via REST"
  else
    log "stop request failed; the pod is still running"
  fi
}
trap stop_pod EXIT

[ -f "$TORNET_ROOT/features_4f/cache.json" ] || { log "no feature cache; run frozen_heads.sh first"; exit 1; }
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
log "syncing environment"
uv sync --quiet || { log "uv sync failed"; exit 1; }
uv run python -c "import torch, lightgbm, herbie; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

log "1. radar metrics ($WORKERS workers)"
uv run tornet-radar-metrics --workers "$WORKERS" || log "RADAR METRICS FAILED"

log "2. pooled embeddings"
uv run tornet-pool-features --features_dir features_4f --out pooled_4f || log "POOLING FAILED"

log "4. HRRR environment fetch in the background ($ENV_WORKERS workers)"
uv run tornet-env-fetch --workers "$ENV_WORKERS" > "$EXP_DIR/env_fetch.log" 2>&1 &
ENV_PID=$!

log "3. heads: configs/cache2/full_*.json"
heads=()
for cfg in configs/cache2/full_*.json; do
  name=$(uv run python -c "import json,sys; print(json.load(open(sys.argv[1]))['exp_name'])" "$cfg")
  if ls -d "$EXP_DIR/${name}_"* >/dev/null 2>&1; then log "  $name exists, skipping"; else heads+=("$cfg"); fi
done
if [ ${#heads[@]} -gt 0 ]; then
  uv run tornet-train-head "${heads[@]}" || log "HEADS FAILED"
fi

log "waiting for the environment fetch (pid $ENV_PID)"
wait "$ENV_PID" || log "ENV FETCH FAILED (see $EXP_DIR/env_fetch.log)"
tail -3 "$EXP_DIR/env_fetch.log"

log "5. meta models: configs/meta/full_*.json"
metas=()
for cfg in configs/meta/full_*.json; do
  name=$(uv run python -c "import json,sys; print(json.load(open(sys.argv[1]))['exp_name'])" "$cfg")
  if ls -d "$EXP_DIR/${name}_"* >/dev/null 2>&1; then log "  $name exists, skipping"; else metas+=("$cfg"); fi
done
if [ ${#metas[@]} -gt 0 ]; then
  uv run tornet-meta-model "${metas[@]}" || log "META MODELS FAILED"
fi

log "6. paired scores"
preds() { ls -d "$EXP_DIR"/$1_* 2>/dev/null | tail -1; }
existing() { for n in "$@"; do p="$(preds "$n")"; [ -n "$p" ] && [ -f "$p/predictions.csv" ] && echo "$p/predictions.csv"; done; }
uv run tornet-score $(existing lead10_maxpool_full lead10_concat_full lead10_spacetime_full lead10_diffconcat_full lead10_onset_full \
    meta_lead10_emb_full meta_lead10_radar_full meta_lead10_env_full meta_lead10_emb_radar_full meta_lead10_all_full) \
    --output "$EXP_DIR/cache2_lead10_full_scores.json" | tee "$EXP_DIR/cache2_lead10_full_scores.txt"
uv run tornet-score $(existing frozen_maxpool_full frozen_temporal_concat_full lead0_spacetime_full lead0_diffconcat_full lead0_onset_full \
    meta_lead0_emb_full meta_lead0_all_full) \
    --output "$EXP_DIR/cache2_lead0_full_scores.json" | tee "$EXP_DIR/cache2_lead0_full_scores.txt"
log "done"
