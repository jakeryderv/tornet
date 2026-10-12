#!/usr/bin/env bash
# Lead-time study on the frozen four-frame feature cache (written by frozen_heads.sh), on a pod
# with the TorNet volume at /workspace: heads that see only the older scans, so detection on the
# newest scan becomes prediction from 5, 10 or 15 minutes earlier. All configs run in one process
# so the 63 GB of train+val features load once. Paired scores against the all-frames control.
#
# Usage:
#   cd /workspace/tornet-repo && nohup bash scripts/runpod/lead_heads.sh > /workspace/lead_heads.log 2>&1 < /dev/null &
set -uo pipefail

export TORNET_ROOT="${TORNET_ROOT:-/workspace/tornet}"
export KERAS_BACKEND=torch
export EXP_DIR="${EXP_DIR:-/workspace/experiments}"
REPO="${REPO:-/workspace/tornet-repo}"
cd "$REPO"
log() { echo "=== $(date -Is) $*"; }

[ -f "$TORNET_ROOT/features_4f/cache.json" ] || { log "no feature cache; run frozen_heads.sh first"; exit 1; }
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
uv sync --quiet || { log "uv sync failed"; exit 1; }

log "train configs/lead/full_*.json"
uv run tornet-train-head configs/lead/full_lead15_maxpool.json configs/lead/full_lead10_maxpool.json \
    configs/lead/full_lead10_concat.json configs/lead/full_lead10_attention.json \
    configs/lead/full_lead5_maxpool.json configs/lead/full_lead5_concat.json \
    configs/lead/full_lead5_attention.json || log "TRAIN FAILED"

preds() { ls -d "$EXP_DIR"/$1_* | tail -1; }
log "paired scores"
uv run tornet-score "$(preds frozen_maxpool_full)/predictions.csv" \
    "$(preds lead15_maxpool_full)/predictions.csv" "$(preds lead10_maxpool_full)/predictions.csv" \
    "$(preds lead10_concat_full)/predictions.csv" "$(preds lead10_attention_full)/predictions.csv" \
    "$(preds lead5_maxpool_full)/predictions.csv" "$(preds lead5_concat_full)/predictions.csv" \
    "$(preds lead5_attention_full)/predictions.csv" \
    --output "$EXP_DIR/lead_full_scores.json" | tee "$EXP_DIR/lead_full_scores.txt"
log "done"
