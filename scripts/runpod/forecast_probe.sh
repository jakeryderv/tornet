#!/usr/bin/env bash
# Forecaster probe on a pod with the TorNet volume at /workspace: two small fold B runs
# (configs/forecast/probe_*.json) back to back, then stop the pod. Idempotent per run.
# Usage: cd /workspace/tornet-repo && UV_PROJECT_ENVIRONMENT=/root/venv setsid nohup \
#   bash scripts/runpod/forecast_probe.sh > /workspace/forecast_probe.log 2>&1 < /dev/null &
set -uo pipefail
export TORNET_ROOT="${TORNET_ROOT:-/workspace/tornet}"
export KERAS_BACKEND=torch
export EXP_DIR="${EXP_DIR:-/workspace/experiments}"
REPO="${REPO:-/workspace/tornet-repo}"
STOP_POD="${STOP_POD:-1}"
cd "$REPO"
log() { echo "=== $(date -Is) $*"; }
stop_pod() {
  [ "$STOP_POD" = "1" ] || { log "STOP_POD=0, leaving the pod running"; return; }
  id=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p')
  [ -n "$id" ] || { log "no pod id"; return; }
  log "stopping pod $id"; sync
  runpodctl pod stop "$id" || runpodctl stop pod "$id" || log "stop failed"
}
trap stop_pod EXIT
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
log "syncing environment"; uv sync --quiet || { log "uv sync failed"; exit 1; }
uv run python -c "import torch, segmentation_models_pytorch as smp; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'smp', smp.__version__)"
for cfg in configs/forecast/probe_a_control.json configs/forecast/probe_a_heatmap_detector.json; do
  name=$(uv run python -c "import json,sys; print(json.load(open(sys.argv[1]))['exp_name'])" "$cfg")
  if ls -d "$EXP_DIR/${name}_"* >/dev/null 2>&1; then log "$name exists, skipping"; continue; fi
  log "run $cfg"; uv run tornet-forecast "$cfg" || log "RUN FAILED: $cfg"
done
log "done"
