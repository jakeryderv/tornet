#!/usr/bin/env bash
# Repackage TorNet as Parquet and publish it on the Hugging Face Hub, from a CPU pod with the
# TorNet volume at /workspace. Phases, each idempotent:
#   convert [years...]  netCDF -> Parquet shards on the container disk ($OUT), given years
#                       (default all ten); resumes per shard, so reruns only rebuild what is missing
#   integrity           read every shard fully, delete unreadable ones (then convert rebuilds them)
#   verify [years...]   row counts vs the catalog and exact round trip of random samples
#   repair              integrity, convert (rebuilds deleted shards), integrity, verify, catalog
#   catalog             merge the per-year metadata into catalog.parquet and copy the card
#   auto                convert 2013, verify it, convert the rest, verify all, catalog (no upload)
#   upload              hf upload (multi-commit, resumable) to $HF_REPO; stops the pod on success only
#   publish             convert, integrity, verify, catalog, upload, stop on success (unattended)
# The pod is NOT stopped after convert/catalog: stopping wipes the container disk.
# Usage: cd /workspace/tornet-repo && UV_PROJECT_ENVIRONMENT=/root/venv setsid nohup \
#   bash scripts/runpod/publish_hf.sh convert 2013 2014 2015 > /workspace/publish_hf.log 2>&1 < /dev/null &
set -uo pipefail
export TORNET_ROOT="${TORNET_ROOT:-/workspace/tornet}"
REPO="${REPO:-/workspace/tornet-repo}"
OUT="${OUT:-/root/hf_tornet}"
HF_REPO="${HF_REPO:-jakeryderv/tornet}"
WORKERS="${WORKERS:-$(( $(free -g | awk '/^Mem:/{print $2}') / 3 ))}"
cd "$REPO"
log() { echo "=== $(date -Is) $*"; }
stop_pod() {
  id=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p')
  [ -n "$id" ] || { log "no pod id"; return; }
  log "stopping pod $id"; sync
  runpodctl pod stop "$id" || runpodctl stop pod "$id" || log "stop failed"
}
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
phase="${1:-convert}"; shift || true
case "$phase" in
  convert)
    log "syncing environment"; uv sync --quiet || { log "uv sync failed"; exit 1; }
    years=("$@"); [ ${#years[@]} -gt 0 ] || years=(2013 2014 2015 2016 2017 2018 2019 2020 2021 2022)
    log "convert years ${years[*]} with $WORKERS workers into $OUT"
    # the converter resumes per shard (complete shards are only read for their metadata), so
    # every partition-year is passed through; deleted or missing shards get rebuilt
    for y in "${years[@]}"; do
      for p in train test; do
        log "convert $p/$y"
        uv run tornet-to-parquet convert --out "$OUT" --partitions "$p" --years "$y" --workers "$WORKERS" \
          || { log "CONVERT FAILED $p/$y"; exit 1; }
      done
    done
    log "convert done: $(du -sh "$OUT" | cut -f1) in $OUT"
    ;;
  integrity)
    uv run tornet-to-parquet integrity --out "$OUT" --workers "${INTEGRITY_WORKERS:-16}" || { log "INTEGRITY FAILED"; exit 1; }
    log "integrity ok"
    ;;
  repair)
    bash "$0" integrity || true
    bash "$0" convert || exit 1
    bash "$0" integrity || exit 1
    bash "$0" verify || exit 1
    bash "$0" catalog || exit 1
    log "repair done: ready for upload"
    ;;
  verify)
    uv run tornet-to-parquet verify --out "$OUT" ${1:+--years "$@"} || { log "VERIFY FAILED"; exit 1; }
    log "verify ok"
    ;;
  auto)
    bash "$0" convert 2013 || exit 1
    bash "$0" verify 2013 || exit 1
    bash "$0" convert 2014 2015 2016 2017 2018 2019 2020 2021 2022 || exit 1
    bash "$0" integrity || { bash "$0" convert || exit 1; bash "$0" integrity || exit 1; }
    bash "$0" verify || exit 1
    bash "$0" catalog || exit 1
    log "auto done: ready for upload (pod left running; container disk holds $OUT)"
    ;;
  catalog)
    uv run tornet-to-parquet catalog --out "$OUT" || { log "CATALOG FAILED"; exit 1; }
    cp docs/hf_dataset_card.md "$OUT/README.md"
    log "catalog done"
    ;;
  upload)
    # the pod stops ONLY after a successful upload: stopping wipes the container disk and the shards
    [ -f "$OUT/catalog.parquet" ] || { log "no catalog.parquet: run the catalog phase first"; exit 1; }
    log "upload $OUT -> $HF_REPO"
    # huggingface_hub 2.x: `hf upload` splits a big folder into several commits and resumes on rerun
    if uv run hf upload "$HF_REPO" "$OUT" . --repo-type dataset --commit-message "TorNet v1.1 as Parquet"; then
      log "upload done"
      [ "${STOP_POD:-1}" = "1" ] && stop_pod
    else
      log "UPLOAD FAILED (pod left running; rerun the upload phase, it resumes)"
      exit 1
    fi
    ;;
  publish)
    # the whole thing unattended: convert, integrity, verify, catalog, upload, stop on success
    bash "$0" convert || exit 1
    bash "$0" integrity || { bash "$0" convert || exit 1; bash "$0" integrity || exit 1; }
    bash "$0" verify || exit 1
    bash "$0" catalog || exit 1
    bash "$0" upload || exit 1
    ;;
  *) log "unknown phase $phase"; exit 2 ;;
esac
log "done $phase"
