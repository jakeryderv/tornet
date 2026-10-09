#!/usr/bin/env bash
# Download and extract the full TorNet dataset from Zenodo into $TORNET_ROOT.
#
# Zenodo throttles to a few MB/s per client, so each archive is fetched with aria2c using
# many parallel connections, resumed if interrupted, and verified against the MD5 that
# Zenodo publishes for the file. Idempotent: a year is skipped once its .done marker
# exists, so the script can simply be re-run after any failure.
#
# Usage (on a pod with the network volume mounted at /workspace):
#   TORNET_ROOT=/workspace/tornet nohup bash download_tornet.sh > /workspace/tornet/download.log 2>&1 &
set -euo pipefail

TORNET_ROOT="${TORNET_ROOT:-/workspace/tornet}"
CONNECTIONS="${CONNECTIONS:-16}"
mkdir -p "$TORNET_ROOT"
cd "$TORNET_ROOT"

# Zenodo record IDs from the upstream README. 2013 also carries catalog.csv.
declare -A RECORDS=(
  [2013]=12636522 [2014]=12637032 [2015]=12655151 [2016]=12655179 [2017]=12655183
  [2018]=12655187 [2019]=12655716 [2020]=12655717 [2021]=12655718 [2022]=12655719
)

for tool in aria2c python3 curl tar; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "[$(date -Is)] installing $tool"
    apt-get update -qq && apt-get install -y -qq aria2 python3 curl >/dev/null
  fi
done

# Print "name md5 url" for every file in a Zenodo record.
record_files() {
  curl -sfL "https://zenodo.org/api/records/$1" | python3 -c '
import json, sys
for f in json.load(sys.stdin)["files"]:
    print(f["key"], f["checksum"].removeprefix("md5:"), f["links"]["self"])
'
}

fetch() {  # fetch <name> <md5> <url>
  aria2c --continue=true --max-tries=0 --retry-wait=30 --timeout=60 \
    -x "$CONNECTIONS" -s "$CONNECTIONS" -k 1M --file-allocation=none \
    --checksum="md5=$2" --console-log-level=warn --summary-interval=120 \
    -d . -o "$1" "$3"
}

for year in $(printf '%s\n' "${!RECORDS[@]}" | sort); do
  rec="${RECORDS[$year]}"
  marker=".${year}.done"
  if [[ -f "$marker" ]]; then
    echo "[$(date -Is)] $year already done, skipping"
    continue
  fi
  echo "[$(date -Is)] $year: listing record $rec"
  while read -r name md5 url; do
    echo "[$(date -Is)] $year: fetching $name"
    fetch "$name" "$md5" "$url"
    if [[ "$name" == *.tar.gz ]]; then
      echo "[$(date -Is)] $year: extracting $name"
      tar -xzf "$name"
      rm -f "$name"
    fi
  done < <(record_files "$rec")
  touch "$marker"
  echo "[$(date -Is)] $year: done; disk used: $(du -sh . | cut -f1)"
done

echo "[$(date -Is)] all years complete"
ls -la "$TORNET_ROOT"
for split in train test; do
  echo "$split: $(find "$TORNET_ROOT/$split" -name '*.nc' | wc -l) files"
done
