#!/usr/bin/env bash
# Download and extract the full TorNet dataset from Zenodo into $TORNET_ROOT.
#
# Idempotent: each year is skipped once its .done marker exists, so the script can be
# re-run after an interruption. Archives are deleted after extraction to cap disk use
# at roughly the extracted size plus one archive.
#
# Usage (on a RunPod pod with the network volume mounted at /workspace):
#   TORNET_ROOT=/workspace/tornet nohup bash download_tornet.sh > /workspace/tornet/download.log 2>&1 &
set -euo pipefail

TORNET_ROOT="${TORNET_ROOT:-/workspace/tornet}"
mkdir -p "$TORNET_ROOT"
cd "$TORNET_ROOT"

# Zenodo record IDs from the upstream README. 2013 also carries catalog.csv.
declare -A RECORDS=(
  [2013]=12636522 [2014]=12637032 [2015]=12655151 [2016]=12655179 [2017]=12655183
  [2018]=12655187 [2019]=12655716 [2020]=12655717 [2021]=12655718 [2022]=12655719
)

# zenodo_get runs via uv so the script does not depend on the image's Python or pip.
if ! command -v uv >/dev/null 2>&1; then
  echo "[$(date -Is)] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
export PATH="$HOME/.local/bin:$PATH"
ZENODO_GET=(uvx zenodo_get)

for year in $(printf '%s\n' "${!RECORDS[@]}" | sort); do
  rec="${RECORDS[$year]}"
  marker=".${year}.done"
  if [[ -f "$marker" ]]; then
    echo "[$(date -Is)] $year already done, skipping"
    continue
  fi
  echo "[$(date -Is)] $year: downloading record $rec"
  # zenodo_get verifies md5 checksums and resumes partial files on retry.
  "${ZENODO_GET[@]}" -r "$rec" -o .
  archive="tornet_${year}.tar.gz"
  echo "[$(date -Is)] $year: extracting $archive"
  tar -xzf "$archive"
  rm -f "$archive"
  touch "$marker"
  echo "[$(date -Is)] $year: done; disk used: $(du -sh . | cut -f1)"
done

echo "[$(date -Is)] all years complete"
ls -la "$TORNET_ROOT"
for split in train test; do
  echo "$split: $(find "$TORNET_ROOT/$split" -name '*.nc' | wc -l) files"
done
