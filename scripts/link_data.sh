#!/usr/bin/env bash
# Link <repo>/data to the channel cache directory and verify its checksums.
#   bash scripts/link_data.sh /path/to/plmfusion_data_2026-11-03
#   SKIP_VERIFY=1 bash scripts/link_data.sh /path/...   # skip checksums (saves about a minute)
#
# Setting PLMFUSION_DATA instead of making this link works equally well and takes priority.
# See DATA.md for what the directory contains.
set -euo pipefail

DATA="${1:-${PLMFUSION_DATA:-}}"
if [ -z "$DATA" ]; then
    echo "usage: bash scripts/link_data.sh /path/to/plmfusion_data_<snapshot>" >&2
    echo "The reference values in reference/ were produced with the 2026-11-03 snapshot." >&2
    exit 1
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -d "$DATA" ]; then
    echo "No such data directory: $DATA" >&2
    exit 1
fi
if [ -e "$REPO/data" ] && [ ! -L "$REPO/data" ]; then
    echo "$REPO/data is a real directory, not a symbolic link. Leaving it alone." >&2
    exit 1
fi
ln -sfn "$DATA" "$REPO/data"
echo "data -> $DATA"

if [ "${SKIP_VERIFY:-0}" != 1 ]; then
    (cd "$DATA" && sha256sum --quiet -c MANIFEST.sha256)
    echo "checksums OK ($(wc -l < "$DATA/MANIFEST.sha256") files)"
fi
