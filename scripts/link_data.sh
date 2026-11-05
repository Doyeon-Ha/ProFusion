#!/usr/bin/env bash
# Link <repo>/data to the channel cache directory, then confirm it is the right snapshot.
#   bash scripts/link_data.sh /path/to/plmfusion_data
#   SKIP_VERIFY=1 bash scripts/link_data.sh /path/...   # skip checksums (saves about a minute)
#
# The directory may be called anything. Nothing in the code reads its name: src/plmfusion/paths.py
# takes the path and nothing else. Which snapshot it holds is decided by SNAPSHOT below, not by
# what the directory is called.
#
# Setting PLMFUSION_DATA instead of making this link works equally well and takes priority.
# See DATA.md for what the directory contains.
set -euo pipefail

# sha256 of MANIFEST.sha256 itself. It says which manifest was distributed, and that is what tells
# the snapshots apart. It says nothing about the current state of the data, because editing a data
# file does not rewrite the manifest: the sha256sum -c run at the end of this script is the check
# that catches listed files which are altered or missing. Two files are in neither check, since the
# manifest does not list them (DATA.md, Integrity).
# reference/ was frozen against the snapshot below. Changing this line means refreezing reference/.
SNAPSHOT=0f1fd5588c6e9366d42e12480822bdf5f841b0c3da1eb54a579d1e2408f8133e

DATA="${1:-${PLMFUSION_DATA:-}}"
if [ -z "$DATA" ]; then
    echo "usage: bash scripts/link_data.sh /path/to/the/channel/cache" >&2
    echo "The expected snapshot has manifest ${SNAPSHOT:0:12}, and reference/ was frozen against it." >&2
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

# Snapshot identity. This costs milliseconds, so it runs even when SKIP_VERIFY is set: pointing at
# an older snapshot is the most likely failure here, and nothing else reports it until the cohort
# check in tests/test_reproduce.py, which is hours of compute away.
if [ ! -f "$DATA/MANIFEST.sha256" ]; then
    echo "WARNING: no MANIFEST.sha256 in $DATA, so the snapshot cannot be identified." >&2
else
    GOT="$(sha256sum "$DATA/MANIFEST.sha256" | cut -d' ' -f1)"
    if [ "$GOT" != "$SNAPSHOT" ]; then
        echo "WARNING: this is not the snapshot reference/ was frozen against." >&2
        echo "  expected manifest ${SNAPSHOT:0:12}, found ${GOT:0:12}" >&2
        echo "  Snapshots differ in which backbones and which assays they hold. The code runs" >&2
        echo "  normally on any of them while the cohort changes silently, so the paper numbers" >&2
        echo "  will not reproduce and pytest tests/ fails on the cohort check." >&2
    else
        echo "snapshot OK (manifest ${GOT:0:12})"
    fi
fi

if [ "${SKIP_VERIFY:-0}" != 1 ]; then
    (cd "$DATA" && sha256sum --quiet -c MANIFEST.sha256)
    echo "checksums OK ($(wc -l < "$DATA/MANIFEST.sha256") files)"
fi
