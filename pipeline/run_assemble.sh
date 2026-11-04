#!/usr/bin/env bash
# Reassemble the channel cache npz from the intermediate files under data/ and compare them array
# by array against the existing cache. CPU only (about 2 minutes). Output goes to results/rebuild/
# and data/ is never touched.
#   bash pipeline/run_assemble.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PYTHON:-python}"
rm -rf results/rebuild
mkdir -p results/logs
"$PY" pipeline/assemble/build_features.py              > results/logs/assemble_pg.log 2>&1
"$PY" pipeline/assemble/build_stability.py             > results/logs/assemble_pg_stability.log 2>&1
"$PY" pipeline/assemble/build_channels.py --variant manual > results/logs/assemble_external.log 2>&1
"$PY" pipeline/assemble/verify.py | tee results/logs/assemble_verify.log
