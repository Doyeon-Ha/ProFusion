#!/usr/bin/env bash
# MSA for the 23 external tuning-set sequences (hhblits, UniRef30_2023_02).
# The same database and parameters as the other MSA runs in this project.
# A node-local copy of the database ("<MSA_DB_DIR>") is used.
# Sequences that already have an a3m are skipped (idempotent, resumable).
#
# Usage: experiments/mavedb_external/run_hhblits.sh [N_PARALLEL] [CPU_PER_JOB]
set -uo pipefail

REPO="<REPO>"
DB="<MSA_DB_DIR>/UniRef30_2023_02/UniRef30_2023_02"
QDIR=$REPO/data/mavedb_external/msa/query
A3M=$REPO/data/mavedb_external/msa/a3m
LOGD=$REPO/data/mavedb_external/msa/logs
HHBLITS="<CONDA_ENVS>/hhsuite/bin/hhblits"

NJ=${1:-8}
CPU=${2:-4}
mkdir -p "$A3M" "$LOGD"

run_one() {
  local q="$1" tag
  tag=$(basename "$q" .fasta)
  [ -s "$A3M/$tag.a3m" ] && { echo "[skip] $tag"; return 0; }
  "$HHBLITS" -i "$q" -d "$DB" -oa3m "$A3M/$tag.a3m" \
             -n 3 -e 1e-3 -maxfilt 100000 -realign_max 100000 -maxseq 1000000 \
             -cpu "$CPU" > "$LOGD/$tag.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$A3M/$tag.a3m" ]; then
    echo "[FAIL rc=$rc] $tag"; return 1
  fi
  echo "[ok] $tag  seqs=$(grep -c '^>' "$A3M/$tag.a3m")"
}
export -f run_one; export A3M LOGD HHBLITS DB CPU

echo "[$(date +%H:%M:%S)] hhblits starting: $(ls "$QDIR"/*.fasta | wc -l) sequences, $NJ at a time x ${CPU} cpu"
# Longest first, to shorten the total wall time
ls -S "$QDIR"/*.fasta | xargs -P "$NJ" -I{} bash -c 'run_one "$@"' _ {}
echo "[$(date +%H:%M:%S)] done: a3m $(ls "$A3M"/*.a3m 2>/dev/null | wc -l) / $(ls "$QDIR"/*.fasta | wc -l)"
