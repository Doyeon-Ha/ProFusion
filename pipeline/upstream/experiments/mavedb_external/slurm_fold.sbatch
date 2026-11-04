#!/usr/bin/env bash
#SBATCH --job-name=mx_fold
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# Structure prediction for the 23 external sequences (ColabFold, using our own hhblits MSA).
# Why they are folded here rather than taken from the AlphaFold database: many of the external
# sequences are constructs (a cohesin fragment, JAG1 exons 1-7, a COMT region, an IgM TM domain),
# whose residue coordinates do not line up with the full-length database structure. Folding the
# sequence the DMS actually used removes the need for any offset mapping.
set -uo pipefail
cd "<REPO>"
export TF_FORCE_UNIFIED_MEMORY=1 XLA_PYTHON_CLIENT_MEM_FRACTION=4.0
CF="<CONDA_ENVS>/cf_clean/bin/colabfold_batch"
IN=data/mavedb_external/fold/in
OUT=data/mavedb_external/fold/out
# Shortest first, so that the rest are secured even if the long TSC2 (1807 aa) dies
for f in $(ls -S -r $IN/*.a3m); do
  tag=$(basename "$f" .a3m)
  ls $OUT/${tag}*_relaxed_*.pdb $OUT/${tag}*_unrelaxed_rank_001*.pdb >/dev/null 2>&1 && { echo "[skip] $tag"; continue; }
  echo "[$(date +%H:%M:%S)] fold $tag"
  timeout 10800 "$CF" --num-models 1 --num-recycle 3 --model-type alphafold2_ptm \
      --random-seed 0 "$f" "$OUT" || echo "[FAIL] $tag rc=$?"
done
echo "[$(date +%H:%M:%S)] done — pdb $(ls $OUT/*_unrelaxed_rank_001*.pdb 2>/dev/null | wc -l)"
