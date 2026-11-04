#!/usr/bin/env bash
#SBATCH --job-name=mx_ladder
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
# The ESM2 capacity ladder (sequence-only), run sequentially in one job to reduce queue occupancy.
set -uo pipefail
cd "<REPO>"
PY="<CONDA_ENVS>/esm3_model_gs/bin/python"
for BB in esm2_8M esm2_35M esm2_150M esm2_3B; do
  echo "=========== $BB ==========="
  PYTHONPATH=src "$PY" experiments/mavedb_external/build_llr.py --backbone "$BB" || echo "FAILED $BB"
done
