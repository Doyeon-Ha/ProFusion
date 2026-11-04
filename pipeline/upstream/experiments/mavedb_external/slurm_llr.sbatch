#!/usr/bin/env bash
#SBATCH --job-name=mx_llr
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
set -euo pipefail
cd "<REPO>"
PY="<CONDA_ENVS>/esm3_model_gs/bin/python"
PYTHONPATH=src "$PY" experiments/mavedb_external/build_llr.py --backbone "${1:-esm2}"
