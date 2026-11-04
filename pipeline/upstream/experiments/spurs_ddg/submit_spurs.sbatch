#!/bin/bash
#SBATCH --job-name=spurs_pg
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
# (The original pinned a specific node, because the model source was only on that node's local
#  disk. On another cluster this line is unnecessary.)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# 186 ProteinGym proteins -> SPURS ddG matrices -> per-DMS variant CSV.
# usage: sbatch experiments/spurs_ddg/submit_spurs.sbatch
set -euo pipefail
PY="<CONDA_ENVS>/spurs/bin/python"
CKPT="<EXTERNAL_TOOLS>/SPURS/data/checkpoints/spurs"   # avoids re-downloading from the hub
cd "<REPO>"
export PROJECT_ROOT=$PWD
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$PY experiments/spurs_ddg/run_spurs_proteingym.py --stage matrix --ckpt-path "$CKPT"
$PY experiments/spurs_ddg/run_spurs_proteingym.py --stage map --device cpu
