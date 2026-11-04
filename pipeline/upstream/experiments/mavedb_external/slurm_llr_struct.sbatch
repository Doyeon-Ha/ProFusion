#!/usr/bin/env bash
#SBATCH --job-name=mx_llrst
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
# (The original pinned a specific node, because the model source was only on that node's local
#  disk. On another cluster this line is unnecessary.)
#SBATCH --output=%x_%j.out
# LLR for the backbones that need structure (ProSST / ESM3 multimodal).
# The original pinned the node: the ProSST source was only on that node's local disk.
set -uo pipefail
cd "<REPO>"
ENVD="<CONDA_ENVS>/merged_prosst_esm3"
# The conda scipy requires CXXABI_1.3.15 while the node system libstdc++ only goes to 1.3.13.
# The environment's own lib has to come first for `import scipy.spatial` to survive.
export LD_LIBRARY_PATH=$ENVD/lib:${LD_LIBRARY_PATH:-}
PY=$ENVD/bin/python
PYTHONPATH=src:external/esm3 "$PY" experiments/mavedb_external/build_llr_struct.py --backbone "$1"
