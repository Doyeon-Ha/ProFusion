#!/bin/bash
#SBATCH --job-name=detango_full
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
# (The original pinned a specific node, because the model source was only on that node's local
#  disk. On another cluster this line is unnecessary.)
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
# Extended to all of ProteinGym: 186 SPURS ddG matrices -> 173 for DETANGO training (L <= 1022).
# Every stage skips existing output, so resubmitting picks up where it left off.
set -euo pipefail
PYS="<CONDA_ENVS>/spurs/bin/python"
PYD="<EXTERNAL_TOOLS>/detango_venv/bin/python"
PROJ="<REPO>"
RUN=$PROJ/data/proteingym/detango
DET="<EXTERNAL_TOOLS>/DETANGO"
CKPT="<EXTERNAL_TOOLS>/SPURS/data/checkpoints/spurs"
COL=${COL:-spurs}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}

cd "$PROJ"
export PROJECT_ROOT=$PWD
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [1/6] SPURS ddG matrices (186 proteins; the existing 7 are skipped)"
$PYS experiments/spurs_ddg/run_spurs_proteingym.py --stage matrix --ckpt-path "$CKPT"
echo "=== [2/6] SPURS to per-DMS variant CSV"
$PYS experiments/spurs_ddg/run_spurs_proteingym.py --stage map --device cpu
echo "matrix=$(ls data/proteingym/spurs/matrix/*.npz | wc -l)  dms=$(ls data/proteingym/spurs/dms/*.csv | wc -l)"

echo "=== [3/6] DETANGO input skeleton (L <= 1022 only)"
$PYD experiments/detango_spurs/prepare_data.py --max-len 1022
ln -sfn "$DET/detango" "$RUN/detango"; ln -sfn "$DET/config" "$RUN/config"
echo "$(ls -d $RUN/data/*/ | wc -l) proteins"

echo "=== [4/6] ESM-1v masked-marginals (5 models) and embeddings"
if [ -f "$RUN/.esm1v_full.done" ]; then echo "skip"; else
  $PYD experiments/detango_spurs/run_esm1v.py --run-root "$RUN"
  touch "$RUN/.esm1v_full.done"
fi

echo "=== [5/6] SPURS ΔΔG → collection.csv"
$PYD experiments/detango_spurs/merge_stability.py --run-root "$RUN"

echo "=== [6/6] DETANGO training (col=$COL)"
cd "$RUN"
n=0; tot=$(ls -d data/*/ | wc -l)
for d in data/*/; do
  p=$(basename "$d"); n=$((n+1))
  seq=$(sed -n '2p' "data/$p/wt.fasta")
  for s in $SEEDS; do
    [ -f "results/$p/intermediates/detango_${COL}_${s}.csv" ] && continue
    $PYD detango/train_inference.py -p "$p" --sequence_wt "$seq" \
        --stability_col "$COL" --sample_seed "$s" --cuda_device 0 >/dev/null 2>&1 \
      || echo "!! FAILED $p seed=$s"
  done
  $PYD detango/merge_results.py -p "$p" --sequence_wt "$seq" --stability_col "$COL" >/dev/null 2>&1 \
    || echo "!! MERGE FAILED $p"
  echo "[$n/$tot] $p (L=${#seq}) done"
done
echo "=== all done"
