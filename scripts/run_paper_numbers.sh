#!/bin/bash
# Paper numbers: run the 24 cells (12 backbones x 2 axes) in parallel, then merge.
#
# The cells are completely independent (both the external fit and the PG evaluation see only that
# cell), and coordinate descent is deterministic, so this gives **the same result** as the serial
# `python scripts/paper_numbers.py`. Use the serial version if the wall time does not matter.
#
# Usage
#   bash scripts/run_paper_numbers.sh            # default: 24 cells at once, 4 threads each
#   JOBS=8 THREADS=6 bash scripts/run_paper_numbers.sh
#
# Environment
#   PLMFUSION_DATA  the data root (otherwise the fusion/data symbolic link)
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
CELLS="${CELLS:-$HERE/results/_cells}"
LOGS="${LOGS:-$HERE/results/logs/cells}"
JOBS="${JOBS:-24}"
THREADS="${THREADS:-4}"

export PYTHONPATH="$HERE/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$THREADS"

BBS=$(python3 -c "import sys; sys.path.insert(0,'$HERE/scripts'); import paper_numbers as P; print(' '.join(P.BBS))")
AXES=$(python3 -c "import sys; sys.path.insert(0,'$HERE/scripts'); import paper_numbers as P; print(' '.join(P.AXES))")

mkdir -p "$CELLS" "$LOGS"
echo "$(echo $BBS | wc -w) backbones x $(echo $AXES | wc -w) axes, $JOBS at a time, $THREADS threads each"

pids=()
for axis in $AXES; do
  for bb in $BBS; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
    out="$CELLS/${axis}__${bb}"
    mkdir -p "$out"
    python3 "$HERE/scripts/paper_numbers.py" --bb "$bb" --axis "$axis" --out "$out" \
      > "$LOGS/${axis}__${bb}.log" 2>&1 &
    pids+=($!)
  done
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
if [ "$fail" -gt 0 ]; then
  echo "ERROR: $fail cells failed. See $LOGS" >&2
  exit 1
fi

echo "all cells done, merging"
python3 "$HERE/scripts/paper_numbers.py" --merge "$CELLS" --out "$HERE/results"
