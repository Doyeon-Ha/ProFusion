#!/usr/bin/env bash
# GEMME for the 23 external tuning-set sequences.
# The same image and arguments as the other GEMME runs in this project.
# a3m -> match-state a2m -> the gemme docker image (once per sequence, over the union of the
# variants of every DMS that uses it).
#
# Usage: experiments/mavedb_external/run_gemme.sh [N_PARALLEL]
set -uo pipefail
REPO="<REPO>"
W=$REPO/data/mavedb_external/gemme/work
NJ=${1:-6}
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock

run_one() {
  # Note: writing `local a="$1" b="$W/$a"` on one line does not see $a yet (a bash local pitfall)
  local tag="$1"
  local wdir="$W/$tag"
  cd "$wdir" || { echo "NODIR $tag"; return 1; }
  local ali="ali${tag}.fasta"            # built directly rather than globbed (the same rule as the build step)
  [ -s "$ali" ] || { echo "NOALI $tag"; return 1; }
  ls ./*_normPred_evolCombi.txt >/dev/null 2>&1 && { echo "SKIP $tag"; return 0; }
  timeout 10800 docker run --rm -v "$wdir":/project -w /project \
    --entrypoint /bin/bash elodielaine/gemme:gemme \
    -c "python2.7 \$GEMME_PATH/gemme.py $ali -r input -f $ali -m mutations.txt -N 20000" \
    > run.log 2>&1
  local rc=$?
  if ls ./*_normPred_evolCombi.txt >/dev/null 2>&1; then echo "OK $tag"; else echo "FAIL $tag rc=$rc"; fi
}
export -f run_one; export W

mapfile -t TAGS < <(for d in "$W"/*/; do t=$(basename "$d"); [ -s "$d/mutations.txt" ] && echo "$t"; done)
echo "[$(date +%H:%M:%S)] GEMME over ${#TAGS[@]} sequences ($NJ at a time)"
printf '%s\n' "${TAGS[@]}" | xargs -P "$NJ" -I{} bash -c 'run_one "$@"' _ {}
echo "[$(date +%H:%M:%S)] done: $(ls "$W"/*/*_normPred_evolCombi.txt 2>/dev/null | wc -l)/${#TAGS[@]}"
