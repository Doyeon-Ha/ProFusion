#!/bin/bash
# Run GEMME for a single DMS via rootless docker. arg: dms_id
dms="$1"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock
ROOT="<REPO>"
wdir="$ROOT/experiments/runs/gemme_all/work/$dms"
cd "$wdir" || { echo "NODIR $dms"; exit 1; }

ali=$(ls ali*.fasta 2>/dev/null | head -1)
[ -z "$ali" ] && { echo "NOALI $dms"; exit 1; }

# idempotent: skip if any combined prediction already exists (GEMME truncates name at first '_')
if ls *_normPred_evolCombi.txt >/dev/null 2>&1; then echo "SKIP $dms"; exit 0; fi

timeout 5400 docker run --rm -v "$wdir":/project -w /project \
  --entrypoint /bin/bash elodielaine/gemme:gemme \
  -c "python2.7 \$GEMME_PATH/gemme.py $ali -r input -f $ali -m mutations.txt -N 20000" \
  > run.log 2>&1
rc=$?

if ls *_normPred_evolCombi.txt >/dev/null 2>&1; then
  echo "OK $dms"
else
  echo "FAIL $dms rc=$rc"
fi
