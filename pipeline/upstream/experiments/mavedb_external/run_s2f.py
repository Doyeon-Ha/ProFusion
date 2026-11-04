#!/usr/bin/env python
"""S2F runner (generalised): the input directory, the device and the output location are arguments.

## Why

`run_s2f_cpu.py` is specific to the MaveDB external set and hard-codes three things.
  (1) the input path is fixed at `EXT/s3f_input`
  (2) the shard cost is computed from **the urn of the external npz** (ProteinGym uses DMS_id, which
      does not match)
  (3) the device is fixed to cpu (chosen while the GPU partition was draining; GPUs are available
      now)
Running the 129 ProteinGym DMS needs all three relaxed. The original produced the existing results,
so it is left untouched and this sits beside it.

## Input layout (the evaluation-script convention)

    <input>/DMS_substitutions.csv   required columns DMS_id, UniProt_ID, target_seq, seq_len,
                                    DMS_filename, pdb_file, pdb_range
    <input>/<DMS_filename>          the per-DMS variant CSVs, flat in the same directory
    <input>/structures/<pdb_file>   the structure PDB (a symbolic link is fine)

⚠ A wrong `pdb_range` dies immediately on the assert in assign_structure. It never fails silently.

## What is measured

The output is one `{DMS_filename}` CSV per DMS, whose `model_score` column is the S2F score.
Larger means a better fit, that is, more tolerated. Unitless (it is z-normalised within an assay
later). The evaluation summary `results.csv` carries the per-DMS spearmanr alongside.

⚠ Each shard needs a different `SLURM_JOB_ID`. util.create_working_directory creates and removes
   "$SLURM_JOB_ID_working_dir.tmp" in a shared cwd, so identical ids race and die.

run
  python run_s2f.py --input data/proteingym/s3f_input --outroot data/proteingym/s3f_out/s2f \
                    --device cuda --shards 1 --wait
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

R0 = Path("<REPO>")
# ⚠ Node-local disks are not visible from the compute nodes. Only the code and checkpoint S2F needs
# were copied to shared storage, and that copy is the default.
# (The 2.6 GB surface dataset is S3F-only and was not copied.)
S3F_LOCAL = Path("<NODE_LOCAL_S3F>")
S3F = Path("<REPO>/external/S3F")
if not S3F.exists():
    S3F = S3F_LOCAL
PYBIN = "<CONDA_ENVS>/s3f/bin/python"
sys.path.insert(0, str(R0 / "experiments/mavedb_external"))
from run_s2f_cpu import TMPL  # the YAML template is **the same one** as the original, to avoid config drift


def costs(input_dir: Path, ids: set[str]) -> dict[str, int]:
    """The cost of a DMS is the number of distinct mutated positions (proportional to the number of
    forwards). Taken from the npz when present, otherwise counted from the variant CSV."""
    out = {}
    for cache in (R0 / "data/proteingym/two_dim", R0 / "data/mavedb_external/two_dim"):
        for f in sorted(cache.glob("*.npz")):
            k = f.stem
            if k in ids and k not in out:
                out[k] = int(np.load(f, allow_pickle=True)["upos"].shape[0])
    ref = pd.read_csv(input_dir / "DMS_substitutions.csv").set_index("DMS_id")
    for k in ids - set(out):
        d = pd.read_csv(input_dir / ref.loc[k, "DMS_filename"], usecols=["mutant"])
        out[k] = d.mutant.astype(str).str[1:-1].nunique()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outroot", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--ckpt", default=str(S3F / "checkpoints/s2f.pth"))
    ap.add_argument("--tag", default="s2f")
    ap.add_argument("--only", default=None, help="a comma-separated subset of DMS_id (for gating)")
    ap.add_argument("--wait", action="store_true", help="block until the shards finish (for Slurm)")
    A = ap.parse_args()

    inp = Path(A.input).resolve()
    outroot = Path(A.outroot).resolve()
    assert Path(A.ckpt).exists(), f"no checkpoint: {A.ckpt}"
    assert (inp / "DMS_substitutions.csv").exists(), f"no summary: {inp}"
    assert (inp / "structures").exists(), f"no structures: {inp}"

    ids = set(pd.read_csv(inp / "DMS_substitutions.csv").DMS_id)
    if A.only:
        want = {x.strip() for x in A.only.split(",")}
        miss = want - ids
        assert not miss, f"DMS in --only that do not exist: {miss}"
        ids = want
    cost = costs(inp, ids)

    buckets: list[list[str]] = [[] for _ in range(A.shards)]
    loads = [0] * A.shards
    for c, k in sorted(((v, k) for k, v in cost.items()), reverse=True):   # LPT
        i = int(np.argmin(loads)); buckets[i].append(k); loads[i] += c
    print(f"[shard] {len(ids)} DMS, {sum(loads):,} positions in total -> " +
          " ".join(f"{l:,}" for l in loads))

    outroot.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, bk in enumerate(buckets):
        if not bk:
            continue
        cfg = S3F / f"config/evaluate/_run_{A.tag}_{i}.yaml"
        y = TMPL.format(out=outroot / f"sh{i}", inp=inp,
                        ids="\n".join(f"  - {x}" for x in bk),
                        bs=A.batch_size, ckpt=A.ckpt)
        y = y.replace("device: cpu", f"device: {A.device}")
        cfg.write_text(y)
        log = open(outroot / f"sh{i}.log", "w")
        env = dict(os.environ, OMP_NUM_THREADS=str(A.threads),
                   MKL_NUM_THREADS=str(A.threads), SLURM_JOB_ID=f"{A.tag}{i}")
        procs.append(subprocess.Popen([PYBIN, "script/evaluate_cpu.py", "-c", str(cfg)],
                                      cwd=S3F, stdout=log, stderr=subprocess.STDOUT, env=env))
        print(f"  shard {i}: {len(bk):3d} DMS, {loads[i]:,} positions, PID {procs[-1].pid}")
    print(f"[launched] {len(procs)} shards -> {outroot}")
    if A.wait:
        rc = [p.wait() for p in procs]
        print(f"[done] exit codes {rc}")
        sys.exit(max(rc) if rc else 0)


if __name__ == "__main__":
    main()
