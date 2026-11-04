#!/usr/bin/env python
"""Run S2F over the 49 external DMS, in parallel on CPU.

This is the version that ran on CPU because no GPU node was available. The idle nodes that could be
used did not have the filesystem holding the code, environment and checkpoint mounted.
Thanks to per-position dedupe the number of forwards is not 248,272 variants but **17,382 distinct
positions**, which CPU can handle. The shards are balanced by position count (INSR has 7 DMS of 928
positions each, so splitting by DMS count piles them into one shard).

run: PYTHONPATH=src python experiments/mavedb_external/run_s2f_cpu.py --shards 8 --threads 28
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

R0 = Path(__file__).resolve().parents[2]
EXT = R0 / "data/mavedb_external"
S3F = Path("<NODE_LOCAL_S3F>")
PYBIN = "<CONDA_ENVS>/s3f/bin/python"
TMPL = """output_dir: {out}
device: cpu
summary:
  class: ProteinGym
  path: {inp}
  csv_file: DMS_substitutions.csv
structure_path: {inp}/structures
id_list:
{ids}
dataset:
  class: DMSAssay
  atom_feature: null
  bond_feature: null
  transform:
    class: Compose
    transforms:
      - class: MutantStructure
        key: graph
        template_key: wild_type
      - class: ProteinView
        view: residue
task:
  class: ResidueTypePrediction
  mask_rate: 0.15
  dropout: 0.5
  plddt_threshold: 70
  model:
    class: FusionNetwork
    sequence_model:
      class: MyESM
      path: ~/scratch/protein-model-weights/esm-model-weights/
      model: ESM-2-650M
    structure_model:
      class: GVPGNN
      node_in_dim: [1280, 0]
      node_h_dim: [256, 16]
      edge_in_dim: [16, 1]
      edge_h_dim: [64, 1]
      num_layers: 5
      vector_gate: True
      readout: mean
      drop_rate: 0.1
  graph_construction_model:
    class: GraphConstruction
    node_layers:
      - class: AlphaCarbonNode
    edge_layers:
      - class: SpatialEdge
        radius: 10.0
        min_distance: 0
    edge_feature: null
gpus: [0]
batch_size: {bs}
model_checkpoint: {ckpt}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--threads", type=int, default=28)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--ckpt", default=str(S3F / "checkpoints/s2f.pth"))
    ap.add_argument("--tag", default="s2f")
    A = ap.parse_args()
    assert Path(A.ckpt).exists(), f"no checkpoint: {A.ckpt}"

    # The shards are balanced by **position count** (the number of forwards is proportional to it)
    cost = {}
    for f in sorted((EXT / "two_dim").glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        cost[str(d["urn"]).replace(":", "-")] = int(d["upos"].shape[0])
    ref = set(pd.read_csv(EXT / "s3f_input/DMS_substitutions.csv").DMS_id)
    items = sorted(((v, k) for k, v in cost.items() if k in ref), reverse=True)
    buckets: list[list[str]] = [[] for _ in range(A.shards)]
    loads = [0] * A.shards
    for c, k in items:                       # LPT: longest first, into the emptiest bin
        i = int(np.argmin(loads)); buckets[i].append(k); loads[i] += c
    print(f"[shard] {sum(loads):,} positions in total -> " + " ".join(f"{l:,}" for l in loads))

    outroot = EXT / f"s3f_out/{A.tag}"
    outroot.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, ids in enumerate(buckets):
        if not ids:
            continue
        cfg = S3F / f"config/evaluate/_mx_{A.tag}_{i}.yaml"
        cfg.write_text(TMPL.format(out=outroot / f"sh{i}", inp=EXT / "s3f_input",
                                   ids="\n".join(f"  - {x}" for x in ids),
                                   bs=A.batch_size, ckpt=A.ckpt))
        log = open(outroot / f"sh{i}.log", "w")
        # The shards share a cwd, and util.create_working_directory creates and removes
        # "$SLURM_JOB_ID_working_dir.tmp". If every id is "null" they race and die with
        # FileNotFoundError (3 of 8 actually died). Each shard gets its own id.
        env = dict(os.environ, OMP_NUM_THREADS=str(A.threads),
                   MKL_NUM_THREADS=str(A.threads), SLURM_JOB_ID=f"{A.tag}{i}")
        procs.append(subprocess.Popen([PYBIN, "script/evaluate_cpu.py", "-c", str(cfg)],
                                      cwd=S3F, stdout=log, stderr=subprocess.STDOUT, env=env))
        print(f"  shard {i}: {len(ids):2d} DMS, {loads[i]:,} positions, PID {procs[-1].pid}")
    print(f"[launched] {len(procs)} shards -> {outroot}")


if __name__ == "__main__":
    main()
