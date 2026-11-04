#!/usr/bin/env python
"""SPURS ddG matrices for the external tuning set.

Uses **the same code path** as the ProteinGym channel: `build_batch` and `ddg_matrix` are imported
straight from the ProteinGym scoring script. Extracting it differently would make w_ddg refer to a
different quantity in the tuning set than in evaluation.

The only difference from ProteinGym is the coordinates. There, the AF2 structure is
`target_seq[pdb_range]` and an offset is needed; here the DMS sequence was folded as it is, so
**pdb_range = 1..L** and `row0 = pos - 1`.

Sign convention: **positive = destabilising**. As a zero-shot score it is used as `-ddg`.

env: a separate `spurs` environment (its `esm` package collides with the esm3 environment)
run: <CONDA_ENVS>/spurs/bin/python experiments/mavedb_external/build_ddg.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

R0 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R0 / "experiments/spurs_ddg"))

FOLD = R0 / "data/mavedb_external/fold/out"
OUT = R0 / "data/mavedb_external/ddg"
CKPT = "<EXTERNAL_TOOLS>/SPURS/data/checkpoints/spurs"


def main() -> None:
    import torch
    from run_spurs_proteingym import ALPHABET, build_batch, ddg_matrix   # noqa: E402
    from spurs.inference import get_SPURS                                # noqa: E402

    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((R0 / "data/mavedb_external/seq_manifest.json").read_text())
    seqs = {}
    for tag in man:
        f = R0 / "data/mavedb_external/msa/query" / f"{tag}.fasta"
        seqs[tag] = f.read_text().split("\n", 1)[1].replace("\n", "").strip()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = get_SPURS(CKPT, device=device)
    model.eval()                                   # required, since dropout is 0.5
    print(f"[model] SPURS on {device}", flush=True)

    done = fail = 0
    for tag, info in man.items():
        dst = OUT / f"{tag}.npz"
        if dst.exists():
            done += 1
            continue
        cands = sorted(glob.glob(str(FOLD / f"{tag}_unrelaxed_rank_001_*.pdb")))
        if not cands:
            print(f"[missing] {tag} {info['tname']}: no structure")
            fail += 1
            continue
        seq = seqs[tag]
        try:
            batch = build_batch(cands[0], tag, "A", cfg, device, seq)
            mat = ddg_matrix(model, batch, chunk=512)
        except Exception as e:                      # noqa: BLE001
            print(f"[failed] {tag} {info['tname']}: {type(e).__name__}: {str(e)[:120]}")
            fail += 1
            continue
        assert mat.shape == (len(seq), 20), f"{tag}: {mat.shape} != ({len(seq)},20)"
        # The ddG at a wild-type residue is 0 by definition, which catches a coordinate mismatch here
        wt_idx = [ALPHABET.index(a) for a in seq]
        wt_vals = mat[np.arange(len(seq)), wt_idx]
        assert np.abs(wt_vals).max() < 1e-4, f"{tag}: wild-type ddG is not 0 (suspect the coordinates)"
        np.savez_compressed(dst, ddg=mat, seq=seq, pdb_range=np.array([1, len(seq)]),
                            alphabet=ALPHABET)
        print(f"[ok] {info['tname'][:28]:30s} L={len(seq):5d} "
              f"median ddG {np.median(mat):.3f} range [{mat.min():.2f},{mat.max():.2f}]", flush=True)
        done += 1
    print(f"\ndone {done}, failed {fail} / {len(man)}")


if __name__ == "__main__":
    main()
