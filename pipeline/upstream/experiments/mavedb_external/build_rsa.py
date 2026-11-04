#!/usr/bin/env python
"""RSA for the external tuning set, via DSSP.

Extracted **the same way** as the ProteinGym channel: `Bio.PDB.DSSP` with `mkdssp`, tuple index 3.
Extracting it differently would make w_rsa refer to a different quantity in the tuning set than in
evaluation.

The structures are ColabFold output
(`data/mavedb_external/fold/out/{tag}_unrelaxed_rank_001_*.pdb`). The DMS sequence was folded as it
is, so **the residue number is the sequence position** and no offset mapping is needed.

output: data/mavedb_external/rsa/{tag}.json  {pos (1-based): rsa}
run:    PYTHONPATH=src python experiments/mavedb_external/build_rsa.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from Bio.PDB import DSSP, PDBParser

R0 = Path(__file__).resolve().parents[2]
FOLD = R0 / "data/mavedb_external/fold/out"
OUT = R0 / "data/mavedb_external/rsa"
DSSP_EXE = "<CONDA_BIN>/mkdssp"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    # The Domainome (277 tags) lives in frozen_domainome.csv rather than in seq_manifest.
    # Passing --frozen builds the manifest from that frozen set's tag, tname and seq_len instead.
    ap.add_argument("--frozen", default=None,
                    help="for example frozen_domainome.csv (a file name relative to EXT, or a path)")
    A = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    EXT = R0 / "data/mavedb_external"
    if A.frozen:
        import pandas as pd
        fz = Path(A.frozen)
        F = pd.read_csv(fz if fz.exists() else EXT / fz.name).drop_duplicates("tag")
        man = {r.tag: dict(tname=r.tname, L=int(r.seq_len)) for r in F.itertuples()}
        print(f"[frozen] {Path(A.frozen).name} → tag {len(man)}")
    else:
        man = json.loads((EXT / "seq_manifest.json").read_text())
    parser = PDBParser(QUIET=True)
    done = fail = 0
    for tag, info in man.items():
        p = OUT / f"{tag}.json"
        if p.exists():
            done += 1
            continue
        cands = sorted(glob.glob(str(FOLD / f"{tag}_unrelaxed_rank_001_*.pdb")))
        if not cands:
            print(f"[missing] {tag} {info['tname']}")
            fail += 1
            continue
        pdb = cands[0]
        try:
            model = next(parser.get_structure(tag, pdb).get_models())
            d = DSSP(model, pdb, dssp=DSSP_EXE)
            rsa = {}
            for key in d.keys():
                rn = key[1][1]                      # residue number = 1-based sequence position
                v = d[key][3]
                rsa[rn] = float(v) if isinstance(v, (int, float)) and v == v else None
        except Exception as e:                      # noqa: BLE001
            print(f"[DSSP failed] {tag} {info['tname']}: {type(e).__name__}: {e}")
            fail += 1
            continue
        cov = sum(v is not None for v in rsa.values()) / max(1, info["L"])
        if cov < 0.9:
            print(f"[warning] {tag} {info['tname']} RSA coverage {cov:.2f} (L={info['L']})")
        p.write_text(json.dumps({str(k): v for k, v in rsa.items()}))
        vals = [v for v in rsa.values() if v is not None]
        print(f"[ok] {info['tname'][:28]:30s} L={info['L']:5d} cov={cov:.2f} "
              f"median RSA {np.median(vals):.3f}")
        done += 1
    print(f"\ndone {done}, failed {fail} / {len(man)}")


if __name__ == "__main__":
    main()
