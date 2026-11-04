#!/usr/bin/env python
"""Structure to per-position relative solvent accessibility (RSA), via DSSP.

Extracted **the same way as the evaluation set**: `Bio.PDB.DSSP` with `mkdssp`, tuple index 3.
Extracting it differently would make the a_RSA weight refer to a different quantity than it was
fitted on.

PDB residue numbers are taken as 1-based sequence positions. AlphaFold DB and ColabFold output
follow that convention. Experimental structures from the PDB often number residues differently
from the sequence, so check with --check-seq.

run: python inference/steps/rsa.py --pdb p.pdb --out rsa.json [--check-seq p.fasta]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

AA3to1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
          "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
          "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def _dssp_exe() -> str:
    for c in ("mkdssp", "dssp"):
        p = shutil.which(c)
        if p:
            return p
    raise FileNotFoundError("mkdssp not found. Install it (for example `conda install -c salilab dssp`, "
                            "or your distribution package) and put it on PATH.")


def compute(pdb: str | Path, check_seq: str | None = None) -> dict[int, float]:
    """{1-based position: RSA}. Positions with a missing RSA are left out."""
    from Bio.PDB import DSSP, PDBParser
    pdb = str(pdb)
    model = next(PDBParser(QUIET=True).get_structure("x", pdb).get_models())
    d = DSSP(model, pdb, dssp=_dssp_exe())
    rsa, aa = {}, {}
    for key in d.keys():
        pos = key[1][1]                       # residue number = 1-based sequence position
        v = d[key][3]
        aa[pos] = d[key][1]
        if isinstance(v, (int, float)) and v == v:
            rsa[pos] = float(v)
    if check_seq:
        bad = [p for p, a in aa.items()
               if 1 <= p <= len(check_seq) and a != "X" and check_seq[p - 1] != a]
        if bad:
            raise ValueError(
                f"The structure's residue numbering does not match the sequence ({len(bad)} positions, "
                f"for example {bad[:5]}). For an experimental structure, renumber against the sequence and rerun.")
        cov = len(rsa) / len(check_seq)
        if cov < 0.9:
            print(f"[warning] RSA coverage {cov:.2f}: the structure has missing regions")
    return rsa


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--check-seq", default=None, help="FASTA. If given, residue numbering is checked against it")
    ap.add_argument("--out", required=True)
    A = ap.parse_args()
    seq = None
    if A.check_seq:
        seq = "".join(l.strip() for l in Path(A.check_seq).read_text().splitlines()
                      if not l.startswith(">"))
    rsa = compute(A.pdb, seq)
    Path(A.out).write_text(json.dumps({str(k): v for k, v in rsa.items()}))
    vals = list(rsa.values())
    print(f"[saved] {A.out}  {len(rsa)} positions, median RSA {np.median(vals):.3f}")


if __name__ == "__main__":
    main()
