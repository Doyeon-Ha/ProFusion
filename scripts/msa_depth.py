#!/usr/bin/env python
"""HHblits alignment depth for the 23 proteins of the external fitting set.

This produces the table behind the Methods sentence about HHblits alignment sizes. That sentence
gave only a range and had no code behind it; the point here is to give a reviewer a path to check
the numbers.

What is measured
  nseq  number of sequences in the alignment (including the query). The same value hhblits
        reports as seqs= in its log.
  neff  effective number of sequences for the same alignment. Redundancy is down-weighted, so it
        is smaller than nseq.
  Both are unitless, and larger means a deeper alignment.

input   <data root>/mavedb_external/neff.json  (recorded when hhblits was run; the original log
        is alongside it as hhblits_run.log)
output  results/msa_depth.csv
run     python scripts/msa_depth.py
"""
from __future__ import annotations
import json
import pandas as pd
from plmfusion.paths import RESULTS, ext_table


def table() -> pd.DataFrame:
    meta = json.loads(ext_table("neff.json").read_text())
    T = pd.DataFrame([dict(hash=k, **v) for k, v in meta.items()])
    return T[["tname", "L", "nseq", "neff", "hash"]].sort_values("nseq").reset_index(drop=True)


def main():
    T = table()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "msa_depth.csv"
    T.to_csv(out, index=False)
    print(f"{len(T)} proteins")
    print(f"  nseq  {T.nseq.min():.0f} to {T.nseq.max():.0f}  (median {T.nseq.median():.0f})")
    print(f"  neff  {T.neff.min():.0f} to {T.neff.max():.0f}  (median {T.neff.median():.0f})")
    print(T.to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
