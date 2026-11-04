#!/usr/bin/env python
"""SPURS ddG matrices to the skeleton of a DETANGO input directory.

Per protein it creates (the DETANGO data/{protein}/ convention):
  wt.fasta                          the header is the protein name (esm_embedding saves as {label}.pt)
  intermediates/collection_raw.csv  the L*19 single-mutant list, 0-based `{wt}{i}{mt}`

The sequence is not the full UniProt one but the sequence of the SPURS matrix (that is,
ProteinGym target_seq[pdb_range]). That puts the ddG and the DMS on one coordinate axis and avoids
needing an AF2 PDB or UniProt accession mapping.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def single_mutants(seq: str) -> list[str]:
    return [f"{wt}{i}{mt}" for i, wt in enumerate(seq) for mt in ALPHABET if mt != wt]


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-dir", default=str(root / "data/proteingym/spurs/matrix"))
    p.add_argument("--run-root", default=str(root / "data/proteingym/detango"))
    p.add_argument("--proteins", nargs="*", default=None)
    p.add_argument("--max-len", type=int, default=None,
                   help="exclude proteins over the ESM-1v positional-embedding limit (1022)")
    a = p.parse_args(argv)

    mdir, run = Path(a.matrix_dir), Path(a.run_root)
    names = a.proteins or sorted(f.stem for f in mdir.glob("*.npz"))

    rows = []
    for name in names:
        z = np.load(mdir / f"{name}.npz", allow_pickle=True)
        seq = str(z["seq"])
        if a.max_len and len(seq) > a.max_len:
            print(f"[skip] {name}: L={len(seq)} > {a.max_len} (beyond ESM-1v)")
            continue
        assert z["ddg"].shape == (len(seq), 20), (name, z["ddg"].shape, len(seq))

        d = run / "data" / name
        (d / "intermediates").mkdir(parents=True, exist_ok=True)
        (d / "wt.fasta").write_text(f">{name}\n{seq}\n")
        muts = single_mutants(seq)
        # index=True: the downstream script reads it with read_csv(index_col=0)
        pd.DataFrame({"mutant": muts}).to_csv(d / "intermediates/collection_raw.csv")
        rows.append({"protein": name, "seq_len": len(seq), "n_mutants": len(muts),
                     "pdb_start": int(z["pdb_range"][0])})
        print(f"[prep] {name}: L={len(seq)} muts={len(muts)}")

    df = pd.DataFrame(rows)
    df.to_csv(run / "proteins.csv", index=False)
    # A combined fasta, so the embeddings are extracted once
    kept = df.protein.tolist()
    fa = "".join((run / "data" / n / "wt.fasta").read_text() for n in kept)
    (run / "all_wt.fasta").write_text(fa)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
