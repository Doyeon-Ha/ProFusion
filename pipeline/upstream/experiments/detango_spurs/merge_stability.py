#!/usr/bin/env python
"""Attach the SPURS ddG as the DETANGO stability column and write collection.csv.

- `spurs`   = -ddg  (larger means more stable, matching the DETANGO `minus_ddg`/`abundance` convention)
- `spurs_z` = spurs z-scored within the protein (to check the scale balance of the three loss terms)
- No `foldx_like` placeholder is written: FoldX is never run.

collection.csv is saved **with its index**, because detango/data.py reads it with
`read_csv(..., index_col=0)`. Saving without an index makes `mutant` be taken as the index and
raises KeyError (index=False in the original initialisation script was a bug).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
A2I = {a: i for i, a in enumerate(ALPHABET)}


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-dir", default=str(root / "data/proteingym/spurs/matrix"))
    p.add_argument("--run-root", default=str(root / "data/proteingym/detango"))
    a = p.parse_args(argv)

    run, mdir = Path(a.run_root), Path(a.matrix_dir)
    names = [d.name for d in sorted((run / "data").iterdir()) if d.is_dir()]
    rows = []
    for n in names:
        z = np.load(mdir / f"{n}.npz", allow_pickle=True)
        ddg, seq = z["ddg"], str(z["seq"])
        df = pd.read_csv(run / "data" / n / "intermediates/collection_esm.csv")

        raw = []
        for m in df["mutant"]:
            wt, idx, mt = m[0], int(m[1:-1]), m[-1]
            assert seq[idx] == wt, (n, m)
            raw.append(float(ddg[idx, A2I[mt]]))
        raw = np.asarray(raw)
        df["spurs_ddg_raw"] = raw
        df["spurs"] = -raw
        df["spurs_z"] = (df["spurs"] - df["spurs"].mean()) / df["spurs"].std(ddof=0)

        df.to_csv(run / "data" / n / "collection.csv")  # index=True, deliberately
        rows.append({"protein": n, "n": len(df), "nan": int(np.isnan(raw).sum()),
                     "ddg_mean": raw.mean(), "ddg_std": raw.std(),
                     "ddg_min": raw.min(), "ddg_max": raw.max(),
                     "esm1v_std": df.esm1v.std()})
        print(f"[merge] {n}: n={len(df)} ddg {raw.min():+.2f}..{raw.max():+.2f} "
              f"(std {raw.std():.2f})  esm1v std {df.esm1v.std():.2f}")
    pd.DataFrame(rows).to_csv(run / "stability_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
