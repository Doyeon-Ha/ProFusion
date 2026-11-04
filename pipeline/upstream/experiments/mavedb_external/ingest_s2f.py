#!/usr/bin/env python
"""Convert S2F output (the per-variant csv from its evaluation script) into the same .pt format as
the other backbones.

⚠️ The S3F `load_dataset` sorts variants **by position** on output, so its row order differs from
   ours. `build_channels.py` assumes every backbone is in **the same order** as esm2 and indexes by
   position, so the rows are reordered to match esm2 here and pinned with an assert.

run: PYTHONPATH=src python experiments/mavedb_external/ingest_s2f.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

R0 = Path(__file__).resolve().parents[2]
EXT = R0 / "data/mavedb_external"
SRC = EXT / "s3f_out/s2f"


def main() -> None:
    out = EXT / "llr/s2f"; out.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("sh*/*/*/*/*/*/urn-*.csv"))
    print(f"[src] {len(files)} csv")
    ok = skip = 0
    for f in files:
        stem = f.stem                                    # urn-mavedb-00001239-a-7
        ref = EXT / "llr/esm2" / f"{stem}.pt"
        if not ref.exists():
            print(f"  no esm2: {stem}"); skip += 1; continue
        R = torch.load(ref, map_location="cpu", weights_only=False)
        order = list(R["mutants"])
        T = pd.read_csv(f)
        s = pd.Series(T.model_score.to_numpy(), index=T.mutant.astype(str))
        s = s[~s.index.duplicated()]
        miss = [m for m in order if m not in s.index]
        if miss:
            print(f"  {stem}: {len(miss)} variants missing (for example {miss[:3]})"); skip += 1; continue
        v = torch.tensor(s.reindex(order).to_numpy(), dtype=torch.float32)
        assert len(v) == len(order) and torch.isfinite(v).all(), stem
        torch.save(dict(mutants=order, scores=v, dms_scores=R["dms_scores"],
                        backbone="s2f", model_name="S2F (NeurIPS'24, masked-marginal)",
                        n_dropped=0, structure_only=False, reordered_to="esm2"), out / f"{stem}.pt")
        ok += 1
    print(f"[done] {ok} written, {skip} skipped -> {out}")


if __name__ == "__main__":
    main()
