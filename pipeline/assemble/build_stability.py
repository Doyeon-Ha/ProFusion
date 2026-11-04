#!/usr/bin/env python
"""Assemble the 65 ProteinGym Stability DMS (axis=stability_ddg). **Not used by the paper
numbers** (they are filtered out by axis). They are in the data cache, so this is kept for
reproducibility. Run it after build_features.py.

## Why they were missing

The axis map is {OrganismalFitness, Activity, Binding -> function; Expression -> stability}, so the
66 assays with `coarse_selection_type == "Stability"` fall into no axis and are discarded. The
"stability axis" in the figures is in fact **Expression**, 18 assays.

The reason for leaving them out is **leakage**: 64 of 66 are Tsuboyama, and the ddG channel (SPURS)
was trained on exactly that data. As long as ddG is used, numbers on this axis read the training
data back.
But **with the ddG channel removed**, the rest (GEMME, RSA, BLOSUM, Delta charge, pLM) never saw
Tsuboyama, so evaluating without that channel gives a leak-free Stability number. This script is
the preprocessing for that.

## Why it is a separate script (e_mix3.py is not edited)

`e_mix3.load()` reads a module-level AXIS, and many scripts use that load. Editing the file would
silently give all of them 65 extra DMS. AXIS is therefore swapped at runtime here only, and the
e_mix3.py on disk is untouched.

## It uses a new axis name

It is not merged into the existing "stability" (Expression, the abundance of full-length human
proteins). ProteinGym Stability is **the folding ddG of domains with a median length of 58
residues**, a different measurement on differently sized proteins. It is kept under the axis name
`stability_ddg`, which affects nothing else because the existing scripts filter by axis (this is
purely an addition).

run:    python pipeline/assemble/build_stability.py
output: results/rebuild/proteingym/two_dim/{dms}.npz (added) plus meta.csv (merged, after a backup)
"""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from plmfusion import paths  # noqa: E402

import e_mix3  # noqa: E402

# Runtime swap: keep Stability only. The e_mix3.py on disk is untouched.
e_mix3.AXIS = {"Stability": "stability_ddg"}

import build_features as BF  # noqa: E402  (AXIS is not read at import time, so order does not matter)

OUT = BF.OUT   # the original wrote straight into the shared cache
META = OUT / "meta.csv"


class _StubEmb:
    """A stand-in that lets the build run **without** embeddings.

    build_features uses H in three places only: the bounds check `row_emb < H.shape[0]`, the stored
    field `hwt=H[upos]`, and `L=H.shape[0]` in meta.
    The 66 Stability assays are 0 of 66 in the sequence-only embedding file, so there is no real H.
    Filling them from the multimodal pickle (66 of 66) would give them **a different provenance
    from the existing 129 (sequence-only) and contaminate them silently**.

    So this returns an array of the right length with **zero columns**. The bounds check and L work
    exactly, and hwt becomes (P, 0), so any code that tries to use it **fails loudly** rather than
    passing silently. The fusion path (common.Rec) never reads hwt, so it is unaffected.
    """

    def __init__(self) -> None:
        ref = pd.read_csv(paths.data_root() / "proteingym/DMS_substitutions.csv")
        self._len = dict(zip(ref.DMS_id, ref.seq_len))

    def __contains__(self, dms: str) -> bool:
        return dms in self._len

    def __getitem__(self, dms: str) -> np.ndarray:
        # [L+2, 0]: the same row count as the 0=BOS, 1..L=residues, L+1=EOS convention
        return np.zeros((int(self._len[dms]) + 2, 0), dtype=np.float32)


class _PickleShim:
    """Intercept only the `pickle.load(f)` inside BF and return the stub."""

    @staticmethod
    def load(f):  # noqa: ANN001
        return _StubEmb()


def main() -> None:
    before = pd.read_csv(META) if META.exists() and META.stat().st_size > 1 else pd.DataFrame()
    if len(before):
        shutil.copy(META, META.with_suffix(".csv.bak"))
        print(f"[backup] meta.csv ({len(before)} rows) -> meta.csv.bak")

    BF.pickle = _PickleShim()          # replaces the embedding load only
    BF.main()                          # writes a fresh meta.csv, merged back below

    new = pd.read_csv(META) if META.stat().st_size > 1 else pd.DataFrame()
    print(f"\n[new] built {len(new)} DMS (axis=stability_ddg)")
    if len(before):
        merged = pd.concat([before, new], ignore_index=True).drop_duplicates(
            subset="dms", keep="last")
        merged.to_csv(META, index=False)
        print(f"[merge] meta.csv = {len(before)} + {len(new)} -> {len(merged)} rows")
        print(merged.axis.value_counts().to_string())
    if new.empty:
        raise SystemExit("[stopped] nothing was built. See the [skip] reasons above")


if __name__ == "__main__":
    main()
