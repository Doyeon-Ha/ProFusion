#!/usr/bin/env python
"""Backbone extension: fit the 7 slot weights on the external MaveDB set and evaluate on
ProteinGym.

## Protocol (the same as the paper)

  fitting     MaveDB external set, separately per axis (function/expression). Coordinate descent,
              3 passes, starting from zero. The grid is symmetric about 0, so every slot can go
              negative.
  evaluation  ProteinGym. Labels are used for evaluation only. There is no CV loop on ProteinGym.
  ⚠ Axis names: both the external and the ProteinGym npz store the expression axis under its old
    name "stability". Only the display name is expression.

## What is measured

  w          the 7 slots (a_GEMME, b_GEMME, a_ddG, b_ddG, a_RSA, b_BLOSUM, b_dchg).
             a_* is the position-mean component, b_* the within-position residual. Channels are
             z-normalised within an assay, so the weights are unitless.
  total      sum of w. The "budget" that backbone spends on auxiliary channels.
  share      w / sum(w). How the budget is allocated.
  tune_obj   mean Spearman over the external DMS. The coordinate-descent objective itself.
  pg_base    mean over ProteinGym DMS of the within-DMS Spearman, pLM LLR alone (w = 0).
  pg_fit7    the same metric for the 7-slot fusion. Both unitless, [-1, 1], higher is better.
  d_rho      pg_fit7 minus pg_base, **paired on the same DMS**. p is a paired two-sided Wilcoxon.
  n_neg      how many slots came out negative. This is where the symmetric grid matters.

## Backbones

Only backbones with an llr in **both** the ProteinGym and the external npz are run. The names on
the two sides are matched through EXTBB (⚠ ProteinGym 'ESM3' corresponds to external 'ESM3mm';
they are not interchangeable).

output  extend_backbones.csv (summary per backbone and axis) and extend_backbones_perdms.csv
run     OMP_NUM_THREADS=6 python scripts/extend_backbones.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))   # paper_numbers lives in this folder
from plmfusion import common as C, grids as P
from plmfusion.paths import RESULTS, ext_cache
import paper_numbers as PN

SL = PN.SL
COMP7 = [0, 1, 2, 3, 4, 5, 6]
GRID = "uni3"          # the confirmed protocol: all slots -3.0 to 3.0, 61 points, NPASS=3, zero start
AXES = ["function", "stability"]          # the names inside the npz; displayed as function / expression
SHOW = {"function": "function", "stability": "expression"}

# ProteinGym npz key <-> external npz key. The paper roster is 12, so paper_numbers.EXTBB is
# used directly and the mapping is defined in exactly one place.
EXTBB = PN.EXTBB

def have(keyset, k):
    return f"llr_{C.llr_key(k)}" in keyset


def key_sets():
    pg = set(np.load(sorted(C.CACHE.glob("*.npz"))[0], allow_pickle=True).files)
    ex = set(np.load(sorted(ext_cache().glob("*.npz"))[0], allow_pickle=True).files)
    return pg, ex


def main():
    pgk, exk = key_sets()
    todo = [b for b in EXTBB if have(pgk, b) and have(exk, EXTBB[b])]
    skip = [b for b in EXTBB if b not in todo]
    print(f"fitting {len(todo)}: {todo}")
    if skip:
        print(f"skipping {len(skip)}: " + ", ".join(
            f"{b}(PG={'o' if have(pgk, b) else 'x'},ext={'o' if have(exk, EXTBB[b]) else 'x'})"
            for b in skip))

    rows, per, t0 = [], [], time.time()
    for axis in AXES:
        for bb in todo:
            E = PN.load_ext(EXTBB[bb], axis)
            if len(E) < 5:
                print(f"  {axis}/{bb}: only {len(E)} external DMS, skipped"); continue
            with P.use_grid(GRID):
                w = PN.descend_tied(E, list(range(len(E))), COMP7)
            tune = float(np.mean([r.rho(w) for r in E]))

            # ⚠ C.load builds a Rec from **every** npz before filtering by axis, so a single npz
            # without that backbone's llr key kills it with KeyError (S2F was run on the 129
            # function and expression assays only and has no key on the 65 stability assays).
            # Temporarily push the missing assays into DROP.
            miss = {f.stem for f in sorted(C.CACHE.glob("*.npz"))
                    if f"llr_{C.llr_key(bb)}" not in np.load(f, allow_pickle=True).files}
            old_drop = set(C.DROP)
            C.DROP |= miss
            try:
                R = C.load(bb, axis)
            finally:
                C.DROP.clear(); C.DROP |= old_drop
            base = np.array([r.rho(np.zeros(7)) for r in R])
            fit = np.array([r.rho(w) for r in R])
            d = fit - base
            p = wilcoxon(fit, base).pvalue if len(R) > 5 else np.nan
            for r, b, f in zip(R, base, fit):
                per.append(dict(axis=axis, bb=bb, dms=r.dms, base=b, fit7=f, d=f - b))
            rows.append(dict(
                axis=axis, bb=bb, n_ext=len(E), n_pg=len(R),
                **{s: w[k] for k, s in enumerate(SL)},
                total=float(w.sum()), n_neg=int((w < 0).sum()),
                tune_obj=tune, pg_base=base.mean(), pg_fit7=fit.mean(),
                d_rho=d.mean(), d_med=np.median(d), win=float((d > 0).mean()), p=p))
            print(f"  {SHOW[axis]:10s} {bb:13s} ext={len(E):2d} pg={len(R):3d}  "
                  f"base {base.mean():.4f} -> fit7 {fit.mean():.4f} ({d.mean():+.4f}, "
                  f"win {(d > 0).mean():.0%}, p={p:.1e})  total={w.sum():.1f} neg={int((w < 0).sum())}")

    D = pd.DataFrame(rows)
    H = RESULTS
    H.mkdir(parents=True, exist_ok=True)
    D.to_csv(H / "extend_backbones.csv", index=False)
    pd.DataFrame(per).to_csv(H / "extend_backbones_perdms.csv", index=False)
    print(f"\ncells that chose a negative slot: {int((D.n_neg > 0).sum())} / {len(D)}   ({time.time()-t0:.0f}s)")
    print("wrote:", H / "extend_backbones.csv")


if __name__ == "__main__":
    main()
