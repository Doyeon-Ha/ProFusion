#!/usr/bin/env python
"""Axis cross-application: do the fusion weights have to be fitted per assay axis?
12 backbones, uni3.

## The arms

  matched  that axis's weights applied to that axis's assays. This is the main model (ProFusion).
  swapped  **the other axis's** weights applied. If the axis distinction meant nothing, this would
           match the matched arm.
  pooled   one vector fitted on all 49 tuning assays with no axis distinction, applied to both
           axes. A fairer control than swapped: it ignores the axis label but uses all the data.

## What is measured

  d_rho  the difference paired by assay (matched minus the comparison). Positive means splitting
         by axis is better. Unitless.
  q      paired two-sided Wilcoxon over assays, Benjamini-Hochberg corrected within a panel.

Weights are fitted on the MaveDB external set only and ProteinGym labels are used for evaluation
only. Grid uni3 (all slots −3.0 to 3.0, steps of 0.1), coordinate descent, 3 passes, zero start.

⚠ The internal axis key "stability" is the expression axis of the manuscript (14 external, 16
  ProteinGym assays). It is a different axis from the 65 folding-ΔG assays (stability_ddg).

output  axis_crossapply_12bb.csv (summary), _perdms.csv, and _w.csv (the weights)
run     OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES="" python scripts/axis_crossapply_12bb.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))   # paper_numbers lives in this folder
from plmfusion import common as C, grids as P
from plmfusion.paths import RESULTS
import paper_numbers as PN                      # noqa: E402

# common.load filters by axis first and skips, with a warning, any assay that has no llr.
# EXTBB is the same 12-backbone mapping, defined once in paper_numbers.
load_pg = C.load
EXTBB = PN.EXTBB

GRID = "uni3"
TIE7 = [0, 1, 2, 3, 4, 5, 6]
AXES = ["function", "stability"]
OTHER = {"function": "stability", "stability": "function"}


def main():
    rows, per, wrows, t0 = [], [], [], time.time()
    for pgk, exk in EXTBB.items():
        E = {a: PN.load_ext(exk, a) for a in AXES}
        if min(len(E[a]) for a in AXES) < 5:
            print(f"  {pgk}: external {[len(E[a]) for a in AXES]}, skipped"); continue
        pool = E["function"] + E["stability"]          # all 49 assays, no axis distinction
        with P.use_grid(GRID):
            W = {a: PN.descend_tied(E[a], list(range(len(E[a]))), TIE7) for a in AXES}
            W["pooled"] = PN.descend_tied(pool, list(range(len(pool))), TIE7)
        for k, w in W.items():
            wrows.append(dict(bb=pgk, fit=k, n_ext=len(pool) if k == "pooled" else len(E[k]),
                              **{s: w[i] for i, s in enumerate(PN.SL)}))
        for axis in AXES:
            R = load_pg(pgk, axis)
            arms = {"matched": W[axis], "swapped": W[OTHER[axis]], "pooled": W["pooled"]}
            v = {k: np.array([r.rho(w) for r in R]) for k, w in arms.items()}
            for r, m, s, p in zip(R, v["matched"], v["swapped"], v["pooled"]):
                per.append(dict(axis=axis, bb=pgk, dms=r.dms,
                                matched=m, swapped=s, pooled=p))
            dmean = {}
            for k in ("swapped", "pooled"):
                d = v["matched"] - v[k]
                dmean[k] = d.mean()
                rows.append(dict(axis=axis, bb=pgk, contrast=k, n=len(R),
                                 mean=d.mean(), median=np.median(d),
                                 pct_improved=100 * float((d > 0).mean()),
                                 p=wilcoxon(v["matched"], v[k]).pvalue if np.any(d != 0) else 1.0))
            # ⚠ This used to print `d.mean() if k=="swapped" else 0` outside the loop. After the
            #   loop k is always "pooled", so **the progress line always read +0.0000**. The CSV
            #   was correct, but the log made the axis distinction look pointless. Both are now
            #   printed with their names.
            print(f"  {axis:10s} {pgk:14s} matched {v['matched'].mean():.4f}  "
                  f"swapped {dmean['swapped']:+.4f}  pooled {dmean['pooled']:+.4f}", flush=True)
        print(f"    ({time.time()-t0:.0f}s)", flush=True)

    D = pd.DataFrame(rows)
    # BH correction within a panel (axis x contrast)
    D["q"] = np.nan
    for (a, c), g in D.groupby(["axis", "contrast"]):
        p = g.p.to_numpy(); n = len(p); o = np.argsort(p); q = np.empty(n); run = 1.0
        for i in range(n - 1, -1, -1):
            run = min(run, p[o[i]] * n / (i + 1)); q[o[i]] = run
        D.loc[g.index, "q"] = q
    H = RESULTS
    H.mkdir(parents=True, exist_ok=True)
    D.to_csv(H / "axis_crossapply_12bb.csv", index=False)
    pd.DataFrame(per).to_csv(H / "axis_crossapply_12bb_perdms.csv", index=False)
    pd.DataFrame(wrows).to_csv(H / "axis_crossapply_12bb_w.csv", index=False)
    print(f"\nwrote output ({time.time()-t0:.0f}s)")
    for (a, c), g in D.groupby(["axis", "contrast"]):
        print(f"  {a:10s} vs {c:8s}  mean d_rho {g['mean'].mean():+.4f}  "
              f"q<0.05 {int((g.q < 0.05).sum())}/{len(g)}")


if __name__ == "__main__":
    main()
