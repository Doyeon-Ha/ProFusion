#!/usr/bin/env python
"""Internal tuning: how much changes if the weights are chosen inside ProteinGym.
12 backbones, uni3.

## Why it is measured

VenusREM chooses its retrieval coefficient on a ProteinGym validation subset, while ProFusion
chooses everything on MaveDB. The text promises that contrast but carries no number for it. The
same fusion model is run with **only the source of the fit changed** to fill that gap.

## The arms

  base  pLM alone. w = 0, which is fusion switched off.
  ext   MaveDB external-set weights (ProFusion as it stands). Read from axis_crossapply_12bb_w.csv.
  int   weights fitted **inside ProteinGym**. The tuning set is the 11-protein SBI cohort.

## Why the SBI cohort is the tuning set

Section 4.3 of the text already uses the labels of those 11 proteins to define SBI positions.
Those assays are already spent, so using them for weight fitting too means **no new labels are
used**. In exchange, those 11 drop out of the evaluation (11 per axis): function goes from 129 to
118 and expression from 18 to 7.

  ⚠ Only 7 assays remain for expression, because 11 of the 18 are in the cohort. No axis-level
    conclusion should be drawn from 7. The number is reported, but the text uses function 118.

## What is measured

  rho        Spearman between the fused score and the label within one DMS. Unitless, [-1, 1],
             higher is better.
  scope      eval = everything outside the cohort (the only fair comparison)
             tune = the cohort itself. int was fitted on those labels, so this is **not** a result
             all  = all 129/18. Used only to check ext against the published values
  d_int_ext  int minus ext on the same DMS. Positive means choosing inside ProteinGym is better

Grid uni3 (all slots −3.0 to 3.0, steps of 0.1), coordinate descent, 3 passes, zero start: the
**same protocol** as ext. The only difference is which labels the objective is measured on.

output  internal_tune_12bb_{perdms,summary,w}.csv
run     OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" python scripts/internal_tune_12bb.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np, pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))   # paper_numbers lives in this folder
from plmfusion import common as C
from plmfusion.paths import RESULTS, pg_cache

GRID = "uni3"
TIE7 = [0, 1, 2, 3, 4, 5, 6]
AXES = ["function", "stability"]          # the internal key stability is the manuscript expression
# The cohort is defined as "proteins with an assay on both axes", so it is reconstructed from the
# cache rather than read from a table. The reconstructed 11 pairs match the original table row for
# row. The weights come from axis_crossapply_12bb_w.csv, whose 7 slots over the 24 cells agree
# with extend_backbones.csv exactly (max |d| = 0).
EXTW_CSV = RESULTS / "extend_backbones.csv"


def cohort():
    """axis -> the set of DMS names used for tuning on that axis."""
    per = {}
    for f in sorted(pg_cache().glob("*.npz")):
        with np.load(f, allow_pickle=True) as z:
            per.setdefault(str(z["axis"]), {}).setdefault(str(z["prot"]), []).append(f.stem)
    both = set(per["function"]) & set(per["stability"])
    return {ax: {d for p in both for d in per[ax][p]} for ax in AXES}


def load_pg(bb, axis):
    """C.load builds a Rec from every npz before filtering by axis, so a single npz without that
    backbone's key kills it with KeyError. The missing assays are pushed into DROP temporarily.
    """
    miss = {f.stem for f in sorted(C.CACHE.glob("*.npz"))
            if f"llr_{bb}" not in np.load(f, allow_pickle=True).files}
    old = set(C.DROP); C.DROP |= miss
    try:
        return C.load(bb, axis)
    finally:
        C.DROP.clear(); C.DROP |= old


def cell(args):
    bb, axis = args
    from plmfusion import grids as P            # noqa: E402
    import paper_numbers as PN                  # noqa: E402

    TUNE = cohort()[axis]
    R = load_pg(bb, axis)
    tune = [i for i, r in enumerate(R) if r.dms in TUNE]
    assert len(tune) == 11, f"{bb}/{axis}: tune has {len(tune)} (should be 11)"

    w_ext = (pd.read_csv(EXTW_CSV).query("bb == @bb and axis == @axis")[PN.SL]
             .to_numpy()[0])
    t0 = time.time()
    with P.use_grid(GRID):
        w_int = PN.descend_tied(R, tune, TIE7)
    secs = time.time() - t0

    rows = []
    for i, r in enumerate(R):
        rows.append(dict(axis=axis, bb=bb, dms=r.dms, prot=r.prot,
                         scope="tune" if i in tune else "eval", n_var=r.n,
                         base=r.rho_plm(), ext=r.rho(w_ext), int=r.rho(w_int)))
    w = dict(bb=bb, axis=axis, n_tune=len(tune), secs=round(secs, 1),
             **{s: float(w_int[i]) for i, s in enumerate(PN.SL)})
    return rows, w


def main():
    import paper_numbers as PN
    EXTBB = PN.EXTBB                  # the same 12-backbone mapping
    cells = [(bb, ax) for ax in AXES for bb in EXTBB]
    print(f"fitting {len(cells)} cells ({GRID}, coordinate descent, 3 passes)")
    per, wrows, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        for rows, w in ex.map(cell, cells):
            per.extend(rows); wrows.append(w)
            print(f"  {w['bb']:14s} {w['axis']:9s} {w['secs']:6.1f}s", flush=True)
    P = pd.DataFrame(per); W = pd.DataFrame(wrows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    P.to_csv(RESULTS / "internal_tune_12bb_perdms.csv", index=False)
    W.to_csv(RESULTS / "internal_tune_12bb_w.csv", index=False)

    rows = []
    for axis in AXES:
        for bb in EXTBB:
            for scope in ("eval", "tune", "all"):
                d = P[(P.axis == axis) & (P.bb == bb)]
                if scope != "all":
                    d = d[d.scope == scope]
                r = dict(axis=axis, bb=bb, scope=scope, n=len(d))
                for a in ("base", "ext", "int"):
                    r[f"mean_{a}"] = d[a].mean()
                    r[f"median_{a}"] = d[a].median()
                for a, b in (("ext", "base"), ("int", "base"), ("int", "ext")):
                    dd = (d[a] - d[b]).to_numpy()
                    r[f"d_{a}_{b}"] = dd.mean()
                    r[f"win_{a}_{b}"] = 100 * float((dd > 0).mean())
                    r[f"p_{a}_{b}"] = (wilcoxon(d[a], d[b]).pvalue
                                       if np.any(dd != 0) else 1.0)
                rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(RESULTS / "internal_tune_12bb_summary.csv", index=False)
    print(f"\ndone in {time.time()-t0:.0f}s")
    for axis in AXES:
        e = S[(S.axis == axis) & (S.scope == "eval")]
        print(f"\n[{axis}] eval n={e.n.iloc[0]}  "
              f"base {e.mean_base.mean():.4f} / ext {e.mean_ext.mean():.4f} / "
              f"int {e.mean_int.mean():.4f}  "
              f"| int−ext {e.d_int_ext.mean():+.4f} "
              f"({int((e.d_int_ext > 0).sum())} of 12 backbones positive)")


if __name__ == "__main__":
    main()
