#!/usr/bin/env python
"""The SBI (stable but function-losing) position cohort: the position-level computation used by
the main text and the appendix.

## Why it is rewritten

The script that produced `sbi_12bb_comp7.csv` (2026-10-16), which holds the published numbers, was
not in the repository. `sbi_quadrant_comp7.py` covers the same topic but its output
(`sbi_quad_comp7_raw.csv`) carries different values (maximum difference 0.22 to 0.48). The
definition written in the paper is therefore reimplemented here, and the **old 10-protein numbers
are reproduced first** before moving to 11 proteins.

## Definition (as in the supplementary section on SBI positions)

  cohort     variants **shared** between the function DMS and the expression DMS of the same
             protein, matched on (wt, pos, mut). Proteins with fewer than 200 shared variants are
             dropped.
  position   each assay averaged over all substitutions at that position.
  quadrant   the two position values split at **their own medians**. No prediction enters this.
             SBI = abundance at or above the median **and** function below it.
  P_m(SBI)   z-normalise the scores over all shared variants within the protein, average per
             position over substitutions, then average over the SBI positions. Unitless. Negative
             means those positions are called more damaging than the protein average.

  ⚠ The median convention is `>=` (`rank >= median`). Using `rank >= n/2` changes the position
    count by one or two.

output  sbi_positions_{tag}.csv (P_m per protein, backbone and arm) and _{tag}_cohort.csv
run     CUDA_VISIBLE_DEVICES="" python scripts/sbi_positions.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))   # paper_numbers lives in this folder
from plmfusion import common as C
from plmfusion.paths import RESULTS, pg_cache
import paper_numbers as PN

CACHE = pg_cache()
OUT = RESULTS

AA = "ACDEFGHIKLMNPQRSTVWY"
SL = C.SLOTS
# The roster is taken from the single place that defines it (paper_numbers.EXTBB) rather than by
# reading this package's own output back in.
BB12 = sorted(PN.EXTBB)
MINSHARED = 200


def key(z):
    return [f"{AA[a]}{p}{AA[b]}" for a, p, b in zip(z["wt_i"], z["pos1"], z["mut_i"])]


def pairs(exclude=()):
    """(prot, function dms, expression dms) with at least MINSHARED shared variants."""
    idx = {}
    for f in sorted(CACHE.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        ax = str(z["axis"])
        if ax in ("function", "stability"):
            idx.setdefault(str(z["prot"]), {}).setdefault(ax, []).append(f.stem)
    out = []
    for p, v in sorted(idx.items()):
        if p in exclude or "function" not in v or "stability" not in v:
            continue
        fd, sd = v["function"][0], v["stability"][0]
        kf = set(key(np.load(CACHE / f"{fd}.npz", allow_pickle=True)))
        ks = set(key(np.load(CACHE / f"{sd}.npz", allow_pickle=True)))
        if len(kf & ks) >= MINSHARED:
            out.append((p, fd, sd))
    return out


def cohort(prot, fd, sd):
    zf = np.load(CACHE / f"{fd}.npz", allow_pickle=True)
    zs_ = np.load(CACHE / f"{sd}.npz", allow_pickle=True)
    kf = {k: i for i, k in enumerate(key(zf))}
    ks = {k: i for i, k in enumerate(key(zs_))}
    com = sorted(set(kf) & set(ks))
    fi = np.array([kf[k] for k in com]); si = np.array([ks[k] for k in com])
    pos = np.asarray(zf["pos1"], int)[fi]
    yf = np.asarray(zf["ry"], float)[fi]        # function experimental values
    ya = np.asarray(zs_["ry"], float)[si]       # abundance experimental values
    up = np.unique(pos)
    pf = np.array([yf[pos == p].mean() for p in up])
    pa = np.array([ya[pos == p].mean() for p in up])
    hi_a = pa >= np.median(pa)
    hi_f = pf >= np.median(pf)
    quad = np.where(hi_a & hi_f, "T_OK", np.where(hi_a & ~hi_f, "T_SBI",
                    np.where(~hi_a & ~hi_f, "T_LOSS", "T_ODD")))
    return dict(prot=prot, func=fd, expr=sd, n_var=len(com), pos=pos, up=up,
                quad=quad, fi=fi, n_pos=len(up))


def main():
    W = pd.read_csv(RESULTS / "extend_backbones.csv")
    W = W[(W.axis == "function") & (W.bb.isin(BB12))].set_index("bb")
    for tag, exclude in [("p10", ("SPIKE_SARS2",)), ("p11", ())]:
        P = pairs(exclude)
        co = [cohort(*p) for p in P]
        nv = sum(c["n_var"] for c in co); np_ = sum(c["n_pos"] for c in co)
        nq = {q: sum(int((c["quad"] == q).sum()) for c in co)
              for q in ("T_OK", "T_SBI", "T_LOSS", "T_ODD")}
        print(f"[{tag}] {len(co)} proteins, {nv:,} shared variants, {np_:,} positions, {nq}")
        rows = []
        for c in co:
            rec = {bb: C.Rec(CACHE / f"{c['func']}.npz", bb) for bb in BB12}
            for bb in BB12:
                w = np.array([W.loc[bb, s] for s in SL], float)
                for arm, wv in (("pLM", np.zeros(7)), ("ProFusion", w)):
                    s = rec[bb].s(wv)[c["fi"]]
                    s = (s - s.mean()) / (s.std() + 1e-12)       # z within the protein
                    pm = np.array([s[c["pos"] == p].mean() for p in c["up"]])
                    for q in ("T_OK", "T_SBI", "T_LOSS", "T_ODD"):
                        m = c["quad"] == q
                        rows.append(dict(prot=c["prot"], bb=bb, arm=arm, quad=q,
                                         Pm=float(pm[m].mean()) if m.any() else np.nan,
                                         n=int(m.sum())))
        D = pd.DataFrame(rows)
        D.to_csv(OUT / f"sbi_positions_{tag}.csv", index=False)
        pd.DataFrame([{k: c[k] for k in ("prot", "func", "expr", "n_var", "n_pos")}
                      for c in co]).to_csv(OUT / f"sbi_positions_{tag}_cohort.csv", index=False)
        # Also written in the shape the figure script reads. p11 is the canonical version.
        # ⚠ The older sbi_12bb_comp7.csv had no generating script in the repository. This replaces it.
        if tag == "p11":
            # GCK T168 percentile (main text, and panel d of the SBI figure). This used to reuse
            # a panel_c12.csv built in a different session, whose values came from **ProteinMPNN
            # before rescoring** and **weights before refitting**. That left panel a on new values
            # and panel d on old ones within the same figure, so it is rebuilt here.
            gck = [c for c in co if c["prot"] == "HXK4_HUMAN"]
            if gck:
                g = gck[0]
                pr = []
                for bb in BB12:
                    r2 = C.Rec(CACHE / f"{g['func']}.npz", bb)
                    wv = np.array([W.loc[bb, s] for s in SL], float)
                    o = {}
                    for arm, vv in (("pLM", np.zeros(7)), ("ProFusion", wv)):
                        s2 = r2.s(vv)[g["fi"]]
                        pm2 = np.array([s2[g["pos"] == q].mean() for q in g["up"]])
                        rk = rankdata(pm2)          # smaller is more damaging
                        o[arm] = float(rk[list(g["up"]).index(168)] / len(g["up"]) * 100)
                    pr.append(dict(bb=bb, **o))
                # Written into the results folder, so nothing needs to be moved aside first.
                # The frozen copy stays at figure_assets("sbi_gck")/panel_c12.csv, and the figure
                # script reads that frozen copy.
                gp = RESULTS / "panel_c12.csv"
                pd.DataFrame(pr).to_csv(gp, index=False)
                print(f"   GCK T168 written: panel_c12.csv ({len(pr)} backbones, {g['n_pos']} positions)")
            # The results folder is rewritten on every run, so no copy is kept aside (running twice
            # gives the same result).
            old = OUT / "sbi_12bb_comp7.csv"
            (D[D.quad == "T_SBI"][["prot", "bb", "arm", "Pm"]]
             .to_csv(old, index=False))
            print(f"   written for the figure: {old.name} ({int((D.quad=='T_SBI').sum())} rows)")
        for q in ("T_OK", "T_SBI", "T_LOSS", "T_ODD"):
            g = D[(D.quad == q)].groupby("arm").Pm.mean()
            print(f"   {q:7s} n={nq[q]:5d}  pLM {g.get('pLM', np.nan):+.4f} "
                  f"-> ProFusion {g.get('ProFusion', np.nan):+.4f}")
        print(f"   wrote sbi_positions_{tag}.csv")


if __name__ == "__main__":
    main()
