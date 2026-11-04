#!/usr/bin/env python
"""Input for the eta-squared criterion figure: per-channel eta-squared, the fraction of a
channel's position-mean component explained by the wild-type amino acid.

A large eta-squared means that channel's position-mean component is effectively a lookup table
over "which amino acid is the wild type". That is why BLOSUM and |Delta charge| drop their
position component and use the residual only, which is the design argument for comp7.

Eta-squared is affine invariant, so (1) position means after z-normalisation and (2) position
means of the raw values must agree. Both are computed as a check.

⚠ Every npz in two_dim/ that has a wt20 key is read (there is no filtering by the direction list),
  so adding files to that directory changes the result.

input:  <data>/mavedb_external/two_dim/*.npz
output: results/fig_data/panelb_eta2_verified.csv
"""
import numpy as np
import pandas as pd

from plmfusion.paths import RESULTS, ext_cache

OUT = RESULTS / "fig_data"
OUT.mkdir(parents=True, exist_ok=True)
AA = "ACDEFGHIKLMNPQRSTVWY"
rows = []
for f in sorted(ext_cache().glob("*.npz")):
    d = np.load(f, allow_pickle=True)
    if 'wt20' not in d.files: continue
    pid = d['pidx'].astype(int); npos = int(d['wt20'].shape[0]); wt = d['wt20'].astype(int)
    cnt = np.bincount(pid, minlength=npos)
    for nm, raw in (("GEMME", d['ch_gemme']), ("ddG", d['ch_ddg']), ("RSA", d['ch_rsa']),
                    ("BLOSUM", d['blosum']), ("Δcharge", -np.abs(d['dchg'].astype(float)))):
        v = np.asarray(raw, float)
        # (1) z at the variant level -> position mean -> eta-squared
        z = (v - v.mean()) / (v.std() + 1e-9)
        p1 = np.bincount(pid, z, minlength=npos) / np.maximum(cnt, 1)
        # (2) position mean of the raw values -> eta-squared (must agree: affine invariance)
        p2 = np.bincount(pid, v, minlength=npos) / np.maximum(cnt, 1)

        # (3) eta-squared straight from the variant level against the wild-type AA (no position mean)
        def eta2(x, g):
            tot = ((x - x.mean())**2).sum()
            if tot < 1e-12: return np.nan
            bet = sum((g == a).sum() * (x[g == a].mean() - x.mean())**2 for a in range(20) if (g == a).sum())
            return bet / tot
        rows.append(dict(dms=str(d['urn']), channel=nm,
                         e_zpos=eta2(p1, wt), e_rawpos=eta2(p2, wt), e_var=eta2(z, wt[pid]),
                         res_var=float(np.var(z - p1[pid]) / (np.var(z) + 1e-12)),
                         nsub=cnt.mean()))
R = pd.DataFrame(rows)
print("=== eta-squared, three ways of computing it (mean over DMS) ===")
print(f"  {'channel':10s} {'(1) z->pos':>12s} {'(2) raw->pos':>14s} {'(3) variant':>12s} {'resid var':>10s}")
for c in ["GEMME", "ddG", "RSA", "BLOSUM", "Δcharge"]:
    q = R[R.channel == c]
    print(f"  {c:10s} {q.e_zpos.mean():12.4f} {q.e_rawpos.mean():14.4f} "
          f"{q.e_var.mean():10.4f} {q.res_var.mean():10.4f}")
print(f"\n  (1) vs (2) max difference {np.nanmax(np.abs(R.e_zpos - R.e_rawpos)):.2e}  (affine invariance confirmed)")
R.to_csv(OUT / "panelb_eta2_verified.csv", index=False)
