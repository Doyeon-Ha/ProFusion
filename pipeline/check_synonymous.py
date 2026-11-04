"""Measure whether the 842 synonymous substitutions (wt == mut) change the paper numbers.

The background is item 1 of "Things to know" in pipeline/README.md. Those rows exist only in the
fitting set (the MaveDB external set); the evaluation set (ProteinGym) has none. An old regular
expression in the pre-assembly step (3b) failed to filter them out.

Method: using the same grid and the same coordinate descent, comp7 is fitted (A) as it is and
(B) with those rows removed. Only the cells whose weights differ are then applied to the
evaluation set and the mean rho over DMS compared.

The records are built straight from the npz, under the same convention as external.ERec but with
the row mask applied **first**, so that the z-normalisation, position split and label ranks are
recomputed on that subset. GEMME and LLR are computed independently per variant, so the raw scores
of the remaining variants do not change when the list gets shorter. This is therefore equivalent
to rerunning steps 3b and 8.

run: OMP_NUM_THREADS=6 python pipeline/check_synonymous.py
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from plmfusion import common as C
from plmfusion.common import descend_tied
from plmfusion.grids import use_grid
from plmfusion.paths import ext_cache, ext_direction

TIE7 = [0, 1, 2, 3, 4, 5, 6]
# The backbone roster comes from the same place as the paper numbers. Writing a separate list here
# would silently measure only 4 backbones as the roster grew, while the log still looked fine.
from paper_numbers import EXTBB  # noqa: E402  (run_all.sh puts scripts/ on PYTHONPATH)


class Rec:
    def __init__(self, d, bb, mask):
        pid = d["pidx"].astype(np.int64)[mask]
        _, inv = np.unique(pid, return_inverse=True)
        npos = int(inv.max()) + 1
        cnt = np.bincount(inv, minlength=npos)

        def split(v):
            v = C.zs(np.asarray(v, float)[mask])
            pm = (np.bincount(inv, v, minlength=npos) / np.maximum(cnt, 1))[inv]
            return pm, v - pm
        g_p, g_r = split(d["ch_gemme"]); dd_p, dd_r = split(d["ch_ddg"])
        r_p, _ = split(d["ch_rsa"]); _, bl_r = split(d["blosum"])
        _, ac_r = split(-np.abs(d["dchg"].astype(float)))
        self.CH = np.ascontiguousarray(np.stack([g_p, g_r, dd_p, dd_r, r_p, bl_r, ac_r]).T)
        self.llr = C.zs(np.asarray(d[f"llr_{bb}"], float)[mask])
        self.ry = C._ry(d["ry"].astype(float)[mask])
        self.n = int(mask.sum())

    def s(self, w): return self.llr + self.CH @ np.asarray(w, float)

    def rho(self, w):
        r = rankdata(self.s(w)); r = r - r.mean(); nr = np.sqrt((r * r).sum())
        return float((r @ self.ry) / nr) if nr else 0.0


keep = set(pd.read_csv(ext_direction()).urn)
print(f"{'axis/backbone':20s} {'n':>3s} {'syn':>6s}  weights")
for axis in ("function", "stability"):
    for bb in EXTBB:
        eb = EXTBB[bb]
        A, B, nsyn = [], [], 0
        for f in sorted(ext_cache().glob("*.npz")):
            d = np.load(f, allow_pickle=True)
            if str(d["urn"]) not in keep or str(d["axis"]) != axis or f"llr_{eb}" not in d.files:
                continue
            syn = d["wt_i"].astype(int) == d["mut_i"].astype(int)
            nsyn += int(syn.sum())
            all_m = np.ones(len(syn), bool)
            A.append(Rec(d, eb, all_m)); B.append(Rec(d, eb, ~syn))
        if len(A) < 8: continue
        # The comparison is only meaningful if the fit uses **the same grid** as the paper weights.
        # The first measurement, on 2026-10-14, used wide, whose GEMME and ddG lower bounds are 0,
        # so slot values hit the boundary (see grids.py). This now uses the confirmed uni3.
        with use_grid("uni3"):
            w1 = descend_tied(A, list(range(len(A))), TIE7)
            w2 = descend_tied(B, list(range(len(B))), TIE7)
        tag = "same" if np.array_equal(w1, w2) else f"differs (max|d| {np.abs(w1 - w2).max():.2f})"
        print(f"{axis+'·'+bb:20s} {len(A):3d} {nsyn:6d}  {tag}", flush=True)
        if np.array_equal(w1, w2):
            continue
        print(f"    as is       {np.round(w1, 2).tolist()}")
        print(f"    syn removed {np.round(w2, 2).tolist()}")
        # Only the cells whose weights changed are compared on the evaluation set (the same PG DMS either way).
        P = C.load(bb, axis=axis)
        ra = np.array([r.rho(w1) for r in P])
        rb = np.array([r.rho(w2) for r in P])
        d = rb - ra
        p = wilcoxon(d)[1] if d.std() > 1e-12 else 1.0
        print(f"    -> PG {len(P)} DMS mean rho {ra.mean():.4f} -> {rb.mean():.4f} "
              f"({d.mean():+.4f}, improved {100 * (d > 0).mean():.0f}%, p={p:.2g})", flush=True)
