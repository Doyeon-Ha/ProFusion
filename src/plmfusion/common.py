#!/usr/bin/env python
"""Seven-slot fusion record, coordinate descent, and protein-wise K-fold.

Only what the paper numbers need is kept here: the record that holds one DMS, the coordinate
descent that fits the weights, and the grouped split used as an internal-tuning control.

## What is measured (state this before quoting any number)

    rho(w)   Spearman between the fused score and the DMS label, within one DMS.
             Unitless, in [-1, 1], higher is better.
    fused    s(w) = z(pLM LLR) + CH^T w
             CH has 7 slots = (a_GEMME, b_GEMME, a_ddG, b_ddG, a_RSA, b_BLOSUM, b_dchg)
             a_* is a channel's **position-mean component**, b_* its **within-position
             residual**. RSA is constant within a position, so there is no b_RSA. For
             BLOSUM and |Delta charge| the position component is essentially the identity
             of the wild-type residue, so only the residual is used.

## Protocol
Grouped K-fold over proteins (UniProt). Weights are fitted on the tune folds only and reported
on test-fold DMS only. **Repeated splits shuffle the groups**, not the samples, because
permuting samples leaves GroupKFold unchanged.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

logger = logging.getLogger(__name__)

# Paths are defined only in paths.py.
from . import paths  # noqa: E402
CACHE = paths.pg_cache()
# 2026-10-28: emptied. It used to be {"CCDB_ECOLI_Adkar_2012"}.
# In that assay the label is measured per position and copied to every substitution at that
# position (1,176 variants, 49 distinct label values, within-position SD of 0 at all 100
# positions). `rho_sub` is therefore blocked at every position by the `np.std(y[m]) > 0` guard
# and comes out NaN.
#
# It is included anyway, for three reasons.
#   1. CCDB_ECOLI_Tripathi_2016 carries the same degeneracy more weakly (40% of positions) and
#      was always included. Excluding only Adkar would be inconsistent.
#   2. The mechanism that handles degeneracy is already the per-position guard in `metrics`.
#      Dropping the whole assay duplicates that guard while discarding rho and rho_pos, which
#      are computed perfectly well.
#   3. The global rho is unaffected and the result does not move: the mean over backbones goes
#      from +0.0980 to +0.0979.
# AIDO-RAG 16B was added to this set on the same date (rho = 0.4912), with no missing llr.
# Numbers quoted above this line are for the 128-assay function cohort; it is now 129.
DROP: set[str] = set()

# Backbone name -> the llr key inside the npz. The names are used by figures, colour tables and
# CSVs, so they stay; only **the key that is read** changes. 2026-10-29: ESM2 was switched to the
# official ProteinGym convention (masked-marginal).
#   llr_ESM2     = wildtype-marginal. Reproduces the official per-DMS rho to within 0.001 on
#                  only 6 of 147 assays.
#   llr_ESM2mask = masked-marginal (official). 121 of 147 by the same test, median |drho| 0.0003.
LLR_ALIAS = {"ESM2": "ESM2mask"}


def llr_key(bb: str) -> str:
    return LLR_ALIAS.get(bb, bb)

SLOTS = ["a_GEMME", "b_GEMME", "a_ddG", "b_ddG", "a_RSA", "b_BLOSUM", "b_dchg"]
W_G = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6]
W_D = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3]
W_R = [-0.2, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
W_S = [-0.2, 0.0, 0.1, 0.2, 0.3, 0.45, 0.6]
GRIDS = [W_G, W_G, W_D, W_D, W_R, W_S, W_S]
NPASS = 3


def zs(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-9)


def _ry(y):
    """Centre and L2-normalise the label ranks, so that the dot product with the centred
    ranks of s is the Spearman correlation."""
    r = rankdata(np.asarray(y, float))
    r = r - r.mean()
    nr = np.sqrt((r * r).sum())
    return r / (nr + 1e-12)


class Rec:
    """One DMS: the 7-slot channel matrix plus the label ranks. No torch, no structure file."""

    def __init__(self, f: Path, bb: str):
        z = np.load(f, allow_pickle=True)
        self.dms, self.prot, self.axis = f.stem, str(z["prot"]), str(z["axis"])
        pid = z["pidx"].astype(np.int64)
        self.n = n = len(pid)
        npos = int(pid.max()) + 1
        cnt = np.bincount(pid, minlength=npos)

        def split(v):
            pm = (np.bincount(pid, v, minlength=npos) / np.maximum(cnt, 1))[pid]
            return pm, v - pm

        g_p, g_r = split(zs(z["ch_gemme"]))
        d_p, d_r = split(zs(z["ch_ddg"]))
        r_p, _ = split(zs(z["ch_rsa"]))
        _, bl_r = split(zs(z["blosum"]))
        _, ac_r = split(zs(-np.abs(z["dchg"].astype(float))))
        self.CH = np.ascontiguousarray(np.stack([g_p, g_r, d_p, d_r, r_p, bl_r, ac_r]).T)  # (n,7)
        self.llr = zs(z[f"llr_{llr_key(bb)}"])
        self.y = z["ry"].astype(float)
        self.ry = _ry(self.y)
        self.npos = int((cnt > 0).sum())

    def s(self, w):
        return self.llr + self.CH @ np.asarray(w, float)

    def rho(self, w):
        r = rankdata(self.s(w))
        r = r - r.mean()
        nr = np.sqrt((r * r).sum())
        return float((r @ self.ry) / nr) if nr else 0.0

    # --- Subset evaluation (for split-half). idx indexes variants; ry_sub holds that subset's label ranks
    def rho_at(self, w, idx, ry_sub):
        s = self.llr[idx] + self.CH[idx] @ np.asarray(w, float)
        r = rankdata(s)
        r = r - r.mean()
        nr = np.sqrt((r * r).sum())
        return float((r @ ry_sub) / nr) if nr else 0.0

    def rho_plm(self):
        return self.rho(np.zeros(7))


def load(bb: str, axis: str = "function"):
    """Read only the DMS on the given axis.

    **Filter by axis before building any Rec.** The 65 stability_ddg assays, which the paper does
    not use, have no llr for some backbones (S2F, VespaG, VenusREM, AIDO_RAG16B), so building
    first raises KeyError. DMS skipped for a missing llr are always logged: if they disappeared
    silently there would be no way to notice that the number of evaluated DMS differs per
    backbone.
    """
    R, skipped = [], []
    for f in sorted(CACHE.glob("*.npz")):
        if f.stem in DROP:
            continue
        with np.load(f, allow_pickle=True) as z:
            if str(z["axis"]) != axis:
                continue
            if f"llr_{llr_key(bb)}" not in z.files:
                skipped.append(f.stem)
                continue
        R.append(Rec(f, bb))
    if skipped:
        logger.warning("load(%s, %s): skipped %d DMS with no llr: %s",
                       bb, axis, len(skipped), ", ".join(skipped[:5]))
    return R


# ---------------------------------------------------------- coordinate descent
def descend(R, idx, w0=None, slots=None, mask=None, ry=None):
    """Coordinate descent maximising the mean rho over the tune set.
    When mask/ry are given, each DMS is measured on that subset only (split-half)."""
    w = list(w0) if w0 is not None else [0.0] * 7
    use = list(range(7)) if slots is None else list(slots)
    for _ in range(NPASS):
        for k in use:
            best, bv = w[k], -9.0
            for v in GRIDS[k]:
                c = list(w)
                c[k] = v
                if mask is None:
                    m = float(np.mean([R[i].rho(c) for i in idx]))
                else:
                    m = float(np.mean([R[i].rho_at(c, mask[i], ry[i]) for i in idx]))
                if m > bv:
                    best, bv = v, m
            w[k] = best
    return np.array(w, float)


def descend_tied(R, idx, tie):
    """Coordinate descent over tied parameters. tie[k] = None pins that slot to 0."""
    free = sorted({t for t in tie if t is not None})
    v = {f: 0.0 for f in free}

    def w_of(vv):
        return np.array([0.0 if t is None else vv[t] for t in tie])

    def grid_of(f):
        k = tie.index(f)
        return GRIDS[k]
    for _ in range(NPASS):
        for f in free:
            best, bv = v[f], -9.0
            for g in grid_of(f):
                c = dict(v); c[f] = g
                w = w_of(c)
                m = float(np.mean([R[i].rho(w) for i in idx]))
                if m > bv: best, bv = g, m
            v[f] = best
    return w_of(v)


# ---------------------------------------------------------------- grouped split
def grouped_folds(groups, k=5, seed=0):
    """Protein-wise K-fold. The split is built by **shuffling the groups themselves**.
    Groups are shuffled and then placed into the smallest fold, rather than being sorted by
    size, so that the folds stay balanced."""
    groups = np.asarray(groups)
    uq, inv = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inv)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uq))
    load_ = np.zeros(k, int)
    assign = np.zeros(len(uq), int)
    for gi in order:
        f = int(np.argmin(load_))
        assign[gi] = f
        load_[f] += sizes[gi]
    fold_of = assign[inv]
    out = []
    for f in range(k):
        te = np.flatnonzero(fold_of == f)
        tr = np.flatnonzero(fold_of != f)
        out.append((tr, te))
    return out


# ---------------------------------------------------------------- feature
