"""MaveDB external-set record. Used for weight fitting only.

Giving an axis reproduces the loader used for the paper numbers; axis=None returns both axes
pooled, which is what the axis cross-application control needs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import common as C
from . import paths

EXT = paths.data_root() / "mavedb_external"


class ERec:
    """One external npz, under the **same 7-slot convention** as common.Rec."""

    def __init__(self, f, bb):
        d = np.load(f, allow_pickle=True)
        self.dms, self.prot, self.axis = str(d["urn"]), str(d["prot"]), str(d["axis"])
        pid = d["pidx"].astype(np.int64)
        self.n = n = len(pid)
        npos = int(pid.max()) + 1
        cnt = np.bincount(pid, minlength=npos)
        self.pid, self.cnt, self.npos = pid, cnt, npos

        def sp(v):
            pm = (np.bincount(pid, v, minlength=npos) / np.maximum(cnt, 1))[pid]
            return pm, v - pm
        g_p, g_r = sp(C.zs(d["ch_gemme"])); d_p, d_r = sp(C.zs(d["ch_ddg"]))
        r_p, _ = sp(C.zs(d["ch_rsa"]))
        _, bl_r = sp(C.zs(d["blosum"])); _, ac_r = sp(C.zs(-np.abs(d["dchg"].astype(float))))
        self.CH = np.ascontiguousarray(np.stack([g_p, g_r, d_p, d_r, r_p, bl_r, ac_r]).T)
        self.llr = C.zs(d[f"llr_{C.llr_key(bb)}"].astype(float))
        self.y = d["ry"].astype(float)                   # normalised rank; monotone, so used directly for rho
        self.ry = C._ry(self.y)
        self.ch_full = {"gem": d["ch_gemme"].astype(float), "ddg": d["ch_ddg"].astype(float),
                        "rsa": d["ch_rsa"].astype(float)}

    def s(self, w): return self.llr + self.CH @ np.asarray(w, float)

    def rho(self, w):
        r = rankdata(self.s(w)); r = r - r.mean(); nr = np.sqrt((r * r).sum())
        return float((r @ self.ry) / nr) if nr else 0.0

    def rho_at(self, w, idx, ry_sub):
        s = self.llr[idx] + self.CH[idx] @ np.asarray(w, float)
        r = rankdata(s); r = r - r.mean(); nr = np.sqrt((r * r).sum())
        return float((r @ ry_sub) / nr) if nr else 0.0

    def rho_plm(self): return self.rho(np.zeros(7))


def load_ext(bb, axis=None):
    keep = set(pd.read_csv(EXT / "direction_manual.csv").urn)
    out = []
    for f in sorted((EXT / "two_dim").glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        if str(d["urn"]) in keep and (axis is None or str(d["axis"]) == axis) and f"llr_{bb}" in d.files:
            out.append(ERec(f, bb))
    return out
