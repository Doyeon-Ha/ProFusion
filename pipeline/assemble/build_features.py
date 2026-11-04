#!/usr/bin/env python
"""Assemble the ProteinGym channel cache, one npz per DMS.

The current snapshot has 212 npz: the two main axes (function 129 and expression 18, 147 together)
plus the appendix Stability axis (stability_ddg, 65). The `axis` field inside each npz says which
one it belongs to, and evaluation selects by axis.

It takes the per-DMS channel bundle from `e_mix3.load()` (4 pLM LLRs, SPURS ddG, RSA, GEMME), adds
what the position/residual decomposition needs, and freezes one npz per DMS.

Stored per DMS
  variant level  : pos1, wt_i, mut_i, ry (rank-centred label), llr_{BB}, ddg, blosum, dvol, dchg, dhyd
  position level : upos, rsa, gemme_pos, hwt[P, 1536]
  position x 20  : ddg20, blo20, dvol20, dchg20, dhyd20, so that zero-sum holds exactly over all 20 AA
  baselines      : ch_gemme, ch_rsa, ch_ddg (variant-level z)

Sign convention: a larger llr means more tolerated (closer to wild type), ddg is the raw SPURS
value (positive = destabilising), and ch_ddg uses z(-ddg).
"""
from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd
import torch

from plmfusion import paths  # noqa: E402

from aa_props import AA20, _GRANTHAM_CPV, blosum62  # noqa: E402
from e_mix3 import load                             # noqa: E402

# The original wrote straight into the shared cache. Here the rebuild goes somewhere separate.
OUT = paths.RESULTS / "rebuild/proteingym/two_dim"
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5,
          L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
CHG = dict(D=-1.0, E=-1.0, K=1.0, R=1.0, H=0.1)
VOL = {a: _GRANTHAM_CPV[a][2] for a in AA20}
AAI = {a: i for i, a in enumerate(AA20)}


def z(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x * 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    D = load()
    print(f"[load] {len(D)} DMS, {sum(d['n'] for d in D)} variants, {time.time()-t0:.0f}s", flush=True)

    with open(paths.data_root() / "proteingym/esm3_seqonly_embeddings.pkl", "rb") as f:
        EMB = pickle.load(f)
    BL = blosum62()
    bl20 = np.array([[BL[(a, b)] for b in AA20] for a in AA20], dtype=np.float32)
    vol = np.array([VOL[a] for a in AA20], dtype=np.float32)
    chg = np.array([CHG.get(a, 0.0) for a in AA20], dtype=np.float32)
    hyd = np.array([KD[a] for a in AA20], dtype=np.float32)

    meta, skipped = [], []
    for d in D:
        dms, prot = d["dms"], d["prot"]
        if dms not in EMB:
            skipped.append((dms, "no_emb")); continue
        H = np.asarray(EMB[dms], dtype=np.float32)                  # [L+2, 1536], including BOS/EOS
        zn = np.load(paths.data_root() / f"proteingym/spurs/matrix/{prot}.npz", allow_pickle=True)
        DDG = zn["ddg"].astype(np.float32)                          # [L, 20], alphabet=AA20
        assert str(zn["alphabet"]) == AA20, dms
        lo = int(zn["pdb_range"][0])                                # ddg row 0 == sequence position lo

        mut = d["mut"]
        pos1 = np.array([int(m[1:-1]) for m in mut], dtype=np.int32)
        wt_i = np.array([AAI[m[0]] for m in mut], dtype=np.int8)
        mt_i = np.array([AAI[m[-1]] for m in mut], dtype=np.int8)
        row_ddg = pos1 - lo                                         # ddg row index
        # The embeddings are [L+2, 1536] (0=BOS, 1..L=residues, L+1=EOS), so the 1-based position is the row number
        row_emb = pos1
        ok = (row_ddg >= 0) & (row_ddg < DDG.shape[0]) & (row_emb >= 0) & (row_emb < H.shape[0])
        if ok.mean() < 0.99:
            skipped.append((dms, f"pos_align {ok.mean():.2f}")); continue

        upos, inv = np.unique(pos1[ok], return_inverse=True)
        P = len(upos)
        gem = d["S"]["gemme"]                                       # variant-level z(GEMME)
        gpos = np.zeros(P, dtype=np.float32)
        np.add.at(gpos, inv, gem[ok]); gpos /= np.bincount(inv, minlength=P)

        np.savez_compressed(
            OUT / f"{dms}.npz",
            prot=prot, axis=d["axis"], lo=lo,
            pos1=pos1[ok], wt_i=wt_i[ok], mut_i=mt_i[ok], pidx=inv.astype(np.int32),
            ry=d["ry"][ok].astype(np.float32),
            **{f"llr_{b}": z(d[b][ok]).astype(np.float32) for b in ("ESM1v", "ESM2", "ESM3", "ProSST")},
            ddg=DDG[row_ddg[ok], mt_i[ok]],
            blosum=bl20[wt_i[ok], mt_i[ok]],
            dvol=(vol[mt_i[ok]] - vol[wt_i[ok]]),
            dchg=(chg[mt_i[ok]] - chg[wt_i[ok]]),
            dhyd=(hyd[mt_i[ok]] - hyd[wt_i[ok]]),
            ch_gemme=gem[ok].astype(np.float32),
            ch_rsa=d["S"]["rsa"][ok].astype(np.float32),
            ch_ddg=d["S"]["ddg"][ok].astype(np.float32),
            upos=upos, rsa=np.asarray([d["S"]["rsa"][ok][inv == p].mean() for p in range(P)], np.float32),
            gemme_pos=gpos, hwt=H[upos].astype(np.float16),
            wt20=np.array([wt_i[ok][inv == p][0] for p in range(P)], np.int8),
            ddg20=DDG[upos - lo], blo20=bl20[[wt_i[ok][inv == p][0] for p in range(P)]],
            dvol20=vol[None, :] - vol[[wt_i[ok][inv == p][0] for p in range(P)]][:, None],
            dchg20=chg[None, :] - chg[[wt_i[ok][inv == p][0] for p in range(P)]][:, None],
            dhyd20=hyd[None, :] - hyd[[wt_i[ok][inv == p][0] for p in range(P)]][:, None],
        )
        meta.append(dict(dms=dms, prot=prot, axis=d["axis"], n=int(ok.sum()), n_pos=P, L=H.shape[0]))
        print(f"  {dms:55s} n={ok.sum():6d} P={P:4d}", flush=True)

    pd.DataFrame(meta).to_csv(OUT / "meta.csv", index=False)
    print(f"\n[done] wrote {len(meta)} DMS -> {OUT}  ({time.time()-t0:.0f}s)")
    if skipped:
        print("[skip]", skipped)


if __name__ == "__main__":
    main()
