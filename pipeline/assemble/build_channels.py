#!/usr/bin/env python
"""Assemble the external tuning-set channels into npz with **the same schema** as the evaluation
cache.

The convention follows build_features.py and e_mix3.py exactly. Any difference would make w* refer
to a different quantity in the tuning set than in evaluation.

    ry        = rank(DMS_score), centred and unit-norm       (after the direction correction)
    ch_gemme  = z(GEMME)                                     NaN filled with the median
    ch_ddg    = z(**-**spurs_ddg)                            note the sign flip
    ch_rsa    = z(RSA)
    ddg       = the raw ddG matrix value (not flipped)
    blosum/dvol/dchg/dhyd = amino acid pair tables

The only difference from ProteinGym is the coordinates: the DMS sequence was folded as it is, so
the ddG row index is pos-1 and the RSA key is pos.

run: python pipeline/assemble/build_channels.py --variant manual
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from plmfusion import paths  # noqa: E402
from aa_props import AA20, _GRANTHAM_CPV, blosum62          # noqa: E402

EXT = paths.data_root() / "mavedb_external"
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5,
          L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
CHG = dict(D=-1.0, E=-1.0, K=1.0, R=1.0, H=0.1)
VOL = {a: _GRANTHAM_CPV[a][2] for a in AA20}
AAI = {a: i for i, a in enumerate(AA20)}
# Directory name -> npz channel name. The evaluation cache uses llr_ESM2/ESM1v/ESM3/ProSST.
# esm3mm is **multimodal** and therefore a different quantity from the cache llr_ESM3, so it gets its own name.
BBNAME = {"esm2": "ESM2", "esm1v": "ESM1v", "prosst": "ProSST", "esm3mm": "ESM3mm", "esm3seq": "ESM3seq",
           "esm2_8M": "ESM2_8M", "esm2_35M": "ESM2_35M",
           "esm2_150M": "ESM2_150M", "esm2_3B": "ESM2_3B",
           "esmif1": "ESMIF1", "proteinmpnn": "ProteinMPNN", "s2f": "S2F", "esm2_masked": "ESM2mask"}


def z(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x * 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="manual")
    # For attaching a frozen set outside the pre-registered 49 (frozen.csv), such as the 3
    # stability-pool assays. It must be paired with direction_{variant}.csv.
    ap.add_argument("--frozen", default="frozen.csv")
    # ⚠ Writing them into the default two_dim/ would silently change the results of every script
    # that **globs that directory unfiltered**. Sets outside the 49 go into their own directory.
    ap.add_argument("--outdir", default="two_dim")
    # Assemble without SPURS ddG. The stability axis drops that channel, so there is no reason to
    # run SPURS on 277 domains. ddg/ch_ddg/ddg20 are filled with NaN and ddg_available=False recorded.
    ap.add_argument("--no-ddg", action="store_true")
    ap.add_argument("--backbones", nargs="*",
                    default=["esm2", "esm1v", "prosst", "esm3mm", "esm3seq",
                             "esmif1", "proteinmpnn", "s2f", "esm2_masked",
                             "esm2_8M", "esm2_35M", "esm2_150M", "esm2_3B"])
    A = ap.parse_args()

    # The original wrote straight into the data directory. Here the rebuild is kept separate.
    OUT = paths.RESULTS / "rebuild/mavedb_external" / A.outdir
    OUT.mkdir(parents=True, exist_ok=True)
    F = pd.read_csv(EXT / A.frozen)
    F["tag"] = [hashlib.md5(s.encode()).hexdigest()[:10] for s in F.pseq]
    D = pd.read_csv(EXT / f"direction_{A.variant}.csv")
    # Some frozen sets already carry a tag in their direction csv (domainome). Merging as is would
    # split it into tag_x/tag_y and raise KeyError, so only missing columns are taken from frozen.
    D = D.merge(F[["urn", *[c for c in ("tag", "pseq") if c not in D.columns]]], on="urn")

    # Every work directory is read immediately. A single directory with no output used to kill the
    # whole run with IndexError, because orphans belonging to no frozen set are mixed in. They are
    # skipped and their names reported. (If a tag really is needed and missing, GEM[tag] below
    # raises KeyError straight away.)
    GEM, gem_missing = {}, []
    for d in glob.glob(str(EXT / "gemme/work") + "/*/"):
        tag = os.path.basename(d.rstrip("/"))
        hit = glob.glob(d + "*_normPred_evolCombi.txt")
        if not hit:
            gem_missing.append(tag); continue
        GEM[tag] = pd.read_csv(hit[0], sep=" ", index_col=0).iloc[:, 0]
    if gem_missing:
        print(f"[no gemme output: {len(gem_missing)}] {sorted(gem_missing)}", flush=True)
    BL = blosum62()
    bl20 = np.array([[BL[(a, b)] for b in AA20] for a in AA20], dtype=np.float32)
    vol = np.array([VOL[a] for a in AA20], np.float32)
    chg = np.array([CHG.get(a, 0.0) for a in AA20], np.float32)
    hyd = np.array([KD[a] for a in AA20], np.float32)

    meta, skipped = [], []
    for r in D.to_dict("records"):
        urn, tag, seq = r["urn"], r["tag"], r["pseq"]
        stem = urn.replace(":", "-")
        llr = {}
        for bb in A.backbones:
            p = EXT / "llr" / bb / f"{stem}.pt"
            llr[bb] = torch.load(p, map_location="cpu", weights_only=False) if p.exists() else None
        if llr.get("esm2") is None:
            skipped.append((urn, "no_esm2")); continue
        base = llr["esm2"]
        mut = list(base["mutants"])
        for bb, t in llr.items():
            assert t is None or list(t["mutants"]) == mut, f"{urn}: {bb} variant order differs from esm2"
        y = base["dms_scores"].numpy().astype(float)
        if r["sign_flip"]:
            y = -y                                   # normalise to higher_is_wt

        rsa_f = EXT / "rsa" / f"{tag}.json"
        ddg_f = EXT / "ddg" / f"{tag}.npz"
        if not rsa_f.exists() or (not ddg_f.exists() and not A.no_ddg):
            skipped.append((urn, "no_struct_channel")); continue
        rsa_map = {int(k): v for k, v in json.loads(rsa_f.read_text()).items()}
        if ddg_f.exists():
            zn = np.load(ddg_f, allow_pickle=True)
            DDG = zn["ddg"].astype(np.float32)
            assert str(zn["alphabet"]) == AA20 and DDG.shape[0] == len(seq), urn
        else:
            # --no-ddg: this is a tuning set for an axis that **does not use ddG** (stability).
            # NaN is written rather than 0, so that an arm reading this channel by mistake fails
            # loudly instead of quietly using 0.
            DDG = np.full((len(seq), 20), np.nan, np.float32)

        pos1 = np.array([int(m[1:-1]) for m in mut], np.int32)
        wt_i = np.array([AAI[m[0]] for m in mut], np.int8)
        mt_i = np.array([AAI[m[-1]] for m in mut], np.int8)
        gem = GEM[tag].reindex(mut).to_numpy()
        rsa_v = np.array([rsa_map.get(int(p), np.nan) for p in pos1], float)

        ok = ((pos1 >= 1) & (pos1 <= len(seq)) & ~np.isnan(rsa_v))
        if ok.mean() < 0.9 or ok.sum() < 100:
            skipped.append((urn, f"coverage {ok.mean():.2f} n={int(ok.sum())}")); continue
        if np.isnan(gem).mean() > 0.05:
            skipped.append((urn, "gemme_nan")); continue
        gem = np.where(np.isnan(gem), np.nanmedian(gem), gem)

        ryv = rankdata(y[ok]); ryv = ryv - ryv.mean(); ryv = ryv / np.sqrt((ryv * ryv).sum())
        upos, inv = np.unique(pos1[ok], return_inverse=True)
        P = len(upos)
        gz = z(gem[ok])
        gpos = np.zeros(P, np.float32); np.add.at(gpos, inv, gz); gpos /= np.bincount(inv, minlength=P)
        row = pos1[ok] - 1                                   # ddG row = pos-1 (lo=1)
        ddg_raw = DDG[row, mt_i[ok]]
        wt_of_pos = np.array([wt_i[ok][inv == p][0] for p in range(P)])

        kw = {f"llr_{BBNAME[b]}": z(llr[b]["scores"].numpy()[ok]).astype(np.float32)
              for b in A.backbones if llr.get(b) is not None}
        np.savez_compressed(
            OUT / f"{stem}.npz",
            prot=r["tname"][:16], axis=r["axis"], lo=1, urn=urn,
            ddg_available=bool(ddg_f.exists()),
            pos1=pos1[ok], wt_i=wt_i[ok], mut_i=mt_i[ok], pidx=inv.astype(np.int32),
            ry=ryv.astype(np.float32), **kw,
            ddg=ddg_raw.astype(np.float32),
            blosum=bl20[wt_i[ok], mt_i[ok]],
            dvol=(vol[mt_i[ok]] - vol[wt_i[ok]]),
            dchg=(chg[mt_i[ok]] - chg[wt_i[ok]]),
            dhyd=(hyd[mt_i[ok]] - hyd[wt_i[ok]]),
            ch_gemme=gz.astype(np.float32),
            ch_rsa=z(rsa_v[ok]).astype(np.float32),
            ch_ddg=z(-ddg_raw).astype(np.float32),           # sign flipped (positive = destabilising)
            upos=upos.astype(np.int32),
            rsa=np.array([rsa_v[ok][inv == p].mean() for p in range(P)], np.float32),
            gemme_pos=gpos, wt20=wt_of_pos.astype(np.int8),
            ddg20=DDG[upos - 1], blo20=bl20[wt_of_pos],
        )
        meta.append(dict(urn=urn, tname=r["tname"], axis=r["axis"], n=int(ok.sum()), n_pos=P,
                         backbones=",".join(k for k in kw)))
        print(f"  {r['tname'][:26]:28s} {urn[-12:]} n={ok.sum():6d} P={P:4d}", flush=True)

    pd.DataFrame(meta).to_csv(OUT / f"meta_{A.variant}.csv", index=False)
    print(f"\n[done] {len(meta)} DMS → {OUT}")
    if skipped:
        print("[skip]", skipped)


if __name__ == "__main__":
    main()
