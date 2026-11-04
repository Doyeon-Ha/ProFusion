"""Three follow-up analyses: how the budget is split between pLM and channels, how the channels
do alone, and how correlated the backbones are after fusion.

## 1. Sweeping the split in w1*s_plm + w2*(channel sum)

Spearman is **invariant to a monotone rescaling**, so w1 and w2 need not be moved independently.
With w1 > 0 the ranking is unchanged, and the only degree of freedom is **lambda = w2/w1**.
lambda = 0 is pLM alone, lambda = 1 is ProFusion as fitted, and lambda -> infinity is channels
only.

Re-choosing lambda on ProteinGym would use ProteinGym labels and therefore **break the
label-free protocol**. This is a post-hoc diagnostic of whether the fitted magnitude transfers,
not a proposal to use the chosen value.

## 2. Channels alone, zero-shot

Drop the pLM and measure the within-assay Spearman from one channel. Signs and transforms are kept
**the same** as the fusion uses (the common.Rec convention): GEMME, −ΔΔG, RSA, BLOSUM,
−|Delta charge|. The 7-slot combination alone (lambda -> infinity) is reported as well.

## 3. Correlation between backbones after fusion, and rank preservation

  r_pair          Spearman between two backbones' **scores** within an assay, averaged over
                  assays, before and after fusion.
  rank preserved  how well the ranking of backbones by mean rho survives fusion (Spearman).

output  plm_vs_channel_{sweep,solo,pairs}.csv
run     OMP_NUM_THREADS=6 python scripts/plm_vs_channel.py
"""
from __future__ import annotations
import time
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from plmfusion import common as C
from plmfusion.paths import RESULTS

H = RESULTS
SL = C.SLOTS
BB = ["AIDO_RAG16B", "VenusREM", "ProSST", "VespaG", "TranceptionLret", "S2F",
      "ESM1v", "SaProt650M", "ESM3", "ESM2", "ESMIF1", "ProteinMPNN"]
AX = ["function", "stability"]
LAM = np.round(np.concatenate([np.arange(0, 2.01, 0.1), [2.5, 3.0, 4.0, 6.0]]), 2)
# Channels alone, with the transforms the fusion uses. (npz key, display name, sign)
SOLO = [("ch_gemme", "GEMME", 1), ("ch_ddg", "-ΔΔG", 1), ("ch_rsa", "RSA", 1),
        ("blosum", "BLOSUM62", 1), ("dchg", "-|Δcharge|", 0)]


def load_bb(bb, axis):
    """⚠ C.load builds a Rec from **every** npz before filtering by axis, so a single npz without
    that backbone's llr key kills it with KeyError (the 65 stability assays have no AIDO or S2F
    key). As in extend_backbones.py, the missing assays are pushed into DROP temporarily."""
    miss = {f.stem for f in sorted(C.CACHE.glob("*.npz"))
            if f"llr_{bb}" not in np.load(f, allow_pickle=True).files}
    old = set(C.DROP)
    C.DROP |= miss
    try:
        return C.load(bb, axis)
    finally:
        C.DROP.clear(); C.DROP |= old


def rho(x, ry):
    r = rankdata(x); r = r - r.mean(); n = np.sqrt((r * r).sum())
    return float((r @ ry) / n) if n else 0.0


def main():
    t0 = time.time()
    W = pd.read_csv(H / "extend_backbones.csv").set_index(["axis", "bb"])
    sweep, solo, pairs = [], [], []
    for ax in AX:
        R0 = load_bb(BB[0], ax)                          # the channels do not depend on the backbone
        dms = [r.dms for r in R0]
        # -- 2. channels alone
        for key, nm, sgn in SOLO:
            for r in R0:
                z = np.load(C.CACHE / f"{r.dms}.npz", allow_pickle=True)
                v = z[key].astype(float)
                v = -np.abs(v) if sgn == 0 else v
                solo.append(dict(axis=ax, channel=nm, dms=r.dms, rho=rho(v, r.ry)))
        # -- 1 and 3, per backbone
        fused = {}
        for bb in BB:
            Rs = load_bb(bb, ax)
            w = W.loc[(ax, bb), SL].to_numpy(float)
            for lam in LAM:
                v = [r.rho(lam * w) for r in Rs]
                sweep.append(dict(axis=ax, bb=bb, lam=float(lam), rho=float(np.mean(v)),
                                  n=len(v)))
            # the channel combination only (pLM removed) = the lambda -> infinity limit
            co = [rho(r.CH @ w, r.ry) for r in Rs]
            sweep.append(dict(axis=ax, bb=bb, lam=np.inf, rho=float(np.mean(co)), n=len(co)))
            fused[bb] = dict(base=[r.s(np.zeros(7)) for r in Rs],
                             fit=[r.s(w) for r in Rs], ry=[r.ry for r in Rs])
        # -- 3. backbone pair correlation (mean over assays of the within-assay score Spearman)
        for i, a in enumerate(BB):
            for b in BB[i + 1:]:
                for tag in ("base", "fit"):
                    v = [spearmanr(x, y).statistic
                         for x, y in zip(fused[a][tag], fused[b][tag])]
                    pairs.append(dict(axis=ax, arm=tag, bb1=a, bb2=b,
                                      r=float(np.mean(v)), n=len(v)))
        print(f"  [{ax}] done ({time.time()-t0:.0f}s)", flush=True)

    S = pd.DataFrame(sweep); S.to_csv(H / "plm_vs_channel_sweep.csv", index=False)
    O = pd.DataFrame(solo); O.to_csv(H / "plm_vs_channel_solo.csv", index=False)
    P = pd.DataFrame(pairs); P.to_csv(H / "plm_vs_channel_pairs.csv", index=False)

    print("\n=== 1. lambda sweep (function). lambda=1 is the current model ===")
    f = S[(S.axis == "function") & np.isfinite(S.lam)]
    piv = f.pivot_table(index="bb", columns="lam", values="rho")
    best = piv.idxmax(axis=1)
    out = pd.DataFrame({"ρ(λ=0)": piv[0.0], "ρ(λ=1)": piv[1.0],
                        "best λ": best, "ρ(best)": piv.max(axis=1)})
    out["Δ(best−1)"] = out["ρ(best)"] - out["ρ(λ=1)"]
    print(out.round(4).sort_values("ρ(λ=1)", ascending=False).to_string())

    print("\n=== 2. channels alone (mean rho over assays) ===")
    print(O.groupby(["axis", "channel"]).rho.agg(["mean", "median"]).round(4).to_string())
    print("\n   (for reference) the 7-slot combination only, pLM removed:")
    print(S[~np.isfinite(S.lam)].groupby(["axis", "bb"]).rho.mean()
          .unstack(0).round(4).to_string())

    print("\n=== 3. backbone pair correlation and rank preservation ===")
    print(P.groupby(["axis", "arm"]).r.agg(["mean", "min", "max"]).round(4).to_string())
    pm = pd.read_csv(H / "extend_backbones_perdms.csv")
    for ax in AX:
        g = pm[(pm.axis == ax) & (pm.bb.isin(BB))].groupby("bb")[["base", "fit7"]].mean()
        rs = spearmanr(g.base, g.fit7)
        print(f"  [{ax}] backbone rank rho(before fusion, after) = {rs.statistic:+.3f}  p={rs.pvalue:.3g}")


if __name__ == "__main__":
    main()
