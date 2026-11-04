#!/usr/bin/env python
"""Run the stability axis **with the structure of the paper**: the main arm uses the external
MaveDB weights w_ext, and the ProteinGym-internal 5-fold is a reference ceiling.

It follows the F2 structure of `paper_numbers.py`:

    main arm        w_ext, fitted on the external set        <- what the paper reports
    [reference]     ProteinGym-internal 5-fold (protein GroupKFold), a ceiling
    [control]       ProteinGym in-sample, a more optimistic diagnostic, not a method

The two ΔΔG slots (a_ddG and b_ddG) are **pinned to 0**: 63 of the 65 ProteinGym `stability_ddg`
assays are Tsuboyama, which is the SPURS training set, so removing that one channel closes the
leak. The grid is the confirmed unified grid, −3.0 to 3.0 in steps of 0.1 (61 points), with
coordinate descent, NPASS = 3, starting from zero.

## External tuning sets (all label-free; none of them sees ProteinGym)

| pool | composition | DMS / proteins | assay |
|---|---|---|---|
| `expr14` | the pre-registered 49 with axis=='stability' | 14 / 9 | abundance |
| `stab17` | plus 3 Stability assays (IGHG1, PyKS, ASPA) | 17 / 12 | abundance |

⚠ The domainome npz were assembled with `--no-ddg`, so `ch_ddg` is **all NaN**. Pinning the ΔΔG
  slots to 0 is not enough, because `0 * NaN = NaN`, so those columns are **replaced by 0**
  (mathematically identical at weight 0, and it stops the NaN propagating).
⚠ The domainome npz carry only ESM2 and ESM1v (no ProSST or ESM3mm LLR was built).
⚠ ESM3 must use **ESM3mm** (multimodal) on the external side to match the ProteinGym cache.

run: OMP_NUM_THREADS=8 PYTHONPATH=src python scripts/stability_protocol.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from plmfusion import common as C
from plmfusion.paths import RESULTS, ext_cache_named, ext_table


# rank_cols originally lived in a neighbouring module whose import chain reaches analyses this
# package does not ship, so the function body is copied here verbatim.
def rank_cols(S):
    """Ranks (1..n) down each column of [n, V]. Scores are continuous, so ties are ignorable."""
    o = np.argsort(S, axis=0)
    r = np.empty(S.shape)
    idx = np.arange(1, S.shape[0] + 1)[:, None]
    np.put_along_axis(r, o, np.broadcast_to(idx, S.shape).copy(), axis=0)
    return r


NPASS = 3
POOL_ARMS = ("expr14", "tsu129_all11", "dom")   # 12 backbones x 3 tuning sets
SLOT7 = ["a_GEMME", "b_GEMME", "a_ddG", "b_ddG", "a_RSA", "b_BLOSUM", "b_dchg"]
FREE5 = [0, 1, 4, 5, 6]                                  # ddG (2, 3) excluded
U3 = np.round(np.arange(-3.0, 3.0 + 1e-9, 0.1), 2)
EQ = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
# 12 backbones. ProteinGym npz key <-> external npz key.
# ⚠ ProteinGym ESM3 is multimodal, so the external side must be ESM3mm.
# ⚠ SaProt's external key is SaProt650M_own.
# ⚠ VenusREM, S2F, AIDO_RAG16B and VespaG had no LLR on the 65 ProteinGym stability_ddg assays,
#   because this axis had been left out of the project. ProteinGym **does not distribute** AF2
#   structures for those 65 (0/65), so structure-input models could not run at all. They were
#   folded here with ColabFold.
#   ⚠ The other 130 assays use the structures ProteinGym distributes, so **the structure source is
#     mixed**. State this in the Methods.
BBS = ["ProteinMPNN", "ProSST", "ESMIF1", "TranceptionLret",
       "ESM3", "SaProt650M", "ESM2", "ESM1v",
       "VespaG", "VenusREM", "S2F", "AIDO_RAG16B"]
# The pair switched to the official ESM2 convention (masked-marginal). Both sides use
# llr_ESM2mask. rho between the wildtype-marginal llr_ESM2 and the official one is 0.9847.
BB_EXTRA = ["ESM2mask"]
EXTBB = {"ESM2": "ESM2", "ESM2mask": "ESM2mask", "ESM1v": "ESM1v", "ESM3": "ESM3mm",
         "ProSST": "ProSST", "ESMIF1": "ESMIF1", "ProteinMPNN": "ProteinMPNN",
         "TranceptionLret": "TranceptionLret", "S2F": "S2F", "VespaG": "VespaG",
         "VenusREM": "VenusREM", "SaProt650M": "SaProt650M_own",
         "AIDO_RAG16B": "AIDO_RAG16B"}
# -- Convention correction (--official)
# On the evaluation side these three backbones are the **official ProteinGym scores**, while the
# fitting set was a different quantity scored by our own code. That is, w was fitted under
# convention A and applied under convention B. The keys whose fitting set was rescored with the
# official formula are *_off. Measured rho (ours, official):
#     ESM-IF1      tsu129 0.978, dom 0.982   (per position -> mean logL over the whole sequence)
#     ProteinMPNN  tsu129 0.753, dom 0.823   (per position -> mean over the AR decoding order)
#     SaProt650M   tsu129 0.991              (all tokens masked -> AA only, summed over 3Di)
# The existing keys are kept and both are reported, so the effect of the correction is measured
# within one frame.
EXTBB_OFF = {"ESMIF1": "ESMIF1_off", "ProteinMPNN": "ProteinMPNN_off",
             "SaProt650M": "SaProt650M_off"}
POOLS = {                                                # (npz directory, direction csv, accepted axis)
    "expr14": [("two_dim", "direction_manual.csv", "stability")],
    "stab17": [("two_dim", "direction_manual.csv", "stability"),
               ("two_dim_stabpool", "direction_stability_pool.csv", "stability")],
    "dom": [("two_dim_domainome", "direction_domainome.csv", "stability_ddg")],
    # The same-assay arm: MaveDB Tsuboyama natural domains that share **no protein at all** with
    # ProteinGym. Every external pool so far was abundance, a different assay from the target
    # (proteolysis dG). This is the direct test of the "the bottleneck is assay mismatch"
    # hypothesis.
    # ⚠ It comes from the **same paper and the same protocol** as the ProteinGym evaluation set
    #   (only the proteins differ). That is a different standard from the other external arms,
    #   which claim weights chosen in an independent study, so the arm is named separately and
    #   its limitation stated.
    "tsu_pilot": [("two_dim_tsu_pilot", "direction_tsu_pilot.csv", "stability_ddg")],
    "tsu129": [("two_dim_tsu129", "direction_tsu129.csv", "stability_ddg")],
    # The **most conservative** variant of the same arm, with all 11 weak mmseqs hits removed.
    # The freeze rule (pid >= 30 and scov >= 0.5 and e < 1e-3) did not count them as homologous,
    # but rather than argue the call, the arm that removes them and remeasures is the main one.
    "tsu129_all11": [("two_dim_tsu129", "direction_tsu129.csv", "stability_ddg")],
}
# Tags of every mmseqs easy-search hit (tsu129 -> ProteinGym Stability 65, -s 7.5 --min-seq-id 0
# -e 1000). The strongest is cylR2 vs ARGR_ECOLI (pid 33.3%, coverage 58%, e 0.22), both HTH
# transcriptional regulators.
HITS11 = ["7e40fe59cf", "0b006c432d", "1aec33427d", "2c53f1f64a", "52bbef7c80",
          "7f4eacdee2", "b964a9d48f", "12e83019d8", "4bc06b8ef0", "5a9dbc4e2e", "90037fadcc"]


def _drop_stems(pool: str) -> set[str]:
    if pool != "tsu129_all11":
        return set()
    F = pd.read_csv(ext_table("frozen_tsu129.csv"))
    return {r.urn.replace(":", "-") for r in F.itertuples() if r.tag in set(HITS11)}


_POOL_CACHE: dict[tuple[str, str], list] = {}


def load_pool(pool, bb):
    """Read an external tuning unit as a common.Rec. A NaN ddG column is replaced by 0.

    ⚠ The same (pool, bb) is requested dozens of times (the coverage guard measures all 12
      backbones). dom has 273 npz, so without a cache this hits the disk hundreds of times."""
    if (pool, bb) in _POOL_CACHE:
        return _POOL_CACHE[(pool, bb)]
    drop = _drop_stems(pool)
    out = []
    for sub, dcsv, want in POOLS[pool]:
        root, dpath = ext_cache_named(sub), ext_table(dcsv)
        if not root.exists() or not dpath.exists():
            continue
        keep = set(pd.read_csv(dpath).urn)
        for f in sorted(root.glob("*.npz")):
            urn = f"urn:{f.stem.split('-')[1]}:{'-'.join(f.stem.split('-')[2:])}"
            if urn not in keep or f.stem in drop:
                continue
            z = np.load(f, allow_pickle=True)
            if str(z["axis"]) != want or f"llr_{bb}" not in z.files:
                continue
            r = C.Rec(f, bb)
            bad = ~np.isfinite(r.CH)
            if bad.any():
                assert set(np.where(bad.any(0))[0]) <= {2, 3}, f"{urn}: a non-ddG column is NaN"
                r.CH[:, [2, 3]] = 0.0            # equivalent at weight 0; only stops NaN spreading
            out.append(r)
    _POOL_CACHE[(pool, bb)] = out
    return out


def rho_many(R, w, k, vals):
    tot = np.zeros(len(vals)); w = np.asarray(w, float)
    for r in R:
        base = r.llr + r.CH @ w - r.CH[:, k] * w[k]
        S = base[:, None] + r.CH[:, k][:, None] * np.asarray(vals)[None, :]
        Rk = rank_cols(S); Rk -= Rk.mean(axis=0, keepdims=True)
        den = np.sqrt((Rk * Rk).sum(axis=0))
        tot += np.divide(Rk.T @ r.ry, den, out=np.zeros(len(vals)), where=den > 0)
    return tot / len(R)


def fit(R, grid=U3, free=FREE5):
    w = [0.0] * 7
    for _ in range(NPASS):
        for k in free:
            w[k] = float(grid[int(np.argmax(rho_many(R, w, k, grid)))])
    return w


def per_dms(R, w):
    return np.array([r.rho(w) for r in R])


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", action="store_true",
                    help="run with the fitting set switched to the official-convention keys (*_off)")
    ap.add_argument("--extra", action="store_true", help="add the ESM2mask rows")
    ap.add_argument("--tag", default="", help="suffix for the output csv")
    A = ap.parse_args()
    if A.official:
        EXTBB.update(EXTBB_OFF)
        print("[convention] fitting set switched to the official keys:", EXTBB_OFF)
    bbs = BBS + (BB_EXTRA if A.extra else [])
    t0, rows = time.time(), []
    for axis in ("stability_ddg", "stability"):
        tag = ("proteolysis ΔG · 63/65 Tsuboyama" if axis == "stability_ddg"
               else "VAMP-seq abundance · Tsuboyama 0")
        print("=" * 112)
        print(f"PG {axis} - {tag}   [grid -3.0 to 3.0 / 0.1, NPASS={NPASS}, ddG slots pinned to 0]")
        print("=" * 112)
        for bb in bbs:
            P = C.load(bb, axis=axis)
            if not P:
                continue
            s_llr = per_dms(P, [0.0] * 7)
            prot = np.array([r.prot for r in P])
            print(f"\n  -- {bb}, PG {len(P)} DMS / {len(set(prot))} proteins, llr alone {s_llr.mean():.4f}")
            print(f"     {'arm':22s} {'tune n':>10s} | " + " ".join(f"{SLOT7[i]:>8s}" for i in FREE5)
                  + f" | {'rho':>7s} {'d vs llr':>8s} {'ceiling':>8s}")

            arms = {}
            for pool in POOL_ARMS:
                E = load_pool(pool, EXTBB[bb])
                # ⚠ A pool where that backbone's LLR covers only part of the set is **discarded**.
                #   Letting it through silently mixes a value fitted on one assay into the table
                #   (dom x ProteinMPNN was 1/1).
                full = max(len(load_pool(pool, EXTBB[b])) for b in bbs)
                if not E or len(E) < 0.8 * full:
                    if E:
                        print(f"     [skipped] ext {pool}: llr_{EXTBB[bb]} present in only "
                              f"{len(E)}/{full} assays", flush=True)
                    continue
                arms[f"ext {pool}"] = (fit(E), f"{len(E)}/{len({r.prot for r in E})}")
            arms["EQ (no fitting)"] = (EQ, "—")

            # [reference] ProteinGym internal 5-fold. Uses target labels. For the ceiling only.
            s_cv = np.zeros(len(P))
            for tu, te in C.grouped_folds(prot, 5, seed=0):
                w_cv = fit([P[i] for i in tu])
                for i in te:
                    s_cv[i] = P[i].rho(w_cv)
            # [control] in-sample
            w_ins = fit(P)
            s_ins = per_dms(P, w_ins)
            ceil = s_ins.mean() - s_llr.mean()

            for name, (w, tn) in arms.items():
                s = per_dms(P, w)
                d = s.mean() - s_llr.mean()
                print(f"     {name:22s} {tn:>10s} │ " + " ".join(f"{w[i]:8.1f}" for i in FREE5)
                      + f" │ {s.mean():7.4f} {d:+8.4f} {100*d/ceil if ceil else np.nan:7.0f}%")
                rows.append(dict(axis=axis, bb=bb, arm=name, tune_n=tn, rho=s.mean(),
                                 d_vs_llr=d, recov=100 * d / ceil if ceil else np.nan,
                                 **{SLOT7[i]: w[i] for i in FREE5}))
            for name, s, w in (("[reference] PG internal 5-fold", s_cv, None),
                               ("[control] PG in-sample", s_ins, w_ins)):
                d = s.mean() - s_llr.mean()
                ws = " ".join(f"{w[i]:8.1f}" for i in FREE5) if w else " " * 44
                print(f"     {name:22s} {'(PG labels)':>10s} | {ws} | {s.mean():7.4f} {d:+8.4f} "
                      f"{100*d/ceil if ceil else np.nan:7.0f}%")
                rows.append(dict(axis=axis, bb=bb, arm=name, tune_n="PG labels", rho=s.mean(),
                                 d_vs_llr=d, recov=100 * d / ceil if ceil else np.nan,
                                 **({SLOT7[i]: w[i] for i in FREE5} if w else {})))
        print()
    RESULTS.mkdir(parents=True, exist_ok=True)
    f = RESULTS / f"stability_protocol{A.tag}.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    print(f"[saved] {f}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
