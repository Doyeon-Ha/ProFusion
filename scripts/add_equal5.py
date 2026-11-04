#!/usr/bin/env python
"""Append the equal5 arm to the per-DMS cache: **the 5 raw channels with weight 1**, no
component split.

## Why it is computed separately

The 7-slot CH in `common.Rec` **discards the position component** of the substitution channels
(BLOSUM and Delta charge):

    _, bl_r = split(zs(z["blosum"]))          # position component discarded
    _, ac_r = split(zs(-|dchg|))              # position component discarded

The position mean of a substitution-type channel is essentially a lookup table over the wild-type
amino acid (eta-squared .79 to .93), which is why it is excluded from the main model. As a result
`a + b = the raw channel` holds for GEMME and ΔΔG, but the substitution channels enter **by half**.
So the older "EQ5", which set all 7 slots to 1, is **not** "the 5 raw channels equally weighted".

The equal5 arm added here z-normalises the five channels and adds them as they are: a plain
baseline with no design in it. RSA is constant within a position, so its residual is 0 and the raw
channel equals its position component.

⚠ **Two** things change at once between equal5 and a fitted arm: (1) the weights are fitted, and
  (2) the wild-type lookup component of the substitution channels is removed. To separate the two,
  the existing EQ5 arm (substitution as residual only, all weights 1) is the midpoint, and it is
  already in the same CSV.

output  adds rows with arm="equal5" to ablation_12bb_perdms.csv (safe to rerun: the same arm is
        removed first)
run     CUDA_VISIBLE_DEVICES="" python scripts/add_equal5.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))   # ablation_12bb lives in this folder
from plmfusion import common as C
from plmfusion.paths import RESULTS
from ablation_12bb import EXTBB, load_pg        # noqa: E402

H = RESULTS
H.mkdir(parents=True, exist_ok=True)


def equal5_rho(dms: str, bb: str) -> float:
    z = np.load(C.CACHE / f"{dms}.npz", allow_pickle=True)
    ch = [C.zs(z["ch_gemme"]), C.zs(z["ch_ddg"]), C.zs(z["ch_rsa"]),
          C.zs(z["blosum"]), C.zs(-np.abs(z["dchg"].astype(float)))]
    s = C.zs(z[f"llr_{bb}"]) + np.sum(ch, axis=0)
    ry = C._ry(z["ry"].astype(float))
    r = rankdata(s); r = r - r.mean(); n = np.sqrt((r * r).sum())
    return float((r @ ry) / n) if n else 0.0


def main():
    f = H / "ablation_12bb_perdms.csv"
    D = pd.read_csv(f)
    D = D[D.arm != "equal5"]                     # so that rerunning does not duplicate
    rows = []
    for axis in ("function", "stability"):
        for bb in EXTBB:
            for r in load_pg(bb, axis):
                rows.append(dict(axis=axis, bb=bb, arm="equal5", dms=r.dms,
                                 rho=equal5_rho(r.dms, bb)))
            print(f"  {axis:10s} {bb}", flush=True)
    out = pd.concat([D, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(f, index=False)
    chk = out[(out.axis == "function")].groupby("arm").rho.mean()
    print(f"\nadded {len(rows):,} rows -> {f}")
    print("function, mean rho:")
    print(chk[["pLM", "equal5", "EQ3", "EQ5", "FW5", "SFW7"]].round(4).to_string())


if __name__ == "__main__":
    main()
