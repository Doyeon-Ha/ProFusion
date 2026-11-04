#!/usr/bin/env python
"""Shapley value per channel: the number that shows why LOO is a lower bound.

## Why Shapley

The channels overlap (ΔΔG with RSA 0.59, BLOSUM with GEMME 0.54, GEMME with ΔΔG 0.51). Under
overlap,

  LOO           the marginal contribution when the channel is added **last** = its unique part
                -> a **lower bound**
  leave-one-in  the marginal contribution when it is added **first** = the whole shared part
                -> an **upper bound**
  Shapley       the mean marginal contribution over **all orderings** -> in between

Shapley splits the shared part fairly among the contributors. Reading the three together
separates "this channel looks small because it is masked" from "this channel is simply small":
a wide gap between the bounds means masking, and bounds that sit close together mean the channel
really is worth that much.

## The computation

v(S) is the mean Spearman over ProteinGym DMS using only the channels in subset S, after
**refitting the remaining slots on the external set**. v(empty) = pLM alone. With 5 channels, all
32 subsets are enumerated.

⚠ By definition the contributions sum to v(all) − v(empty) (efficiency). That identity is used as
  a check.

output  shapley_channels.csv (axis, bb, channel, shapley, loo, loi)
run     python scripts/shapley_channels.py
"""
from __future__ import annotations
from math import factorial
from pathlib import Path
import itertools
import numpy as np, pandas as pd

from plmfusion.paths import RESULTS

H = RESULTS
H.mkdir(parents=True, exist_ok=True)
NAMES = ["GEMME", "ddG", "RSA", "BLOSUM", "dchg"]
SHOW = {"GEMME": "GEMME", "ddG": "ΔΔG", "RSA": "RSA", "BLOSUM": "BLOSUM", "dchg": "Δcharge"}
N = len(NAMES)


def key(sub):
    return "+".join(n for n in NAMES if n in sub) or "none"


def main():
    D = pd.read_csv(H / "ablation_lattice.csv")
    rows = []
    for (axis, bb), g in D.groupby(["axis", "bb"]):
        v = dict(zip(g.subset, g.rho))
        full, empty = v[key(set(NAMES))], v["none"]
        for k in NAMES:
            rest = [n for n in NAMES if n != k]
            phi = 0.0
            for m in range(N):
                for S in itertools.combinations(rest, m):
                    w = factorial(m) * factorial(N - m - 1) / factorial(N)
                    phi += w * (v[key(set(S) | {k})] - v[key(set(S))])
            rows.append(dict(axis=axis, bb=bb, channel=SHOW[k], shapley=phi,
                             loo=full - v[key(set(rest))],          # added last (lower bound)
                             loi=v[key({k})] - empty,               # added first (upper bound)
                             total_gain=full - empty))
    R = pd.DataFrame(rows)
    # efficiency check: the channel contributions must sum to the total gain
    chk = R.groupby(["axis", "bb"]).apply(
        lambda d: abs(d.shapley.sum() - d.total_gain.iloc[0]), include_groups=False)
    assert chk.max() < 1e-9, f"efficiency violated, max={chk.max():.2e}"
    print(f"efficiency check passed (max error {chk.max():.1e})\n")
    R.to_csv(H / "shapley_channels.csv", index=False)

    for axis, lab in [("function", "function"), ("stability", "expression")]:
        d = R[R.axis == axis]
        t = d.groupby("channel").agg(LOO=("loo", "mean"), Shapley=("shapley", "mean"),
                                     LOI=("loi", "mean"))
        t["masked share"] = t.Shapley - t.LOO
        t = t.reindex(["GEMME", "ΔΔG", "RSA", "BLOSUM", "Δcharge"])
        print(f"=== {lab} axis (mean over 12 backbones) ===")
        print(t.round(4).to_string())
        print(f"  Shapley sum {t.Shapley.sum():.4f} = total gain "
              f"{d.groupby('bb').total_gain.first().mean():.4f}\n")
    print("wrote:", H / "shapley_channels.csv")


if __name__ == "__main__":
    main()
