#!/usr/bin/env python
"""All channel subsets (2^5 = 32): the numbers that show why LOO is a lower bound.

## Why LOO alone is not enough

The channels overlap (measured on ESM2, function: ΔΔG with RSA 0.59, BLOSUM with GEMME 0.54,
GEMME with ΔΔG 0.51). Under overlap LOO measures only a channel's **unique contribution**, because
the remaining channels absorb the rest when the weights are refitted. Leave-one-in does the
opposite and credits the whole shared part to that channel, giving an upper bound. The Shapley
value fills the gap: the marginal contribution averaged over every insertion order.

  LOO            = the "added last" term of the Shapley value   (lower bound)
  leave-one-in   = the "added first" term                       (upper bound)
  Shapley        = the mean over every order in between

⚠ Shapley does not rescue every channel. Delta charge correlates at most 0.21 with the others, so
  there is no shared part to recover: LOO is already a fair measurement and Shapley will be
  similarly small. That is the channel's worth, not a measurement artefact.

## Convention

For each subset the **remaining slots are refitted on the external set** (the position/residual
split is kept). Grid uni3, coordinate descent, 3 passes, starting from zero. The empty subset is
pLM alone (w = 0) and is not fitted. Weights are fitted on the MaveDB external set only;
ProteinGym labels are used for evaluation only.

output  ablation_lattice.csv  (axis, bb, subset, k, rho), rho being the mean over ProteinGym DMS
run     OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES="" python scripts/ablation_lattice.py
"""
from __future__ import annotations
import itertools, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))   # ablation_12bb lives in this folder
from plmfusion import common as C, grids as P
from plmfusion.paths import RESULTS
import paper_numbers as PN                    # noqa: E402
from ablation_12bb import EXTBB, load_pg      # noqa: E402

H = RESULTS
H.mkdir(parents=True, exist_ok=True)

GRID = "uni3"
CH = {"GEMME": [0, 1], "ddG": [2, 3], "RSA": [4], "BLOSUM": [5], "dchg": [6]}
NAMES = list(CH)


def tie_for(sub):
    """Only the slots of the channels in subset sub are free; the rest are None (pinned to 0)."""
    tie, nxt = [None] * 7, 0
    for nm in NAMES:
        if nm in sub:
            for k in CH[nm]:
                tie[k] = nxt; nxt += 1
    return tie


def main():
    subs = [frozenset(c) for k in range(6) for c in itertools.combinations(NAMES, k)]
    rows, per, t0 = [], [], time.time()
    for axis in ("function", "stability"):
        for bb in EXTBB:
            E = PN.load_ext(EXTBB[bb], axis)
            if len(E) < 5:
                continue
            R = load_pg(bb, axis)
            for sub in subs:
                if sub:
                    with P.use_grid(GRID):
                        w = PN.descend_tied(E, list(range(len(E))), tie_for(sub))
                else:
                    w = np.zeros(7)
                key = "+".join(n for n in NAMES if n in sub) or "none"
                v = [r.rho(w) for r in R]
                rows.append(dict(axis=axis, bb=bb, subset=key, k=len(sub), rho=float(np.mean(v))))
                for r, x in zip(R, v):
                    per.append(dict(axis=axis, bb=bb, subset=key, dms=r.dms, rho=x))
            print(f"  {axis:10s} {bb:14s} 32 subsets  ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(H / "ablation_lattice.csv", index=False)
    pd.DataFrame(per).to_csv(H / "ablation_lattice_perdms.csv", index=False)
    print(f"\nwrote ablation_lattice.csv ({len(rows)} rows) and _perdms.csv ({len(per):,} rows)")


if __name__ == "__main__":
    main()
