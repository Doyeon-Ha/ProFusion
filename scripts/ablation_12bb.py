#!/usr/bin/env python
"""Component ablations over 12 backbones and 2 axes. Three questions, fitted in one pass.

## The three questions

  A1 channels   how much is lost by removing one of the 5 channels (leave-one-out)
  A2 weights    is equal weighting (no fitting) enough, and how much does fitting buy
  A3 position/residual  is it worth keeping a and b separate

## Control conventions (do not change these)

Weights are fitted on the **MaveDB external set only**, and ProteinGym labels are used for
evaluation only. The ablation and merge arms remove or tie slots and then **refit the remaining
weights on the external set**. Without the refit, "information that disappeared" and "weights left
at the wrong scale" would be mixed together. The grid is uni3 (all slots −3.0 to 3.0 in steps of
0.1), coordinate descent, 3 passes, starting from zero, fitted separately per axis.

## The arms

  pLM        w = 0. No auxiliary channels.
  EQ3 / EQ5  **unfitted**. Every slot in use is 1, every other slot is 0.
             ⚠ The script that produced the EQ5 value (0.4836) in the older ladder CSV is not in
             the repository. Here a single rule is used, "every slot in use is 1". Under that
             definition EQ3 matches the stored 0.4942 and EQ5 matches the stored EQ5_slot 0.4888
             exactly.
  FW3/FW5    fitted, but with position tied to residual. FW3 also drops the substitution channels
             (BLOSUM and Delta charge).
  SFW7       all 7 slots fitted separately. This is ProFusion.
  merge X    channel X has position tied to residual, the rest stay separate. merge both == FW5.
  −X         every slot of channel X pinned to 0, the rest refitted.

## What is measured

  rho   Spearman between the fused score and the experimental label within one DMS. Unitless,
        [-1, 1], higher is better. Stored per DMS, so any paired comparison can be built later.

output  ablation_12bb_perdms.csv (axis, bb, arm, dms, rho) and ablation_12bb_w.csv (fitted weights)
run     OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES="" python scripts/ablation_12bb.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))   # paper_numbers lives in this folder
from plmfusion import common as C, grids as P
from plmfusion.paths import RESULTS
import paper_numbers as PN                    # noqa: E402

H = RESULTS
H.mkdir(parents=True, exist_ok=True)

GRID = "uni3"
SL = PN.SL
AXES = ["function", "stability"]
# ProteinGym npz key <-> external npz key (the main 12 of extend_backbones.EXTBB)
EXTBB = {"ESM2": "ESM2", "ESM1v": "ESM1v", "ESM3": "ESM3mm", "ProSST": "ProSST",
         "ESMIF1": "ESMIF1", "ProteinMPNN": "ProteinMPNN",
         "TranceptionLret": "TranceptionLret", "S2F": "S2F", "VespaG": "VespaG",
         "VenusREM": "VenusREM", "SaProt650M": "SaProt650M_own",
         "AIDO_RAG16B": "AIDO_RAG16B"}

# tie[k] = which free parameter that slot uses. None pins it to 0.
FITTED = {
    # A2 ladder
    "FW3":          [0, 0, 1, 1, 2, None, None],
    "FW5":          [0, 0, 1, 1, 2, 3, 4],
    "SFW7":         [0, 1, 2, 3, 4, 5, 6],
    # A3 position/residual. merge both is the same model as FW5, so it is not fitted twice
    "merge GEMME":  [0, 0, 1, 2, 3, 4, 5],
    "merge ddG":    [0, 1, 2, 2, 3, 4, 5],
    # A1 channel leave-one-out
    "−GEMME":       [None, None, 0, 1, 2, 3, 4],
    "−ddG":         [0, 1, None, None, 2, 3, 4],
    "−RSA":         [0, 1, 2, 3, None, 4, 5],
    "−BLOSUM":      [0, 1, 2, 3, 4, None, 5],
    "−dchg":        [0, 1, 2, 3, 4, 5, None],
}
FIXED = {                      # unfitted arms. Every slot in use is 1.
    "pLM": np.zeros(7),
    "EQ3": np.array([1., 1., 1., 1., 1., 0., 0.]),
    "EQ5": np.ones(7),
}


def load_pg(bb, axis):
    """C.load builds a Rec from every npz before filtering by axis, so a single npz without that
    backbone's key kills it with KeyError. Temporarily push the missing assays into DROP."""
    miss = {f.stem for f in sorted(C.CACHE.glob("*.npz"))
            if f"llr_{bb}" not in np.load(f, allow_pickle=True).files}
    old = set(C.DROP); C.DROP |= miss
    try:
        return C.load(bb, axis)
    finally:
        C.DROP.clear(); C.DROP |= old


def main():
    rows, wrows, t0 = [], [], time.time()
    for axis in AXES:
        for bb in EXTBB:
            E = PN.load_ext(EXTBB[bb], axis)
            if len(E) < 5:
                print(f"  {axis}/{bb}: only {len(E)} external assays, skipped"); continue
            R = load_pg(bb, axis)
            W = dict(FIXED)
            with P.use_grid(GRID):
                for nm, tie in FITTED.items():
                    W[nm] = PN.descend_tied(E, list(range(len(E))), tie)
            W["merge both"] = W["FW5"]          # the same model under two names
            for nm, w in W.items():
                for r in R:
                    rows.append(dict(axis=axis, bb=bb, arm=nm, dms=r.dms, rho=r.rho(w)))
                wrows.append(dict(axis=axis, bb=bb, arm=nm, total=float(np.sum(w)),
                                  **{s: w[k] for k, s in enumerate(SL)}))
            print(f"  {axis:10s} {bb:14s} ext={len(E):2d} pg={len(R):3d}  "
                  f"pLM {np.mean([r.rho(W['pLM']) for r in R]):.4f} -> "
                  f"SFW7 {np.mean([r.rho(W['SFW7']) for r in R]):.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)
    pd.DataFrame(rows).to_csv(H / "ablation_12bb_perdms.csv", index=False)
    pd.DataFrame(wrows).to_csv(H / "ablation_12bb_w.csv", index=False)
    print(f"\nwrote ablation_12bb_perdms.csv ({len(rows):,} rows) and ablation_12bb_w.csv")


if __name__ == "__main__":
    main()
