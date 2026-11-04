#!/usr/bin/env python
"""Size-matched random control: is the axis gain due to the axis label or to how many assays
were drawn?

Because the matched arm is fitted per axis, its tuning set is smaller than the pooled one (49).
To see whether that size difference alone explains the gain, an arm is needed that draws the
**same number of assays out of the 49 while ignoring the axis** and then fits by the same
procedure. If that arm lands next to the pooled arm, the gain comes from the label and not from
the size.

Two things differ from the first version of this control:
  grid        wide -> **uni3** (the confirmed grid of the paper)
  evaluation  common.DROP was emptied on 2026-10-28, taking function from 112 to 129 and
              expression from 16 to 18, so the same code now sees the cohort used in the text.

⚠ A random subset does not carry zero axis information. The axis mix comes along by chance, so
  the matching axis fraction is recorded alongside and the result is read as a dose-response.

output  control_random_uni3.csv
run     OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="" python scripts/control_random_uni3.py
"""
from __future__ import annotations
import sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))   # paper_numbers lives in this folder
from plmfusion import common as C, grids as P
from plmfusion.external import ERec, EXT
from plmfusion.paths import RESULTS
import paper_numbers as PN                         # noqa: E402

# common.load filters by axis first and skips, with a warning, any assay that has no llr.
load_pg = C.load

# Figure 3b in the text uses 12 backbones, so the control uses 12 as well. With 4 backbones
#   n = 80, which has too little power and leaves the dose-response at p = 0.06 (this is a power
#   problem, not an effect-size one).
EXTBB = PN.EXTBB                           # the canonical 12-backbone mapping
BBS = list(EXTBB)
NSEED = 10
GRID = "uni3"


def load_ext(bb):
    keep = set(pd.read_csv(EXT / "direction_manual.csv").urn)
    out = []
    for f in sorted((EXT / "two_dim").glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        if str(d["urn"]) in keep and f"llr_{bb}" in d.files:
            out.append(ERec(f, bb))
    return out


def cell(arg):
    bb, evax = arg
    E = load_ext(EXTBB[bb])
    ax = np.array([r.axis for r in E])
    k = int((ax == evax).sum())             # the same size as the matched arm
    PG = load_pg(bb, evax)
    out = []
    for seed in range(NSEED):
        rng = np.random.default_rng(1000 + seed)
        idx = list(rng.choice(len(E), size=k, replace=False))
        with P.use_grid(GRID):
            w = C.descend(E, idx)
        rho = [r.rho(w) for r in PG]
        out.append(dict(bb=bb, eval_axis=evax, k=k, seed=seed,
                        frac_same_axis=float((ax[idx] == evax).mean()),
                        n_eval=len(PG), mean=float(np.mean(rho)),
                        median=float(np.median(rho)), w=list(np.round(w, 2))))
    return out


def main():
    jobs = [(bb, ax) for bb in BBS for ax in ("function", "stability")]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=12) as ex:
        res = list(ex.map(cell, jobs))
    R = pd.DataFrame([r for chunk in res for r in chunk])
    H = RESULTS
    H.mkdir(parents=True, exist_ok=True)
    R.to_csv(H / "control_random_uni3.csv", index=False)
    print(f"{len(R)} refits · {time.time()-t0:.0f}s")
    print(R.groupby(["eval_axis", "bb"]).agg(
        k=("k", "first"), n_eval=("n_eval", "first"),
        rand_mean=("mean", "mean"), rand_sd=("mean", "std")).to_string())


if __name__ == "__main__":
    main()
