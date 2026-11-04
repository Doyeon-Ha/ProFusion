#!/usr/bin/env python
"""Recompute panel d (the comparison of fusion methods). The in-text number and Fig. 2d read this.

## Why it runs separately

The chain (run_all.sh) recomputes only the ProteinGym evaluation set. Panel d is a **paired
comparison** against VenusREM and VenusREM2 on the same backbone, which needs that harness output,
so it used to live outside the chain. As a result it still carried the weights as of 2026-10-15
(function 1.3/1.0/0.0/0.5/0.7/0.3/0.3) hard-coded. It now **reads** the fitting output
(extend_backbones.csv).

## What is measured

    ProSST2048        our llr_ProSST (r = 1.0000 against the harness ProSST-2048 raw)
    ProSSTens_raw     the z-mean of 6 raw backbones. No fusion
    VenusREM_v1       the score from the original VenusREM repository (single backbone)
    v1ens             the v1 **recipe** (alpha 0.8) applied to the 6-model ensemble, rebuilt here
    vrh_ProSST2048    the VenusREM2 recipe applied to ProSST-2048
    VenusREM2         VenusREM2 (6-model ensemble)
    comp7_*           ProFusion, 7 slots, on the same backbone

All of them are within-DMS Spearman rho. Unitless, higher is better.

## The set

Only assays the VenusREM2 harness was run on are included. As of 2026-10-29 that is **128**, a
subset of the evaluation set (129 function + 18 expression). The remaining 19 have not been run
through the harness. One of them, Q6WV12_9MAXI_Somermeyer_2022, is absent from the harness
alignment set and would drop out anyway, so the ceiling is 146.

⚠ Once that job runs, this script **automatically** grows to 146, and the in-text number and the
  Fig. 2d caption must change with it. The figure then has to be redrawn (plot_fig1_ext_v3.py).
  Running only one of the two leaves the text and the figure disagreeing.

## What differs in the public version

The raw VenusREM2 harness scores are 152 MB and are not shipped with the public data. Three
derived files (paneld_perdms.csv, FINAL_perdms.csv, v1recipe_ens_perdms.csv) are frozen under
frozen/ instead, and this script reads those. There is therefore no step that walks the harness
CSVs to build D.

Both control checks survive. The original checked that rerunning the harness gave the same rho as
the stored copy. The public version applies the same rho-level check to the **public cache**: the
two single-backbone ProSST columns (ProSST2048 and comp7_ProSST2048) are recomputed on both axes
and matched against the frozen copy. Because that covers a column the weights enter, it is
guaranteed to catch the state where extend_backbones.csv has been refitted but the frozen copy has
not (that is, numbers going quietly stale).

run  OMP_NUM_THREADS=8 python scripts/paneld_numbers.py
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from plmfusion import common as C
from plmfusion.paths import RESULTS, frozen

H = OUT = RESULTS
# The harness raw-score folders are not in the public version, because frozen/paneld_perdms.csv is
# read instead of the raw scores.


def main():
    EB = pd.read_csv(H / "extend_backbones.csv")
    W = {ax: EB[(EB.bb == "ProSST") & (EB.axis == ax)][C.SLOTS].to_numpy(float)[0]
         for ax in ("function", "stability")}
    print("comp7 ProSST w:", {k: list(v) for k, v in W.items()})

    D = pd.read_csv(frozen("paneld_perdms.csv"))

    # -- Control reproduction check. The frozen copy must agree with the current public cache and fitted weights.
    ref = pd.read_csv(frozen("FINAL_perdms.csv")).set_index("dms")
    refe = pd.read_csv(frozen("v1recipe_ens_perdms.csv")).set_index("dms")
    # ⚠ Compared **at the level of rho**. The scores themselves differ by about 1e-5 depending on
    #   the card and precision the model ran on (the stored copy is CPU, a rerun may be GPU). What
    #   is used is the within-DMS Spearman, which depends only on the ranking, so this has no
    #   effect: the measured d_rho is exactly 0. Comparing the scores instead would flag a
    #   perfectly correct input configuration as wrong.
    worst_rho, n_ctrl = 0.0, 0
    for axis in ("function", "stability"):
        sub = D[D.ax == axis].set_index("dms")
        for r in C.load("ProSST", axis=axis):
            if r.dms not in sub.index:
                continue
            worst_rho = max(worst_rho,
                            abs(r.rho(np.zeros(7)) - float(sub.at[r.dms, "ProSST2048"])),
                            abs(r.rho(W[axis]) - float(sub.at[r.dms, "comp7_ProSST2048"])))
            n_ctrl += 1
        print(f"  control {axis}: max |d_rho| against the frozen copy {worst_rho:.2e} ({n_ctrl} DMS)")
    assert worst_rho < 1e-9, f"control rho reproduction failed at {worst_rho:.3e}: the input configuration or the fitted weights differ"

    # -- The **weight-independent** columns of the old 128 must not move by a single digit
    old = D[D.dms.isin(ref.index)].set_index("dms")
    for c in ("ProSST2048", "ProSSTens_raw", "vrh_ProSST2048", "VenusREM2", "VenusREM_v1"):
        m = float(np.abs(old[c] - ref.loc[old.index, c]).max())
        assert m < 1e-9, f"{c} differs from the stored copy ({m:.2e})"
    m = float(np.abs(old["v1ens"] - refe.loc[old.index, "v1ens"]).max())
    assert m < 1e-9, f"v1ens differs from the stored copy ({m:.2e})"
    print(f"  old 128 stored copy reproduced (6 weight-independent columns)")

    D.to_csv(OUT / "paneld_perdms.csv", index=False)

    def pair(a, b):
        return dict(n=int(len(D)), a=float(D[a].mean()), b=float(D[b].mean()),
                    d=float((D[a] - D[b]).mean()), win=int((D[a] > D[b]).sum()),
                    p=float(wilcoxon(D[a], D[b]).pvalue))

    J = dict(
        n=int(len(D)), n_function=int((D.ax == "function").sum()),
        n_expression=int((D.ax == "stability").sum()),
        w={k: list(map(float, v)) for k, v in W.items()},
        mean={c: float(D[c].mean()) for c in
              ["ProSST2048", "comp7_ProSST2048", "ProSSTens_raw", "comp7_ProSSTens",
               "vrh_ProSST2048", "VenusREM2", "VenusREM_v1", "v1ens"]},
        pf_vs_v1_single=pair("comp7_ProSST2048", "VenusREM_v1"),
        pf_vs_v2_ens=pair("comp7_ProSSTens", "VenusREM2"),
        v1_gain_single=float(D.VenusREM_v1.mean() - D.ProSST2048.mean()),
        pf_gain_single=float(D.comp7_ProSST2048.mean() - D.ProSST2048.mean()),
        cross=float(D.comp7_ProSST2048.mean() - D.VenusREM2.mean()),
    )
    json.dump(J, open(OUT / "paneld_numbers.json", "w"), indent=2)
    print(f"\n=== panel d, n = {J['n']} (function {J['n_function']} + expression {J['n_expression']}) ===")
    for k, v in J["mean"].items():
        print(f"  {k:18s} {v:.4f}")
    for k in ("pf_vs_v1_single", "pf_vs_v2_ens"):
        s = J[k]
        print(f"  {k}: {s['a']:.4f} vs {s['b']:.4f}  win {s['win']}/{s['n']}  p={s['p']:.2e}")
    print(f"  VenusREM single gain +{J['v1_gain_single']:.4f}, ProFusion single gain +{J['pf_gain_single']:.4f}")
    print(f"  cross (single ProFusion minus ensemble VenusREM2) {J['cross']:+.4f}")
    print(f"\nwrote {OUT/'paneld_perdms.csv'} and {OUT/'paneld_numbers.json'}")


if __name__ == "__main__":
    main()
