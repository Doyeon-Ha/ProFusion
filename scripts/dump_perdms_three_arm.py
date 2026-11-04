#!/usr/bin/env python
"""Dump **per-DMS** rho over the 65 ProteinGym Stability assays for three arms (for the figure).

  plm   pLM alone (all weights 0)
  eq    the unfitted baseline: 1 on the GEMME position and residual slots and on the RSA position
        slot, 0 elsewhere
  tsu   weights fitted on tsu-all11, the 115 Tsuboyama natural domains that share no protein with
        ProteinGym

The weights are read back from what stability_protocol already chose; nothing is refitted here.
ESM-IF1, ProteinMPNN and SaProt650M use the variant whose fitting set was rescored with the
**official ProteinGym formula** (`_off`). For the others the two variants give the same values.

output  perdms_three_arm.csv  (bb, dms, prot, plm, eq, tsu)
run     OMP_NUM_THREADS=8 PYTHONPATH=src python scripts/dump_perdms_three_arm.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from plmfusion import common as C
from plmfusion.paths import RESULTS

SLOT = ["a_GEMME", "b_GEMME", "a_ddG", "b_ddG", "a_RSA", "b_BLOSUM", "b_dchg"]
FREE5 = [0, 1, 4, 5, 6]                      # the two ΔΔG slots are pinned to 0 (SPURS leakage)
EQ = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
FIX = {"ESMIF1", "ProteinMPNN", "SaProt650M"}   # backbones that use the corrected convention
ARM = "ext tsu129_all11"


def main() -> None:
    pre = pd.read_csv(RESULTS / "stability_protocol_pre.csv")
    off = pd.read_csv(RESULTS / "stability_protocol_off.csv")
    rows = []
    for bb in sorted(set(pre.bb)):
        T = off if bb in FIX else pre
        r = T[(T.axis == "stability_ddg") & (T.bb == bb) & (T.arm == ARM)]
        if not len(r):
            print(f"[skipped] {bb}: no {ARM} row")
            continue
        w = [0.0] * 7
        for k in FREE5:
            w[k] = float(r[SLOT[k]].iloc[0])
        P = C.load(bb, axis="stability_ddg")
        for x in P:
            rows.append(dict(bb=bb, dms=x.dms, prot=x.prot,
                             plm=x.rho([0.0] * 7), eq=x.rho(EQ), tsu=x.rho(w)))
        print(f"{bb:16s} {len(P):3d} DMS · w {[w[k] for k in FREE5]}"
              f"  {'official formula' if bb in FIX else 'single formula'}")
    D = pd.DataFrame(rows)
    f = RESULTS / "perdms_three_arm.csv"
    D.to_csv(f, index=False)
    print(f"\n[saved] {f}  ({len(D)} rows, {D.bb.nunique()} backbones)")


if __name__ == "__main__":
    main()
