#!/usr/bin/env python
"""Build the input CSVs for the Figure 5 case study (B2L11 Gly156).

## Why this is rewritten

`case_positions_scored.csv` and `case_study_variants.csv` were produced ad hoc on 2026-10-06 and
**no script for them existed in the repository**. They carry the fusion scores **precomputed** in
the `champ5` and `comp7` columns, so the figure would keep drawing old values after the weights
changed. Concretely, champ5 = −0.2074 at position 156 is a value only the old WIDE grid (4.0, 2.2)
produces; with the current fit (3.0, 1.6) it is −0.23.

This script reads the fitting output (`ablation_12bb_w.csv`) and rebuilds them every time.

## Column convention (recovered from the old files)

  positions
    pos        position number
    gem, ddg   the **position components** of GEMME and ΔΔG at that position (columns 0 and 2 of
               CH in common.Rec)
    lab        mean label percentile of the variants at that position
    n          number of variants measured at that position
    champ5     the position-channel contribution of FW5  = a_GEMME*gem + a_ddG*ddg
    comp7      the same for SFW7, with its separately fitted weights
    *_rank     rank within the 10 positions. 1 = most damaging (smaller means more damaging).

  variants
    label_pct, champ5_pct, comp7_pct   percentile within the assay (smaller is more damaging)
    ddg_pos/res, gem_pos/res, llr      the components and the pLM score

output  case_positions_scored.csv and case_study_variants.csv
run     CUDA_VISIBLE_DEVICES="" python scripts/build_case156.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import rankdata

from plmfusion import common as C
from plmfusion.paths import RESULTS

DMS = "B2L11_HUMAN_Dutta_2010_binding-Mcl-1"
BB = "ESM2"
AA20 = "ACDEFGHIKLMNPQRSTVWY"
SL = C.SLOTS


def pct(v):
    """Percentile under the convention that smaller is more damaging (0 to 100)."""
    return (rankdata(v) - 1) / (len(v) - 1) * 100


def main():
    W = pd.read_csv(RESULTS / "ablation_12bb_w.csv")
    w = {}
    for arm, key in (("FW5", "champ5"), ("SFW7", "comp7")):
        r = W[(W.axis == "function") & (W.bb == BB) & (W.arm == arm)]
        assert len(r) == 1, f"{arm}: {len(r)} rows"
        w[key] = np.array([float(r.iloc[0][s]) for s in SL])
        assert np.abs(w[key]).max() <= 3.0 + 1e-9, f"{arm} lies outside the grid"

    rec = C.Rec(C.CACHE / f"{DMS}.npz", BB)
    z = np.load(C.CACHE / f"{DMS}.npz", allow_pickle=True)
    pos = np.asarray(z["pos1"], int)
    mut = [f"{AA20[a]}{p}{AA20[b]}" for a, p, b in zip(z["wt_i"], z["pos1"], z["mut_i"])]
    gem_pos, gem_res = rec.CH[:, 0], rec.CH[:, 1]
    ddg_pos, ddg_res = rec.CH[:, 2], rec.CH[:, 3]
    llr = rec.llr

    V = pd.DataFrame(dict(variant=mut, pos=pos, label_pct=pct(rec.y),
                          ddg_pos=ddg_pos, ddg_res=ddg_res,
                          gem_pos=gem_pos, gem_res=gem_res, llr=llr))
    for key in ("champ5", "comp7"):
        V[f"{key}_pct"] = pct(rec.s(w[key]))
    V["improve"] = V.comp7_pct - V.champ5_pct
    V = V[["variant", "pos", "label_pct", "champ5_pct", "comp7_pct", "improve",
           "ddg_pos", "ddg_res", "gem_pos", "gem_res", "llr"]]
    RESULTS.mkdir(parents=True, exist_ok=True)
    V.to_csv(RESULTS / "case_study_variants.csv", index=False)

    up = np.unique(pos)
    rows = []
    for p in up:
        m = pos == p
        rows.append(dict(pos=int(p), gem=float(gem_pos[m][0]), ddg=float(ddg_pos[m][0]),
                         lab=float(V.label_pct[m].mean()), n=int(m.sum())))
    S = pd.DataFrame(rows)
    for key in ("champ5", "comp7"):
        S[key] = w[key][0] * S.gem + w[key][2] * S.ddg
    S["truth_rank"] = rankdata(S.lab)              # smaller is more damaging, so rank 1 is worst
    for key in ("champ5", "comp7"):
        S[f"{key}_rank"] = rankdata(S[key])
    S = S[["pos", "gem", "ddg", "lab", "n", "champ5", "comp7",
           "truth_rank", "champ5_rank", "comp7_rank"]]
    S.to_csv(RESULTS / "case_positions_scored.csv", index=False)

    print(f"weights  FW5 a_GEMME {w['champ5'][0]:.1f} a_ddG {w['champ5'][2]:.1f}  |  "
          f"SFW7 a_GEMME {w['comp7'][0]:.1f} a_ddG {w['comp7'][2]:.1f}")
    print(f"{len(S)} positions, {len(V)} variants")
    f156 = S[S.pos == 156].iloc[0]
    print(f"Gly156  gem {f156.gem:+.4f} ddg {f156.ddg:+.4f}  "
          f"champ5 {f156.champ5:+.4f} (rank {f156.champ5_rank:.0f})  "
          f"comp7 {f156.comp7:+.4f} (rank {f156.comp7_rank:.0f})  truth {f156.truth_rank:.0f}")
    print("wrote case_positions_scored.csv and case_study_variants.csv")


if __name__ == "__main__":
    main()
