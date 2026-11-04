#!/usr/bin/env python
"""The internal-tuning figure: choosing the weights inside ProteinGym changes almost nothing.

Three boxes per backbone: pLM alone, ProFusion fitted on MaveDB, and ProFusion fitted on
ProteinGym.

## What is measured

  rho   Spearman between the fused score and the label within one DMS. Unitless, [-1, 1], higher
        is better. Each box is the distribution over that backbone's 118 evaluation assays. The
        centre line is the median, the box the quartiles, the whiskers 1.5 IQR. Individual points
        are not drawn: the main figure jitters per-assay points under 24 boxes, but there are 36
        here and at the same width they read as speckle.

The evaluation set is the 118 function assays left after removing the 11 tuning-cohort assays.
Expression has only 7, which is not a distribution worth drawing, so it appears as a number in the
text only.

⚠ The width of 6.5 in matches \textwidth in the supplementary (letter, 1 in margins), so the scale
  is 1.0. Exporting at 7.2 in makes LaTeX shrink it to 0.90 and 6 pt text lands at 5.4 pt.

input   results/internal_tune_12bb_perdms.csv  (produced by scripts/internal_tune_12bb.py)
output  results/figures/internal_tune.{pdf,png} and internal_tune_data.csv
        ⚠ The paper includes this PDF copied to Figures/fig_pg_tuned.pdf.
run     CUDA_VISIBLE_DEVICES="" python figures/plot_internal_tune.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from scipy.stats import wilcoxon         # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plmfusion.paths import RESULTS      # noqa: E402

H = RESULTS / "figures"
AXIS = "function"
# Colours and box conventions match the main figure, where pLM alone is baseline grey and
# ProFusion is dodgerblue. There are three arms here, so the two ProFusion arms share one cool
# family and dodgerblue goes to the model of the paper (fitted on MaveDB). Their lightness values
# are 82 / 60 / 33, so they still separate in greyscale.
ARMS = [("base", "pLM alone", "#CFCECE"),
        ("ext", "ProFusion (MaveDB-tuned)", "#1E90FF"),
        ("int", "ProFusion (ProteinGym-tuned)", "#0F4D92")]
NAME = {"AIDO_RAG16B": "AIDO-RAG 16B", "VenusREM": "VenusREM", "ProSST": "ProSST",
        "S2F": "S2F-MSA", "SaProt650M": "SaProt 650M", "ESM3": "ESM3-open",
        "ESMIF1": "ESM-IF1", "ProteinMPNN": "ProteinMPNN", "VespaG": "VespaG",
        "TranceptionLret": "Tranception L", "ESM1v": "ESM-1v", "ESM2": "ESM2 650M"}


def bh(ps):
    """Benjamini-Hochberg, grouped within one figure."""
    p = np.asarray(ps, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); run = 1.0
    for i in range(n - 1, -1, -1):
        run = min(run, p[o[i]] * n / (i + 1)); q[o[i]] = run
    return q


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Source Sans Pro", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.0, "axes.labelsize": 7.0, "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0, "legend.fontsize": 6.5, "legend.frameon": False,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "pdf.fonttype": 42, "ps.fonttype": 42})


def table(P):
    """The numbers the figure drew. The boxes show distributions, but the text quotes the mean, so both are written."""
    rows = []
    for ax in sorted(P.axis.unique()):
        d0 = P[(P.axis == ax) & (P.scope == "eval")]
        for bb in d0.bb.unique():
            d = d0[d0.bb == bb]
            dd = (d["int"] - d["ext"]).to_numpy()
            r = dict(axis=ax, bb=bb, name=NAME[bb], n=len(d),
                     gain_ext=(d.ext - d.base).mean(), d_int_ext=dd.mean(),
                     p=wilcoxon(d["int"], d["ext"]).pvalue if np.any(dd != 0) else 1.0)
            for a, _, _ in ARMS:
                r[f"mean_{a}"], r[f"median_{a}"] = d[a].mean(), d[a].median()
            rows.append(r)
    S = pd.DataFrame(rows)
    S["q"] = np.concatenate([bh(S[S.axis == a].p) for a in sorted(S.axis.unique())])
    return S


def main():
    style()
    P = pd.read_csv(RESULTS / "internal_tune_12bb_perdms.csv")
    S = table(P)
    H.mkdir(parents=True, exist_ok=True)
    S.to_csv(H / "internal_tune_data.csv", index=False)

    D = P[(P.axis == AXIS) & (P.scope == "eval")]
    fn = S[S.axis == AXIS].sort_values("median_ext", ascending=False)
    order = list(fn.bb)
    x = np.arange(len(order))
    off = np.array([-0.26, 0.0, 0.26])

    fig = plt.figure(figsize=(6.5, 2.85))
    ax = fig.add_subplot(111)
    for (arm, lab, col), o in zip(ARMS, off):
        vals = [D[D.bb == b][arm].to_numpy() for b in order]
        bp = ax.boxplot(vals, positions=x + o, widths=0.23, showfliers=False,
                        patch_artist=True, zorder=4,
                        medianprops=dict(color="black", lw=0.8),
                        whiskerprops=dict(lw=0.55), capprops=dict(lw=0.55),
                        boxprops=dict(lw=0.45, edgecolor="black"))
        for box in bp["boxes"]:
            box.set_facecolor(col)                 # opaque. The main figure uses no transparency either
        ax.add_patch(plt.Rectangle((0, 0), 0, 0, fc=col, ec="black", lw=0.45,
                                   label=lab))

    ax.set_xticks(x)
    ax.set_xticklabels([NAME[b] for b in order], rotation=38, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("Spearman ρ per assay")
    ax.set_xlim(-0.62, len(order) - 0.38)
    ax.set_ylim(-0.15, 0.90)        # so the lowest whisker (-0.1198, ESM2) is not clipped
    # The legend goes **above the axes**, on one line. Inside, it would hide the whiskers below 0.
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, handlelength=0.9,
              handletextpad=0.4, columnspacing=1.4, borderpad=0.2)
    fig.subplots_adjust(left=0.072, right=0.995, top=0.915, bottom=0.235)
    for fmt in ("pdf", "png"):
        fig.savefig(H / f"internal_tune.{fmt}", dpi=300)
    plt.close(fig)
    print("wrote:", H / "internal_tune.pdf")
    print(fn[["name", "n", "median_base", "median_ext", "median_int",
              "mean_ext", "mean_int", "d_int_ext", "q"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
