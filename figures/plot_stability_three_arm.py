#!/usr/bin/env python
"""ProteinGym Stability 65, 13 backbones x 3 arms: the distribution over assays.

The conventions match the legacy zero-shot versus fusion figure (box = quartiles, whiskers =
1.5 IQR, outliers hidden, grey point = one assay).

Metric definitions
  rho        Spearman between the score and the measurement within one DMS (assay). Unitless,
             higher is better. One box is the distribution over 65 assays, and every point in the
             figure is one assay.
             ⚠ ProteinGym Stability has 66 assays but only 65 are used.
             PSAE_PICP2_Tsuboyama_2023_1PSE has neither ProSST structure tokens nor GEMME output
             distributed by ProteinGym, so its channels cannot be built.
  pLM alone  the backbone score with no fusion. 12 backbones, and ESM2 appears in **one variant
             only, masked-marginal** (the official ProteinGym ESM baseline convention). The
             wildtype-marginal variant is not drawn.
  ProFusion  fusion with weights fitted on the 115 Tsuboyama natural domains that share no protein
             with ProteinGym (tsu-all11). The two ΔΔG slots are pinned to 0, because SPURS was
             trained on this data and would leak.
  stars      Wilcoxon paired on the same assay (two-sided). *** p<0.001, ** p<0.01, * p<0.05, ns.
             ProFusion against pLM alone.

The unfitted EQ baseline is computed (and kept in the data csv) but not drawn.

ESM-IF1, ProteinMPNN and SaProt650M use the values whose fitting set was rescored with the
official ProteinGym formula.

run  python figures/plot_stability_three_arm.py
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plmfusion.paths import RESULTS  # noqa: E402

# ---------------------------------------------------------------- style
PALETTE = {"main": "#1E90FF", "alt": "#B64342", "baseline": "#CFCECE",   # main = dodgerblue
           "extra1": "#42949E", "extra2": "#9A4D8E", "gray_line": "#767676"}


@dataclass(frozen=True)
class FigureStyle:
    font_size: float = 7.0
    tick_size: float = 6.0
    axes_linewidth: float = 0.8
    font_family: tuple[str, ...] = ("Source Sans Pro", "Helvetica", "Arial", "DejaVu Sans")


def apply_publication_style(style: FigureStyle = FigureStyle()) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": list(style.font_family),
        "font.size": style.font_size, "axes.labelsize": style.font_size,
        "xtick.labelsize": style.tick_size, "ytick.labelsize": style.tick_size,
        "legend.fontsize": style.tick_size + 0.5, "legend.frameon": False,
        "axes.linewidth": style.axes_linewidth, "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": style.axes_linewidth, "ytick.major.width": style.axes_linewidth,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        # ⚠ Without matching the math font, only the italic p falls back to DejaVu and the
        #   figure mixes typefaces. Greek letters exist in Source Sans Pro, so they are written
        #   directly as Unicode.
        "mathtext.fontset": "custom", "mathtext.rm": "Source Sans Pro",
        "mathtext.it": "Source Sans Pro:italic", "mathtext.bf": "Source Sans Pro:bold",
    })


SRC = RESULTS / "perdms_three_arm.csv"
OUT = RESULTS / "figures"
# The labels match the main 12-backbone figure.
# ⚠ ESM2 appears in **one variant only, masked-marginal**. The official ProteinGym ESM baseline
#   uses masked-marginal, and our wildtype-marginal variant is a different quantity (rho = 0.9847
#   against the official one). Showing two conventions under one name would turn a comparison of
#   backbones into a comparison of conventions.
NAME = {"AIDO_RAG16B": "AIDO RAG 16B", "VenusREM": "VenusREM", "ProSST": "ProSST",
        "VespaG": "VespaG", "TranceptionLret": "Tranception L (ret)", "S2F": "S2F",
        "ESM1v": "ESM-1v", "SaProt650M": "SaProt 650M", "ESM3": "ESM3-open",
        "ESM2mask": "ESM2 650M", "ESMIF1": "ESM-IF1", "ProteinMPNN": "ProteinMPNN"}
DROP_BB = {"ESM2"}      # the wildtype-marginal variant. See the comment above.
ARMS = [("plm", "pLM alone", PALETTE["baseline"]),
        ("tsu", "ProFusion", PALETTE["main"])]


def stars(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def main() -> None:
    apply_publication_style()
    D = pd.read_csv(SRC)
    D = D[~D.bb.isin(DROP_BB)]
    missing = sorted(set(D.bb) - set(NAME))
    assert not missing, f"backbones missing from NAME: {missing}"
    # Backbones are ordered by descending tsu-all11 median (the same convention as the manuscript figures).
    med = D.groupby("bb")["tsu"].median().sort_values(ascending=False)
    bbs = list(med.index)
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    fig.subplots_adjust(left=0.072, right=0.995, top=0.885, bottom=0.235)

    pos, data, colors, rows = [], [], [], []
    for i, bb in enumerate(bbs):
        g = D[D.bb == bb].sort_values("dms")
        for j, (k, _lab, col) in enumerate(ARMS):
            pos.append(i + (j - 0.5) * 0.38)
            data.append(g[k].to_numpy())
            colors.append(col)
        p_plm = wilcoxon(g.tsu, g.plm).pvalue
        p_eq = wilcoxon(g.tsu, g["eq"]).pvalue
        rows.append(dict(bb=bb, label=NAME[bb], n=len(g),
                         plm_med=g.plm.median(), eq_med=g["eq"].median(), tsu_med=g.tsu.median(),
                         delta_vs_plm=float((g.tsu - g.plm).mean()),
                         delta_vs_eq=float((g.tsu - g["eq"]).mean()),
                         p_vs_plm=p_plm, sig_vs_plm=stars(p_plm),
                         p_vs_eq=p_eq, sig_vs_eq=stars(p_eq)))

    bp = ax.boxplot(data, positions=pos, widths=0.33, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=0.8),
                    whiskerprops=dict(linewidth=0.6), capprops=dict(linewidth=0.6),
                    boxprops=dict(linewidth=0.5), zorder=2)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor("black")
    # One point per assay, drawn over the box so the tail the box hides stays visible.
    for x, v in zip(pos, data):
        ax.scatter(x + rng.uniform(-0.105, 0.105, len(v)), v, s=1.6,
                   color="#555555", alpha=0.45, linewidths=0, zorder=3)

    ax.axhline(0, color=PALETTE["gray_line"], linewidth=0.6, linestyle=":", zorder=0)
    top = max(np.max(d) for d in data)
    bot = min(np.min(d) for d in data)
    span = top - bot
    ax.set_ylim(bot - 0.06 * span, top + 0.17 * span)
    for i, r in enumerate(rows):
        ax.text(i, top + 0.05 * span, r["sig_vs_plm"], ha="center", va="bottom",
                fontsize=FigureStyle.tick_size, color="black")

    ax.set_xticks(range(len(bbs)))
    ax.set_xticklabels([NAME[b] for b in bbs], rotation=35, ha="right")
    ax.set_ylabel("Spearman ρ per assay")
    ax.set_xlim(-0.62, len(bbs) - 0.38)
    ax.set_title(f"Stability eval (n={rows[0]['n']})", loc="left",
                 fontsize=FigureStyle.font_size, fontweight="bold", pad=6)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", linewidth=0.5)
               for _k, _l, c in ARMS]
    # ⚠ Placed at the lower left inside the panel it would sit on top of the ESM-IF1 point for
    #   A4_HUMAN_Seuma_2022 (rho = −0.118). It goes above the axes, opposite the title.
    ax.legend(handles, [l for _k, l, _c in ARMS], loc="lower right",
              bbox_to_anchor=(1.0, 1.0), ncol=2,
              handlelength=1.2, handleheight=0.9, columnspacing=1.0, borderaxespad=0.1)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "stability_13bb_three_arm"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    pd.DataFrame(rows).to_csv(OUT / "stability_13bb_three_arm_data.csv", index=False)
    print(f"[saved] {stem}.pdf · .png · _data.csv")
    R = pd.DataFrame(rows)
    print(f"  tsu vs pLM  significant {(R.p_vs_plm < 0.05).sum()}/{len(R)}")
    print(f"  tsu vs EQ   significant {(R.p_vs_eq < 0.05).sum()}/{len(R)}"
          f"  (ns: {', '.join(R[R.p_vs_eq >= 0.05].label)})")


if __name__ == "__main__":
    main()
