#!/usr/bin/env python
"""The criterion figure for deciding whether to split a channel into position and residual.

a: per-channel eta-squared against the wild-type amino acid, the fraction of the position
   component explained by the wild-type residue alone. The labels on the right say how each
   channel is actually used (position + residual, position only, residual only).
   ⚠ The labels avoid a and b: a reader seeing only the figure cannot tell which is position and
     which is residual.
   ⚠ No threshold such as 0.7 is drawn. It would read as a boundary chosen after the fact, and the
     actual choice was made from the nature of the channel (substitution-type or position-type),
     not from one threshold.
b: where that lookup table comes from. As substitution coverage grows (measured substitutions per
   position, at most 19), the eta-squared of a substitution-type channel approaches 1. A
   position-type channel (GEMME) does not behave that way.
   ⚠ The x distribution is bimodal (nothing between 7 and 13 among the 49 assays). Do not read the
     fitted slope as a continuous trend.

data: results/fig_data/panelb_eta2_verified.csv (49 DMS, cross-checked with three independent implementations)
run:  python figures/plot_eta2_criterion.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402
from scipy.stats import spearmanr                                  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plmfusion.paths import RESULTS                               # noqa: E402

FIG_DATA = RESULTS / "fig_data"      # produced by figures/data/eta2.py
OUTDIR = RESULTS / "figures"
P = {"main": "#0F4D92", "alt": "#B64342", "mid": "#D4897F",
     "extra1": "#42949E", "gray": "#767676"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Source Sans Pro", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.6, "axes.labelsize": 8.6, "xtick.labelsize": 7.6, "ytick.labelsize": 7.6,
    "legend.fontsize": 8.0, "legend.frameon": False, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
})


def panel_a(ax, Q):
    order = ["GEMME", "RSA", "ddG", "BLOSUM", "Δcharge"]
    # The labels avoid a and b: a reader seeing only the figure cannot tell which is which.
    USE = {"GEMME": ("position + residual", P["main"]),
           "ddG": ("position + residual", P["main"]),
           "RSA": ("position only\n(constant within a position)", P["extra1"]),
           "BLOSUM": ("residual only", P["alt"]),
           "Δcharge": ("residual only", P["alt"])}
    rng = np.random.default_rng(1)
    for i, c in enumerate(order):
        v = Q[Q.channel == c].e_zpos.to_numpy()
        med, q1, q3 = np.median(v), np.quantile(v, .25), np.quantile(v, .75)
        ax.barh(i, med, height=.6, color=USE[c][1], edgecolor="white", lw=.5, zorder=2)
        ax.scatter(v, i + rng.uniform(-.17, .17, len(v)), s=1.8, color="#2B2B2B",
                   alpha=.35, lw=0, zorder=3)
        ax.hlines(i, q1, q3, color="#1A1A1A", lw=1.0, zorder=4)
        ax.vlines([q1, q3], i - .13, i + .13, color="#1A1A1A", lw=1.0, zorder=4)
        ax.text(1.045, i, USE[c][0], va="center", fontsize=7.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(["ΔΔG" if c == "ddG" else c for c in order])
    ax.invert_yaxis(); ax.set_xlim(0, 1.92); ax.set_xticks([0, .25, .5, .75, 1.0])
    ax.set_xlabel(r"$\eta^2$ of the position component explained by the WT amino acid", labelpad=2)
    ax.set_title("a   Wild-type dependence by channel",
                 loc="left", fontsize=9.2, fontweight="bold", pad=5)


def panel_b(axes, Q):
    sat = Q.groupby("dms").nsub.first()
    spec = [("Δcharge", "Δcharge", P["alt"], "s"), ("BLOSUM", "BLOSUM", P["mid"], "^"),
            ("GEMME", "GEMME", P["main"], "o")]
    for ax, (c, lab, col, mk) in zip(axes, spec):
        q = Q[Q.channel == c].merge(sat.rename("sat"), left_on="dms", right_index=True)
        x, y = q.sat.to_numpy(), q.e_zpos.to_numpy()
        ax.scatter(x, y, s=5.5, color=col, marker=mk, lw=0, alpha=.8, zorder=3)
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(0, 20.5, 200)
        ys = np.clip(a + b * xs, 0, 1)          # eta-squared is in [0, 1], so the line is clipped too
        ax.plot(xs, ys, color=col, lw=1.1, zorder=4)
        r = spearmanr(x, y).statistic
        ax.text(.015, 1.0, f"{lab}  (ρ = {r:+.2f})", transform=ax.transAxes,
                fontsize=8.2, fontweight="bold", color=col, va="top")
        ax.set_ylim(-.10, 1.52); ax.set_yticks([0, .5, 1.0]); ax.set_xlim(0, 20.8)
        ax.set_ylabel(r"$\eta^2$", labelpad=1)
        if ax is axes[-1]:
            ax.set_xlabel("substitution coverage (of 19 possible substitutions)", labelpad=2)
            ax.set_xticks([0, 5, 10, 15, 20])
        else:
            ax.set_xticks([0, 5, 10, 15, 20]); ax.set_xticklabels([])
            ax.spines["bottom"].set_visible(False); ax.tick_params(bottom=False)
    axes[0].set_title("b   Effect of substitution coverage",
                      loc="left", fontsize=9.2, fontweight="bold", pad=5)


def main() -> None:
    Q = pd.read_csv(FIG_DATA / "panelb_eta2_verified.csv")
    fig = plt.figure(figsize=(7.2, 3.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0],
                          left=.095, right=.988, top=.885, bottom=.225, wspace=.26)
    panel_a(fig.add_subplot(gs[0, 0]), Q)
    gb = gs[0, 1].subgridspec(3, 1, hspace=.30)
    panel_b([fig.add_subplot(gb[i]) for i in range(3)], Q)
    fig.text(.095, .035,
             "49 external tuning assays. a: bar = median, whisker = IQR, dots = assays.  "
             "b: lines are least-squares fits clipped to [0, 1].",
             fontsize=7.0, color=P["gray"])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"eta2_criterion.{ext}", dpi=300)
    print(f"[saved] {OUTDIR}/eta2_criterion.pdf / .png")


if __name__ == "__main__":
    main()
