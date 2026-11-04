#!/usr/bin/env python
"""The comp7 design figure: (a) the position/residual split of a real channel, (b) how separable
it is, judged by eta-squared against the wild-type amino acid, (c) the weight allocation per
backbone.

Data
  a: the GEMME channel of one external npz (SGCB, 19.8 substitutions per position)
  c: panelc_comp7_weights.csv, the comp7 rows of paper_weights.csv (fitted on the MaveDB external set)
run: python figures/plot_comp7_design.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from scipy.stats import spearmanr                                  # noqa: E402

from plmfusion.paths import RESULTS, ext_cache
DATA_IN = RESULTS / "fig_data"  # the inputs built by figures/data/*.py
HERE = RESULTS / "figures"  # where the figure goes
HERE.mkdir(parents=True, exist_ok=True)
P = {"main": "#0F4D92", "alt": "#B64342", "baseline": "#CFCECE",
     "extra1": "#42949E", "extra2": "#9A4D8E", "gray": "#767676"}
BB = ["ESM2", "ESM1v", "ESM3", "ProSST"]
BBC = {"ESM2": P["main"], "ESM1v": P["alt"], "ESM3": P["extra1"], "ProSST": P["extra2"]}
BBM = {"ESM2": "o", "ESM1v": "s", "ESM3": "^", "ProSST": "D"}
BBLAB = {"ESM2": "ESM2", "ESM1v": "ESM-1v", "ESM3": "ESM3", "ProSST": "ProSST"}
SLOTS = ["a_GEMME", "b_GEMME", "a_ddG", "b_ddG", "a_RSA", "b_BLOSUM", "b_dchg"]
SLAB = ["GEMME\nposition", "GEMME\nresidual", "ΔΔG\nposition", "ΔΔG\nresidual",
        "RSA\nposition", "BLOSUM\nresidual", "Δcharge\nresidual"]
# The confirmed uni3 grid puts all 7 slots on −3.0 to 3.0, so there is one ceiling (wide differed per slot).
UNI3_CEIL = 3.0
SHOW_BB = {
    "ESM2": "ESM2 650M", "ESM1v": "ESM-1v", "ESM3": "ESM3-open", "ProSST": "ProSST",
    "ESMIF1": "ESM-IF1", "ProteinMPNN": "ProteinMPNN", "TranceptionLret": "Tranception L (ret)",
    "S2F": "S2F", "VespaG": "VespaG", "VenusREM": "VenusREM",
    "SaProt650M": "SaProt 650M", "AIDO_RAG16B": "AIDO RAG 16B",
}
# Negative red, zero white, positive blue. The pair separates under red-green colour vision deficiency.
CMAP_SHARE = LinearSegmentedColormap.from_list("share", [P["alt"], "#FFFFFF", P["main"]])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Source Sans Pro", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.0, "axes.labelsize": 7.0, "xtick.labelsize": 6.0, "ytick.labelsize": 6.0,
    "legend.fontsize": 6.5, "legend.frameon": False, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
})
AA = "ACDEFGHIKLMNPQRSTVWY"


# ─────────────────────────────────────────────────────────── panel a
def load_a(p0=120, npos=14):
    """The GEMME channel of one DMS as a (20 AA x npos) matrix. Empty cells are NaN."""
    d = np.load(ext_cache() / "urn-mavedb-00000659-a-1.npz",
                allow_pickle=True)
    pid = d["pidx"].astype(int); mut = d["mut_i"].astype(int)
    v = d["ch_gemme"].astype(float); v = (v - v.mean()) / v.std()
    sel = (pid >= p0) & (pid < p0 + npos)
    M = np.full((20, npos), np.nan)
    M[mut[sel], pid[sel] - p0] = v[sel]
    pos = np.nanmean(M, 0)
    return M, np.tile(pos, (20, 1)), M - pos, d["upos"].astype(int)[p0:p0 + npos]


def panel_a(axes, cax):
    M, Mp, Mr, upos = load_a()
    vmax = np.nanmax(np.abs(M))
    ttl = ["channel  z(GEMME)", "position part  (constant down a column)",
           "residual part  (deviation within a position)"]
    for ax, Z, t in zip(axes, [M, Mp, Mr], ttl):
        im = ax.imshow(Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
                       interpolation="nearest")
        ax.set_title(t, fontsize=6.6, pad=2.5, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(.5); s.set_color("#BBBBBB")
    axes[0].set_yticks([0, 9, 19]); axes[0].set_yticklabels([AA[0], AA[9], AA[19]])
    axes[0].set_ylabel("substitution", labelpad=1)
    axes[-1].set_xlabel(f"residue position  ({upos[0]}–{upos[-1]}, SGCB)", labelpad=2)
    cb = plt.colorbar(im, cax=cax); cb.outline.set_linewidth(.5)
    cb.set_ticks([-2, 0, 2]); cb.ax.tick_params(labelsize=5.5, width=.6, length=2)
    cb.set_label("z", fontsize=6, labelpad=1)


# ─────────────────────────────────────────────────────────── panel b
def panel_b(ax, ax2):
    """Eta-squared bars and the relation to saturation. Medians and IQRs are used (the substitution-channel distribution has a long left tail)."""
    Q = pd.read_csv(DATA_IN / "panelb_eta2_verified.csv")
    order = ["GEMME", "RSA", "ddG", "BLOSUM", "Δcharge"]
    USE = {"GEMME": ("split a + b", P["main"]), "ddG": ("split a + b", P["main"]),
           "RSA": ("a only (no residual)", P["extra1"]),
           "BLOSUM": ("b only", P["alt"]), "Δcharge": ("b only", P["alt"])}
    rng = np.random.default_rng(1)
    for i, c in enumerate(order):
        v = Q[Q.channel == c].e_zpos.to_numpy()
        med, q1, q3 = np.median(v), np.quantile(v, .25), np.quantile(v, .75)
        ax.barh(i, med, height=.6, color=USE[c][1], edgecolor="white", lw=.5, zorder=2)
        ax.scatter(v, i + rng.uniform(-.17, .17, len(v)), s=1.6, color="#2B2B2B",
                   alpha=.35, lw=0, zorder=3)
        ax.hlines(i, q1, q3, color="#1A1A1A", lw=1.0, zorder=4)
        ax.vlines([q1, q3], i - .13, i + .13, color="#1A1A1A", lw=1.0, zorder=4)
        ax.text(1.045, i, USE[c][0], va="center", fontsize=6.0)
    ax.axvline(.7, color=P["gray"], lw=.9, ls=(0, (3, 2)), zorder=5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(["ΔΔG" if c == "ddG" else c for c in order])
    ax.invert_yaxis(); ax.set_xlim(0, 1.60); ax.set_xticks([0, .25, .5, .75, 1.0])
    ax.set_xlabel(r"$\eta^2$ of the position part explained by the WT amino acid", labelpad=2)
    ax.text(.713, .975, "0.7", transform=ax.get_xaxis_transform(), fontsize=5.8,
            color=P["gray"], ha="left", va="top")
    ax.set_title("b   A substitution channel's position part is only a WT look-up",
                 loc="left", fontsize=7.6, fontweight="bold", pad=5)
    ax.text(0, -.70, "bar = median, whisker = IQR, dots = the 49 tuning assays",
            transform=ax.transAxes, fontsize=5.8, color=P["gray"])

    # Mechanism: as saturation rises, the eta-squared of a substitution-type channel converges to 1
    sat = Q.groupby("dms").nsub.first()
    for c, col, mk in (("Δcharge", P["alt"], "s"), ("BLOSUM", "#D4897F", "^"),
                       ("GEMME", P["main"], "o")):
        q = Q[Q.channel == c].merge(sat.rename("sat"), left_on="dms", right_index=True)
        ax2.scatter(q.sat, q.e_zpos, s=4.5, color=col, marker=mk, lw=0, alpha=.75)
        r = spearmanr(q.sat, q.e_zpos).statistic
        ax2.scatter([], [], s=9, color=col, marker=mk, lw=0,
                    label=f"{c}   ρ = {r:+.2f}")
    ax2.set_xlabel("substitutions assayed per position", labelpad=2)
    ax2.set_ylabel(r"$\eta^2$", labelpad=1)
    ax2.set_ylim(-.06, 1.14); ax2.set_xlim(0, 21.5)
    ax2.set_yticks([0, .5, 1.0]); ax2.set_xticks([0, 5, 10, 15, 20])
    ax2.text(0, 1.07, "saturation is what creates the look-up", transform=ax2.transAxes,
             fontsize=6.4, va="bottom")
    ax2.legend(loc="center", bbox_to_anchor=(.47, .47), fontsize=5.8,
               handletextpad=.3, labelspacing=.5, borderpad=.25)


# ─────────────────────────────────────────────────────────── panel c
def panel_c(axes):
    """A 12 backbone x 7 slot heatmap, showing that the axis decides the allocation.

    The cell value is share = w_slot / sum(w), so each row sums to 1. Rows are ordered by
    descending sum(w) on the function axis.
    ⚠ Fitted on the uni3 grid (−3.0 to 3.0). The earlier wide grid had a lower bound of 0 on the
    GEMME and ΔΔG slots, so those could not go negative. State the grid whenever an individual
    slot value is quoted.
    """
    W = pd.read_csv(DATA_IN / "panelc_comp7_weights.csv")
    have = [b for b in SHOW_BB if b in set(W.bb)]
    q0 = W[W.axis == "function"].set_index("bb")
    tot0 = {b: float(q0.loc[b, SLOTS].to_numpy(float).sum()) for b in have}
    bbs = sorted(have, key=lambda b: -tot0[b])

    mats, rs = [], []
    for axis in ("function", "stability"):
        q = W[W.axis == axis].set_index("bb")
        M = np.vstack([q.loc[b, SLOTS].to_numpy(float) for b in bbs])
        M = M / M.sum(axis=1, keepdims=True)
        mats.append(M)
        R = np.corrcoef(M)
        rs.append(R[np.triu_indices(len(bbs), 1)].mean())
    vmax = max(np.abs(m).max() for m in mats)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    for ax, axis, ttl, M, r in zip(axes, ("function", "stability"),
                                   ("Function axis", "Expression axis"), mats, rs):
        q = W[W.axis == axis].set_index("bb")
        im = ax.imshow(M, cmap=CMAP_SHARE, norm=norm, aspect="auto")
        for i, b in enumerate(bbs):
            for j, sl in enumerate(SLOTS):
                v = M[i, j]
                mark = "†" if abs(float(q.loc[b, sl])) >= UNI3_CEIL - 1e-9 else ""
                ax.text(j, i, f"{v:.2f}".replace("0.", ".") + mark, ha="center", va="center",
                        fontsize=5.2, color="white" if abs(v) > .62 * vmax else "black")
        ax.set_xticks(range(7)); ax.set_xticklabels(SLAB, fontsize=5.4)
        ax.set_yticks(range(len(bbs)))
        if ax is axes[0]:
            ax.set_yticklabels([f"{SHOW_BB[b]} ({tot0[b]:.1f})" for b in bbs], fontsize=5.4)
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{ttl}      profiles alike: mean pairwise r = {r:.2f}",
                     fontsize=6.4, pad=3)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    return im


def main() -> None:
    fig = plt.figure(figsize=(7.2, 6.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0],
                             left=.085, right=.985, top=.905, bottom=.545, wspace=.26)
    ga = outer[0, 0].subgridspec(3, 2, hspace=.42, wspace=.04, width_ratios=[1, .035])
    axes_a = [fig.add_subplot(ga[i, 0]) for i in range(3)]
    panel_a(axes_a, fig.add_subplot(ga[:, 1]))
    axes_a[0].text(0, 1.47, "a   Every channel splits into a position part and a residual part",
                   transform=axes_a[0].transAxes, fontsize=7.6, fontweight="bold")
    gb = outer[0, 1].subgridspec(2, 1, hspace=1.30, height_ratios=[1.3, 1.0])
    panel_b(fig.add_subplot(gb[0]), fig.add_subplot(gb[1]))
    fig.text(.020, .495, "c   The assay axis sets the allocation, not the backbone",
             fontsize=7.6, fontweight="bold")
    gc = fig.add_gridspec(1, 3, left=.155, right=.925, top=.455, bottom=.155,
                          wspace=.06, width_ratios=[1, 1, .028])
    ax_c = [fig.add_subplot(gc[0]), fig.add_subplot(gc[1])]
    im_c = panel_c(ax_c)
    cb = fig.colorbar(im_c, cax=fig.add_subplot(gc[2]))
    cb.set_label("w / Σw", fontsize=5.6)
    cb.ax.tick_params(labelsize=5.2, length=2); cb.outline.set_visible(False)
    fig.text(.020, .085,
             r"score $=$ pLM LLR $+\ \sum_k \mathbf{a}_k\cdot$position$_k$ "
             r"$+\ \mathbf{b}_k\cdot$residual$_k$ "
             ".  Weights fitted on the MaveDB external set only, per axis", fontsize=6.0)
    fig.text(.020, .050, "(function 35 DMS over 18 proteins, expression 14 DMS over 9 proteins). "
             "ProteinGym is never seen during fitting.", fontsize=6.0)
    fig.text(.020, .016, "† at the uni3 grid ceiling (3.0): a bound, not an optimum. Refitting those two "
             "cells on a wider grid moves the tuning objective by 0.0008 or less.   "
             "Weight magnitude is not channel value. The ablation measures value.",
             fontsize=5.6, color=P["gray"])
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"comp7_design.{ext}", dpi=300)
    print(f"[saved] {HERE}/comp7_design.pdf / .png")


if __name__ == "__main__":
    main()
