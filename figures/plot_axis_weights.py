"""Result 2: the weights differ by assay axis, and the right axis has to be used.

## The two panels split one claim

  a  the fitted weight allocation itself. For the same backbone, changing the axis changes where
     the budget goes.
  b  whether that difference is real: how much is lost by using the other axis's weights (swapped)
     or a single vector fitted over both axes (pooled). Panel a says "they differ", panel b says
     "and that matters".

**The rows are the same in both panels**, so one backbone can be followed across and its
allocation read alongside its loss.

## What is measured

  share   w_k / sum(w): which slot the backbone spent its budget on. The 7 slots sum to 1,
          unitless. The colour splits at 1/7 (the black line on the colour bar), which is the
          value for an equal split across 7 slots and the only natural reference point in this
          figure. Red is more than its share, blue less.
  d_rho   the difference paired by assay (matched minus the comparison). Positive means splitting
          by axis is better. swapped = the other axis's weights; pooled = one vector fitted on all
          49 tuning assays with no axis distinction. pooled is the fairer control: it ignores the
          axis label but uses all the data.
  q       paired two-sided Wilcoxon over assays, Benjamini-Hochberg corrected within a panel.

Weights are fitted on the MaveDB external set only, and ProteinGym labels are used for evaluation
only.

⚠ Cells sitting on the grid ceiling (±3.0) are not marked here. There are only 2 of 168 and the
  caption says so.
⚠ The internal axis key "stability" is the expression axis of the manuscript.

input   extend_backbones.csv (a) and axis_crossapply_12bb.csv (b)
output  fig_axis_weights.{pdf,png} and fig_axis_weights_data.csv
"""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plmfusion.paths import RESULTS  # noqa: E402

PAL = {"main": "#0F4D92", "alt": "#B64342", "gray": "#3A3A3A", "soft": "#9DBBD8"}
H = RESULTS                     # every input and output lives under here
SL = ["a_GEMME", "b_GEMME", "a_ddG", "b_ddG", "a_RSA", "b_BLOSUM", "b_dchg"]
SLAB = ["GEMME pos", "GEMME res", "ΔΔG pos", "ΔΔG res", "RSA pos", "BLOSUM res", "Δcharge res"]
SHOW = {"AIDO_RAG16B": "AIDO-RAG 16B", "VenusREM": "VenusREM", "ProSST": "ProSST",
        "VespaG": "VespaG", "TranceptionLret": "Tranception L (ret)", "S2F": "S2F",
        "ESM1v": "ESM-1v", "SaProt650M": "SaProt 650M", "ESM3": "ESM3-open",
        "ESM2": "ESM2 650M", "ESMIF1": "ESM-IF1", "ProteinMPNN": "ProteinMPNN"}
AX = [("function", "Function"), ("stability", "Expression")]
EQ = 1.0 / 7
# Following heatmap convention, large values are red. Reversing it would force a trip to the colour bar.
CMAP = LinearSegmentedColormap.from_list(
    "div", ["#0F4D92", "#8FB2D5", "#FFFFFF", "#DFA9A8", "#B64342"])
VMIN, VMAX = -0.05, 0.35


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Source Sans Pro", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.6,
        "ytick.labelsize": 7.0, "legend.fontsize": 6.8, "legend.frameon": False,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "svg.fonttype": "none", "pdf.fonttype": 42,
        "ps.fonttype": 42})


def heat(ax, D, axis, order, ylab, rows, alab):
    f = D[D.axis == axis].set_index("bb").loc[order]
    W = f[SL].to_numpy(float)
    S = W / f.total.to_numpy(float)[:, None]
    im = ax.imshow(S, cmap=CMAP, aspect="auto",
                   norm=TwoSlopeNorm(vcenter=EQ, vmin=VMIN, vmax=VMAX))
    for i in range(len(f)):
        for j in range(7):
            v = S[i, j]
            ax.text(j, i, f"{v:.2f}".lstrip("0") if v >= 0 else f"{v:.2f}",
                    ha="center", va="center", fontsize=5.3,
                    color="white" if (v > 0.27 or v < 0.01) else "#222222")
        rows.append(dict(panel="a", axis=alab, backbone=SHOW[order[i]],
                         **{f"share_{k}": S[i, j] for j, k in enumerate(SL)}))
    ax.set_xticks(range(7))
    ax.set_xticklabels(SLAB, fontsize=5.6, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(f)))
    ax.set_yticklabels([SHOW[b] for b in order] if ylab else [])
    ax.set_xticks(np.arange(-.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(f), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.7); ax.tick_params(which="minor", length=0)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def cross(ax, X, axis, order, rows, alab, xlab):
    d = X[X.axis == axis].set_index(["contrast", "bb"])
    for ci, (con, col, mk) in enumerate([("swapped", PAL["alt"], "o"),
                                         ("pooled", PAL["main"], "s")]):
        for i, bb in enumerate(order):
            r = d.loc[(con, bb)]
            y = i + (ci - 0.5) * 0.30
            ax.scatter(r["mean"], y, s=13, marker=mk, zorder=3, linewidths=0.5,
                       facecolor=col if r.q < 0.05 else "white", edgecolor=col)
            rows.append(dict(panel="b", axis=alab, backbone=SHOW[bb], contrast=con,
                             delta=r["mean"], pct_improved=r.pct_improved, q=r.q, n=r.n))
    ax.axvline(0, color="black", lw=0.7)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_yticks([]); ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlab)
    ax.spines["left"].set_visible(False)


def main():
    style()
    D = pd.read_csv(H / "extend_backbones.csv"); D = D[D.bb.isin(SHOW)]
    X = pd.read_csv(H / "axis_crossapply_12bb.csv")
    order = (D[D.axis == "function"].set_index("bb").pg_base
             .sort_values(ascending=False).index.tolist())
    n = len(order); rows = []

    fig = plt.figure(figsize=(7.2, 0.80 + 0.195 * n))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.0, 0.10, 0.52, 0.52], wspace=0.07)
    im = heat(fig.add_subplot(gs[0]), D, "function", order, True, rows, "Function")
    heat(fig.add_subplot(gs[1]), D, "stability", order, False, rows, "Expression")
    axc = fig.add_subplot(gs[3]); cross(axc, X, "function", order, rows, "Function",
                                        "gain in per-assay ρ")
    axd = fig.add_subplot(gs[4]); cross(axd, X, "stability", order, rows, "Expression",
                                        "gain in per-assay ρ")
    lim = max(abs(np.array(ax.get_xlim())).max() for ax in (axc, axd))
    for ax in (axc, axd):
        ax.set_xlim(-0.004, lim)
        ax.set_xticks([0, 0.01, 0.02, 0.03])
        ax.set_xticklabels(["0", ".01", ".02", ".03"])

    # There are two panels. The axis names go one line below as grey block subtitles.
    # a is the number of MaveDB assays the weights were fitted on; b is the number of ProteinGym
    # assays they were evaluated on.
    # ⚠ The assay counts are counted from the data. Hard-coding them leaves the subtitle stale when
    #   the evaluation set grows (112/16 did exactly that). The fitting side is read the same way.
    nf = int(D[D.axis == "function"].n_pg.iloc[0]); ne = int(D[D.axis == "stability"].n_pg.iloc[0])
    tf = int(D[D.axis == "function"].n_ext.iloc[0]); te = int(D[D.axis == "stability"].n_ext.iloc[0])
    for ax, lab in [(fig.axes[0], f"Function tuning set (n={tf})"),
                    (fig.axes[1], f"Expression tuning set (n={te})"),
                    (axc, f"Function eval (n={nf})"),
                    (axd, f"Expression eval (n={ne})")]:
        ax.set_title(lab, loc="left", fontsize=7.2, color=PAL["gray"], pad=3)
    fig.axes[0].text(0.0, 1.085, "a   Fitted weight allocation", transform=fig.axes[0].transAxes,
                     fontsize=8.0, fontweight="bold", ha="left", va="bottom")
    axc.text(0.0, 1.085, "b   Gain from phenotype matching", transform=axc.transAxes,
             fontsize=8.0, fontweight="bold", ha="left", va="bottom")

    b0 = fig.subplotpars.bottom; h0 = fig.subplotpars.top - b0
    cax = fig.add_axes([0.155, 0.070, 0.112, 0.026])
    # The middle tick is written as **"1/7"**. It is the only text inside the figure that answers
    # why the black vertical line is there. Writing 0.15 looks tidier but moves the meaning into
    # the caption, and readers look at the figure before the caption. (As a decimal it would have
    # to be 0.14, not 0.15.) ".3" is written "0.3" so that it matches "0".
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0.0, EQ, 0.3])
    cb.ax.set_xticklabels(["0", "1/7", "0.3"])
    cb.ax.tick_params(labelsize=6.8, length=2, pad=1.5)
    cb.ax.axvline(EQ, color="#222222", lw=0.8)
    fig.text(0.278, 0.084, "share of total weight w$_k$/Σw", fontsize=7.6,
             ha="left", va="center")
    hh = [plt.Line2D([], [], marker="o", ls="", mfc=PAL["alt"], mec=PAL["alt"], ms=4.2),
          plt.Line2D([], [], marker="s", ls="", mfc=PAL["main"], mec=PAL["main"], ms=4.2)]
    # The legend sits at **the same height as the colour bar**. Misaligning them leaves the bottom
    # margin untidy. x is the centre of the two panels.
    fig.legend(hh, ["vs swapped axis", "vs pooled fit"], loc="center", ncol=2,
               bbox_to_anchor=(0.855, 0.084), handletextpad=0.4, columnspacing=1.6,
               fontsize=7.6)

    (H / "figures").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(H / "figures/fig_axis_weights_data.csv", index=False)
    out = H / "figures/fig_axis_weights"
    fig.subplots_adjust(left=0.152, right=0.995, top=0.868, bottom=0.250)
    for fmt in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{fmt}"), dpi=300)
    plt.close(fig)
    print("wrote:", out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
