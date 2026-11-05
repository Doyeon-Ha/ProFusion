"""Figure 2, extended: pLM alone versus ProFusion across 13 backbones.

## What changed from the earlier version

With 13 backbones instead of 4, **the boxes are laid horizontally**. Cramming 26 vertical boxes
into 7.2 in overlaps the labels and hides the whiskers. Laid horizontally, each backbone name gets
its own row on the y axis, and **ordering by descending base makes the convergence visible
directly**: the grey (pLM) column falls steeply from top to bottom while the blue (ProFusion) one
stands almost vertical.

Panel c draws **every backbone line** and lays the median over them in bold. With only two colours
(grey = pLM, blue = ProFusion), 20 lines still separate into two bundles. The claim is that
ProFusion stays below wherever the threshold is put, so all that has to be visible is that the two
bundles do not mix.

## What is measured

  rho     Spearman between the score and the experimental label within one DMS. Unitless, [-1, 1],
          higher is better.
  base    pLM LLR alone (auxiliary channel weights w = 0).
  fit7    the 7-slot fusion. Weights are fitted **on the MaveDB external set only** and ProteinGym
          labels are used for evaluation only.
  p       Wilcoxon paired on the same DMS, two-sided.
  panel c the number of DMS whose per-DMS rho falls below a threshold, that is, how many assays
          that backbone gave up on. Lower is better. Sweeping the threshold from 0.1 to 0.4 shows
          whether the ordering holds, which means the conclusion does not hang on one value such
          as 0.2.

⚠ Left out of the main figure: the three small ESM2 models (8M, 35M, 150M) exist only for the size
   ladder and 650M represents them, and Tranception L keeps only the retrieval variant. Both are
   still in the CSV.
⚠ S2F stays **S2F**. The official comparison target S2F-MSA is z(S2F) + z(EVE), and the 49
   external assays have no EVE, so it cannot be fitted under the same definition. The comparison
   against official S2F-MSA is made in the text from the published per-DMS Spearman values
   (ProFusion 0.5127 vs S2F-MSA 0.4942, paired p = 0.0013).
⚠ The axis key "stability" is the old name; its contents are the 16 abundance and surface
   expression assays.

input   results/extend_backbones_perdms.csv and frozen/paneld_perdms.csv
output  fig1_ext.{pdf,png} and fig1_ext_data.csv
"""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plmfusion.paths import RESULTS, frozen  # noqa: E402

# The ProFusion colour is dodgerblue (#1E90FF). Matching the value of the navy in
# fig_common.PALETTE["main"] does not make it the same thing: that one is the first colour of a
# 4-backbone series, so do not change this one to follow it.
PALETTE = {"main": "#1E90FF", "baseline": "#CFCECE", "gray_line": "#767676", "plm_bold": "#1C1C1C",
           "rival": "#8C8C8C"}
# Panel d compares post-hoc fusion methods on the same 128 DMS. The comparison only holds if the
# base is the same: the harness ProSST-2048 raw matches our llr_ProSST at r = 1.0000.
# ⚠ Do not compare against their published 0.556 directly. Our 128 assays are harder (0.013 by the
#   mean over the 97 methods in the official per-assay table, and 0.536 -> 0.493 for VenusREM alone).
VRC = frozen()
PANELD = frozen("paneld_perdms.csv")
ARMS = [  # (column, display name, design category, is it ours) -- stacked bottom to top
    # The baseline (VenusREM-Harness) has a paired difference of 0, so it has no box. The zero line
    # is its position.
    # ⚠ Keep the base: panel d is a result **on top of the ProSST ensemble**, so that ensemble has
    #   to be visible for the panel to read.
    ("ProSSTens_raw",   "ProSST ens.",      "no fusion",              False),
    ("VenusREM_v1",     "VenusREM",         "fit on benchmark split", False),
    ("VenusREM2",       "VenusREM2",        "harness, adaptive",      False),
    ("comp7_ProSSTens", "ProFusion",        "axis-matched, split",    True),
]
SHOW = {"ESM2": "ESM2 650M", "ESM1v": "ESM-1v", "ESM3": "ESM3-open", "ProSST": "ProSST",
        "ESMIF1": "ESM-IF1", "ProteinMPNN": "ProteinMPNN",
        "TranceptionLret": "Tranception L (ret)",
        "S2F": "S2F", "VespaG": "VespaG", "VenusREM": "VenusREM",
        "SaProt650M": "SaProt 650M", "AIDO_RAG16B": "AIDO-RAG 16B"}
CUTS = [0.1, 0.2, 0.3, 0.4]
H = RESULTS


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Source Sans Pro", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.0, "axes.labelsize": 8.0, "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0, "legend.fontsize": 7.5, "legend.frameon": False,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "svg.fonttype": "none", "pdf.fonttype": 42,
        "ps.fonttype": 42})


def stars(p):
    return "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else "ns"))


def box_panel(ax, D, axis, title, order, rows, ylab=True):
    s = D[D.axis == axis]
    data, pos, cols = [], [], []
    for i, bb in enumerate(order):
        d = s[s.bb == bb]
        a, c = d.base.to_numpy(), d.fit7.to_numpy()
        y = len(order) - 1 - i                       # top to bottom in order
        data += [a, c]; pos += [y + 0.19, y - 0.19]
        cols += [PALETTE["baseline"], PALETTE["main"]]
        p = wilcoxon(c, a).pvalue
        # ⚠ Every value the text quotes must be here (Source Data). Leaving out the median and the
        #   win rate once made "71.4% to 98.2%" and "0.156 -> 0.501" untraceable from the CSV.
        rows.append(dict(panel=title[0], axis=axis, backbone=SHOW[bb], n=len(a),
                         plm_mean=a.mean(), plm_sd=a.std(), profusion_mean=c.mean(),
                         profusion_sd=c.std(), plm_median=np.median(a),
                         profusion_median=np.median(c), delta=(c - a).mean(),
                         win_frac=float((c - a > 0).mean()), p=p))
        ax.text(0.985, y, stars(p), ha="right", va="center", fontsize=6.6,
                color=PALETTE["gray_line"], transform=ax.get_yaxis_transform())
    bp = ax.boxplot(data, positions=pos, widths=0.33, patch_artist=True, vert=False,
                    showfliers=False, medianprops=dict(color="black", lw=0.8),
                    whiskerprops=dict(lw=0.55), capprops=dict(lw=0.55),
                    boxprops=dict(lw=0.45, edgecolor="black"))
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col)
    rng = np.random.default_rng(0)
    for dat, pp in zip(data, pos):
        ax.scatter(dat, pp + rng.uniform(-0.085, 0.085, len(dat)), s=0.9,
                   color="black", alpha=0.22, zorder=3, linewidths=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHOW[b] for b in order][::-1] if ylab else [])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(-0.28, 1.0); ax.set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Spearman ρ per assay")
    ax.axvline(0, color=PALETTE["gray_line"], lw=0.5, ls=":")
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", pad=5)


def arms_panel(ax, rows, tag="d"):
    """Panel d: what separates the methods is the recipe, not the backbone (ensemble).

    x holds two backbone positions, y the mean rho over assays, and each line is one recipe. The
    vertical gap is the recipe effect and the slope is the ensemble effect, so both are read on one
    axis. A box plot cannot make this comparison (the within-assay spread is three times the
    between-method difference, so all the boxes overlap).

    The two arrows are, in each column, the gap between ProFusion and **whichever competitor is in
    its published configuration there**. On the left that is +0.047 against VenusREM (ProSST-2048
    only), on the right +0.018 against VenusREM2 (the 6-model ensemble).

    The crossing: the **left end** of the blue line (single-backbone ProFusion, 0.5377) is above
    the **right end** of the grey one (6-model ensemble VenusREM2, 0.5226). No guide line is drawn:
    ProFusion moves only 0.0068 between single and ensemble, so a guide would sit on the solid line
    and look redundant, and the crossing is read straight off the y axis.

    ⚠ The VenusREM x ensemble cell is **our own reconstruction** (the published v1 is ProSST-2048
      only). Applying the same recipe to a single backbone and comparing against the original
      repository gives 0.4921 vs 0.4906 with af2cf alignments (+0.0014, 128 assays) and 0.4587 vs
      0.4740 once the alignments match too (−0.0153, 37 assays). The implementation error is
      therefore between −0.015 and +0.001, far smaller than the gap to ProFusion (+0.0359), so it
      can carry that comparison. But
      ⚠ **the +0.0140 difference from VenusREM2 (0.5226) in the ensemble column is smaller than
      that error, so no ordering can be claimed there.** Per-assay claims are also ruled out
      (per-DMS |d_rho| up to 0.108). Hence the faint dashed line and the dagger.
    """
    def ink(c, k=0.45, cap=0x6A):
        """Using the line colour for the text leaves pale grey unreadable. Mix towards black and cap it."""
        r, g, b = (int(c[i:i + 2], 16) for i in (1, 3, 5))
        return "#%02x%02x%02x" % tuple(min(int(v * (1 - k)), cap) for v in (r, g, b))

    # ⚠ Reads the file produced by scripts/paneld_numbers.py. The older FINAL_perdms.csv had the
    #   **old weights** of 2026-10-15 baked in and disagreed with the text.
    V = pd.read_csv(PANELD)
    v1e = V.v1ens.mean()
    LINES = [  # (name, single, ensemble, colour, line width, is it a reconstruction)
        ("none",       V.ProSST2048.mean(),       V.ProSSTens_raw.mean(),   PALETTE["baseline"], 1.2, False),
        ("VenusREM",   V.VenusREM_v1.mean(),      v1e,                      PALETTE["rival"],    1.2, True),
        ("VenusREM2",  V.vrh_ProSST2048.mean(),   V.VenusREM2.mean(),       PALETTE["gray_line"], 1.5, False),
        ("ProFusion",  V.comp7_ProSST2048.mean(), V.comp7_ProSSTens.mean(), PALETTE["main"],     2.1, False),
    ]
    pf1, vr2 = V.comp7_ProSST2048.mean(), V.VenusREM2.mean()
    for nm, a, b, col, lw, recon in LINES:
        bold = nm == "ProFusion"
        if np.isfinite(b):
            ax.plot([0, 1], [a, b], color=col, lw=lw, ls=":" if recon else "-",
                    alpha=0.55 if recon else 1.0, zorder=3, solid_capstyle="round")
            ax.plot([1], [b], "o", ms=3.6, color=col, mec="white", mew=0.6, zorder=4,
                    alpha=0.55 if recon else 1.0)
            ax.annotate(nm + ("†" if recon else ""), (0.46, a + (b - a) * 0.46),
                        textcoords="offset points", xytext=(0, 3.5), va="bottom",
                        ha="center", fontsize=6.2, color=col if bold else ink(col),
                        fontweight="bold" if bold else "normal", zorder=6)
        else:
            ax.annotate(nm, (0, a), textcoords="offset points", xytext=(2, 4), va="bottom",
                        ha="left", fontsize=6.2, color=ink(col))
        ax.plot([0], [a], "o", ms=3.6, color=col, mec="white", mew=0.6, zorder=4)
        rows.append(dict(panel=tag, axis="function+expression", method=nm,
                         design="recipe x backbone", n=len(V),
                         rho_single=a, rho_ensemble=b, reconstructed=recon))
    # ⚠ An arrow must span **two lines at the same x**. Standing it at the ensemble position but
    #    measuring only to the dashed line (single-backbone ProFusion) makes the reader compare it
    #    against the blue line there, and the number will not match. The crossing (+0.013) is shown
    #    by the dashed line itself, so no number is attached to it.
    pf2 = V.comp7_ProSSTens.mean()
    ax.annotate("", xy=(0.97, pf2), xytext=(0.97, vr2),
                arrowprops=dict(arrowstyle="<->", color=PALETTE["main"], lw=0.6,
                                shrinkA=1, shrinkB=1))
    ax.annotate(f"+{pf2 - vr2:.3f}", (0.97, (pf2 + vr2) / 2), textcoords="offset points",
                xytext=(-13, 0), ha="center", va="center", fontsize=6.0,
                color=PALETTE["main"], fontweight="bold")
    # The left arrow: VenusREM's **published configuration is ProSST-2048**, so the gap is measured
    #   only in that column. VenusREM in the ensemble column is a reconstruction, so the same
    #   comparison must not be made there. The dashed line passes through the middle, but both ends
    #   are at the same x, so the gap itself still reads correctly.
    v1s = V.VenusREM_v1.mean()
    ax.annotate("", xy=(0.03, pf1), xytext=(0.03, v1s),
                arrowprops=dict(arrowstyle="<->", color=PALETTE["main"], lw=0.6,
                                shrinkA=1, shrinkB=1))
    ax.annotate(f"+{pf1 - v1s:.3f}", (0.03, (pf1 + v1s) / 2), textcoords="offset points",
                xytext=(13, 0), ha="center", va="center", fontsize=6.0,
                color=PALETTE["main"], fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ProSST-2048\n(1 model)", "ProSST-ensemble\n(6 models)"], fontsize=6.4)
    ax.set_xlim(-0.10, 1.10)
    ax.set_ylabel("mean ρ per assay", labelpad=2)
    # ⚠ This number is over both axes (function + expression). Panel c is function only, so its range differs.
    ax.set_title(f"{tag}   Fusion methods (both axes, n={len(V)})", loc="left",
                 fontsize=8.5, fontweight="bold", pad=5)


def cut_panel(ax, D, order, rows, ymax=None):
    """Panel c: the threshold sweep. The 12 backbone lines are drawn faintly with the median in bold."""
    # -- c threshold sweep: the median over all backbones plus a full-range band
    s = D[D.axis == "function"]
    N0 = np.array([[int((s[s.bb == b].base < t).sum()) for t in CUTS] for b in order])
    N1 = np.array([[int((s[s.bb == b].fit7 < t).sum()) for t in CUTS] for b in order])
    for M, thin, bold, ls, mk, lab in [
            (N0, PALETTE["gray_line"], PALETTE["plm_bold"], "--", "o", "pLM alone"),
            (N1, PALETTE["main"], PALETTE["main"], "-", "s", "ProFusion")]:
        for r in M:                                  # one line per backbone
            ax.plot(CUTS, r, ls, color=thin, lw=0.55, alpha=0.28)
        ax.plot(CUTS, np.median(M, 0), ls, marker=mk, color=bold, lw=1.9, ms=4.0,
                mfc="white" if ls == "--" else bold, mew=1.0, label=lab, zorder=5)
    for b, r0, r1 in zip(order, N0, N1):
        for t, x0, x1 in zip(CUTS, r0, r1):
            rows.append(dict(panel="c", axis="function", backbone=SHOW[b], cutoff=t,
                             plm_n=int(x0), profusion_n=int(x1)))
    # Attached **just below the last marker** instead of in a legend box. To the right of the line
    # end it would force the axis out to 0.6 and widen the panel for nothing (and collide with the
    # neighbouring panel).
    for M, col, lab in [(N0, PALETTE["plm_bold"], "pLM alone"),
                        (N1, PALETTE["main"], "ProFusion")]:
        ax.text(CUTS[-1] + 0.004, np.median(M, 0)[-1] - 4.5, lab, color=col,
                fontsize=7.5, va="top", ha="right", fontweight="bold", zorder=8,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.82))
    # Passing ymax clips the top. Only 5 of the 96 values exceed 50, and most are ProteinMPNN
    # (pLM alone 30/74/102/109). Without clipping the axis runs to 110 and squashes every other
    # curve into the bottom third. For the clipped curves, **how many go how far** is stated in text.
    if ymax is not None:
        over = sorted({SHOW[b] for b, r in zip(order, N0) if max(r) > ymax} |
                      {SHOW[b] for b, r in zip(order, N1) if max(r) > ymax})
        hi = max(N0.max(), N1.max())
        ax.set_ylim(0, ymax)
        if over:
            ax.text(0.5, 0.985, f"{len(over)} pLM curves exit above (max {hi})",
                    transform=ax.transAxes, ha="center", va="top", fontsize=5.8,
                    color=PALETTE["gray_line"])
    ax.set_xticks(CUTS); ax.set_xlim(0.082, 0.432)
    ax.set_xlabel("threshold on ρ")
    ax.set_ylabel("assays with ρ below threshold")
    # ⚠ This panel counts **the function axis only**. Panel d counts both axes, so the ranges
    #   differ. Both put n in the title so that a reader does not take them for the same set.
    nc = s.dms.nunique()
    ax.set_title(f"c   Fewer low-ρ assays (function, n={nc})", loc="left", fontsize=8.5,
                 fontweight="bold", pad=5)


def main(drop_c: bool = False):
    """drop_c=True removes panel c (the threshold sweep) and d becomes c. Output is fig_ext_3."""
    style()
    D = pd.read_csv(H / "extend_backbones_perdms.csv")
    D = D[D.bb.isin(SHOW)]
    order = (D[D.axis == "function"].groupby("bb").base.mean()
             .sort_values(ascending=False).index.tolist())
    n_f = D[(D.axis == "function") & (D.bb == order[0])].dms.nunique()
    n_e = D[(D.axis == "stability") & (D.bb == order[0])].dms.nunique()
    print(f"{len(order)} backbones, function {n_f} DMS, expression {n_e} DMS")

    # Layout: a and b are the main result. The backbone names overflow **outside the axes** of a
    # (into the left margin of the figure), so giving both panels the same width ratio makes their
    # drawing areas match 1:1.
    if drop_c:
        # Three panels in a row. An empty column (index 2) keeps a and b together while separating
        # c: a and b are read across the same 12 rows, while c is an unrelated panel that needs room
        # for its y label.
        fig = plt.figure(figsize=(7.2, 3.60))
        gs = fig.add_gridspec(1, 4, width_ratios=[1.12, 1.12, 0.14, 0.95], wspace=0.10)
        top, slope_slot = [gs[0], gs[1]], gs[3]
    else:
        fig = plt.figure(figsize=(7.2, 5.35))
        outer = fig.add_gridspec(2, 1, height_ratios=[1.34, 1.0], hspace=0.30)
        top = outer[0].subgridspec(1, 2, wspace=0.075)
        bot = outer[1].subgridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.24)
        slope_slot = bot[1]
    rows = []
    axa = fig.add_subplot(top[0])
    box_panel(axa, D, "function", f"a   Function eval (n={n_f})", order, rows)
    axb = fig.add_subplot(top[1])
    box_panel(axb, D, "stability", f"b   Expression eval (n={n_e})", order, rows, ylab=False)
    h = [plt.Rectangle((0, 0), 1, 1, fc=PALETTE["baseline"], ec="black", lw=0.5),
         plt.Rectangle((0, 0), 1, 1, fc=PALETTE["main"], ec="black", lw=0.5)]
    # Placed at the upper left, where there are no boxes (the top rows of the rho < 0.25 region).
    # Outside the panel it would collide with the c and d titles in the two-row layout.
    axb.legend(h, ["pLM alone", "ProFusion"], loc="upper left", ncol=1,
               handlelength=1.0, handletextpad=0.4, labelspacing=0.3,
               borderpad=0.25, fontsize=6.8)

    if not drop_c:
        cut_panel(fig.add_subplot(bot[0]), D, order, rows)

    # -- d, comparison of fusion methods (same 128 DMS, same base)
    arms_panel(fig.add_subplot(slope_slot), rows, tag="c" if drop_c else "d")

    out = H / ("figures/fig_ext_3" if drop_c else "figures/fig1_ext")
    out.parent.mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame(rows)
    R["axis"] = R["axis"].replace({"stability": "expression"})
    R.to_csv(out.with_name(out.name + "_data.csv"), index=False)
    if drop_c:
        fig.subplots_adjust(left=0.122, right=0.992, top=0.905, bottom=0.145)
    else:
        fig.subplots_adjust(left=0.122, right=0.988, top=0.945, bottom=0.088)
    for fmt in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{fmt}"), dpi=300)
    plt.close(fig)
    print("wrote:", out.with_suffix(".pdf"))


if __name__ == "__main__":
    import sys
    main(drop_c="--no-c" in sys.argv)
