"""Result 3: a backbone that already reads structure buys less of the ΔΔG channel.
**Two-axis version (function and expression).**

The single-panel, single-column version is plot_model_needs.py. This file extends it to two axes
and writes under a different name, so it does not overwrite that figure.

## Why the accuracy axis was dropped

Putting standalone accuracy on the x axis leaves ProteinMPNN far out on its own, forcing a broken
axis, and 12 labels float around 12 points so it is unclear which name belongs to which. Accuracy
is a **control**, not the claim, so it goes into the text instead (rho = 0.154, p = 0.63; group
accuracy rho = −0.205, p = 0.52; partial rho after controlling for accuracy = −0.678, p = 0.015).
The figure says only that the two groups separate.

## What this figure answers

If the weights track "what that backbone lacks", a backbone that already reads structure should
spend less of its budget on the structure channel. The two groups separate vertically **without
overlap**, and do not separate horizontally (accuracy). That is what makes the vertical separation
something other than a restatement of accuracy.

## What is measured

  share     (a_ddG + b_ddG) / sum(|w|) * 100: the fraction of that backbone's budget taken by ΔΔG.
  p         Mann-Whitney U between the two groups (two-sided). With 8 against 4, a rank test is used.
            ⚠ Not printed inside the figure. It goes in the caption and the text (function p = 0.0040).

Weights are fitted on the MaveDB function-axis external set only, and ProteinGym labels are used
for evaluation only.

⚠ The GEMME-against-conservation and RSA-against-structure controls, and the Shapley contribution
  breakdown, were moved to the supplementary. That is where the reason lives for not stating
  "models buy less of what they already know" as a general principle in the text.

input   extend_backbones.csv   output  fig_model_needs.{pdf,png} and _data.csv
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))             # backbone_key lives in this folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402
from backbone_key import CMAP, MARKER, STRUCT  # noqa: E402
from plmfusion.paths import RESULTS  # noqa: E402

PAL = {"main": "#0F4D92", "alt": "#B64342", "gray": "#3A3A3A"}
H = RESULTS                     # every input and output lives under here
SL = ["a_GEMME", "b_GEMME", "a_ddG", "b_ddG", "a_RSA", "b_BLOSUM", "b_dchg"]
SHOW = {"AIDO_RAG16B": "AIDO-RAG", "VenusREM": "VenusREM", "ProSST": "ProSST",
        "VespaG": "VespaG", "TranceptionLret": "Tranception", "S2F": "S2F",
        "ESM1v": "ESM-1v", "SaProt650M": "SaProt", "ESM3": "ESM3",
        "ESM2": "ESM2", "ESMIF1": "ESM-IF1", "ProteinMPNN": "MPNN"}
# Full names, used only in the legend. The abbreviations in SHOW exist because the stack and wide
# layouts write text **next to the point** and have no room; bymodel has a legend for the names, so
# there is no reason to shorten them. The names match the other figures character for character,
# so the two legends agree.
FULL = {"AIDO_RAG16B": "AIDO-RAG 16B", "VenusREM": "VenusREM", "ProSST": "ProSST",
        "VespaG": "VespaG", "TranceptionLret": "Tranception L (ret)", "S2F": "S2F",
        "ESM1v": "ESM-1v", "SaProt650M": "SaProt 650M", "ESM3": "ESM3-open",
        "ESM2": "ESM2 650M", "ESMIF1": "ESM-IF1", "ProteinMPNN": "ProteinMPNN"}
# Only labels that overlap are nudged by hand. (dx, dy) in points.
NUDGE = {"ESM2": (-21, -10), "SaProt650M": (4, 6), "ESM1v": (-22, -3),
         "VespaG": (3, 7), "S2F": (19, -3), "ProSST": (-21, -3),
         "VenusREM": (-2, -13), "AIDO_RAG16B": (-6, 7), "ESM3": (-3, 7),
         "ESMIF1": (0, 7), "ProteinMPNN": (0, 7), "TranceptionLret": (0, 7)}
XL, XR = (0.138, 0.178), (0.358, 0.530)
LEFT, RIGHT = 0.215, 0.985   # margins. The x axis label is centred between them


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Source Sans Pro", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.0, "axes.labelsize": 8.0, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.frameon": False,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "svg.fonttype": "none", "pdf.fonttype": 42,
        "ps.fonttype": 42})



def needs_panel(ax, D, axis_name, tag, title, rows, compact=False, tfs=8.2):
    """The ΔΔG budget fraction for one axis. Returns (p, whether they separate, the two groups)."""
    f = D[(D.axis == axis_name) & (D.bb.isin(SHOW))].set_index("bb")
    tot = f[SL].abs().sum(axis=1)
    f["ddg"] = (f.a_ddG + f.b_ddG) / tot * 100
    f["struct"] = [b in STRUCT for b in f.index]
    a, b = f[f.struct].sort_values("ddg"), f[~f.struct].sort_values("ddg")
    sep = a.ddg.max() < b.ddg.min()
    p = mannwhitneyu(a.ddg, b.ddg).pvalue

    def spread(v, gap):
        """Walk the label y positions from bottom to top, enforcing a minimum gap. The points stay
        where they are.

        ⚠ Using the values as dict keys **merges ties into one entry** and the labels overprint.
          On the expression axis S2F, SaProt and AIDO-RAG are all exactly 47.3684%. This therefore
          returns **the sort index**, not the value.
        """
        order = sorted(range(len(v)), key=lambda i: v[i])
        out, prev = [0.0] * len(v), None
        for i in order:
            y = v[i] if prev is None else max(v[i], prev + gap)
            out[i] = y
            prev = y
        return out

    # The minimum label gap is derived from **the displayed height**. The value range differs per
    # axis (function 8.9 to 31.6, expression 39.1 to 58.8), so a fixed value would let the text
    # touch on one of them.
    #   gap(data) = required gap (pt) / axis height (pt) x displayed y range
    # The axis is about 138 pt tall with margins(y) = 0.15, so the displayed range is span x 1.30;
    # 9 pt is used for 6.8 pt text with some slack, giving a factor of 9/138*1.30 = 0.085.
    span = (f.ddg.max() - f.ddg.min()) * 0.085
    for k, (g, col, mk) in enumerate([(a, PAL["main"], "o"), (b, PAL["alt"], "s")]):
        xb = k - 0.235
        ax.plot([xb, xb], [g.ddg.min(), g.ddg.max()], color=col, lw=1.6, alpha=0.5,
                solid_capstyle="round", zorder=2)
        ax.plot([xb - 0.055, xb + 0.055], [g.ddg.median()] * 2, color=col, lw=2.2, zorder=3)
        # In compact the panel is only 1 in, so putting the median text **above** the bar collides
        # with the points on the right. Attached to the left of the bar it never collides, however
        # the points scatter.
        if compact:
            ax.annotate(f"{g.ddg.median():.1f}%", (xb - 0.075, g.ddg.median()),
                        textcoords="offset points", xytext=(-1, 0), ha="right",
                        va="center", fontsize=6.5, color=col, fontweight="bold",
                        annotation_clip=False)
        else:
            ax.annotate(f"{g.ddg.median():.1f}%", (xb, g.ddg.median()),
                        textcoords="offset points", xytext=(0, 4.5), ha="center",
                        va="bottom", fontsize=6.8, color=col, fontweight="bold",
                        annotation_clip=False)
        ax.scatter([k] * len(g), g.ddg, s=26, marker=mk, facecolor=col,
                   edgecolor="white", linewidths=0.6, zorder=4)

        if compact:
            # Without a label, coincident points look like one. Points with equal (or very close)
            # values are **spread symmetrically along x** so the count is visible. The vertical
            # position is untouched.
            v = g.ddg.to_numpy()
            tol = (f.ddg.max() - f.ddg.min()) * 0.012
            dx = np.zeros(len(v))
            i = 0
            while i < len(v):
                j = i
                while j + 1 < len(v) and abs(v[j + 1] - v[i]) <= tol:
                    j += 1
                m = j - i + 1
                if m > 1:
                    dx[i:j + 1] = (np.arange(m) - (m - 1) / 2) * 0.085
                i = j + 1
            ax.scatter(k + dx, v, s=20, marker=mk, facecolor=col,
                       edgecolor="white", linewidths=0.55, zorder=4,
                       label=f"structure-{'aware' if k == 0 else 'unaware'} ({len(g)})")
            for bb, r in g.iterrows():
                rows.append(dict(panel=tag, axis=axis_name, backbone=SHOW[bb],
                                 ddg_share_pct=r.ddg, structure_aware=bool(r.struct),
                                 p_mannwhitney=p, ranges_disjoint=sep))
            continue

        # Backbones with **exactly equal** values have fully coincident points. Separating their
        # labels would make it impossible to tell which point each belongs to and would only cost
        # height, so they go on one line (S2F, SaProt and AIDO-RAG are all 47.3684% on expression).
        # The source CSV keeps them separate, one row per backbone.
        tie = {}
        for bb, r in g.iterrows():
            tie.setdefault(round(r.ddg, 6), []).append(SHOW[bb])
            rows.append(dict(panel=tag, axis=axis_name, backbone=SHOW[bb],
                             ddg_share_pct=r.ddg, structure_aware=bool(r.struct),
                             p_mannwhitney=p, ranges_disjoint=sep))
        vals = sorted(tie)
        pos = spread(vals, span)
        for j, v in enumerate(vals):
            ax.annotate(" / ".join(tie[v]), (k, pos[j]), textcoords="offset points",
                        xytext=(7, -2.2), ha="left", fontsize=6.8, color=PAL["gray"],
                        annotation_clip=False)
    if sep:
        ax.axhspan(a.ddg.max(), b.ddg.min(), color=PAL["gray"], alpha=0.14, lw=0, zorder=1)
    if compact:            # the legend names the groups, so the x axis is emptied entirely
        ax.set_xlim(-0.92, 1.30); ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    else:
        ax.set_xlim(-0.44, 1.72); ax.set_xticks([0, 1])
        ax.set_xticklabels([f"structure-aware\n({len(a)})",
                            f"structure-unaware\n({len(b)})"])
    ax.margins(y=0.15)
    ax.set_title(f"{tag}   {title}", loc="left", fontsize=tfs, fontweight="bold", pad=5)
    return p, sep



def needs_panel_bymodel(ax, D, axis_name, tag, title, cmap, rows, tfs=8.2):
    """The same geometry as compact, but the point colour names the model. main builds the legend once."""
    f = D[(D.axis == axis_name) & (D.bb.isin(SHOW))].set_index("bb")
    tot = f[SL].abs().sum(axis=1)
    f["ddg"] = (f.a_ddG + f.b_ddG) / tot * 100
    f["struct"] = [b in STRUCT for b in f.index]
    a, b = f[f.struct].sort_values("ddg"), f[~f.struct].sort_values("ddg")
    sep = a.ddg.max() < b.ddg.min()
    p = mannwhitneyu(a.ddg, b.ddg).pvalue
    # Overlap has to be judged by **how much the markers actually overlap**. The panel is 2 in tall
    # over a value range of 20 to 23, so one unit is about 6 pt and the marker is 5.2 pt: 0.012
    # (about 0.25) does not catch coincident points. At 0.05 (about 1.0) every visually overlapping
    # point is spread along x.
    tol = (f.ddg.max() - f.ddg.min()) * 0.05
    for k, g in enumerate([a, b]):
        xb = k - 0.235
        # Bars and medians are **group summaries**, so they stay neutral grey. Colour here would collide with the model colours.
        ax.plot([xb, xb], [g.ddg.min(), g.ddg.max()], color=PAL["gray"], lw=1.5,
                alpha=0.38, solid_capstyle="round", zorder=2)
        ax.plot([xb - 0.07, xb + 0.07], [g.ddg.median()] * 2, color=PAL["gray"],
                lw=2.0, zorder=3)
        ax.annotate(f"{g.ddg.median():.1f}%", (xb - 0.09, g.ddg.median()),
                    textcoords="offset points", xytext=(-1, 0), ha="right", va="center",
                    fontsize=6.5, color=PAL["gray"], fontweight="bold",
                    annotation_clip=False)
        v = g.ddg.to_numpy()
        dx, i = np.zeros(len(v)), 0
        while i < len(v):                      # spread coincident points symmetrically along x
            j = i
            while j + 1 < len(v) and abs(v[j + 1] - v[i]) <= tol:
                j += 1
            m = j - i + 1
            if m > 1:
                dx[i:j + 1] = (np.arange(m) - (m - 1) / 2) * 0.128
            i = j + 1
        for q, (bb, r) in enumerate(g.iterrows()):
            ax.scatter([k + dx[q]], [r.ddg], s=27, marker=MARKER[bb],
                       facecolor=cmap[bb], edgecolor="white", linewidths=0.55, zorder=4)
            rows.append(dict(panel=tag, axis=axis_name, backbone=SHOW[bb],
                             ddg_share_pct=r.ddg, structure_aware=bool(r.struct),
                             p_mannwhitney=p, ranges_disjoint=sep))
    if sep:
        ax.axhspan(a.ddg.max(), b.ddg.min(), color=PAL["gray"], alpha=0.12, lw=0, zorder=1)
    ax.set_xlim(-0.92, 1.42)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"structure\naware ({len(a)})", f"structure\nunaware ({len(b)})"],
                       fontsize=6.6)
    ax.margins(y=0.15)
    ax.set_title(f"{tag}   {title}", loc="left", fontsize=tfs, fontweight="bold", pad=5)
    return p, sep


def main(mode: str = "stack"):
    """mode = stack | wide | compact.

      stack    one column, stacked vertically. Every model label is drawn. (default)
      wide     two columns, side by side. Labels kept.
      compact  one column, side by side. Labels dropped; the legend distinguishes the groups only.
      bymodel  one column, side by side. **The point colour names the model and the legend lists
               all 12.** The group is told by the x position and the colour family (cool/warm),
               and the marker shape is a secondary channel that separates models of similar colour.

    Side by side with labels does not fit in one column (3.42 in): each panel is 1.43 in and the
    longest label ("S2F / SaProt / AIDO-RAG") is 0.94 in, which would cover the neighbouring
    group's points. So the only single-column side-by-side layout is compact, without labels.
    """
    wide = mode == "wide"
    compact = mode == "compact"
    bymodel = mode == "bymodel"
    style()
    (H / "figures").mkdir(parents=True, exist_ok=True)
    D = pd.read_csv(H / "extend_backbones.csv")
    # ⚠ The y axis must not be shared. The two groups are 0.3 percentage points apart on function,
    #   and pooling the axes widens the range to 8.9 to 58.8, which turns that gap into 0.013 in.
    if bymodel:
        # The legend is ordered by ascending function-axis value, which is how panel a reads bottom to top.
        g = D[(D.axis == "function") & (D.bb.isin(SHOW))].set_index("bb")
        g = (g.a_ddG + g.b_ddG) / g[SL].abs().sum(axis=1)
        aw_o = [b for b in g.sort_values().index if b in STRUCT]
        un_o = [b for b in g.sort_values().index if b not in STRUCT]
        cmap = CMAP
        order = aw_o + un_o          # the legend is grouped the same way
        fig, axes = plt.subplots(1, 2, figsize=(3.42, 3.05))
        rows = []
        p1, s1 = needs_panel_bymodel(axes[0], D, "function", "a", "Function tuning set", cmap, rows, 7.2)
        p2, s2 = needs_panel_bymodel(axes[1], D, "stability", "b", "Expression tuning set", cmap, rows, 7.2)
        fig.supylabel("weight share on the $\\Delta\\Delta G$ channel  (%)",
                      x=0.016, fontsize=7.5)
        hh = [plt.Line2D([], [], ls="", marker=MARKER[bb], ms=4.6,
                         mfc=cmap[bb], mec="white", mew=0.5, label=FULL[bb])
              for bb in order]
        # 4 columns x 3 rows. Even with full names, 7.0 pt gives 3.04 in, inside the 3.30 in
        # available. The last column is "Tranception L (ret)" and so is the widest; at 0.545 the
        # closing bracket is clipped on the right, so the centre is pulled left and the column
        # spacing tightened.
        fig.legend(hh, [FULL[bb] for bb in order], loc="lower center", ncol=4,
                   bbox_to_anchor=(0.487, 0.0), handletextpad=0.28,
                   columnspacing=0.62, labelspacing=0.38, fontsize=7.0)
        out = H / "figures/fig_model_needs_2ax_bymodel"
        pd.DataFrame(rows).to_csv(out.with_name(out.name + "_data.csv"), index=False)
        fig.subplots_adjust(left=0.150, right=0.995, top=0.928, bottom=0.295, wspace=0.38)
        for fmt in ("pdf", "png"):
            fig.savefig(out.with_suffix(f".{fmt}"), dpi=300)
        plt.close(fig)
        print(f"wrote: {out.with_suffix('.pdf')}")
        print(f"  function   p={p1:.4f}  separated={s1}")
        print(f"  expression p={p2:.4f}  separated={s2}")
        return

    if wide:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.70))
    elif compact:
        fig, axes = plt.subplots(1, 2, figsize=(3.42, 2.25))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(3.42, 3.90))
    rows = []
    p1, s1 = needs_panel(axes[0], D, "function", "a", "Function tuning set", rows, compact, 7.2 if compact else 8.2)
    p2, s2 = needs_panel(axes[1], D, "stability", "b", "Expression tuning set", rows, compact, 7.2 if compact else 8.2)
    YL = "weight share on the $\\Delta\\Delta G$ channel  (%)"
    if wide:
        axes[0].set_ylabel(YL)
        fig.subplots_adjust(left=0.098, right=0.995, top=0.905, bottom=0.195, wspace=0.235)
    elif compact:
        fig.supylabel(YL, x=0.016, fontsize=7.5)
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=2, bbox_to_anchor=(0.56, -0.012),
                   handletextpad=0.35, columnspacing=1.3, fontsize=7.0, scatterpoints=1)
        fig.subplots_adjust(left=0.150, right=0.995, top=0.885, bottom=0.165, wspace=0.40)
    else:
        # Same quantity, so the axis label appears once. The x tick text of the upper panel duplicates the lower one and is removed.
        axes[0].set_xticklabels([])
        fig.supylabel("weight share on the\n$\\Delta\\Delta G$ channel  (%)",
                      x=0.012, fontsize=8.0)   # stacked panels are short, so two lines are kept
        fig.subplots_adjust(left=0.205, right=0.995, top=0.945, bottom=0.115, hspace=0.26)
    out = H / ("figures/fig_model_needs_2ax_wide" if wide else
               "figures/fig_model_needs_2ax_compact" if compact else
               "figures/fig_model_needs_2ax")
    pd.DataFrame(rows).to_csv(out.with_name(out.name + "_data.csv"), index=False)
    for fmt in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{fmt}"), dpi=300)
    plt.close(fig)
    print(f"wrote: {out.with_suffix('.pdf')}")
    print(f"  function   p={p1:.4f}  separated={s1}")
    print(f"  expression p={p2:.4f}  separated={s2}")


if __name__ == "__main__":
    main("wide" if "--wide" in sys.argv else
         "compact" if "--compact" in sys.argv else
         "bymodel" if "--bymodel" in sys.argv else "stack")
