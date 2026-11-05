"""Figure 2, layout v3. Panels a and b are **stacked vertically to share the backbone axis**,
with c and d standing to the right.

## Why the layout changed

  1x5 row    a and b were the narrowest panels (1.4 to 2.1 in), and the backbone names were drawn
             twice.
  2x2        a and b grew (2.9 in), but the figure became 5.35 in tall and the names were still
             only under a, so b had to be read across the row.
  v3 (here)  a and b **share one backbone axis**. Drawing the names once returns that width to the
             panels, and stacking the two puts the two axes of the same backbone side by side.
             c and d both suit a tall shape (c is a monotone curve, d a two-point slope).

## The resulting geometry

a and b change from horizontal boxes to **vertical boxes**: backbones on x, rho on y. Two boxes
(pLM, ProFusion) stand together at each backbone position.

input   results/extend_backbones_perdms.csv and frozen/paneld_perdms.csv
output  fig1_ext_v3.{pdf,png} and fig1_ext_v3_data.csv
run     python figures/plot_fig1_ext_v3.py
"""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_fig1_ext import (  # noqa: E402
    PALETTE, SHOW, CUTS, H, VRC, style, stars, arms_panel, cut_panel)


def box_panel_v(ax, D, axis, tag, title, order, rows, xlab, ylim, yticks):
    """Vertical boxes: backbone on x, rho on y, with (pLM, ProFusion) paired at each position."""
    data, pos, cols = [], [], []
    for i, bb in enumerate(order):
        d = D[(D.axis == axis) & (D.bb == bb)]
        a, c = d.base.to_numpy(), d.fit7.to_numpy()
        data += [a, c]
        pos += [i - 0.19, i + 0.19]
        cols += [PALETTE["baseline"], PALETTE["main"]]
        p = wilcoxon(c, a).pvalue
        rows.append(dict(panel=tag, axis=axis, backbone=SHOW[bb], n=len(a),
                         plm_mean=a.mean(), plm_sd=a.std(), profusion_mean=c.mean(),
                         profusion_sd=c.std(), plm_median=np.median(a),
                         profusion_median=np.median(c), delta=(c - a).mean(),
                         win_frac=float((c - a > 0).mean()), p=p))
        # Stars sit at a **fixed height near the top of the panel**. Attaching them to the top of
        #   each box makes the height vary per backbone, and they collide where a whisker is long.
        ax.text(i, 0.975, stars(p), ha="center", va="top", fontsize=6.0,
                color=PALETTE["gray_line"], transform=ax.get_xaxis_transform())
    bp = ax.boxplot(data, positions=pos, widths=0.33, patch_artist=True, vert=True,
                    showfliers=False, medianprops=dict(color="black", lw=0.8),
                    whiskerprops=dict(lw=0.55), capprops=dict(lw=0.55),
                    boxprops=dict(lw=0.45, edgecolor="black"))
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col)
    rng = np.random.default_rng(0)
    for dat, pp in zip(data, pos):
        ax.scatter(pp + rng.uniform(-0.085, 0.085, len(dat)), dat, s=0.9,
                   color="black", alpha=0.22, zorder=3, linewidths=0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SHOW[b] for b in order] if xlab else [],
                       rotation=38, ha="right", rotation_mode="anchor", fontsize=6.6)
    ax.set_xlim(-0.62, len(order) - 0.38)
    ax.set_ylim(*ylim); ax.set_yticks(yticks)
    ax.set_ylabel("Spearman ρ per assay")
    if ylim[0] < 0:
        ax.axhline(0, color=PALETTE["gray_line"], lw=0.5, ls=":")
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", pad=4)


def main():
    style()
    D = pd.read_csv(H / "extend_backbones_perdms.csv")
    D = D[D.bb.isin(SHOW)]
    order = (D[D.axis == "function"].groupby("bb").base.mean()
             .sort_values(ascending=False).index.tolist())
    n_f = D[(D.axis == "function") & (D.bb == order[0])].dms.nunique()
    n_e = D[(D.axis == "stability") & (D.bb == order[0])].dms.nunique()
    print(f"{len(order)} backbones, function {n_f} DMS, expression {n_e} DMS")

    fig = plt.figure(figsize=(7.2, 4.95))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.80, 1.0], wspace=0.19)
    # The two left panels **share the x axis**: the backbone names are drawn under b only.
    # The gap is still generous, because a crowded bottom axis of a against the title of b reads badly.
    left = outer[0].subgridspec(2, 1, hspace=0.22)
    right = outer[1].subgridspec(2, 1, hspace=0.46)   # room for the x axis label of c
    rows = []
    axa = fig.add_subplot(left[0])
    box_panel_v(axa, D, "function", "a", f"a   Function eval (n={n_f})", order, rows,
                xlab=False, ylim=(-0.28, 1.06),
                yticks=[-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axb = fig.add_subplot(left[1])
    box_panel_v(axb, D, "stability", "b", f"b   Expression eval (n={n_e})", order, rows,
                xlab=True, ylim=(0.0, 1.06),
                yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    h = [plt.Rectangle((0, 0), 1, 1, fc=PALETTE["baseline"], ec="black", lw=0.5),
         plt.Rectangle((0, 0), 1, 1, fc=PALETTE["main"], ec="black", lw=0.5)]
    # Because the stars sit at a **fixed height near the top**, the legend cannot go there. The
    # lower left (rho < 0) is empty for the four leftmost backbones, whose minima are +0.01 to +0.08.
    axa.legend(h, ["pLM alone", "ProFusion"], loc="lower left", ncol=2,
               handlelength=1.0, handletextpad=0.4, columnspacing=1.0,
               borderpad=0.25, fontsize=6.8)

    cut_panel(fig.add_subplot(right[0]), D, order, rows, ymax=55)
    arms_panel(fig.add_subplot(right[1]), rows, tag="d")

    R = pd.DataFrame(rows)
    R["axis"] = R["axis"].replace({"stability": "expression"})
    out = H / "figures/fig1_ext_v3"
    out.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(out.with_name("fig1_ext_v3_data.csv"), index=False)
    fig.subplots_adjust(left=0.068, right=0.978, top=0.945, bottom=0.145)
    for fmt in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{fmt}"), dpi=300)
    plt.close(fig)
    print("wrote:", out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
