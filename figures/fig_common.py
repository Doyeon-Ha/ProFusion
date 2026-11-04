"""Style, statistics and box-plot helpers shared by the fusion ablation figures.

Kept in one place so that Figure 3 (the control ladder, what fitting buys, the position/residual
split) and the supplementary channel-removal figure use the same visual system.

Conventions
  Axis names: 'stability' in the code is the ProteinGym Expression set, so the figures say
  Expression.
  Names: FIT3 = champ3, FIT5 = champ5, FIT7 = comp7 (the same 7 slots with different parameter
  tying).
  Significance: Wilcoxon signed-rank (paired over DMS) plus Benjamini-Hochberg within a panel,
  shown as box fill at q < 0.05.
"""
from dataclasses import dataclass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon, t as tdist  # noqa: E402

PALETTE = {"main": "#0F4D92", "alt": "#B64342", "extra1": "#42949E", "extra2": "#9A4D8E",
           "gray_line": "#767676", "shade": "#EDEDED"}
BBS = ["ESM2", "ESM-1v", "ESM3", "ProSST"]
KEY = {"ESM2": "ESM2", "ESM-1v": "ESM1v", "ESM3": "ESM3", "ProSST": "ProSST"}
BBC = [PALETTE["main"], PALETTE["alt"], PALETTE["extra1"], PALETTE["extra2"]]
AXL = {"function": "Function", "stability": "Expression"}
SIG_NOTE = ("box and whiskers show the spread over DMS (quartiles, 1.5 IQR, outliers hidden), "
            "and the black bar is the 95% CI of the mean.\n"
            "filled box = q<0.05, open = n.s.   Wilcoxon signed-rank paired by DMS, "
            "Benjamini-Hochberg within panel.")


@dataclass(frozen=True)
class FigureStyle:
    font_size: float = 8.0
    tick_size: float = 7.0
    axes_linewidth: float = 0.8
    font_family: tuple = ("Source Sans Pro", "Helvetica", "Arial", "DejaVu Sans")


def apply_publication_style(style=FigureStyle()):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": list(style.font_family),
        "font.size": style.font_size, "axes.labelsize": style.font_size,
        "xtick.labelsize": style.tick_size, "ytick.labelsize": style.tick_size,
        "legend.fontsize": style.tick_size + 0.5, "legend.frameon": False,
        "axes.linewidth": style.axes_linewidth, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.major.width": style.axes_linewidth,
        "ytick.major.width": style.axes_linewidth, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})


def bh(ps):
    """Benjamini-Hochberg, applied within a panel only."""
    p = np.asarray(ps, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); run = 1.0
    for rank, i in zip(range(n, 0, -1), o[::-1]):
        run = min(run, p[i] * n / rank); q[i] = run
    return q


def panel_title(ax, letter, text):
    ax.set_title(f"{letter}   {text}", loc="left", fontsize=8.5, fontweight="bold", pad=4)


def whisker_span(items, pad=0.06):
    """A y range that fits the whole 1.5 IQR whisker. A clipped whisker invites misreading."""
    lo, hi = [], []
    for _, vals, _ in items:
        for v in vals.values():
            if v is None or len(v) == 0:
                continue
            q1, q3 = np.percentile(v, [25, 75]); i = q3 - q1
            lo.append(float(v[v >= q1 - 1.5 * i].min()))
            hi.append(float(v[v <= q3 + 1.5 * i].max()))
    a, b = min(lo), max(hi); m = (b - a) * pad
    return a - m, b + m


def box_group(ax, items, ylim=None, width=0.19, mean_ci=True):
    """items: [(x centre, dict of per-backbone arrays, q array)]. The caller decides the
    significance fill from q.

    mean_ci: overlay the **95% CI of the mean** on the box as a black bar. The box shows the spread
    across DMS (SD), but the claim is about where the mean sits, and that uncertainty is SD/sqrt(n),
    about ten times narrower. Without the overlay a q = 1e-7 result looks like noise.
    """
    ylim = ylim or whisker_span(items)
    for base, vals, qs in items:
        for bi, b in enumerate(BBS):
            v = vals.get(b)
            if v is None or len(v) == 0:
                continue
            x = base + (bi - 1.5) * width
            sig = qs[bi] < 0.05
            bp = ax.boxplot([v], positions=[x], widths=width * 0.84, patch_artist=True,
                            showfliers=False, medianprops=dict(color="black", lw=0.8),
                            whiskerprops=dict(lw=0.5, color="black" if sig else BBC[bi]),
                            capprops=dict(lw=0.5, color="black" if sig else BBC[bi]),
                            boxprops=dict(lw=0.4 if sig else 0.9,
                                          edgecolor="black" if sig else BBC[bi]))
            bp["boxes"][0].set_facecolor(BBC[bi] if sig else "white")
            if mean_ci and len(v) > 2 and np.std(v) > 1e-12:
                mu = float(np.mean(v))
                h = float(np.std(v, ddof=1) / np.sqrt(len(v)) * tdist.ppf(0.975, len(v) - 1))
                ax.plot([x, x], [mu - h, mu + h], color="black", lw=1.5,
                        solid_capstyle="butt", zorder=6)
                ax.plot([x], [mu], marker="o", ms=1.9, mfc="white", mec="black",
                        mew=0.5, zorder=7)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylim(*ylim)


def collect(getter, groups, rows, panel, ref, axis_tag):
    """Walk the groups, build the per-backbone arrays and BH q values, and record every number in rows."""
    cells, praw = [], []
    for key, lab in groups:
        vals = {b: getter(key, b) for b in BBS}
        ps = []
        for b in BBS:
            v = vals[b]
            ps.append(wilcoxon(v)[1] if v is not None and len(v) > 2 and np.std(v) > 1e-12 else 1.0)
        cells.append((lab, vals, ps)); praw += ps
    qs = bh(praw).reshape(len(groups), len(BBS))
    out = []
    for gi, (lab, vals, ps) in enumerate(cells):
        out.append((vals, qs[gi]))
        for bi, b in enumerate(BBS):
            v = vals[b]
            if v is None:
                continue
            rows.append(dict(panel=panel, axis=axis_tag, reference=ref, group=lab, backbone=b,
                             n=len(v), mean=float(np.mean(v)), median=float(np.median(v)),
                             p_wilcoxon=ps[bi], q_bh=float(qs[gi, bi])))
    return out


def two_block_x(n_per_block, gap_between=1.15):
    """The x centres of the Function and Expression blocks, and where the divider goes."""
    left = list(range(n_per_block))
    off = n_per_block - 1 + gap_between
    right = [off + i for i in range(n_per_block)]
    return left, right, (left[-1] + right[0]) / 2


def backbone_legend(fig, y=1.003, fontsize=8.0):
    hb = [plt.Line2D([], [], color=c, lw=1.6, marker="s", ms=4.2, mew=0) for c in BBC]
    fig.legend(hb, BBS, ncol=4, loc="upper center", bbox_to_anchor=(0.5, y),
               handlelength=1.1, columnspacing=1.8, handletextpad=0.45, fontsize=fontsize)
