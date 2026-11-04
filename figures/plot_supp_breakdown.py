"""Supplementary figure: backbones separate in the **left tail** of the distribution, not at its
centre.

## What it shows

The default is the 129 function assays, drawn as the per-assay rho distribution of each backbone.
The 11 sequence and structure-conditioned pLMs have medians clustered between 0.366 and 0.503
(span 0.136), while their 10th percentiles run from 0.045 to 0.330 (span 0.285), 2.1 times wider.
The quartile and upper spans are 0.132 to 0.169, so they separate **only at the left end**. No
threshold (such as rho < 0.2) is used; the statement is made in quantile spans alone.
The same holds for `--both` (147), which is what the paper carries: span 0.138 against 0.280,
2.0 times.

⚠ ProteinMPNN is excluded from the numeric claim. It is an inverse-folding model, so variant
  effect prediction is not its purpose. It is drawn **separately, in another colour**, and the
  numbers are stated over the other 11.
  On the old cohort (112) it was lowest at every quantile and including it reversed the claim.
  On the current cohort it does not: it is no longer lowest at the 10th percentile, so that span
  stays at 0.285 and only the median span grows, from 0.136 to 0.180. The ratio **falls from 2.1
  to 1.6 but does not reverse**. Drawing it separately is still easier to read.

## What is measured

  rho            Spearman between the zero-shot LLR and the experimental value within one assay.
                 Unitless, [-1, 1].
  quantile span  max minus min over the 11 backbones at that quantile. Larger means the choice of
                 backbone matters more at that point.
  GEMME          a reference line: the mean rho of the MSA-based conservation score alone, not a pLM.

## Panels

  a  violins per backbone with the individual assay points. The bar is the IQR and the white dot
     the median, ordered by descending median.
  b  12 cumulative distributions. rho is cut at steps of 0.1, the fraction of assays at or below
     each cut is averaged over the 12 backbones and marked with a diamond (the value is written
     too). The lines fanning out on the left and converging on the right is visible in the bundle
     itself.

input   results/extend_backbones_perdms.csv
output  results/figures/figS_breakdown.{pdf,png}, figS_breakdown_data.csv, figS_breakdown_stats.csv
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plmfusion.paths import RESULTS  # noqa: E402


PAL = {"main": "#0F4D92", "alt": "#B64342", "gray": "#767676", "pale": "#CFCECE"}
H = RESULTS          # reads extend_backbones_perdms.csv, writes under figures/
SHOW = {"AIDO_RAG16B": "AIDO-RAG 16B", "VenusREM": "VenusREM", "ProSST": "ProSST",
        "VespaG": "VespaG", "TranceptionLret": "Tranception L", "S2F": "S2F-MSA",
        "ESM1v": "ESM-1v", "SaProt650M": "SaProt 650M", "ESM3": "ESM3-open",
        "ESM2": "ESM2 650M", "ESMIF1": "ESM-IF1", "ProteinMPNN": "ProteinMPNN"}
SCOPE = "function"
ODD = "ProteinMPNN"          # excluded from the numeric claim only
QS = [10, 25, 50, 75, 90]
CUTS = np.round(np.arange(0.0, 0.81, 0.1), 2)


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Source Sans Pro", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.2, "axes.labelsize": 7.2, "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8, "legend.fontsize": 6.5, "legend.frameon": False,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "svg.fonttype": "none", "pdf.fonttype": 42,
        "ps.fonttype": 42})


def panel_violin(ax, V, order, gem_unused=None, tag="a   "):
    """Per-assay rho distribution per backbone. Passing tag="" gives the standalone version (no panel letter)."""
    rng = np.random.default_rng(0)
    rng = np.random.default_rng(0)
    for i, b in enumerate(order):
        v = V[b]
        col = PAL["main"]
        pc = ax.violinplot(v, positions=[i], widths=0.78, showextrema=False)
        for body in pc["bodies"]:
            body.set_facecolor(col); body.set_alpha(0.16)
            body.set_edgecolor(col); body.set_linewidth(0.6); body.set_zorder(2)
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=2.0, color=col,
                   alpha=0.50, linewidths=0, zorder=3)
        q1, md, q3 = np.percentile(v, [25, 50, 75])
        ax.vlines(i, q1, q3, color="black", lw=2.4, zorder=4)
        ax.plot([i], [md], marker="o", ms=2.8, color="white",
                markeredgecolor="black", markeredgewidth=0.5, zorder=5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SHOW[b] for b in order], rotation=38, ha="right",
                       rotation_mode="anchor")
    ax.set_xlim(-0.75, len(order) - 0.25)
    ax.set_ylim(-0.25, 0.95)
    ax.set_ylabel("Spearman ρ per assay")
    # A title longer than the panel gets clipped. "112 function" (2.80 in) fits, but
    # "128 ProteinGym" (2.97 in) exceeds the 2.86 in panel. **The width is measured and the text
    # shrunk** (7.8 -> 6.8 pt) so that a change of set never clips it.
    t = (f"{tag}Per-assay accuracy of each backbone "
         f"({len(V[order[0]])} {SCOPE} assays)")
    fs = 7.8
    fig = ax.figure
    fig.canvas.draw()
    avail = ax.get_window_extent().width / fig.dpi
    while fs > 6.8:
        w = fig.text(0, 0, t, fontsize=fs, fontweight="bold").get_window_extent(
            renderer=fig.canvas.get_renderer()).width / fig.dpi
        fig.texts.pop()
        if w <= avail:
            break
        fs -= 0.2
    ax.set_title(t, loc="left", fontsize=fs, fontweight="bold", pad=4)



def panel_violin_h(ax, V, order):
    """Single-column standalone: **horizontal violins**. The names stand along the y axis, so
    nothing is rotated.

    The vertical version rotated the 12 backbone names by 38 degrees under the x axis, and that
    text took 30% of the figure height, making it almost square at 3.42 x 3.45 in. In one column
    that fills much of a page, so LaTeX shrinks the width and the text with it.
    Laid horizontally, (1) there are no rotated labels to read, (2) the height is only 12 rows, and
    (3) the width that frees up all goes to the rho axis.

    No title is drawn inside the figure: it would duplicate the caption and cost height.
    """
    rng = np.random.default_rng(0)
    for i, b in enumerate(order):
        v = V[b]
        y = len(order) - 1 - i                    # top to bottom in order
        pc = ax.violinplot(v, positions=[y], widths=0.80, vert=False,
                           showextrema=False)
        for body in pc["bodies"]:
            body.set_facecolor(PAL["main"]); body.set_alpha(0.16)
            body.set_edgecolor(PAL["main"]); body.set_linewidth(0.5); body.set_zorder(2)
        ax.scatter(v, y + rng.uniform(-0.17, 0.17, len(v)), s=1.6, color=PAL["main"],
                   alpha=0.45, linewidths=0, zorder=3)
        q1, md, q3 = np.percentile(v, [25, 50, 75])
        ax.hlines(y, q1, q3, color="black", lw=2.2, zorder=4)
        ax.plot([md], [y], marker="o", ms=2.6, color="white", markeredgecolor="black",
                markeredgewidth=0.5, zorder=5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHOW[b] for b in order][::-1])
    ax.set_ylim(-0.62, len(order) - 0.38)
    ax.set_xlim(-0.25, 0.92)
    ax.set_xlabel("Spearman ρ per assay")
    ax.tick_params(axis="y", length=0, pad=1.5)


def panel_ecdf(ax, V, order, tag="b   ", fs_title=7.8, fs_note=6.2, ms=4.2):
    """Cumulative distribution of the same values, with mean diamonds at steps of 0.1.

    fs_title, fs_note and ms are **sizes as printed**. Shrinking the figure in LaTeX shrinks them
    too, so they are raised only by the caller that inserts it at full size (ecdf_b).
    """
    for b in order:
        v = np.sort(V[b])
        y = np.arange(1, len(v) + 1) / len(v) * 100
        ax.step(v, y, where="post", color=PAL["main"], lw=0.7, alpha=0.30, zorder=3)
    mean_pct = np.array([np.mean([(V[b] <= c).mean() for b in order]) * 100
                         for c in CUTS])
    ax.plot(CUTS, mean_pct, color=PAL["main"], lw=0.9, zorder=5)
    ax.plot(CUTS, mean_pct, marker="D", ms=ms, lw=0, color=PAL["main"],
            markeredgecolor="white", markeredgewidth=0.6, zorder=6)
    for c, m in zip(CUTS, mean_pct):             # values are written outside the curve
        lo = m < 12                              # the two that sit on the floor go above
        ax.annotate(f"{m:.0f}", (c, m), textcoords="offset points",
                    xytext=(5.5, 4.5 if lo else -4.5), ha="left",
                    va="bottom" if lo else "top", fontsize=fs_note,
                    fontweight="bold", color=PAL["main"], zorder=7)
    ax.set_xlim(-0.25, 0.95); ax.set_ylim(-2, 105)   # so the 100% diamond is not clipped
    ax.set_xlabel("Spearman ρ per assay")
    ax.set_ylabel("assays at or below ρ  (%)")
    ax.set_xticks(np.arange(-0.2, 0.81, 0.2))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.spines["left"].set_bounds(0, 100)
    hdl = [plt.Line2D([], [], color=PAL["main"], lw=1.0, alpha=0.45,
                      label="one backbone"),
           plt.Line2D([], [], color=PAL["main"], lw=0.9, marker="D", ms=ms,
                      markeredgecolor="white", markeredgewidth=0.6,
                      label="mean of the 12")]
    ax.legend(handles=hdl, loc="upper left", handlelength=1.5, handletextpad=0.5,
              labelspacing=0.3, borderpad=0.2)
    ax.set_title(f"{tag}Cumulative distribution", loc="left",
                 fontsize=fs_title, fontweight="bold", pad=4)



def main(split: bool = False, both: bool = False, ecdf_b: bool = False,
         v113: bool = False):
    """split=True draws panels a and b as **two separate single-column figures** (no panel letters).

    This is for the case where the appendix carries the two panels separately. A standalone figure
    has no partner for "a"/"b" to point at, so the tag is left empty.

    both=True draws over **all 147**, function 129 plus expression 18. This is what the paper
    carries. The conclusion is the same (11 pLM median span 0.138 against p10 span 0.280; on the
    function axis alone it is 0.136 against 0.285). The output name gains _147: the file name
    always carries **the actual assay count**.
    """
    style()
    (H / "figures").mkdir(parents=True, exist_ok=True)
    # v113 fills Q6WV12 into the 113-assay function axis. Every other column and backbone matches the 112 version.
    P = pd.read_csv(H / ("extend_backbones_perdms_113.csv" if v113
                         else "extend_backbones_perdms.csv"))
    AXES = ["function", "stability"] if both else ["function"]
    global SCOPE
    SCOPE = "ProteinGym" if both else "function"
    f = P[P.axis.isin(AXES) & P.bb.isin(SHOW)]
    V = {b: f[f.bb == b].base.to_numpy() for b in SHOW}
    order = sorted(SHOW, key=lambda b: -np.median(V[b]))        # descending median
    core = [b for b in order if b != ODD]                        # the 11 used for the claim

    # Quantile spans: the claim of the figure itself
    QT = {q: np.array([np.percentile(V[b], q) for b in core]) for q in QS}
    span = {q: QT[q].max() - QT[q].min() for q in QS}

    # The file name carries **the actual assay count**; only the 112 default goes without a suffix.
    n_tot = f.dms.nunique()
    SFX = "" if n_tot == 112 else f"_{n_tot}"
    if split:
        # Two single-column standalone figures. The violin one gets more height, because the 12
        # backbones stand along its x axis.
        # ⚠ Standalone figures carry no "a"/"b": there is no partner to point at.
        for nm, fn, hgt, adj in [
                (f"figS_breakdown_violin{SFX}", panel_violin_h, 2.55,
                 dict(left=0.268, right=0.988, top=0.985, bottom=0.138)),
                (f"figS_breakdown_ecdf{SFX}", panel_ecdf, 2.70,
                 dict(left=0.165, right=0.985, top=0.940, bottom=0.165))]:
            f1 = plt.figure(figsize=(3.42, hgt))
            axn = f1.add_subplot(111)
            fn(axn, V, order) if fn is panel_violin_h else fn(axn, V, order, tag="")
            f1.subplots_adjust(**adj)
            o1 = H / f"figures/{nm}"
            for fmt in ("pdf", "png"):
                f1.savefig(o1.with_suffix(f".{fmt}"), dpi=300)
            plt.close(f1)
            print("wrote:", o1.with_suffix(".pdf"))
        return

    if ecdf_b:
        # For the supplementary figure whose panel a became a table: the ECDF alone. The table is a
        # LaTeX tabular, so the two stand side by side inside one float.
        # ⚠ The size is made **exactly as printed**. The old version exported 3.30 x 2.85 in and
        # LaTeX shrank it to 0.385\textwidth (2.50 in), a scale of 0.758, so the tick text landed
        # at 5.2 pt instead of 6.8 pt while the table beside it was 9.96 pt because the class file
        # fixes \footnotesize at 10 pt. The text differed by a factor of two (table 192 pt against
        # figure 156 pt) and the heights did not match either, so panel b alone looked small.
        # It is now 2.50 in wide = 0.385\textwidth, a scale of 1.0, and the height of 2.89 in
        # matches the table (from the top of the panel letter to \bottomrule, 208 pt).
        # When changing these dimensions, check the minipage width in the supplementary source too.
        f2 = plt.figure(figsize=(2.50, 2.89))
        with plt.rc_context({"axes.labelsize": 8.0, "xtick.labelsize": 7.4,
                             "ytick.labelsize": 7.4, "legend.fontsize": 7.2}):
            panel_ecdf(f2.add_subplot(111), V, order, tag="b   ",
                       fs_title=9.4, fs_note=6.8, ms=4.0)
        # top is chosen so that the top of the panel letter "b" sits level with the "a" on the table side.
        f2.subplots_adjust(left=0.205, right=0.995, top=0.930, bottom=0.125)
        o2 = H / f"figures/figS_breakdown_ecdf_b{SFX}"
        for fmt in ("pdf", "png"):
            f2.savefig(o2.with_suffix(f".{fmt}"), dpi=300)
        plt.close(f2)
        print("wrote:", o2.with_suffix(".pdf"))
        return

    fig = plt.figure(figsize=(7.2, 2.95))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.30, 1.0], wspace=0.22)

    # -- a, distribution per backbone
    panel_violin(fig.add_subplot(gs[0]), V, order, tag="a   ")
    panel_ecdf(fig.add_subplot(gs[1]), V, order, tag="b   ")
    rows = [dict(backbone=SHOW[b], **{f"p{q}": float(np.percentile(V[b], q)) for q in QS},
                 mean=float(V[b].mean()), n=len(V[b]),
                 in_11pLM_claim=(b != ODD)) for b in order]
    rows.append(dict(backbone="SPAN across the 11 pLMs",
                     **{f"p{q}": float(span[q]) for q in QS},
                     mean=float("nan"), n=11, in_11pLM_claim=True))
    D = pd.DataFrame(rows)
    D.to_csv(H / f"figures/figS_breakdown{SFX}_data.csv", index=False)
    pd.DataFrame([dict(cutoff=float(c),
                       mean_pct_at_or_below=float(np.mean([(V[b] <= c).mean()
                                                           for b in order]) * 100),
                       min_pct=float(min((V[b] <= c).mean() for b in order) * 100),
                       max_pct=float(max((V[b] <= c).mean() for b in order) * 100))
                  for c in CUTS]).to_csv(H / f"figures/figS_breakdown{SFX}_cutoffs.csv",
                                         index=False)

    # The tests the text cites: quantile spans compared by bootstrapping over assays, and whether
    # the lower-tail ranking follows the median ranking.
    dms = sorted(f.dms.unique())
    M = np.column_stack([f[f.bb == b].set_index("dms").loc[dms, "base"].to_numpy()
                         for b in core])                        # 112 x 11
    sp = lambda X, q: np.ptp(np.percentile(X, q, axis=0))
    rng2 = np.random.default_rng(0); B = 10000
    boot = {q: np.empty(B) for q in QS}
    for t in range(B):
        X = M[rng2.integers(0, M.shape[0], M.shape[0])]
        for q in QS:
            boot[q][t] = sp(X, q)
    st = [dict(stat=f"span_p{q}", value=span[q],
               ci_lo=float(np.percentile(boot[q], 2.5)),
               ci_hi=float(np.percentile(boot[q], 97.5)),
               p_span10_greater=float((boot[10] > boot[q]).mean()) if q != 10 else np.nan)
          for q in QS]
    rs = spearmanr(np.percentile(M, 10, axis=0), np.percentile(M, 50, axis=0))
    st.append(dict(stat="spearman_p10_vs_median_across_11pLM", value=float(rs.statistic),
                   ci_lo=np.nan, ci_hi=np.nan, p_span10_greater=float(rs.pvalue)))
    pd.DataFrame(st).to_csv(H / f"figures/figS_breakdown{SFX}_stats.csv", index=False)
    print(pd.DataFrame(st).round(4).to_string(index=False))
    out = H / f"figures/figS_breakdown{SFX}"
    fig.subplots_adjust(left=0.063, right=0.995, top=0.915, bottom=0.265)
    for fmt in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{fmt}"), dpi=300)
    plt.close(fig)
    print("wrote:", out.with_suffix(".pdf"))
    print(D.round(3).to_string(index=False))


if __name__ == "__main__":
    import sys
    main(split="--split" in sys.argv, both="--both" in sys.argv,
         ecdf_b="--ecdf-b" in sys.argv, v113="--v113" in sys.argv)
