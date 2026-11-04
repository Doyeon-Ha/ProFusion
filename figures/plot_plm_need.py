"""Does the pLM earn its place on top of the auxiliary channels? (all function-axis assays)

## Why the baseline changes

The fusion gain has so far been reported **relative to pLM alone**. That leaves "the channels
alone reach 0.496, so why is a pLM needed?" unanswered. The baseline is therefore raised to
**channels only**, and what the pLM adds on top of that is measured. Panel a raises the weakness
first and b, c and d answer it.

  a  the 5 channels alone, plus the 7-slot combination **refitted for channels only** (0.502).
     ⚠ Using "the w fitted jointly with the pLM, applied without the pLM" (0.496) as the baseline
       is unfair to the channels. Refitting for channels only raises it by +0.0054. The fair one
       is used.
  b  the lambda sweep. The dashed line (channels only) is the **floor**, not the ceiling, and
     every curve peaks above it.
  c  d (fusion minus channels only) per backbone. **10 of 12 are significantly positive.** The
     exact numbers are printed at run time rather than written here, where they would go stale.
     ESM-IF1 is not significant (+0.0015) and ProteinMPNN is negative (−0.0094).
  d  because the channel baseline is nearly the same for every backbone (0.499), d is the fusion
     result shifted down by a constant. So d does not show "contribution": it shows that **fusion
     does not erase the differences between backbones** (Spearman +0.85). Dropping the two
     inverse-folding models still leaves +0.75 (n = 10, p = 0.017), so it is not a line made by
     two points.

⚠ Two readings to avoid
  - "it does not work without a pLM": the channels alone reach 0.496.
  - "the pLM props up shallow MSAs": the direction is the opposite (High +0.021 > Low +0.012).
⚠ ProteinMPNN is not removed. It lands exactly on the left end of the regression line in d, which
  makes it a confirmation of "the contribution tracks pLM quality", not a counterexample.

input   results/plm_need_perdms.csv and results/plm_vs_channel_{solo,sweep}.csv
output  results/figures/fig_plm_need.{pdf,png} and _data.csv
run     python figures/plot_plm_need.py   (after scripts/plm_need.py and scripts/plm_vs_channel.py)
"""
import sys  # noqa: E402
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.cm as cm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plmfusion.paths import RESULTS  # noqa: E402

H = RESULTS
FIGS = RESULTS / "figures"
SHOW = {"AIDO_RAG16B": "AIDO-RAG 16B", "VenusREM": "VenusREM", "ProSST": "ProSST",
        "VespaG": "VespaG", "TranceptionLret": "Tranception L", "S2F": "S2F-MSA",
        "ESM1v": "ESM-1v", "SaProt650M": "SaProt 650M", "ESM3": "ESM3-open",
        "ESM2": "ESM2 650M", "ESMIF1": "ESM-IF1", "ProteinMPNN": "ProteinMPNN"}
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbone_key import CMAP, MARKER, STRUCT  # noqa: E402

INK, SOFT = "#3A3A3A", "#BDBDBD"
MAIN, NEG = "#0F4D92", "#B64342"


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Source Sans Pro", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.2, "axes.labelsize": 7.2, "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0, "legend.fontsize": 6.4, "legend.frameon": False,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "svg.fonttype": "none", "pdf.fonttype": 42,
        "ps.fonttype": 42})


def ink(c, k=0.45, cap=0x6A):
    """Darken a hex colour for text. backbone_key supplies hex strings."""
    r, g, b = (int(c[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(min(int(v * (1 - k)), cap) for v in (r, g, b))


def main():
    style()
    D = pd.read_csv(H / "plm_need_perdms.csv")
    O = pd.read_csv(H / "plm_vs_channel_solo.csv")
    S = pd.read_csv(H / "plm_vs_channel_sweep.csv")
    # The baseline is the **channel-only refit** (chan_refit). Using the current w without the pLM
    # (chan) is unfair to the channels and inflates d (0.4962 vs 0.5016).
    g = D.groupby("bb").agg(plm=("plm", "mean"), chan=("chan_refit", "mean"),
                            fused=("fused", "mean"), d=("d_refit", "mean"))
    g["ci"] = D.groupby("bb").d_refit.apply(
        lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x)))
    g["p"] = D.groupby("bb").apply(
        lambda t: wilcoxon(t.fused, t.chan_refit).pvalue, include_groups=False)
    # ⚠ Colours and markers are **pinned by name** in backbone_key. Zipping a ramp against the
    #   sort order here would give the same model a different colour in each figure.
    order = g.plm.sort_values(ascending=False).index.tolist()
    cmap = CMAP
    chan_mean = g.chan.mean()

    # Layout: a has 6 rows and c has 12, so there is no reason to give them equal height.
    # Shrinking the top row and giving it to the bottom one unsqueezes the labels of c and takes
    # 0.3 in off the figure.
    fig = plt.figure(figsize=(7.2, 5.20))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.86, 1.0], hspace=0.40, wspace=0.26)

    # -- a, channels alone
    ax = fig.add_subplot(gs[0, 0])
    so = O[O.axis == "function"].groupby("channel").rho.mean().sort_values()
    lab = list(so.index) + ["five channels,\nweights refitted"]
    val = list(so.values) + [chan_mean]
    col = [SOFT] * len(so) + [INK]
    ax.barh(range(len(val)), val, color=col, height=0.62, zorder=3)
    for i, v in enumerate(val):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=6.4,
                color=INK, fontweight="bold" if i == len(val) - 1 else "normal")
    ax.set_yticks(range(len(val))); ax.set_yticklabels(lab)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_xlim(0, 0.60); ax.set_xlabel("mean Spearman ρ per assay")
    ax.set_title("a   External channels alone", loc="left", fontsize=8.0,
                 fontweight="bold", pad=5)

    # -- b, lambda sweep
    ax = fig.add_subplot(gs[0, 1])
    f = S[(S.axis == "function") & np.isfinite(S.lam)]
    piv = f.pivot_table(index="lam", columns="bb", values="rho")
    x = piv.index.to_numpy(); m = x <= 3.0
    for b in order:
        ax.plot(x[m], piv[b].to_numpy()[m], color=cmap[b], lw=1.0, zorder=3)
    ax.axvline(1.0, color=INK, lw=0.7, ls="--", zorder=2)
    ax.axhline(chan_mean, color=INK, lw=0.7, ls=":", zorder=2)
    ax.annotate("λ = 1", (1.0, 0.17), textcoords="offset points", xytext=(3, 0),
                ha="left", fontsize=6.4, color=INK)
    ax.annotate(f"channels alone ({chan_mean:.3f})", (3.0, chan_mean),
                textcoords="offset points", xytext=(-2, -3), ha="right", va="top",
                fontsize=6.2, color=INK)
    ax.set_xlim(-0.08, 3.08); ax.set_ylim(0.14, 0.56)
    ax.set_xlabel("λ   (channel weight relative to the pLM)")
    ax.set_ylabel("mean Spearman ρ per assay")
    ax.set_title("b   Trading the pLM against the channels", loc="left", fontsize=8.0,
                 fontweight="bold", pad=5)

    # -- c, d (fusion minus channels only)
    ax = fig.add_subplot(gs[1, 0])
    gc = g.sort_values("d")
    y = np.arange(len(gc))
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    for yy, (b, r) in zip(y, gc.iterrows()):
        ax.plot([0, r.d], [yy, yy], color=cmap[b], lw=1.1, alpha=0.45, zorder=2)
        ax.plot([r.d - r.ci, r.d + r.ci], [yy, yy], color=INK, lw=0.6, alpha=0.55,
                zorder=3)
        ax.scatter(r.d, yy, s=22, marker=MARKER[b],
                   facecolor=cmap[b] if r.p < 0.05 else "white",
                   edgecolor=cmap[b], linewidths=0.9, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([SHOW[b] for b in gc.index])
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_ylim(-0.7, len(gc) - 0.3)
    ax.set_xlabel("Δρ  (ProFusion − channels alone)")
    ax.set_title("c   What the pLM adds on top of the channels", loc="left",
                 fontsize=8.0, fontweight="bold", pad=5)

    # -- d, d against pLM quality
    ax = fig.add_subplot(gs[1, 1])
    rs = spearmanr(g.plm, g.d)
    b1, b0 = np.polyfit(g.plm, g.d, 1)
    xs = np.linspace(g.plm.min() - 0.02, g.plm.max() + 0.02, 10)
    ax.plot(xs, b1 * xs + b0, color=INK, lw=0.8, ls="--", zorder=2)
    ax.axhline(0, color=INK, lw=0.6, alpha=0.5, zorder=1)
    for b, r in g.iterrows():
        ax.scatter(r.plm, r.d, s=30, marker=MARKER[b], facecolor=cmap[b],
                   edgecolor="white", linewidths=0.6, zorder=4)
    for b in ("ProteinMPNN", "VenusREM", "ProSST", "AIDO_RAG16B"):
        ax.annotate(SHOW[b], (g.plm[b], g.d[b]), textcoords="offset points",
                    xytext=(5, -1), ha="left", va="center", fontsize=6.2,
                    color=ink(cmap[b]))
    ax.text(0.03, 0.96, f"Spearman {rs.statistic:+.2f}\n$p$ = {rs.pvalue:.3f}",
            transform=ax.transAxes, va="top", fontsize=6.6, color=INK,
            linespacing=1.3)
    ax.set_xlim(0.12, 0.60)
    ax.set_xlabel("pLM alone, mean Spearman ρ")
    ax.set_ylabel("Δρ  (ProFusion − channels alone)")
    ax.set_title("d   Fusion does not level the backbones", loc="left", fontsize=8.0,
                 fontweight="bold", pad=5)

    # The 12 curves in panel b carry no names. The legend appears once, below the figure: it uses
    # the same 6 x 2 layout and the same colours and markers as the ablation figure, so a reader
    # learns it only once.
    hh = [plt.Line2D([], [], ls="", marker=MARKER[b], ms=4.4, mfc=cmap[b],
                     mec=cmap[b], mew=0.5, label=SHOW[b]) for b in order]
    fig.legend(hh, [SHOW[b] for b in order], loc="lower center", ncol=6,
               bbox_to_anchor=(0.55, -0.006), handletextpad=0.35, columnspacing=1.5,
               labelspacing=0.42, fontsize=7.5)

    FIGS.mkdir(parents=True, exist_ok=True)
    g.assign(backbone=[SHOW[b] for b in g.index]).to_csv(
        FIGS / "fig_plm_need_data.csv", index=False)
    out = FIGS / "fig_plm_need"
    fig.subplots_adjust(left=0.135, right=0.985, top=0.950, bottom=0.172)
    for fmt in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{fmt}"), dpi=300)
    plt.close(fig)
    print("wrote:", out.with_suffix(".pdf"))
    pool = D.groupby("dms").d_refit.mean()
    print(f"  pooled Δ {pool.mean():+.4f}  {int((pool>0).sum())}/{len(pool)}  "
          f"p={wilcoxon(pool, np.zeros(len(pool))).pvalue:.2g}")
    print(f"  significant positives {int(((g.d>0)&(g.p<0.05)).sum())}/12, negative "
          f"{list(g[g.d<0].index)}")


if __name__ == "__main__":
    main()
