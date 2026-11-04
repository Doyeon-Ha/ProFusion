#!/usr/bin/env python
"""Main figure: zero-shot (pLM alone) against ProFusion (comp7), 12 backbones x 2 axes.

## What is measured (state this before any number)

    rho   Spearman between the score and the experimental label within one DMS. Unitless, [-1, 1],
          higher is better. One box is the DMS distribution of that axis.
    stars Wilcoxon paired on the same DMS (two-sided). *** p<0.001, ** p<0.01, * p<0.05, ns.
          **It tests the paired difference, not the overlap of the distributions.** Boxes can
          overlap and still be significant.

input   results/paper_perdms.csv  (produced by scripts/paper_numbers.py)
output  results/figures/zeroshot_vs_fusion.{pdf,png} and zeroshot_vs_fusion_data.csv
run     python figures/plot_zeroshot_vs_fusion.py [perdms.csv] [output directory]
"""
from __future__ import annotations

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
PALETTE = {"main": "#0F4D92", "baseline": "#CFCECE", "gray_line": "#767676"}
WIDTH_IN = 7.2


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
        "axes.linewidth": style.axes_linewidth, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": style.axes_linewidth, "ytick.major.width": style.axes_linewidth,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def finalize_figure(fig, out_path: Path, formats=("pdf", "png"), dpi=300) -> list[Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.4)
    saved = []
    for fmt in formats:
        p = out_path.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=dpi)
        saved.append(p)
    plt.close(fig)
    return saved


# ---------------------------------------------------------------- conventions
BASE_ARM = "pLM alone"
FUSE_ARM = "comp7 (position/residual split)"
# The axis name inside the npz to the name used in the figure. The npz stores the expression axis under its old name, stability.
PANELS = [("function", "Function eval"), ("stability", "Expression eval")]
# Internal key to the notation used in the paper
SHOW_BB = {
    "AIDO_RAG16B": "AIDO RAG 16B", "VenusREM": "VenusREM", "ProSST": "ProSST",
    "VespaG": "VespaG", "TranceptionLret": "Tranception L (ret)", "S2F": "S2F",
    "ESM1v": "ESM-1v", "SaProt650M": "SaProt 650M", "ESM3": "ESM3-open",
    "ESM2": "ESM2 650M", "ESMIF1": "ESM-IF1", "ProteinMPNN": "ProteinMPNN",
}


def stars(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def paired(df: pd.DataFrame, axis: str, bb: str) -> tuple[np.ndarray, np.ndarray, float]:
    """(base, fused, p) paired on the same DMS. If they do not pair, only the intersection is used."""
    d = df[(df.axis == axis) & (df.bb == bb)]
    b = d[d.arm == BASE_ARM].set_index("dms")["rho"]
    f = d[d.arm == FUSE_ARM].set_index("dms")["rho"]
    common = b.index.intersection(f.index)
    b, f = b.reindex(common), f.reindex(common)
    diff = (f - b).dropna()
    p = wilcoxon(diff)[1] if len(diff) > 1 and diff.std() > 1e-12 else 1.0
    return b.to_numpy(), f.to_numpy(), float(p)


def main(perdms: Path, outdir: Path) -> None:
    df = pd.read_csv(perdms)
    for arm in (BASE_ARM, FUSE_ARM):
        if arm not in set(df.arm):
            raise SystemExit(f"arm '{arm}' is not in {perdms}. Run paper_numbers.py first.")

    panels = [(a, t) for a, t in PANELS if a in set(df.axis)]
    # Backbone order: descending ProFusion median on the function panel. Both panels use that order.
    order_axis = panels[0][0]
    present = [b for b in SHOW_BB if b in set(df.bb)]
    med = {b: np.median(paired(df, order_axis, b)[1]) for b in present}
    bbs = sorted(present, key=lambda b: -med[b])

    apply_publication_style()
    fig, axes = plt.subplots(len(panels), 1, figsize=(WIDTH_IN, 2.9 * len(panels)))
    axes = np.atleast_1d(axes)
    rows = []

    for ax, (axis, title), tag in zip(axes, panels, "abcdef"):
        pos, data, colors = [], [], []
        n_dms = 0
        for i, bb in enumerate(bbs):
            b, f, p = paired(df, axis, bb)
            n_dms = max(n_dms, len(b))
            pos += [i - 0.19, i + 0.19]
            data += [b, f]
            colors += [PALETTE["baseline"], PALETTE["main"]]
            rows.append(dict(
                axis=axis, bb=bb, label=SHOW_BB[bb], n=len(b),
                base_med=np.median(b), base_q1=np.percentile(b, 25), base_q3=np.percentile(b, 75),
                fuse_med=np.median(f), fuse_q1=np.percentile(f, 25), fuse_q3=np.percentile(f, 75),
                delta_mean=float(np.mean(f - b)), p_wilcoxon=p, sig=stars(p)))

        bp = ax.boxplot(data, positions=pos, widths=0.33, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=0.8),
                        whiskerprops=dict(linewidth=0.6), capprops=dict(linewidth=0.6),
                        boxprops=dict(linewidth=0.5))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_edgecolor("black")

        # Clear the space for the stars first, or they collide with the panel title.
        top = max(np.max(d) for d in data)
        bot = min(np.min(d) for d in data)
        span = max(top - bot, 1e-9)
        ax.set_ylim(bot - 0.06 * span, top + 0.17 * span)
        for i, bb in enumerate(bbs):
            r = rows[-len(bbs) + i]
            ax.text(i, top + 0.045 * span, r["sig"], ha="center", va="bottom",
                    fontsize=FigureStyle.tick_size, color="black")

        ax.axhline(0, color=PALETTE["gray_line"], linewidth=0.6, linestyle=":", zorder=0)
        ax.set_xticks(range(len(bbs)))
        ax.set_xticklabels([SHOW_BB[b] for b in bbs], rotation=35, ha="right")
        ax.set_ylabel("Spearman ρ per assay")
        ax.set_xlim(-0.6, len(bbs) - 0.4)
        ax.set_title(f"{tag}  {title} (n={n_dms})", loc="left", fontsize=FigureStyle.font_size,
                     fontweight="bold", pad=6)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["baseline"], edgecolor="black", linewidth=0.5),
               plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE["main"], edgecolor="black", linewidth=0.5)]
    axes[0].legend(handles, ["pLM alone", "ProFusion"], loc="lower left", ncol=2,
                   handlelength=1.2, handleheight=0.9, columnspacing=1.0)

    out = outdir / "zeroshot_vs_fusion"
    saved = finalize_figure(fig, out)
    data_csv = outdir / "zeroshot_vs_fusion_data.csv"
    pd.DataFrame(rows).to_csv(data_csv, index=False)
    for p in saved + [data_csv]:
        print(f"wrote: {p}")


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "paper_perdms.csv"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS / "figures"
    main(src, dst)
