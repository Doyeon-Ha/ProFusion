#!/usr/bin/env python
"""The comp7 weight allocation: a 12 backbone x 7 slot heatmap.

## What is drawn

The 7 slots are the 5 auxiliary channels split into position (a_*) and residual (b_*) components.
  a_*  how important that position is overall (the position mean of the channel's within-DMS z)
  b_*  whether this substitution is special within that position (z minus the position mean)
RSA is constant within a position, so it has no b; BLOSUM and Delta charge have a position
component that is effectively a wild-type lookup table (eta-squared .79 to .93), so only their
residual is used.

## What is measured (defined before any number)

The cell value is the **allocation share**, share = w_slot / sum(w). Unitless. Positive means
budget is spent on that slot and negative means the channel is used in the opposite direction.
Each row sums to 1. Dividing by the total is necessary because the raw total differs by more than
a factor of three between backbones, which otherwise mixes "how much is bought" with "what it is
spent on". The raw weights w and the total sum(w) are kept in the row labels and in _data.csv.

Rows are ordered by descending total sum(w) on the function axis, and both panels use that order.

## Data

results/fig_data/panelc_comp7_weights.csv, the comp7 rows of paper_weights.csv.
Every weight was fitted on the MaveDB external set only, and ProteinGym was never seen during
fitting. The fitting units differ per axis: function 35 DMS over 18 proteins, expression 14 DMS
over 9 proteins.
⚠ ESM-1v alone has function 34 DMS over 17 proteins (one external npz has no llr_ESM1v).

## Changing the grid changes individual slot values

This figure uses values fitted on the uni3 grid (all slots −3.0 to 3.0 in steps of 0.1). The
earlier wide grid had a lower bound of 0 on the GEMME and ΔΔG slots, so those could not go
negative, which is what allowed the statement that "a_ddG is exactly 0 for ESM3 and ProSST". Under
uni3 it is −0.1 for ESM3 and 0.2 for ProSST.
**State which grid a value came from whenever an individual slot is used to argue a mechanism.**
The ordering of the arms by performance is the same under both grids.

output: results/figures/comp7_weights.{pdf,png} and comp7_weights_data.csv
run:    python figures/plot_comp7_weights.py
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plmfusion.paths import RESULTS  # noqa: E402

PALETTE = {"main": "#0F4D92", "alt": "#B64342"}


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


SLOTS = ["a_GEMME", "b_GEMME", "a_ddG", "b_ddG", "a_RSA", "b_BLOSUM", "b_dchg"]
COLLAB = ["GEMME\nposition", "GEMME\nresidual", "ΔΔG\nposition", "ΔΔG\nresidual",
          "RSA\nposition", "BLOSUM\nresidual", "Δcharge\nresidual"]
SHOW_BB = {
    "ESM2": "ESM2 650M", "ESM1v": "ESM-1v", "ESM3": "ESM3-open", "ProSST": "ProSST",
    "ESMIF1": "ESM-IF1", "ProteinMPNN": "ProteinMPNN", "TranceptionLret": "Tranception L (ret)",
    "S2F": "S2F", "VespaG": "VespaG", "VenusREM": "VenusREM",
    "SaProt650M": "SaProt 650M", "AIDO_RAG16B": "AIDO RAG 16B",
}
PANELS = [("function", "Function axis"), ("stability", "Expression axis")]
# Fitting units. ESM-1v alone has one fewer on function (one external npz has no llr).
NFIT = {"function": "35 DMS over 18 proteins", "stability": "14 DMS over 9 proteins"}

# Negative red, zero white, positive blue. The pair separates under red-green colour vision deficiency.
CMAP = LinearSegmentedColormap.from_list("share", [PALETTE["alt"], "#FFFFFF", PALETTE["main"]])


def main(src: Path, outdir: Path) -> None:
    W = pd.read_csv(src)
    have = [b for b in SHOW_BB if b in set(W.bb)]
    panels = [(a, t) for a, t in PANELS if a in set(W.axis)]

    q0 = W[W.axis == panels[0][0]].set_index("bb")
    total0 = {b: float(q0.loc[b, SLOTS].to_numpy(float).sum()) for b in have}
    bbs = sorted(have, key=lambda b: -total0[b])

    apply_publication_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 3.3),
                             gridspec_kw=dict(wspace=0.08))
    axes = np.atleast_1d(axes)
    rows, mats = [], []

    for (axis, _), _ in zip(panels, axes):
        q = W[W.axis == axis].set_index("bb")
        M = np.vstack([q.loc[b, SLOTS].to_numpy(float) for b in bbs])
        mats.append(M / M.sum(axis=1, keepdims=True))
    vmax = max(np.abs(m).max() for m in mats)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    for ax, (axis, title), M in zip(axes, panels, mats):
        q = W[W.axis == axis].set_index("bb")
        im = ax.imshow(M, cmap=CMAP, norm=norm, aspect="auto")
        for i, b in enumerate(bbs):
            for j, s in enumerate(SLOTS):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center", va="center",
                        fontsize=FigureStyle.tick_size - 0.5,
                        color="white" if abs(v) > 0.62 * vmax else "black")
                rows.append(dict(axis=axis, bb=b, label=SHOW_BB[b], slot=s,
                                 weight=float(q.loc[b, s]), share=float(v),
                                 total=float(q.loc[b, SLOTS].to_numpy(float).sum())))
        ax.set_xticks(range(len(SLOTS)))
        ax.set_xticklabels(COLLAB, fontsize=FigureStyle.tick_size - 0.5)
        ax.set_yticks(range(len(bbs)))
        if ax is axes[0]:
            tot = {b: float(q.loc[b, SLOTS].to_numpy(float).sum()) for b in bbs}
            ax.set_yticklabels([f"{SHOW_BB[b]}  ({tot[b]:.1f})" for b in bbs])
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{title}\n{NFIT[axis]}", fontsize=FigureStyle.font_size, pad=4)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)

    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.015)
    cb.set_label("allocated weight  (w / Σw)", fontsize=FigureStyle.tick_size)
    cb.ax.tick_params(labelsize=FigureStyle.tick_size - 0.5, length=2)
    cb.outline.set_visible(False)
    axes[0].set_ylabel("backbone  (Σw on function axis)", fontsize=FigureStyle.font_size)

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "comp7_weights"
    for fmt in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(outdir / "comp7_weights_data.csv", index=False)
    print(f"[saved] {out}.pdf / .png / _data.csv")


if __name__ == "__main__":
    main(RESULTS / "fig_data" / "panelc_comp7_weights.csv", RESULTS / "figures")
