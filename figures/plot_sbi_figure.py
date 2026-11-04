#!/usr/bin/env python
"""Combined SBI figure: group result + GCK T168 case study. 4 panels.

  a  Delta P_m(T_SBI) split by whether the backbone takes structure as input. One point per
     backbone (12). Group labels tested by permutation.
  b  the 8,255 GCK variants on stability against function percentiles. The 18 substitutions at
     T168 fall in the "stable but function-losing" quadrant.
  c  structure. The T168 side-chain OH hydrogen bonds to glucose O2 at 2.90 A, while the nearest
     polar protein atom is 4.58 A away.
  d  how tolerant the T168 position is among the 463 GCK positions, for 12 backbones, before and
     after fusion.

Metric definitions
  P_m(T_SBI)  z-normalise the variant scores within a DMS, average per position, then average over
              the T_SBI positions. Unitless.
              **Negative means the model calls that position more damaging than the protein average.**
  quadrant    the variants shared between the function and stability DMS of one protein, split at
              the medians of the two **experimental** axes. No prediction enters it. 10 proteins,
              3,249 positions, 495 T_SBI.
  percentile  b is over the 8,255 variants, d over the 463 GCK positions. In d, 100 is the most
              tolerant.
structure  PDB 1V4S (human glucokinase with alpha-D-glucose), chain A. Distances are measured
           between heavy atoms; there are no hydrogens.
weights    fitted on the MaveDB external set (49 DMS, 23 proteins, no overlap with the evaluation
           cohort) on the uni3 grid, then frozen.
⚠ d is a **single** position. The group claim rests on a, not on d.
run: python figures/plot_sbi_figure.py
"""
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plmfusion.paths import RESULTS, figure_assets  # noqa: E402

PAL = {"fus": "#0F4D92", "seq": "#767676", "alt": "#B64342", "gray": "#767676",
       "glc": "#D4A017", "focal": "#3C9D9B"}
SRC = figure_assets("sbi_gck")                              # ⚠ This used to read a scratch
# directory. One cleanup of that directory would have lost the paper figure permanently.
EXP = RESULTS                       # the two position-level CSVs written by sbi_positions.py
OUT = RESULTS / "figures"
BB12 = ["ProteinMPNN", "ProSST", "ESMIF1", "VenusREM", "S2F", "TranceptionLret",
        "ESM3", "AIDO_RAG16B", "SaProt650M", "VespaG", "ESM2", "ESM1v"]
# Split by whether structure is read at inference time. S2F has a GVP structure encoder, and
# AIDO-RAG takes struc_data in its input tree.
STRUCT = {"ProSST", "ESM3", "VenusREM", "SaProt650M", "ESMIF1", "ProteinMPNN",
          "S2F", "AIDO_RAG16B"}
# Panel a plots **−P_m** (up = calls SBI more damaging). Plotting P_m directly would make the
# claim (structure-aware models improve) a descending line, fighting the "up is better" convention.
# The values are unchanged and only the viewing direction flips, so the text reports P_m as it is
# and states after the definition that the figure shows −P_m.

NICE = {"AIDO_RAG16B": "AIDO-RAG 16B", "TranceptionLret": "Tranception L (ret)",
        "ESM1v": "ESM-1v", "ESM2": "ESM2 650M", "ESM3": "ESM3-open",
        "SaProt650M": "SaProt 650M", "ESMIF1": "ESM-IF1"}


@dataclass(frozen=True)
class FigureStyle:
    font_size: float = 7.2
    tick_size: float = 6.5
    axes_linewidth: float = 0.8
    font_family: tuple = ("Source Sans Pro", "Helvetica", "Arial", "DejaVu Sans")


def apply_publication_style(style: FigureStyle = FigureStyle()) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": list(style.font_family),
        "font.size": style.font_size, "axes.labelsize": style.font_size,
        "xtick.labelsize": style.tick_size, "ytick.labelsize": style.tick_size,
        "legend.fontsize": style.tick_size + 0.3, "legend.frameon": False,
        "axes.linewidth": style.axes_linewidth, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.major.width": style.axes_linewidth,
        "ytick.major.width": style.axes_linewidth, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.rm": "Source Sans Pro",
        "mathtext.it": "Source Sans Pro:italic",
        "mathtext.cal": "Source Sans Pro:italic",
        "mathtext.sf": "Source Sans Pro",
        "mathtext.tt": "Source Sans Pro", "mathtext.bf": "Source Sans Pro:bold",
    })


def pl(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            ha="left", va="bottom")


def lab(ax, x, y, s, color="black", size=6.4, ha="center", va="center", box=True,
        weight="normal"):
    bb = dict(fc="white", ec="none", alpha=0.85, pad=1.0) if box else None
    ax.text(x, y, s, transform=ax.transAxes, fontsize=size, color=color, ha=ha, va=va,
            bbox=bb, fontweight=weight, zorder=5)


def autocrop(path, pad=0.02, thresh=0.93, cut_bottom=0.0):
    im = mpimg.imread(path)
    mask = im[..., :3].min(axis=2) < thresh
    ys, xs = np.where(mask)
    H, W = mask.shape
    py, px = int(pad * H), int(pad * W)
    im = im[max(0, ys.min() - py):min(H, ys.max() + py),
            max(0, xs.min() - px):min(W, xs.max() + px)]
    return im[: int((1 - cut_bottom) * im.shape[0])] if cut_bottom else im


def main():
    apply_publication_style()
    rng = np.random.default_rng(0)
    T = pd.read_csv(EXP / "sbi_12bb_comp7.csv")
    W = T.pivot_table(index=["bb", "prot"], columns="arm", values="Pm").reset_index()
    g = W.groupby("bb").agg(pLM=("pLM", "mean"), fus=("ProFusion", "mean"))
    g["delta"] = g.fus - g.pLM
    g["grp"] = ["structure-aware" if b in STRUCT else "structure-unaware"
                for b in g.index]
    d = np.load(SRC / "panel_data.npz")
    pf = 100 * rankdata(d["yf"]) / len(d["yf"])
    pa = 100 * rankdata(d["ya"]) / len(d["ya"])
    m = d["pos"] == 168
    C = pd.read_csv(SRC / "panel_c12.csv").set_index("bb").reindex(BB12)
    # The strip under panel c: the position-level percentile of the residues that hydrogen bond
    # to glucose.
    # Criterion: a side-chain N or O within 3.5 A of a glucose oxygen in every observed GCK chain.
    # Verified exhaustively over 35 PDB entries and 37 GCK chains.
    # ⚠ Gln287 bonds in only 21 of the 35 chains (60%), which is borderline, so it is excluded.
    #   Including it would make SBI 3.
    # Only Thr168, which continues into the structure panel, is named on the strip; the rest go in
    # the caption.
    HB8 = [(168, "Thr168"), (169, None), (204, None), (205, None),
           (231, None), (256, None), (290, None)]
    upos, inv = np.unique(d["pos"], return_inverse=True)
    pcnt = np.bincount(inv)
    pa_pos = 100 * rankdata(np.bincount(inv, d["ya"]) / pcnt) / len(upos)
    pf_pos = 100 * rankdata(np.bincount(inv, d["yf"]) / pcnt) / len(upos)
    SITE8 = []
    for r, nm in HB8:
        k = np.where(upos == r)[0][0]
        SITE8.append((r, nm, pa_pos[k], pa_pos[k] >= 50 and pf_pos[k] < 50))

    # The structure image in panel c is portrait (W/H 0.756), so the box matches that ratio.
    # An even gridspec split would letterbox it, so the coordinates are given explicitly, in inches.
    # The canvas is sized to the ink bounding box. The old 3.42 left 0.75 in empty above and below,
    # which is what opened the vertical gaps around the figure in LaTeX. The target margin is
    # 0.05 in.
    FW, FH = 7.2, 2.77
    H_IN = 1.95                      # the shared axis height of every panel
    Y0, HH = 0.623 / FH, H_IN / FH
    XS = {"a": (0.50, 1.30), "b": (2.24, 1.34), "c": (3.73, 1.23), "d": (5.86, 1.26)}
    fig = plt.figure(figsize=(FW, FH))

    def box(k, dh=0.0):
        """dh is in inches. The baseline is fixed and only the top shrinks, making room for the legend."""
        x, w = XS[k]
        return fig.add_axes([x / FW, Y0, w / FW, (H_IN - dh) / FH])

    # ---------------- (a) group result (before and after fusion, paired)
    # Testing the difference would make the group with the higher baseline change more by
    # construction, so both levels are drawn and the group difference is tested at each.
    #
    # ⚠ The axis is **broken into three segments**. The two reference points (WT-like −0.60 and
    #   total-loss +0.57, both in −P_m coordinates) sit far outside the data band (−0.16 to +0.26),
    #   so one axis would widen the range by 2.6 times and squeeze the group shift from 21% of the
    #   panel height to 8% (measured). The middle segment keeps the data at 72% of its present
    #   resolution, and two thin segments above and below carry the reference lines. This folds the
    #   empty space between the data and the annotation lines; it does not hide any difference
    #   between the things being compared.
    def perm_two_sided(v, is_st):
        a, b = v[is_st], v[~is_st]
        obs = a.mean() - b.mean()
        allv = np.concatenate([a, b]); idx = range(len(allv))
        pm = np.array([allv[list(c)].mean() - allv[[i for i in idx if i not in c]].mean()
                       for c in combinations(idx, len(a))])
        return obs, float(np.mean(np.abs(pm) >= abs(obs))), len(pm)

    st = (g.grp == "structure-aware").values
    gap0, p0, nperm = perm_two_sided(g.pLM.values, st)
    gap1, p1, _ = perm_two_sided(g.fus.values, st)
    g["pLM"], g["fus"] = -g.pLM, -g.fus          # drawn as −P_m

    # The reference points are exactly the values the text quotes, from the same source and formula as final_numbers.py.
    Q = pd.read_csv(EXP / "sbi_positions_p11.csv")
    REF = {nm: -float(Q[(Q.quad == qd) & (Q.arm == "ProFusion")].Pm.mean())
           for qd, nm in (("T_OK", "WT-like"), ("T_LOSS", "total loss"))}

    # Three segments: bottom (WT-like) 0.24, middle (data) 1.40, top (total loss) 0.24, 0.035 between
    xa, wa = XS["a"]
    HB, GAP, HM = 0.24, 0.035, 1.40
    def seg(y_in, h_in):
        return fig.add_axes([xa / FW, (0.623 + y_in) / FH, wa / FW, h_in / FH])
    a_lo = seg(0.0, HB)
    ax = seg(HB + GAP, HM)                        # the data segment
    a_hi = seg(HB + GAP + HM + GAP, HB)

    X = [0.0, 1.0]
    for bb in g.index:                                   # one line per backbone
        c = PAL["fus"] if g.loc[bb, "grp"] == "structure-aware" else PAL["seq"]
        ax.plot(X, [g.loc[bb, "pLM"], g.loc[bb, "fus"]], color=c, lw=0.7, alpha=0.5,
                zorder=2, solid_capstyle="round")
        ax.scatter(X, [g.loc[bb, "pLM"], g.loc[bb, "fus"]], s=8, color=c, lw=0,
                   alpha=0.75, zorder=3)
    for grp, c in [("structure-aware", PAL["fus"]), ("structure-unaware", PAL["seq"])]:
        q = g[g.grp == grp]
        ax.plot(X, [q.pLM.mean(), q.fus.mean()], color=c, lw=1.8, zorder=4,
                solid_capstyle="round")
        ax.scatter(X, [q.pLM.mean(), q.fus.mean()], marker="D", s=17, facecolor="white",
                   edgecolor=c, linewidths=1.1, zorder=5)
    ax.axhline(0, color="black", lw=0.8, zorder=1)
    ax.set_ylim(-0.19, 0.33); ax.set_yticks([-0.1, 0.0, 0.1, 0.2, 0.3])
    # The axis label states the unit and the reference point directly. z is taken over all shared
    # variants of that protein, so 0 is the average position of that protein (measured +0.011).
    ax.set_ylabel("SBI,  SD below the protein mean", labelpad=1.5)
    lab(ax, 0.995, 0.995, "structure-aware (8)", color=PAL["fus"], size=6.5, box=False,
        ha="right", va="top", weight="bold")
    lab(ax, 0.995, 0.905, "structure-unaware (4)", color=PAL["seq"], size=6.5, box=False,
        ha="right", va="top", weight="bold")

    # The two reference segments: a dashed line with a name, and a single tick at that value.
    # ⚠ The name is set **3 pt clear of the line**. Placing the text baseline on the line itself
    #   makes it run into the dashes. That clearance needs room, so ylim is pushed towards the text
    #   (0.09 on one side, 0.03 on the other).
    for a_seg, nm, sgn in ((a_hi, "total loss", +1), (a_lo, "WT-like", -1)):
        v = REF[nm]
        a_seg.axhline(v, color="black", lw=0.8, ls=(0, (1.6, 1.6)), zorder=1)
        a_seg.annotate(nm, (1.30, v), textcoords="offset points", xytext=(0, 3 * sgn),
                       fontsize=6.0, color="#1F1F1F", ha="right",
                       va="bottom" if sgn > 0 else "top", zorder=6)
        a_seg.set_ylim(v - (0.03 if sgn > 0 else 0.09),
                       v + (0.09 if sgn > 0 else 0.03))
        a_seg.set_yticks([round(v, 2)])

    for a_seg in (a_lo, ax, a_hi):
        a_seg.set_xlim(-0.32, 1.32)
    ax.set_xticks([]); a_hi.set_xticks([])
    a_lo.set_xticks(X); a_lo.set_xticklabels(["pLM\nalone", "ProFusion"])
    ax.spines["bottom"].set_visible(False); a_hi.spines["bottom"].set_visible(False)
    # Break marks: one diagonal at the upper left of each segment.
    for a_seg in (a_lo, ax):
        a_seg.plot([-0.018, 0.018], [0.985, 1.015], transform=a_seg.transAxes,
                   color="black", lw=0.7, clip_on=False, zorder=7)
    a_hi.set_title("\u2191 more deleterious", loc="right", fontsize=6.0,
                   color="#4A4A4A", pad=2.5)
    pl(a_hi, "a", x=-0.33)

    # ---------------- (b) the DMS evidence
    ax = box("b", dh=0.26)
    ax.add_patch(plt.Rectangle((50, 0), 50, 50, facecolor="#F3E7E6", lw=0, zorder=0))
    ax.scatter(pa[~m], pf[~m], s=0.5, c="#C9CDD2", lw=0, alpha=0.5, zorder=1,
               rasterized=True)
    ax.axvline(50, color="black", lw=0.7, zorder=2)
    ax.axhline(50, color="black", lw=0.7, zorder=2)
    ax.scatter(pa[m], pf[m], s=9, facecolor=PAL["alt"], edgecolor="white",
               linewidths=0.35, zorder=4)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xticks([0, 50, 100]); ax.set_yticks([0, 50, 100])
    ax.set_xlabel("abundance percentile", labelpad=1.5)
    ax.set_ylabel("activity percentile", labelpad=1.5)
    # All four quadrants are named, at the exact centre of each.
    # The labels sit in front of the scatter (zorder 4). The centre of the SBI quadrant coincides
    # with the T168 point (76.6, 25.4), so that one point is hidden by the box. Legibility wins.
    for x, y, t, c, w in [
            (0.75, 0.25, "SBI", "#1F1F1F", "bold"),
            (0.75, 0.75, "WT-like", "#4A4A4A", "normal"),
            (0.25, 0.25, "total loss", "#4A4A4A", "normal"),
            (0.25, 0.75, "low abundance,\nhigh activity", "#4A4A4A", "normal")]:
        ax.text(x, y, t, transform=ax.transAxes, fontsize=6.5, color=c, ha="center",
                va="center", fontweight=w, zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.60, pad=1.0))
    # The axis was shortened by 0.26 in, so the panel letter rises by the same amount to line up
    # with the others. 1.95*1.02/1.69 = 1.177. x is pulled in so it does not sit beside the panel a legend.
    pl(ax, "b", x=-0.22, y=1.177)
    # T168 is a legend entry, not a quadrant name, so it goes outside the axes.
    # Writing just "T168" would not be recognised, since this is the first time the name appears.
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="none", ms=3.0,
                                  mfc=PAL["alt"], mec="white", mew=0.35,
                                  label="Thr168 substitutions")],
              loc="lower right", bbox_to_anchor=(1.01, 1.05), handletextpad=0.3,
              borderpad=0.0, borderaxespad=0.0)
    # The title rises to the panel-letter line, so it separates from the legend vertically rather than horizontally.
    ax.text(0.0, 1.177, "glucokinase (n = 8,255)", transform=ax.transAxes,
            fontsize=7.0, fontweight="bold", color="black", ha="left", va="bottom")

    # ---------------- (c) structure plus the apo control strip
    # One panel used as two stacked axes. The top is the structure render (1100x1300, ZM=3.8) and
    # the bottom is a strip placing the 7 glucose-binding residues on the abundance percentile.
    CX, CW = XS["c"]
    # Showing the background cartoon (alpha 0.55) fills the lower area, so no cropping is needed.
    H_IMG = CW / 0.846
    # The strip shrinks slightly to make room for a caption line above it. The total is unchanged, so the structure stays put.
    H_STRIP, C_GAP = 0.34, 0.16
    # The structure sits directly above the strip and the leftover margin moves under the title.
    C_BOT = 0.623 + H_STRIP + C_GAP
    ax = fig.add_axes([CX / FW, C_BOT / FH, CW / FW, H_IMG / FH])
    ax.imshow(mpimg.imread(SRC / "r_c5.png"), interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    # Atom positions measured from a marker render taken with the same camera (axes fraction)
    A = {"cb": (0.358, 0.729), "og1": (0.407, 0.667), "o2": (0.554, 0.585)}
    BOND = tuple((A["og1"][i] + A["o2"][i]) / 2 for i in (0, 1))

    def leader(x0, y0, x1, y1, c="#3f3f3f"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords="axes fraction",
                    textcoords="axes fraction", zorder=4,
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=c,
                                    shrinkA=1.5, shrinkB=0))

    def clab(x, y, t, c, size=6.8, ha="center", va="center", w="normal"):
        # The box is fully opaque. Semi-transparent, the leader line shows through the text and reads as a stroke.
        ax.text(x, y, t, transform=ax.transAxes, fontsize=size, color=c, ha=ha, va=va,
                fontweight=w, zorder=5,
                bbox=dict(fc="white", ec="none", alpha=1.0, pad=1.2))

    clab(0.02, 0.720, "Thr168", PAL["focal"], 7.0, ha="left", w="bold")
    leader(0.215, 0.722, *A["cb"], c=PAL["focal"])
    clab(0.985, 0.300, "glucose", "#8A6810", 7.0, ha="right", w="bold")
    leader(0.790, 0.318, 0.690, 0.375, c="#8A6810")
    clab(0.985, 0.560, "2.90 \u00c5", "black", 6.8, ha="right", w="bold")
    leader(0.855, 0.572, *BOND)
    # The ribbon passes just above and is the same colour, so no leader line is needed.
    # Not apo-specific: it is unobserved in 8 of the 37 chains, and 6 of those are high-resolution
    # glucose complexes.
    clab(0.985, 0.845, "157 to 179\nflexible segment", "#4A5272", 6.4,
         ha="right", va="top")
    # The letter and title stay at the same height as the other panels (compensating for the lowered axes)
    pl(ax, "c", x=-0.01, y=1 + (0.623 + H_IN - C_BOT - H_IMG) / H_IMG + 0.02)
    ax.set_title("PDB 1V4S", fontsize=7.0, color="black", fontweight="bold",
                 pad=4 + 72 * (0.623 + H_IN - C_BOT - H_IMG))

    # The strip: all 8 residues that hydrogen bond to glucose are functionally critical, yet they
    # split into two groups on abundance. The gap is 24 percentile points with no overlap.
    axs = fig.add_axes([CX / FW, 0.623 / FH, CW / FW, H_STRIP / FH])
    axs.axvspan(50, 100, color="#F3E7E6", lw=0, zorder=0)
    # The 50 line spans only the height of the points. 5.9 pt text takes about 0.55 in data
    # coordinates, so the upper label uses y 0.42 to 0.97 and the lower one −0.45 to −1.0. Only the
    # space between them is left. The boundary still reads, because the pink shaded edge continues it.
    axs.plot([50, 50], [-0.38, 0.38], color="black", lw=0.7, zorder=2)
    for res, nm, pa_, is_sbi in SITE8:
        axs.scatter(pa_, 0, s=22, zorder=4,
                    facecolor=PAL["alt"] if is_sbi else "white",
                    edgecolor=PAL["alt"] if is_sbi else PAL["seq"], linewidths=0.9)
        if nm:
            axs.text(pa_, 0.42, nm, fontsize=5.9, ha="center", va="bottom",
                     color=PAL["alt"], fontweight="bold")
    # The width is only 1.23 in, so every label is shortened. The details of the criterion go in the caption.
    axs.text(38, -0.45, "total loss (5)", fontsize=5.9, color=PAL["seq"],
             ha="center", va="top")
    axs.text(84, -0.45, "SBI (2)", fontsize=5.9, color=PAL["alt"],
             ha="center", va="top", fontweight="bold")
    # The strip is taller now, so the points sit towards the top. The axis line stays on the same baseline as the other panels.
    axs.set_yticks([]); axs.set_ylim(-1.35, 1.35)
    axs.set_xlim(25, 103); axs.set_xticks([50, 100])
    axs.set_xlabel("abundance percentile", labelpad=1.5)
    # The empty band above the strip is where the caption line goes. Inside, it would compete with the points and labels.
    axs.set_title("7 glucose H-bond residues", fontsize=5.9, color="#4A4A4A",
                  loc="right", pad=2.5)
    for sp in ("left", "right", "top"):
        axs.spines[sp].set_visible(False)

    # ---------------- (d) model scores
    # The rows are grouped, and each median line is drawn only across its own group. Spanning the
    # full height mixes the four lines, and in particular hides the two post-fusion medians
    # (28.9 and 29.4), which overlap.
    ax = box("d")
    GAP = 0.9                                  # the gap between the two blocks
    order = [b for b in BB12 if b in STRUCT] + [b for b in BB12 if b not in STRUCT]
    ypos, yy = {}, 0.0
    for k, bb in enumerate(order):
        if k and (bb in STRUCT) != (order[k - 1] in STRUCT):
            yy += GAP
        ypos[bb] = yy
        yy += 1.0
    ytop = yy - 1.0
    for bb in order:                           # reversed so it reads top to bottom
        ypos[bb] = ytop - ypos[bb]

    ax.axvspan(50, 100, color="#EDEFF1", lw=0, zorder=0)
    # The 50 line stops below the rows. Above it is where the median arrow goes.
    ax.plot([50, 50], [-1.95, ytop + 0.50], color="black", lw=0.7, zorder=1)
    for grp, mem in [(True, [b for b in order if b in STRUCT]),
                     (False, [b for b in order if b not in STRUCT])]:
        col = PAL["fus"] if grp else PAL["seq"]
        lo, hi = min(ypos[b] for b in mem) - 0.45, max(ypos[b] for b in mem) + 0.45
        m0, m1 = C.loc[mem, "pLM"].median(), C.loc[mem, "ProFusion"].median()
        for val, ls in [(m0, (0, (1, 1.4))), (m1, (0, (3.2, 1.6)))]:
            ax.plot([val, val], [lo, hi], ls=ls, color=col, lw=0.9, zorder=2,
                    dash_capstyle="butt")
        # An arrow joins the two medians. The label always sits outside the arrow.
        ya = hi + 0.32
        ax.annotate("", xy=(m1, ya), xytext=(m0, ya), zorder=6,
                    arrowprops=dict(arrowstyle="-|>,head_width=0.12,head_length=0.30",
                                    color=col, lw=0.9, shrinkA=0, shrinkB=0))
        out = 1.8 if m1 > m0 else -1.8          # the direction the arrow travels
        for val, dx, ha in [(m0, -out, "right" if out > 0 else "left"),
                            (m1, +out, "left" if out > 0 else "right")]:
            ax.text(val + dx, ya, f"{val:.0f}", fontsize=6.0, color=col, ha=ha,
                    va="center", zorder=7,
                    bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.5))
    for bb in order:
        r, yi = C.loc[bb], ypos[bb]
        col = PAL["fus"] if bb in STRUCT else PAL["seq"]
        ax.annotate("", xy=(r.ProFusion, yi), xytext=(r.pLM, yi),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.15,head_length=0.38",
                                    color=col, lw=1.0, shrinkA=2.4, shrinkB=0), zorder=3)
        ax.scatter(r.pLM, yi, s=11, facecolor="white", edgecolor=col, linewidths=0.8,
                   zorder=4)
    ax.set_yticks([ypos[b] for b in order])
    ax.set_yticklabels([NICE.get(b, b) for b in order])
    for t, bb in zip(ax.get_yticklabels(), order):
        if bb in STRUCT:
            t.set_fontweight("bold")
    ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100])
    ax.set_ylim(-1.95, ytop + 1.35)
    ax.set_xlabel("percentile among\n463 GCK positions", labelpad=15)
    # The two labels that floated inside the axes are attached to the two ends of the x axis
    ax.text(0.0, -0.115, "\u2190 deleterious", transform=ax.transAxes,
            fontsize=6.6, color="#1F1F1F", ha="left", va="top")
    ax.text(1.0, -0.115, "tolerant \u2192", transform=ax.transAxes,
            fontsize=6.6, color="#1F1F1F", ha="right", va="top")
    h = [plt.Line2D([], [], marker="o", ls="none", mfc="white", mec="#555555", ms=3.4,
                    mew=0.9, label="pLM alone"),
         plt.Line2D([], [], marker="<", ls="none", mfc="#555555", mec="#555555", ms=3.2,
                    label="ProFusion")]
    # The median lines carry their numbers, so they are left out of the legend. The line styles are explained in the caption.
    ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.03, -0.022), ncol=1,
              handletextpad=0.35, borderpad=0.2, labelspacing=0.35)
    pl(ax, "d", x=-0.58)
    ax.set_title("T168 only", fontsize=7.0, pad=4, color="black",
                 fontweight="bold")

    OUT.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(OUT / f"sbi_figure.{fmt}", dpi=300)
    plt.close(fig)
    g.round(4).to_csv(OUT / "sbi_figure_data.csv")
    print("saved:", OUT / "sbi_figure.pdf", OUT / "sbi_figure.png", sep="\n  ")
    print(f"before gap {gap0:+.4f} p={p0:.4f} | after gap {gap1:+.4f} "
          f"p={p1:.4f} | nperm={nperm}")


if __name__ == "__main__":
    main()
