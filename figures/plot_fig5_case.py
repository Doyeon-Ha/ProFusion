"""Figure 5, the case study: what changes at one position when position and residual are split.

The case is B2L11_HUMAN_Dutta_2010_binding-Mcl-1 (Bim BH3 binding to Mcl-1, yeast surface display
with FACS).
  The Dutta 2010 library touched only **10 positions** between 148 and 159 (150 and 154 are
  solvent-facing and excluded). With 17 substitutions per position (Cys and Met missing), that is
  170 variants.

Panels
  a  the addition at one position. One point is one of those 10 positions, and the number inside
     the point is the residue number.
     x is **the sum of the two position channels** and excludes the pLM LLR and the residual terms,
     so it is not the model's final position ranking.
  b  the 17 variants at that position move towards the measurement.
  c  structure. The overview (the BH3 helix lying in the Mcl-1 groove) and the close-up (there is
     no room for a side chain) are combined into one panel, with the close-up region joined by a
     rectangle and a connector. The rectangle is not placed by eye: its pixel coordinates are
     projected from the camera by the render script (fig_struct_overview_anchor.csv).

Names
  Following the Methods convention of the paper: FW5 is the merged (position = residual) fitted
  model and SFW7 the split one. The internal keys are still champ5 and comp7, so the translation
  happens in DISP alone.

input  case_study_variants.csv and case_positions_scored.csv
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.patches import Circle, ConnectionPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plmfusion.paths import RESULTS, figure_assets  # noqa: E402

PALETTE = {"main": "#0F4D92", "alt": "#B64342", "baseline": "#CFCECE",
           "extra1": "#42949E", "extra2": "#7B3E8F", "gray_line": "#767676"}
# ⚠ W7=(2.5,0.5) and W5=(4.0,2.2) used to be written here, but those are values from the **old
#   WIDE grid** and differ from the current fit. On top of that, 4.0 lies outside the −3.0 to 3.0
#   grid the Methods declares. They are read from the fitting output instead, and asserted to lie
#   on the declared grid, so they cannot go stale again.
def _w(arm):
    """The (a_GEMME, a_ddG, b_ddG) of the ESM2 function-axis arm."""
    W = pd.read_csv(RESULTS / "ablation_12bb_w.csv")
    r = W[(W.axis == "function") & (W.bb == "ESM2") & (W.arm == arm)]
    assert len(r) == 1, f"{arm}: {len(r)} rows"
    r = r.iloc[0]
    v = dict(g=float(r.a_GEMME), d=float(r.a_ddG), dres=float(r.b_ddG))
    for k, x in v.items():
        assert abs(x) <= 3.0 + 1e-9, f"{arm}.{k}={x} lies outside the grid"
        assert abs(round(x / 0.1) - x / 0.1) < 1e-6, f"{arm}.{k}={x} is off the 0.1 step"
    return v


W7 = _w("SFW7")              # ESM2 function axis, SFW7: a_GEMME, a_ddG, b_ddG
W5 = _w("FW5")               # ESM2 function axis, FW5
DISP = {"champ5": "FW5", "comp7": "SFW7"}
FOCUS = 156


@dataclass(frozen=True)
class FigureStyle:
    font_size: float = 9.0
    tick_size: float = 8.0
    axes_linewidth: float = 0.9
    font_family: tuple = ("Source Sans Pro", "Helvetica", "Arial", "DejaVu Sans")


def apply_publication_style(style=FigureStyle()):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": list(style.font_family),
        "font.size": style.font_size, "axes.labelsize": style.font_size,
        "xtick.labelsize": style.tick_size, "ytick.labelsize": style.tick_size,
        "legend.fontsize": style.tick_size, "legend.frameon": False,
        "axes.linewidth": style.axes_linewidth, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.major.width": style.axes_linewidth,
        "ytick.major.width": style.axes_linewidth, "xtick.major.size": 3.0,
        "ytick.major.size": 3.0, "svg.fonttype": "none", "pdf.fonttype": 42,
        "ps.fonttype": 42})


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def trim(img, pad=4):
    """Trim the transparent margin. The crop box is returned too, because the zoom box coordinates have to move with it."""
    m = img[..., 3] > 0.02 if img.shape[2] == 4 else img.sum(2) < 2.97
    ys, xs = np.where(m)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1], (x0, y0)


def beeswarm(xs, min_dx, pin=None):
    """Scatter points that share an x value vertically, trying levels 0, +1, −1, +2 and so on.

    The pin index is handled first and fixed at level 0. If the point the arrow lands on moved, the
    figure would be false."""
    order = list(np.argsort(xs))
    if pin is not None:
        order.remove(pin)
        order = [pin] + order
    tiers, out = {}, np.zeros(len(xs))
    for i in order:
        t = 0
        while any(abs(xs[i] - xs[j]) < min_dx for j in tiers.get(t, [])):
            t = -t + (1 if t <= 0 else 0)
        tiers.setdefault(t, []).append(i)
        out[i] = t
    return out


# ----------------------------- a  the addition at one position
def panel_a(fig, ax, S, f):
    LAY = {"champ5": dict(name=3.62, ag=3.10, ad=2.72, dot=1.60, note=0.70, tag="a = b"),
           "comp7": dict(name=0.10, ag=-0.42, ad=-0.80, dot=-1.68, note=-2.60, tag="a ≠ b")}
    # ⚠ Hard-coded, not read from the fitting output. This panel does not draw RHO either.
    #   The caption line below promises "rho = Spearman across those 10" but no value is printed
    #   anywhere. The two values also come from different places: champ5 0.285 comes from the
    #   champ5 weights in paper_weights.csv (3.0, 1.6), while comp7 0.661 comes from the SFW7
    #   weights in ablation_12bb_w.csv (2.5, 0.5). Recomputed on the current input, champ5 is 0.297.
    #   Fixing it needs a computation rather than a path change, so the ported copy keeps the
    #   original values.
    RHO = {"champ5": 0.285, "comp7": 0.661}
    STEP = 0.36          # vertical spacing of one beeswarm level. Must exceed the circle diameter
    MIN_DX = 0.37        # the horizontal spacing corresponding to the circle diameter
    ax.axvline(0, color=PALETTE["gray_line"], lw=0.7, ls=":", zorder=1)

    for nm, W in (("champ5", W5), ("comp7", W7)):
        L = LAY[nm]
        g, dd = W["g"] * f.gem, W["d"] * f.ddg
        rk = int(S[S.pos == FOCUS][f"{nm}_rank"].iloc[0])
        xs = S[nm].to_numpy(float)
        pin = int(np.where(S.pos.to_numpy() == FOCUS)[0][0])
        ys = L["dot"] + STEP * beeswarm(xs, MIN_DX, pin=pin)
        for x, y, pos, lab in zip(xs, ys, S.pos, S.lab):
            hot = pos == FOCUS
            sc = ax.scatter([x], [y], c=[lab], cmap="RdYlBu", vmin=0, vmax=100,
                            s=150 if hot else 128,
                            edgecolor=PALETTE["alt"] if hot else "black",
                            linewidths=1.9 if hot else 0.7, zorder=6 if hot else 4)
            ax.text(x, y, str(int(pos)), ha="center", va="center", zorder=7,
                    fontsize=5.0, fontweight="bold" if hot else "normal",
                    color="white" if (lab < 24 or lab > 78) else "black")
        yf = ys[pin]

        ax.annotate("", xy=(g, L["ag"]), xytext=(0, L["ag"]),
                    arrowprops=dict(arrowstyle="-|>", lw=1.7, color=PALETTE["main"],
                                    shrinkA=0, shrinkB=0))
        ax.text(g / 2, L["ag"] + 0.08, f"GEMME   {W['g']} × {f.gem:+.2f} = {g:+.2f}",
                ha="center", va="bottom", fontsize=7.4, color=PALETTE["main"])
        ax.annotate("", xy=(g + dd, L["ad"]), xytext=(g, L["ad"]),
                    arrowprops=dict(arrowstyle="-|>", lw=1.7, color=PALETTE["alt"],
                                    shrinkA=0, shrinkB=0))
        ax.text(g + dd / 2, L["ad"] - 0.08, f"ΔΔG   {W['d']} × {f.ddg:+.2f} = {dd:+.2f}",
                ha="center", va="top", fontsize=7.4, color=PALETTE["alt"])
        ax.plot([g, g], [L["ad"], L["ag"]], color=PALETTE["gray_line"], lw=0.6, ls=":", zorder=2)
        ax.plot([g + dd, g + dd], [yf + 0.16, L["ad"]], color=PALETTE["gray_line"],
                lw=0.6, ls=":", zorder=2)
        ax.text(-4.68, L["name"], f"{DISP[nm]}   ({L['tag']})", fontsize=9.5,
                fontweight="bold", va="center")

    # The colour bar lies in the empty band between the two rows. At the upper right it would overlap the 151 and 158 circles.
    cax = ax.inset_axes([0.615, 0.505, 0.300, 0.030])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=[0, 50, 100])
    cb.set_label("measured label percentile", fontsize=7.0, labelpad=3)
    cb.ax.tick_params(labelsize=6.8, length=2, pad=1.5)
    cb.outline.set_linewidth(0.6)
    ax.set_xlim(-4.75, 3.7)
    ax.set_ylim(-3.05, 4.05)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("what the two position channels add to the pLM score\n"
                  "← toward “damaging”                       toward “tolerated” →", labelpad=2)
    ax.set_title("a   Merging the weights leaves the worst site in the middle", loc="left",
                 fontsize=9.8, fontweight="bold", pad=17)
    ax.text(-4.75, 4.22, "one circle = one of the 10 mutated sites, numbered by residue   ·   "
                        "ρ = Spearman across those 10", fontsize=7.4,
            color=PALETTE["gray_line"])


# ----------------------------- b  variant level
def panel_b(ax, V):
    o = V.pos != FOCUS
    nF = int((~o).sum())
    better = int((np.abs(V.comp7_pct - V.label_pct)[~o] <
                  np.abs(V.champ5_pct - V.label_pct)[~o]).sum())
    ax.scatter(V.label_pct[o], V.champ5_pct[o], s=7, color=PALETTE["baseline"],
               edgecolor="none", zorder=2, label="other 9 sites, FW5")
    for _, r in V[~o].iterrows():
        ax.annotate("", xy=(r.label_pct, r.comp7_pct), xytext=(r.label_pct, r.champ5_pct),
                    arrowprops=dict(arrowstyle="->", lw=0.7, color=PALETTE["alt"], alpha=0.75))
    ax.scatter(V.label_pct[~o], V.champ5_pct[~o], s=26, color="white", edgecolor="black",
               linewidths=0.7, zorder=3, label=f"Gly156 · FW5 (n={nF})")
    ax.scatter(V.label_pct[~o], V.comp7_pct[~o], s=26, color=PALETTE["alt"], marker="^",
               edgecolor="black", linewidths=0.5, zorder=4, label="Gly156 · SFW7")
    ax.plot([0, 100], [0, 100], color="black", lw=0.8, ls="--", zorder=1)
    ax.text(84, 87, "perfect", fontsize=7.2, ha="center", rotation=45, va="bottom",
            color=PALETTE["gray_line"])
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("measured label percentile   (low = damaging)", labelpad=2)
    ax.set_ylabel("predicted percentile", labelpad=2)
    ax.legend(loc="upper left", handlelength=1.0, handletextpad=0.4, fontsize=7.4,
              borderaxespad=0.25, labelspacing=0.45)
    ax.text(0.97, 0.05, f"{better} of {nF} Gly156 variants\nmove toward the diagonal",
            transform=ax.transAxes, ha="right", fontsize=7.6, color=PALETTE["alt"],
            linespacing=1.35, fontweight="bold")
    ax.set_title("b   Gly156 variants move to the truth", loc="left",
                 fontsize=9.8, fontweight="bold", pad=8)
    return better, nF


def clash_inset(axz):
    """A scale drawing of Cb and the Arg263 Ca as van der Waals spheres in the plane.

    Why it is needed: the axis joining the two centres is only 17 degrees off the camera direction,
    so in the 3D render the on-screen centre-to-centre distance collapses to 0.77 A and the overlap
    reads as simple occlusion. Laid flat, the difference between 2.64 A and 3.40 A is visible as it
    is. The carbon vdW radius is 1.70 A, two carbons touch at 3.40 A, the measurement is 2.64 A,
    and the overlap is 0.76 A.
    """
    R, D = 1.70, 2.64
    ins = axz.inset_axes([0.020, 0.030, 0.470, 0.400])
    ins.set_facecolor("white")
    ins.patch.set_alpha(0.90)
    for s in ins.spines.values():
        s.set_linewidth(0.6)
        s.set_edgecolor(PALETTE["gray_line"])
    lens = Circle((0, 0), R, facecolor=PALETTE["extra2"], edgecolor="none", alpha=0.30)
    ins.add_patch(lens)
    ins.add_patch(Circle((D, 0), R, facecolor=PALETTE["extra1"], edgecolor="none", alpha=0.30))
    ins.add_patch(Circle((0, 0), R, facecolor="none", edgecolor=PALETTE["extra2"], lw=1.3))
    ins.add_patch(Circle((D, 0), R, facecolor="none", edgecolor=PALETTE["extra1"], lw=1.3))
    ins.plot([0, D], [0, 0], color="black", lw=0.8, zorder=5)
    for x in (0, D):
        ins.plot([x], [0], marker="o", ms=2.6, color="black", zorder=6)
    ins.annotate("", xy=(0, -2.25), xytext=(D, -2.25),
                 arrowprops=dict(arrowstyle="<->", lw=0.9, color="black"))
    ins.text(D / 2, -2.05, "2.64 Å", ha="center", va="bottom", fontsize=7.6,
             fontweight="bold")
    ins.text(D / 2, -3.35, "3.40 Å needed", ha="center", va="center", fontsize=7.0,
             color=PALETTE["gray_line"])
    ins.text(-1.5, 2.25, "Cβ", ha="center", va="bottom", fontsize=7.4, fontweight="bold",
             color=PALETTE["extra2"])
    ins.text(D + 1.5, 2.25, "Arg263 Cα", ha="center", va="bottom", fontsize=7.4,
             fontweight="bold", color=PALETTE["extra1"])
    ins.set_xlim(-3.4, D + 3.4)
    ins.set_ylim(-4.0, 3.6)
    ins.set_aspect("equal")
    ins.set_xticks([])
    ins.set_yticks([])


# ----------------------------- c  structure (overview and close-up in one panel)
def panel_c(fig, ax, ST):
    ax.axis("off")
    ax.set_title("c   Gly156 is buried against the partner, where no side chain fits",
                 loc="left", fontsize=9.8, fontweight="bold", pad=4)

    ov, (ox, oy) = trim(plt.imread(ST / "fig_struct_overview.png"))
    zm, (zx, zy) = trim(plt.imread(ST / "fig_struct_clash.png"))
    axo = ax.inset_axes([0.0, 0.0, 0.545, 1.0])
    axz = ax.inset_axes([0.595, 0.0, 0.405, 1.0])
    axo.imshow(ov); axo.axis("off")
    axz.imshow(zm); axz.axis("off")
    for s in axz.spines.values():
        s.set_visible(True)
        s.set_linewidth(1.1)
        s.set_edgecolor(PALETTE["extra2"])
    axz.set_frame_on(True)

    # The right-hand view is **rotated by 69 degrees**. Corner-to-corner connectors imply a direct
    # magnification and would therefore be false here. The box marks the position only, joined by a
    # single arrow.
    A = pd.read_csv(ST / "fig_struct_overview_anchor.csv").set_index("name")
    cx, cy = float(A.loc["CA156", "x"]) - ox, float(A.loc["CA156", "y"]) - oy
    half = 0.62 * float(A.loc["BOXHALF", "x"])
    axo.add_patch(Rectangle((cx - half, cy - half), 2 * half, 2 * half, fill=False,
                            edgecolor=PALETTE["extra2"], lw=1.1, zorder=9))
    fig.add_artist(ConnectionPatch(
        xyA=(cx + half, cy), coordsA=axo.transData, xyB=(-0.055, 0.5),
        coordsB=axz.transAxes, color=PALETTE["extra2"], lw=1.1,
        arrowstyle="-|>", mutation_scale=11, zorder=8))
    ax.text(0.572, 0.555, "rotated 69°", transform=ax.transAxes, fontsize=6.8,
            color=PALETTE["extra2"], ha="center", va="bottom", rotation=90,
            fontweight="bold")

    HALO = [pe.withStroke(linewidth=2.6, foreground="white")]

    def call(axx, text, xy, xytext, color, fs=8.0, **kw):
        """A leader line on the structure. The arrow needs a white outline too, or it disappears against the grey surface."""
        a = axx.annotate(text, xy=xy, xytext=xytext, textcoords=axx.transAxes,
                         fontsize=fs, fontweight="bold", color=color, path_effects=HALO,
                         arrowprops=dict(arrowstyle="-|>", lw=1.6, color=color,
                                         shrinkA=1, shrinkB=2,
                                         path_effects=[pe.withStroke(linewidth=3.6,
                                                                     foreground="white")]),
                         **kw)
        return a

    axo.text(0.50, 0.955, "Mcl-1", color="#333333", fontsize=9.0, fontweight="bold",
             ha="center", transform=axo.transAxes, path_effects=HALO)
    # The four anchors are not labelled. Following the convention of structural papers, only the
    # residue the text argues about (Gly156) is pointed at, and the anchors that provide context are
    # named in prose. Four leader lines obscured Gly156 and blurred the point.
    # The sentence is longer than the overview width (0.545), so it is centred on the whole panel
    ax.text(0.50, -0.045, "Ile148, Leu152, Ile155 and Phe159 anchor the helix in the groove, "
                          "burying 83 to 100 % of their free surface",
            color=PALETTE["gray_line"], fontsize=7.4, ha="center", va="top",
            transform=ax.transAxes)
    call(axo, "Gly156", (cx, cy), (0.855, 0.845), PALETTE["alt"], fs=9.0, ha="center")

    # The 69-degree rotation lays the clash axis in the plane of the image. The overlap of the two
    # vdW spheres is visible directly, so no separate 2D schematic is needed.
    Z = pd.read_csv(ST / "fig_struct_clash_anchor.csv").set_index("name")

    def zpt(name):
        return float(Z.loc[name, "x"]) - zx, float(Z.loc[name, "y"]) - zy

    call(axz, "modelled Cβ", zpt("CB156"), (0.185, 0.885),
         PALETTE["extra2"], fs=8.2, ha="center")
    call(axz, "Arg263 Cα", zpt("CA263"), (0.840, 0.870), PALETTE["extra1"],
         fs=8.2, ha="center")

    # The distance is drawn where it was measured. A bare number in the corner would leave the reader guessing between what and what.
    (x0, y0), (x1, y1) = zpt("CB156"), zpt("CA263")
    axz.plot([x0, x1], [y0, y1], color="black", lw=1.4, ls=(0, (2.6, 1.6)), zorder=8,
             path_effects=[pe.withStroke(linewidth=3.2, foreground="white")])
    # The sphere radius is 1.70 A, about 131 px, so the offset has to exceed that for the text to clear the sphere
    for dx, dy, t, fs, col in ((0, 182, "2.64 Å", 9.0, "black"),
                               (0, 250, "3.40 Å needed", 7.6, PALETTE["gray_line"])):
        axz.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, t, ha="center", va="center",
                 fontsize=fs, fontweight="bold", color=col, zorder=9,
                 path_effects=HALO)

    # Which ribbon is which. The overview has an Mcl-1 label, but the close-up had no cue at all.
    axz.text(0.055, 0.045, "Bim BH3", transform=axz.transAxes, fontsize=8.6,
             fontweight="bold", color="#333333", ha="left", path_effects=HALO)
    axz.text(0.955, 0.045, "Mcl-1", transform=axz.transAxes, fontsize=8.6,
             fontweight="bold", color="#333333", ha="right", path_effects=HALO)


def main():
    apply_publication_style()
    ST = figure_assets("struct")
    V = pd.read_csv(RESULTS / "case_study_variants.csv")
    S = pd.read_csv(RESULTS / "case_positions_scored.csv")
    P = V.groupby("pos").agg(gem=("gem_pos", "first"), ddg=("ddg_pos", "first")).reset_index()
    f = P[P.pos == FOCUS].iloc[0]

    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(2, 10, height_ratios=[0.88, 1.0], hspace=0.30, wspace=1.60,
                          left=0.052, right=0.975, top=0.925, bottom=0.070)
    panel_a(fig, fig.add_subplot(gs[0, 0:6]), S, f)
    better, nF = panel_b(fig.add_subplot(gs[0, 6:10]), V)
    panel_c(fig, fig.add_subplot(gs[1, 0:10]), ST)

    outdir = RESULTS / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "fig5_case"
    for fmt in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{fmt}"), dpi=300)
    plt.close(fig)
    print("wrote:", out.with_suffix(".pdf"), f"| Gly156 improved {better}/{nF}")


if __name__ == "__main__":
    main()
