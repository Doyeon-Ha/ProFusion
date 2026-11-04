#!/usr/bin/env python
"""Does the gain from axis-specific weights come from the **phenotype label of the assay** or
from **how many assays were drawn**?

## Metrics (defined before any number)

    rho        Spearman between prediction and measurement within one assay. One value per assay.
    matched    function weights on function assays, expression weights on expression assays.
    pooled     **one vector** fitted on all 49 tuning assays and applied to both axes. This is the
               zero line.
    random     drawn from the 49 ignoring the axis, at **the same size** as matched, fitted by the
               same procedure. A capacity control that tests whether size alone explains the gain.
               10 seeds.

    y = (mean rho of that arm) − (mean rho of pooled). Positive means better than pooled.

⚠ A random subset does not carry zero axis information; it comes along by chance. The point colour
  therefore shows the matching-axis fraction, so the panel is read as a **dose-response**. It is
  not an "axis-information-free control".

## Why a single panel

The earlier version drew the distribution of matched vs swapped/pooled in panel a. That was with
4 backbones. At 12 that panel becomes 48 boxes and says the same thing as Figure 3b in the main
text. The duplicate is dropped and only the control remains, which is all this appendix section
claims. The matched vs swapped/pooled numbers are in axis_crossapply_12bb.csv.

The internal axis key "stability" is the expression axis of the manuscript (14 tuning, 18
ProteinGym assays). It is a different axis from stability_ddg (the 65 folding-ΔG assays).

input   results/axis_crossapply_12bb_perdms.csv (matched and pooled baselines)
        results/control_random_uni3.csv         (240 random points)
output  results/figures/axis_crossapply.{pdf,png} and axis_crossapply_panelb.csv
run:    python figures/plot_axis_crossapply.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

# An earlier version read a scratch directory, which disappears when cleaned, and whose
#   common.DROP left the evaluation set at function 112 and expression 16. This reads the
#   repository output instead, where DROP is empty and the cohort matches the text (129 and 18).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plmfusion.paths import RESULTS           # noqa: E402

H = RESULTS
PER = H / "axis_crossapply_12bb_perdms.csv"   # matched/swapped/pooled, 12 backbones
CTL = H / "control_random_uni3.csv"           # size-matched random, 12 backbones x 10 seeds
OUT = RESULTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbone_key import CMAP                 # noqa: E402  colours are pinned by name

# -- Style (copied from the template rather than imported) --------------------
for _p in Path.home().joinpath(".fonts").glob("*.ttf"):
    try:
        fm.fontManager.addfont(str(_p))
    except Exception:
        pass
_HAVE = {f.name for f in fm.fontManager.ttflist}
_FAMILY = next((f for f in ("Source Sans Pro", "Helvetica", "Arial", "DejaVu Sans") if f in _HAVE),
               "DejaVu Sans")
plt.rcParams.update({
    "font.family": _FAMILY, "font.size": 8.5,
    "axes.labelsize": 9.5, "axes.titlesize": 10, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})
INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#d8d8d4"
FRACMAP = plt.get_cmap("YlGnBu")   # matching-axis fraction: pale = low, dark = high
SHOW = {"AIDO_RAG16B": "AIDO-RAG 16B", "VenusREM": "VenusREM", "ProSST": "ProSST",
        "VespaG": "VespaG", "TranceptionLret": "Tranception L (ret)", "S2F": "S2F",
        "ESM1v": "ESM-1v", "SaProt650M": "SaProt 650M", "ESM3": "ESM3-open",
        "ESM2": "ESM2 650M", "ESMIF1": "ESM-IF1", "ProteinMPNN": "ProteinMPNN"}
AXES = ["function", "stability"]
AXNAME = {"function": "Function", "stability": "Expression"}

# -- Data ---------------------------------------------------------------------
per = pd.read_csv(PER)
ctl = pd.read_csv(CTL)
# The same ordering as Figure 3 in the main text: unfused function accuracy, descending.
BB = (pd.read_csv(H / "extend_backbones.csv").query("axis == 'function' and bb in @SHOW")
      .set_index("bb").pg_base.sort_values(ascending=False).index.tolist())
assert set(BB) == set(ctl.bb) == set(per.bb), "the backbone sets differ between the three inputs"

# Check that the two inputs saw the same evaluation set. A mismatch would mix them silently.
NEVAL = {}
for a in AXES:
    n_per = per[(per.axis == a) & (per.bb == BB[0])].dms.nunique()
    n_ctl = int(ctl[ctl.eval_axis == a].n_eval.iloc[0])
    assert n_per == n_ctl, f"{a}: perdms {n_per} vs control {n_ctl}"
    NEVAL[a] = n_per

mu = per.groupby(["axis", "bb"])[["matched", "pooled"]].mean()
B = []
for ax_name in AXES:
    for bb in BB:
        m = mu.loc[(ax_name, bb)]
        c = ctl[(ctl.eval_axis == ax_name) & (ctl.bb == bb)].sort_values("frac_same_axis")
        B.append(dict(axis=ax_name, bb=bb, matched=m["matched"] - m["pooled"],
                      rand=(c["mean"] - m["pooled"]).to_numpy(),
                      frac=c["frac_same_axis"].to_numpy()))

# -- Figure -------------------------------------------------------------------
fig = plt.figure(figsize=(7.2, 3.25))
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.082, right=0.995, top=0.835, bottom=0.265)

x, tick, cen, sep = 0.0, [], [], None
for ai, ax_name in enumerate(AXES):
    blk = []
    for bb in BB:
        r = next(q for q in B if q["axis"] == ax_name and q["bb"] == bb)
        jit = np.linspace(-0.26, 0.26, r["rand"].size)
        ax.scatter(x + jit, r["rand"], s=13, c=r["frac"], cmap=FRACMAP, vmin=0.0, vmax=1.0,
                   linewidths=0.4, edgecolors=INK2, zorder=2)
        ax.plot([x], [r["matched"]], marker="D", ms=5.6, mfc=CMAP[bb], mec=INK,
                mew=0.8, zorder=4)
        tick.append(x)
        blk.append(x)
        x += 1.0
    cen.append(float(np.mean(blk)))
    if ai == 0:
        sep = x - 0.5
        x += 0.5
ax.axvline(sep, color=GRIDC, lw=0.9, zorder=1)
ax.axhline(0, color=INK2, lw=1.0, zorder=1)
ax.set_xlim(-0.72, x - 0.28)

# ⚠ A fixed range clips points silently when the cohort changes (−0.030 once clipped −0.040).
#   The top is given more room than the bottom, because the legend sits there.
_b = np.concatenate([r["rand"] for r in B] + [np.array([r["matched"] for r in B])])
_rng = _b.max() - _b.min()
ax.set_ylim(_b.min() - 0.08 * _rng, _b.max() + 0.34 * _rng)

for p_, a_ in zip(cen, AXES):
    ax.text(p_, 1.035, f"{AXNAME[a_]}  ({NEVAL[a_]} assays)",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=8.5, color=INK2)
ax.set_xticks(tick)
ax.set_xticklabels([SHOW[b] for _ in AXES for b in BB], fontsize=7.4, rotation=38,
                   ha="right", rotation_mode="anchor")
ax.set_ylabel("mean ρ  relative to pooled fit")
ax.set_title("The gain is the axis label, not the set size", loc="left",
             fontweight="bold", pad=17)
ax.legend(handles=[
    Line2D([], [], marker="D", ls="none", ms=5.6, mfc=INK2, mec=INK, mew=0.8,
           label="axis-matched"),
    Line2D([], [], marker="o", ls="none", ms=4.0, mfc=FRACMAP(0.55), mec=INK2, mew=0.4,
           label="random subset, same size (10 seeds)"),
], loc="upper left", frameon=False, ncol=2, columnspacing=1.4, handletextpad=0.4,
    borderpad=0.0, bbox_to_anchor=(-0.010, 1.015))

cax = ax.inset_axes([0.068, 0.075, 0.140, 0.032])
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=FRACMAP),
                  cax=cax, orientation="horizontal")
cb.set_ticks([0, 0.5, 1.0])
cb.ax.tick_params(labelsize=7.0, length=1.8, width=0.7, pad=1.0)
cb.outline.set(linewidth=0.6, edgecolor=INK2)
cax.set_title("same-axis fraction of the subset", fontsize=7.4, color=INK2,
              pad=3.0, loc="left")

for p_ in (OUT / "axis_crossapply.pdf", OUT / "axis_crossapply.png"):
    fig.savefig(p_)
print("wrote", OUT / "axis_crossapply.pdf")

# -- The numbers the figure drew ----------------------------------------------
S = pd.DataFrame([
    dict(axis=r["axis"], bb=r["bb"], matched_minus_pooled=r["matched"],
         rand_mean=float(np.mean(r["rand"])), rand_sd=float(np.std(r["rand"], ddof=1)),
         rand_min=float(np.min(r["rand"])), rand_max=float(np.max(r["rand"])),
         n_seeds_below_pooled=int(np.sum(r["rand"] < 0)),
         n_seeds_ge_matched=int(np.sum(r["rand"] >= r["matched"])),
         frac_same_axis_min=float(np.min(r["frac"])), frac_same_axis_max=float(np.max(r["frac"])),
         rho_frac_vs_perf=float(spearmanr(r["frac"], r["rand"])[0]))
    for r in B])
S.to_csv(OUT / "axis_crossapply_panelb.csv", index=False)
print(S.round(4).to_string(index=False))

# Pooled dose-response. ⚠ The rho level differs per cell, so the values must be **z-normalised
#   within a cell** before pooling. Pooling the raw values lets the between-cell differences
#   dominate and flips the sign.
c = ctl.copy()
c["z"] = c.groupby(["eval_axis", "bb"])["mean"].transform(lambda v: (v - v.mean()) / v.std(ddof=1))
rz, pz = spearmanr(c.frac_same_axis, c.z)
rr, _ = spearmanr(c.frac_same_axis, c["mean"])
print(f"\npooled dose-response  within-cell z: rho={rz:+.3f} p={pz:.2e} (n={len(c)})"
      f"  | raw: rho={rr:+.3f} (for reference; dominated by cell effects)")
print("seeds at or above matched:", int(S.n_seeds_ge_matched.sum()), f"/ {len(c)}")
print("cells where rand_mean is closer to pooled than to matched:",
      int((S.rand_mean.abs() < (S.rand_mean - S.matched_minus_pooled).abs()).sum()), f"/ {len(S)}")
for a_ in AXES:
    v = S[S.axis == a_].rho_frac_vs_perf
    print(f"  {AXNAME[a_]:11s} per-backbone rho {v.min():+.2f} to {v.max():+.2f}")
