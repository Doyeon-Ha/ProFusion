"""The **numeric table** behind panel a of the supplementary variability figure: the spread of
per-assay rho for each backbone.

A table replaces the violin plot that used to sit here. The violin was dropped not because it was
wrong but because it **reads as the same picture** as the grey boxes in panels a and b of Figure 2.
Showing the same values twice in two different plots is worse than a table.

The scope matches panel b (the ECDF). The default is **function 129 + expression 18 = 147**
(extend_backbones_perdms.csv). Older cohorts (129 = 113+16, 128 = 112+16, 112) are kept in SETS
below and are produced only when their input file is present, for comparison against earlier
numbers.

⚠ The file tag is **the sum over both axes**, so it collides with the function count. The default
  tag is now 147, and the old tag "129" means 113+16, not today's function 129.

What is measured (all on the backbone score alone, before fusion)
  rho      Spearman between the zero-shot LLR and the experimental value within one assay.
           Unitless, [-1, 1].
  SD       sample standard deviation (ddof=1). How much that backbone varies across assays.
  rho<0.4  fraction of assays with rho below 0.4. The 0.4 cutoff was requested and differs from
           the 0.3 boundary used in the text, so it is not carried outside this table.

Rows are ordered by descending median, as in the figure.

output  backbone_variability_147.{csv,tex} (and _129, _128, _112 when those inputs exist)
run     python scripts/backbone_variability.py [TAG]   # one tag only, for example 147
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plmfusion.paths import RESULTS

H = RESULTS          # reads extend_backbones_perdms.csv, writes .csv/.tex
SHOW = {"AIDO_RAG16B": "AIDO-RAG 16B", "VenusREM": "VenusREM", "ProSST": "ProSST",
        # ⚠ This is **S2F**, not "S2F-MSA". Our llr_S2F reproduces the S2F column of the
        #   official table (median |d_rho| 0.0003 on 133/147) and is far from the S2F-MSA column
        #   (0.0387). S2F-MSA is z(S2F) + z(EVE), and the 49 external assays have no EVE, so it
        #   cannot be fitted under the same definition. Every figure writes "S2F"; only this file
        #   differed.
        "VespaG": "VespaG", "TranceptionLret": "Tranception L", "S2F": "S2F",
        "ESM1v": "ESM-1v", "SaProt650M": "SaProt 650M", "ESM3": "ESM3-open",
        "ESM2": "ESM2 650M", "ESMIF1": "ESM-IF1", "ProteinMPNN": "ProteinMPNN"}
CUT = 0.4


def table(P, axes, tag, note):
    f = P[P.axis.isin(axes) & P.bb.isin(SHOW)]
    V = {b: f[f.bb == b].base.to_numpy() for b in SHOW}
    n = {b: len(v) for b, v in V.items()}
    assert len(set(n.values())) == 1, f"assay counts differ between backbones: {n}"
    order = sorted(SHOW, key=lambda b: -np.median(V[b]))

    R = pd.DataFrame([dict(backbone=SHOW[b], n=len(V[b]), max=V[b].max(),
                           min=V[b].min(), mean=V[b].mean(),
                           median=np.median(V[b]), std=V[b].std(ddof=1),
                           below=float((V[b] < CUT).mean())) for b in order])
    R.to_csv(H / f"backbone_variability_{tag}.csv", index=False)

    print(f"\n=== backbone alone, per-assay Spearman rho, {R.n.iloc[0]} DMS ({note}) ===")
    print(f"{'backbone':15s} {'n':>4s} {'max':>7s} {'min':>7s} {'mean':>7s} "
          f"{'median':>7s} {'std':>7s} {'<0.4':>7s}")
    for r in R.itertuples():
        print(f"{r.backbone:15s} {r.n:4d} {r.max:7.3f} {r.min:7.3f} {r.mean:7.3f} "
              f"{r.median:7.3f} {r.std:7.3f} {r.below*100:6.1f}%")

    # Emit the tabular only; the caption and float go in the manuscript.
    tex = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
           r"Backbone & Max & Min & Mean & Median & SD & $\rho<0.4$ \\", r"\midrule"]
    # Use a real minus sign, not a hyphen: in a table "-0.010" typesets as a hyphen.
    m = lambda x: (f"$-${abs(x):.3f}" if x < 0 else f"{x:.3f}")
    for r in R.itertuples():
        tex.append(f"{r.backbone} & {m(r.max)} & {m(r.min)} & {m(r.mean)} & "
                   f"{m(r.median)} & {m(r.std)} & {r.below*100:.1f}\\% \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (H / f"backbone_variability_{tag}.tex").write_text("\n".join(tex) + "\n")
    print("wrote:", H / f"backbone_variability_{tag}.tex")


# (source csv, axes to use, file tag, description, expected function count, expected expression count)
# ⚠ The tag is **the sum over both axes** and is easy to confuse with the function count. When
#   function became 129 on 2026-10-28 it collided with the old tag "129" (= 113+16). The tag is
#   therefore not trusted: the expected counts below are checked against the actual assays.
SETS = [
    ("extend_backbones_perdms.csv",        ["function", "stability"], "147",
     "function 129 + expression 18", 129, 18),
    ("extend_backbones_perdms_113.csv",    ["function", "stability"], "129",
     "function 113 + expression 16", 113, 16),
    ("extend_backbones_perdms_pre129.csv", ["function"],              "112",
     "function 112", 112, 0),
    ("extend_backbones_perdms_pre129.csv", ["function", "stability"], "128",
     "function 112 + expression 16", 112, 16),
]


def main(only=None):
    H.mkdir(parents=True, exist_ok=True)
    for src, axes, tag, note, n_fun, n_exp in SETS:
        if only and tag != only:
            continue
        f = H / src
        if not f.exists():
            print(f"skipping {tag}: {src} not found"); continue
        P = pd.read_csv(f)
        got_f = P[P.axis == "function"].dms.nunique()
        got_e = P[P.axis == "stability"].dms.nunique()
        assert got_f == n_fun, f"{src}: function {got_f} != expected {n_fun}"
        if "stability" in axes:
            assert got_e == n_exp, f"{src}: expression {got_e} != expected {n_exp}"
        want = n_fun + (n_exp if "stability" in axes else 0)
        assert int(tag) == want, f"tag {tag} != actual total {want}"
        table(P, axes, tag, note)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(only=args[0] if args else None)
