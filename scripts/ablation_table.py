r"""Build the ablation table (\Cref{tab:abl}) directly from two CSVs. It replaces Fig 6.

Two outputs
  ablation_table.tex   the table* block to paste into the paper
  ablation_table.json  every number, so that the tex audit can check each cell of the text
                       and the table

The aggregation convention matches the figure script and final_numbers.py.
  panel a  channel Shapley divided by that backbone's total_gain, then the median, range and
           count of positives over the 12 backbones
  panel b  per backbone, the mean over DMS of (SFW7 - arm), then the median and range over the
           12 backbones. Wilcoxon is paired over DMS and BH is applied within a panel.
           ⚠ Counting n_sig without regard to sign mixes in cells where the reduced model wins.
             Both counts are reported.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from plmfusion.paths import RESULTS

H = RESULTS
H.mkdir(parents=True, exist_ok=True)
BB12 = ["AIDO_RAG16B", "VenusREM", "ProSST", "VespaG", "TranceptionLret", "S2F",
        "ESM1v", "SaProt650M", "ESM3", "ESM2", "ESMIF1", "ProteinMPNN"]
PRETTY = {"AIDO_RAG16B": "AIDO-RAG 16B", "VenusREM": "VenusREM", "ProSST": "ProSST",
          "VespaG": "VespaG", "TranceptionLret": "Tranception L", "S2F": "S2F",
          "ESM1v": "ESM-1v", "SaProt650M": "SaProt 650M", "ESM3": "ESM3-open",
          "ESM2": "ESM2 650M", "ESMIF1": "ESM-IF1", "ProteinMPNN": "ProteinMPNN"}
AXIS = [("function", "Function", 129), ("stability", "Expression", 18)]
CH = [("GEMME", "GEMME"), ("ΔΔG", r"$\Delta\Delta G$"), ("RSA", "RSA"),
      ("BLOSUM", "BLOSUM62"), ("Δcharge", r"$\Delta$charge")]
ARMS = [("equal5", "No fitted weights", 0),
        ("merge both", "No decomposition", 5),
        ("merge GEMME", r"\texttt{merge GEMME}", 6),
        ("merge ddG", r"\texttt{merge $\Delta\Delta G$}", 6)]
NB = len(BB12)


BETTER = (r"\begin{tabular}[b]{@{}c@{}}Better\\($q<0.05$)\end{tabular}")


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty(len(p))
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return q


def tex_num(x, dec, signed=False):
    s = f"{abs(x):.{dec}f}"
    if x < 0:
        return "$-$" + s
    return ("$+$" if signed else "") + s


def tex_rng(lo, hi, dec):
    """Range cells carry a sign only when negative. It saves width, and 'to' already says range."""
    return f"{tex_num(lo, dec)} to {tex_num(hi, dec)}"


def build():
    S = pd.read_csv(H / "shapley_channels.csv")
    D = pd.read_csv(H / "ablation_12bb_perdms.csv")
    S, D = S[S.bb.isin(BB12)], D[D.bb.isin(BB12)]
    assert set(S.bb) == set(D.bb) == set(BB12), "backbone set drifted"
    assert set(S.channel) == {c for c, _ in CH}, "channel set drifted"
    for a, _, n in AXIS:
        assert D[D.axis == a].groupby("bb").dms.nunique().eq(n).all(), f"{a} assay count"
    S = S.assign(pct=100 * S.shapley / S.total_gain)
    assert np.allclose(S.groupby(["axis", "bb"]).pct.sum(), 100.0), "shares sum to 100"

    out = {"panel_a": {}, "panel_b": {}, "n_backbones": NB,
           "n_assays": {lab: n for _, lab, n in AXIS}}

    for key, _ in CH:
        for ax, lab, _ in AXIS:
            v = S[(S.axis == ax) & (S.channel == key)].pct.values
            assert len(v) == NB
            out["panel_a"][f"{lab}_{key}"] = dict(
                med=float(np.median(v)), lo=float(v.min()), hi=float(v.max()),
                pos=int((v > 0).sum()))

    for arm, _, nw in ARMS:
        for ax, lab, _ in AXIS:
            m, p, tied, bad = [], [], [], []
            for bb in BB12:
                d = D[(D.axis == ax) & (D.bb == bb)]
                f = d[d.arm == "SFW7"].set_index("dms").rho
                dd = (f - d[d.arm == arm].set_index("dms").rho.reindex(f.index)).dropna()
                assert len(dd) == dict((a, n) for a, _, n in AXIS)[ax], f"{arm} {bb} n"
                m.append(float(dd.mean()))
                if (dd == 0).all():
                    p.append(1.0)
                    tied.append(bb)
                else:
                    p.append(float(wilcoxon(dd).pvalue))
            m, q = np.array(m), bh(p)
            bad = [BB12[i] for i in range(NB) if q[i] < 0.05 and m[i] < 0]
            out["panel_b"][f"{lab}_{arm}"] = dict(
                fitted=nw, med=float(np.median(m)), lo=float(m.min()), hi=float(m.max()),
                win=int((m > 0).sum()), sig=int(((q < 0.05) & (m > 0)).sum()),
                sig_any=int((q < 0.05).sum()), against=bad, tied=tied)

    SHOWN = ["equal5", "merge both"]          # the two rows kept in the table
    g = lambda ax, a, k: out["panel_b"][f"{ax}_{a}"][k]
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\caption{\textbf{Channel contributions and parameterization ablation.}",
         r"Every row summarises the 12 backbones. Backbones counts those that support",
         r"the row, and in \textbf{(b)} the number in parentheses counts how many of",
         r"those reach $q<0.05$ in favour of full ProFusion, by a two-sided Wilcoxon",
         r"signed-rank test paired by assay with Benjamini--Hochberg adjustment.",
         r"\textbf{(a)} Proportional Shapley values over all 32 channel subsets,",
         r"counted as supporting when above zero. \textbf{(b)} Full ProFusion fits 7",
         r"weights. No fitted weights averages the 5 standardized channels",
         r"(\texttt{equal5}), and no decomposition fits 5 weights that tie the",
         r"position and residual terms of GEMME and $\Delta\Delta G$",
         r"(\texttt{merge GEMME,}\,\texttt{$\Delta\Delta G$}). Variants releasing one",
         r"channel's split are given in the text, per-backbone values in the source",
         r"data file.}",
         r"\label{tab:abl}", "",
         r"\begin{tabular*}{0.86\textwidth}{@{\extracolsep{\fill}}l rc rc@{}}",
         r"\toprule",
         r"& \multicolumn{2}{c}{Function, 129 assays}"
         r" & \multicolumn{2}{c}{Expression, 18 assays} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"& Median & Backbones & Median & Backbones \\", r"\midrule",
         r"\multicolumn{5}{@{}l}{\textbf{(a)}~Share of each backbone's own fusion"
         r" gain (\%)} \\"]
    for key, nm in CH:
        c = [out["panel_a"][f"{lab}_{key}"] for _, lab, _ in AXIS]
        L.append(r"\quad " + nm + " & " + " & ".join(
            f"{tex_num(x['med'], 1)} & {x['pos']}/{NB}" for x in c) + r" \\")
    L += [r"\addlinespace[4pt]",
          r"\multicolumn{5}{@{}l}{\textbf{(b)}~Advantage of full ProFusion over a"
          r" reduced model ($\Delta\rho$)} \\"]
    for arm, nm, nw in ARMS:
        if arm not in SHOWN:
            continue
        c = [out["panel_b"][f"{lab}_{arm}"] for _, lab, _ in AXIS]
        L.append(r"\quad " + nm + " & " + " & ".join(
            f"{tex_num(x['med'], 4, True)} & {x['win']}/{NB} ({x['sig']})"
            for x in c) + r" \\")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]

    return out, L


def main():
    out, L = build()
    (H / "ablation_table.tex").write_text("\n".join(L) + "\n")
    (H / "ablation_table.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for k, v in out["panel_b"].items():
        print(f"{k:34s} med {v['med']:+.4f}  win {v['win']}/{NB}  "
              f"sig+ {v['sig']}  sig(any) {v['sig_any']}  against {v['against']}  "
              f"tied {v['tied']}")
    print("wrote ablation_table.tex / .json")


if __name__ == "__main__":
    main()
