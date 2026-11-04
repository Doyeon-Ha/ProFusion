#!/usr/bin/env python
"""Collect the **final numbers** for the paper into one file. The tex substitution script reads
only this.

Numbers produced in two places always drift apart, so the code that edits the tex reads only the
final_numbers.json this script writes, never the CSVs directly.

Evaluation set 129/18. ProteinMPNN and ESM-IF1 are the values after fitting and evaluation were
unified on the official formula.

output  results/final_numbers.json and results/FINAL.md
run     OMP_NUM_THREADS=6 python scripts/final_numbers.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr, mannwhitneyu
import itertools

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
from plmfusion.paths import RESULTS, REFERENCE, data_root, figure_assets   # noqa: E402
from backbone_key import STRUCT                                    # noqa: E402

# Every intermediate CSV written by the other scripts in the chain lands in RESULTS, and this
# script reads only those.
H = OUT = FGA = RESULTS

SL = ["a_GEMME", "b_GEMME", "a_ddG", "b_ddG", "a_RSA", "b_BLOSUM", "b_dchg"]
IF2 = ["ESMIF1", "ProteinMPNN"]
BB12 = sorted(set(pd.read_csv(REFERENCE / "paper_perdms.csv").bb))
# ⚠ The 12 backbones the figures use. extend_backbones_perdms.csv holds 14, of which TranceptionL
#   and S2F_MSA are **not** paper backbones (the paper's "Tranception L" is TranceptionLret and its
#   "S2F-MSA" is S2F). Without this filter the ranges come out wrong.
SHOW12 = set(pd.read_csv(REFERENCE / "paper_perdms.csv").bb)
NAME = {"AIDO_RAG16B": "AIDO-RAG 16B", "ESM3": "ESM3-open", "ESMIF1": "ESM-IF1",
        "ProteinMPNN": "ProteinMPNN", "ProSST": "ProSST", "ESM2": "ESM2",
        "ESM1v": "ESM-1v", "S2F": "S2F", "VespaG": "VespaG", "VenusREM": "VenusREM",
        "TranceptionLret": "Tranception L", "SaProt650M": "SaProt 650M"}
F = {}


def bh(ps):
    """Benjamini-Hochberg, under the same convention as the figure script (grouped within a panel)."""
    p = np.asarray(ps, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); run = 1.0
    for i in range(n - 1, -1, -1):
        run = min(run, p[o[i]] * n / (i + 1)); q[o[i]] = run
    return q


def perm_p(a, u):
    """12C8 = 495 exhaustive label permutations. This is an exact p, not an approximation."""
    obs = abs(np.mean(a) - np.mean(u)); pool = np.array(list(a) + list(u)); k = len(a)
    hit = tot = 0
    for c in itertools.combinations(range(len(pool)), k):
        m = np.zeros(len(pool), bool); m[list(c)] = True
        hit += abs(pool[m].mean() - pool[~m].mean()) >= obs - 1e-12
        tot += 1
    return hit / tot


def main():
    D = pd.read_csv(H / "extend_backbones.csv"); D = D[D.bb.isin(BB12)]
    P = pd.read_csv(H / "extend_backbones_perdms.csv"); P = P[P.bb.isin(BB12)]
    ref = pd.read_csv(data_root() / "proteingym" / "DMS_substitutions.csv").set_index("DMS_id")

    # ── 4.1 · Datasets
    for axk, lab in [("function", "function"), ("stability", "expression")]:
        q = P[P.axis == axk]
        s = ref.loc[[d for d in sorted(set(q.dms)) if d in ref.index]]
        g = q.groupby("bb").agg(b=("base", "median"), f=("fit7", "median"))
        rel = (g.f - g.b) / g.b; r10 = rel.drop(index=IF2)
        lo, hi = r10.idxmin(), r10.idxmax()
        F[lab] = dict(
            n=int(q.dms.nunique()), prot=int(s["UniProt_ID"].nunique()),
            types={k: int(v) for k, v in s["coarse_selection_type"].value_counts().items()},
            rel10=float(r10.median()),
            # ⚠ The abstract and the contributions paragraph say "across twelve backbones", so the
            #   number quoted there must be the median over 12. Only one sentence in the text
            #   splits it 10/2.
            rel12=float(rel.median()),
            lo_bb=NAME[lo], lo=float(r10[lo]), lo_base=float(g.b[lo]), lo_fit=float(g.f[lo]),
            hi_bb=NAME[hi], hi=float(r10[hi]), hi_base=float(g.b[hi]), hi_fit=float(g.f[hi]),
            r10_min=float(r10.min()), r10_max=float(r10.max()),
            esmif1=float(rel["ESMIF1"]), esmif1_base=float(g.b["ESMIF1"]),
            esmif1_fit=float(g.f["ESMIF1"]),
            mpnn=float(rel["ProteinMPNN"]), mpnn_base=float(g.b["ProteinMPNN"]),
            mpnn_fit=float(g.f["ProteinMPNN"]),
            max_p=float(max(wilcoxon(t.fit7, t.base).pvalue for _, t in q.groupby("bb"))),
            n_up=int((rel > 0).sum()),
            strongest=NAME[g.b.idxmax()], strongest_base=float(g.b.max()),
            # The text says the two inverse-folding models started from the lowest baselines and
            # gained the most. That is a claim about **rank**, not a number, so the per-backbone
            # values have to be kept for it to be checkable.
            base_med={k: float(v) for k, v in g.b.items()},
            rel_all={k: float(v) for k, v in rel.items()})
        # Channel share (this comes from the weights alone)
        f = D[D.axis == axk].set_index("bb")
        # ⚠ The denominator is the sum of **|w|**. f["total"] is the signed sum, which diverges as
        #   soon as one slot is negative (9.3% vs 8.9% for a_ddG = −0.1 on ESM3 function). The
        #   figure the text cites is drawn by plot_model_needs_2ax.py, which uses the absolute sum,
        #   so this matches that.
        den = f[SL].abs().sum(axis=1)
        W = f[SL].to_numpy(float); S = W / den.to_numpy()[:, None]
        med = {k: float(v) for k, v in zip(SL, np.median(S, 0) * 100)}
        F[lab]["share"] = med
        # ⚠ The paper says "the median of the per-backbone sum". Summing per-slot medians is a different statistic
        #   (function GEMME 46.3 vs 46.7).
        F[lab]["share_GEMME"] = float((((f["a_GEMME"] + f["b_GEMME"]) / den) * 100).median())
        F[lab]["share_ddG"] = float((((f["a_ddG"] + f["b_ddG"]) / den) * 100).median())
        F[lab]["share_RSA"] = float(((f["a_RSA"] / den) * 100).median())
        F[lab]["share_a_ddG"] = float(((f["a_ddG"] / den) * 100).median())
        sh = ((f["a_ddG"] + f["b_ddG"]) / den) * 100
        aw = [sh[b] for b in f.index if b in STRUCT]; un = [sh[b] for b in f.index if b not in STRUCT]
        F[lab]["ddg_group"] = dict(
            aware_med=float(np.median(aw)), aware_lo=float(min(aw)), aware_hi=float(max(aw)),
            un_med=float(np.median(un)), un_lo=float(min(un)), un_hi=float(max(un)),
            p=float(mannwhitneyu(aw, un, alternative="two-sided").pvalue))

    # -- rho thresholds (median count over 12 backbones)
    q = P[P.axis == "function"]
    F["thr"] = {}
    for t in (0.1, 0.2, 0.3, 0.4):
        nb = float(q.groupby("bb").base.apply(lambda x: (x < t).sum()).median())
        nf = float(q.groupby("bb").fit7.apply(lambda x: (x < t).sum()).median())
        F["thr"][str(t)] = dict(base=nb, fus=nf, red=1 - nf / nb)

    # -- accuracy correlation (accuracy = the per-assay median)
    f = D[D.axis == "function"].set_index("bb")
    sh = (f["a_ddG"] + f["b_ddG"]) / f[SL].abs().sum(axis=1)
    med = q.groupby("bb").base.median().reindex(f.index)
    st = pd.Series({b: (1 if b in STRUCT else 0) for b in f.index})
    F["corr"] = dict(share_acc=float(spearmanr(sh, med).statistic),
                     share_acc_p=float(spearmanr(sh, med).pvalue),
                     struct_acc=float(spearmanr(st, med).statistic),
                     struct_acc_p=float(spearmanr(st, med).pvalue))

    # -- 4.2 axis cross-application
    X = pd.read_csv(H / "axis_crossapply_12bb.csv"); X = X[X.bb.isin(BB12)]
    F["cross"] = {}
    for axk, lab in [("function", "function"), ("stability", "expression")]:
        for con in ("swapped", "pooled"):
            g = X[(X.axis == axk) & (X.contrast == con)]
            sig = g[g.q < 0.05]
            F["cross"][f"{lab}_{con}"] = dict(
                lo=float(g["mean"].min()), hi=float(g["mean"].max()),
                n_sig=int(len(sig)), n=int(len(g)),
                q_max=float(g.q.max()), q_max_sig=float(sig.q.max()) if len(sig) else None,
                sig_lo=float(sig["mean"].min()) if len(sig) else None,
                sig_hi=float(sig["mean"].max()) if len(sig) else None,
                exceptions=[NAME.get(b, b) for b in g[g.q >= 0.05].bb])

    # ── 4.4 Shapley · ablation
    # ⚠ What the paper uses is the **proportion of the total gain** (proportional Shapley). The raw
    #   data is long over (axis, bb, channel), so averaging without splitting by channel gives a
    #   meaningless value.
    S = pd.read_csv(H / "shapley_channels.csv"); S = S[S.bb.isin(BB12)]
    S["share"] = S.shapley / S.total_gain
    F["shapley"] = {}
    for axk, lab in [("function", "function"), ("stability", "expression")]:
        g = S[S.axis == axk]
        F["shapley"][lab] = {str(c): float(v * 100)
                             for c, v in g.groupby("channel").share.median().items()}
        F["shapley"][lab + "_pos_all12"] = {
            str(c): int((gg.share > 0).all())
            for c, gg in g.groupby("channel")}
    A = pd.read_csv(H / "ablation_12bb_perdms.csv"); A = A[A.bb.isin(BB12)]
    F["ablation"] = {}
    for axk, lab in [("function", "function"), ("stability", "expression")]:
        for arm in ("equal5", "merge both", "merge GEMME", "merge ddG"):
            v, ps = [], []
            for bb in BB12:
                d = A[(A.axis == axk) & (A.bb == bb)]
                fr = d[d.arm == "SFW7"].set_index("dms").rho
                w = d[d.arm == arm].set_index("dms").rho.reindex(fr.index)
                if w.isna().all():
                    continue
                v.append(float((fr - w).mean()))
                g = (fr - w).to_numpy()
                ps.append(wilcoxon(fr, w).pvalue if np.any(g != 0) else 1.0)
            if v:
                F["ablation"][f"{lab}_{arm}"] = dict(med=float(np.median(v)),
                                                     lo=float(min(v)), hi=float(max(v)),
                                                     n_sig=int(sum(q < 0.05 for q in bh(ps))),
                                                     n=len(v))

    # -- 4.3 SBI (11 proteins)
    Q = pd.read_csv(FGA / "sbi_positions_p11.csv")
    co = pd.read_csv(FGA / "sbi_positions_p11_cohort.csv")
    T = Q[Q.quad == "T_SBI"].groupby(["bb", "arm"]).Pm.mean().unstack()
    aw = [b for b in T.index if b in STRUCT]; un = [b for b in T.index if b not in STRUCT]
    sbi = dict(n_prot=int(co.prot.nunique()), n_var=int(co.n_var.sum()),
               n_pos=int(co.n_pos.sum()),
               n_sbi=int(Q[(Q.quad == "T_SBI") & (Q.bb == aw[0]) & (Q.arm == "pLM")].n.sum()))
    for arm in ("pLM", "ProFusion"):
        a = T.loc[aw, arm].to_numpy(); u = T.loc[un, arm].to_numpy()
        sbi[arm] = dict(aware=float(a.mean()), un=float(u.mean()),
                        gap=float(abs(a.mean() - u.mean())), p=float(perm_p(a, u)),
                        n_pos_aware=int((a > 0).sum()), n_neg_aware=int((a < 0).sum()),
                        mpnn=float(T.loc["ProteinMPNN", arm]))
    for qd, nm in (("T_OK", "wtlike"), ("T_LOSS", "totalloss")):
        sbi[nm] = float(Q[(Q.quad == qd) & (Q.arm == "ProFusion")].Pm.mean())
    # -- GCK Thr168 position percentile (panel d of the text). It moves with the backbone weights.
    gpath = figure_assets("sbi_gck") / "panel_c12.csv"
    if gpath.exists():
        G = pd.read_csv(gpath)
        G["aware"] = G.bb.isin(STRUCT)
        for arm in ("pLM", "ProFusion"):
            sbi[f"gck_{arm}_aware"] = float(G[G.aware][arm].median())
            sbi[f"gck_{arm}_un"] = float(G[~G.aware][arm].median())
    F["sbi"] = sbi

    # -- supplementary: breakdown (147 = function 129 + expression 18)
    FIG = H / "figures"
    nt = F["function"]["n"] + F["expression"]["n"]   # ⚠ f was overwritten as a DataFrame above
    F["supp"] = dict(n_tot=nt)
    dpath = FIG / f"figS_breakdown_{nt}_data.csv"
    if dpath.exists():
        B = pd.read_csv(dpath)
        # ⚠ The last row, "SPAN across the 11 pLMs", is a summary row. Its in_11pLM_claim is True,
        #   so filtering on that alone lets it in and the span comes out 0.373 instead of 0.138.
        core = B[B.in_11pLM_claim & (B.backbone != "ProteinMPNN")
                 & ~B.backbone.astype(str).str.startswith("SPAN")]
        odd = B[B.backbone == "ProteinMPNN"]
        F["supp"].update(
            med_lo=float(core.p50.min()), med_hi=float(core.p50.max()),
            med_span=float(core.p50.max() - core.p50.min()),
            p10_lo=float(core.p10.min()), p10_hi=float(core.p10.max()),
            p10_span=float(core.p10.max() - core.p10.min()),
            span_p25=float(core.p25.max() - core.p25.min()),
            span_p75=float(core.p75.max() - core.p75.min()),
            span_p90=float(core.p90.max() - core.p90.min()),
            odd_med=float(odd.p50.iloc[0]), odd_p10=float(odd.p10.iloc[0]),
            odd_p90=float(odd.p90.iloc[0]),
            n_below_odd_p10=int((core.p10 < odd.p10.iloc[0]).sum()),
            n_below_odd_p25=int((core.p25 < odd.p25.iloc[0]).sum()),
            lowest_p10_name=str(core.loc[core.p10.idxmin(), "backbone"]))
    spath = FIG / f"figS_breakdown_{nt}_stats.csv"
    if spath.exists():
        S2 = pd.read_csv(spath).set_index("stat")
        for k in ("span_p25", "span_p50", "span_p75", "span_p90"):
            if k in S2.index:
                F["supp"][f"boot_{k}"] = float(S2.loc[k, "p_span10_greater"])
        if "span_p10" in S2.index:
            F["supp"]["p10_ci"] = [float(S2.loc["span_p10", "ci_lo"]),
                                   float(S2.loc["span_p10", "ci_hi"])]
        if "spearman_p10_vs_median_across_11pLM" in S2.index:
            r = S2.loc["spearman_p10_vs_median_across_11pLM"]
            F["supp"]["p10_vs_med_rho"] = float(r["value"])
            F["supp"]["p10_vs_med_p"] = float(r["p_span10_greater"])
        # ⚠ The supplementary text says ProteinMPNN is below every other backbone **at every
        #   percentile**. Rescoring raises its base substantially, which can break that claim. If it
        #   breaks, the prose has to be rewritten, so a flag is raised here.
        import numpy as _np
        QS = ["p10", "p25", "p50", "p75", "p90"]
        below = {q: bool(odd[q].iloc[0] < core[q].min()) for q in QS}
        F["supp"]["odd_below_all"] = below
        F["supp"]["odd_below_all_ok"] = all(below.values())
        # ⚠ This span is over the 12 backbones **including ProteinMPNN** (that is what the
        #   supplementary sentence says). Filtering on in_11pLM_claim drops MPNN and gives the
        #   11-backbone span.
        allb = B[~B.backbone.astype(str).str.startswith("SPAN")]
        with_odd = {q: float(allb[q].max() - allb[q].min()) for q in QS}
        F["supp"]["span_with_odd"] = with_odd
        F["supp"]["span_with_odd_monotone"] = bool(
            all(with_odd[a] <= with_odd[b] for a, b in zip(QS, QS[1:])))

    cpath = FIG / f"figS_breakdown_{nt}_cutoffs.csv"
    if cpath.exists():
        C2 = pd.read_csv(cpath).set_index("cutoff")
        for c in (0.2, 0.3, 0.5):
            if c in C2.index:
                F["supp"][f"cut_{c}"] = float(C2.loc[c, "mean_pct_at_or_below"])
    # ⚠ The text says "from ... to ... among eleven models". min_pct/max_pct in cutoffs.csv are
    #   over **12 backbones** and so include ProteinMPNN. They are recounted over 11 here.
        PD = H / "extend_backbones_perdms.csv"
        if PD.exists():
            E = pd.read_csv(PD)
            E = E[E.bb.isin(SHOW12)].copy()
            bs = E.groupby(["bb", "dms"]).base.first().reset_index()
            bs = bs[bs.bb != "ProteinMPNN"]
            for c in (0.2, 0.5):
                fr = bs.groupby("bb").base.apply(lambda s: float((s <= c).mean() * 100))
                F["supp"][f"cut_{c}_lo"] = float(fr.min())
                F["supp"][f"cut_{c}_hi"] = float(fr.max())
                F["supp"][f"cut_{c}_fold"] = float(fr.max() / fr.min())

    # ⚠ The text also cites backbone **ranks** (ESM-1v 6th and so on). Those are places, not numbers, so they are counted separately.
    if dpath.exists():
        _c = B[B.in_11pLM_claim & ~B.backbone.astype(str).str.startswith("SPAN")]
        F["supp"]["rank_p50"] = {str(r.backbone): int(v) for r, v
                                 in zip(_c.itertuples(), _c.p50.rank(ascending=False))}
        F["supp"]["rank_p10"] = {str(r.backbone): int(v) for r, v
                                 in zip(_c.itertuples(), _c.p10.rank(ascending=False))}
    # -- supplementary: case study (B2L11 Gly156), recomputed with the fitted weights
    try:
        from plmfusion import common as C
        from scipy.stats import spearmanr as _sp
        DMS = "B2L11_HUMAN_Dutta_2010_binding-Mcl-1"
        Wc = pd.read_csv(H / "ablation_12bb_w.csv")
        rec = C.Rec(C.CACHE / f"{DMS}.npz", "ESM2")
        z = np.load(C.CACHE / f"{DMS}.npz", allow_pickle=True)
        pos = np.asarray(z["pos1"], int); up = sorted(set(pos.tolist()))
        gp = {p_: rec.CH[:, 0][pos == p_][0] for p_ in up}
        dp = {p_: rec.CH[:, 2][pos == p_][0] for p_ in up}
        ym = np.array([rec.y[pos == p_].mean() for p_ in up])
        cs = dict(gemme_pos=float(gp[156]), ddg_pos=float(dp[156]), n_pos=len(up))
        for arm in ("FW5", "SFW7"):
            r = Wc[(Wc.axis == "function") & (Wc.bb == "ESM2") & (Wc.arm == arm)].iloc[0]
            con = {p_: r.a_GEMME * gp[p_] + r.a_ddG * dp[p_] for p_ in up}
            order = sorted(up, key=lambda x: con[x])
            s = rec.s(np.array([r[x] for x in SL], float))
            sm = np.array([s[pos == p_].mean() for p_ in up])
            cs[arm] = dict(a_GEMME=float(r.a_GEMME), a_ddG=float(r.a_ddG),
                           b_ddG=float(r.b_ddG),
                           c_gemme=float(r.a_GEMME * gp[156]),
                           c_ddg=float(r.a_ddG * dp[156]),
                           rank=order.index(156) + 1,
                           rho_pos=float(_sp([con[p_] for p_ in up], ym).statistic),
                           rho_full=float(_sp(sm, ym).statistic))
        A2 = pd.read_csv(H / "ablation_12bb_perdms.csv")
        d2 = A2[(A2.axis == "function") & (A2.bb == "ESM2")]
        fr = d2[d2.arm == "SFW7"].set_index("dms").rho
        w2 = d2[d2.arm == "FW5"].set_index("dms").rho.reindex(fr.index)
        g2 = (fr - w2).sort_values(ascending=False)
        cs["split_gain"] = float(g2.loc[DMS]); cs["split_rank"] = int(list(g2.index).index(DMS) + 1)
        cs["split_corpus_mean"] = float(g2.mean()); cs["n_dms"] = int(len(g2))
        cs["bb4"] = {}
        for bb in ("ESM2", "ESM1v", "ESM3", "ProSST"):
            tt = A2[(A2.axis == "function") & (A2.bb == bb) & (A2.dms == DMS)]
            pl = float(tt[tt.arm == "pLM"].rho.iloc[0]); sf = float(tt[tt.arm == "SFW7"].rho.iloc[0])
            cs["bb4"][bb] = dict(plm=pl, sfw7=sf, d=sf - pl, rel=(sf - pl) / pl)
        F["case"] = cs
    except Exception as exc:                       # the text still works without the case study
        F["case"] = dict(error=str(exc))

    # -- supplementary app:plm_need: channel-only performance and the refitted baseline
    try:
        SO = pd.read_csv(H / "plm_vs_channel_solo.csv")
        SO = SO[SO.axis == "function"]
        solo = {str(c): float(g.rho.mean()) for c, g in SO.groupby("channel")}
        PN2 = pd.read_csv(H / "plm_need_perdms.csv")
        gb = PN2.groupby("bb").agg(plm=("plm", "mean"), chan=("chan", "mean"),
                                   chan_refit=("chan_refit", "mean"))
        pr = PN2.groupby("dms").d_refit.mean()
        per = PN2.groupby("bb").d_refit.agg(["mean", lambda s: (s > 0).sum()])
        pv = {bb: float(wilcoxon(g.fused, g.chan_refit).pvalue)
              for bb, g in PN2.groupby("bb")}
        sig = {bb: (per.loc[bb, "mean"], pv[bb]) for bb in per.index}
        F["plm_need"] = dict(
            solo=solo,
            gemme=solo.get("GEMME"),
            n_below_gemme=int((gb.plm < solo.get("GEMME", 9)).sum()),
            chan=float(gb.chan.mean()), chan_refit=float(gb.chan_refit.mean()),
            refit_lo=float(gb.chan_refit.min()), refit_hi=float(gb.chan_refit.max()),
            d_mean=float(pr.mean()), d_win=int((pr > 0).sum()), d_n=int(len(pr)),
            d_p=float(wilcoxon(pr, np.zeros(len(pr))).pvalue),
            n_sig_pos=int(sum(1 for bb, (m, q) in sig.items() if m > 0 and q < 0.05)),
            best_bb=max(sig, key=lambda b: sig[b][0]),
            best_d=float(max(v[0] for v in sig.values())),
            worst_pos_bb=min((b for b in sig if sig[b][0] > 0 and pv[b] < 0.05),
                             key=lambda b: sig[b][0], default=None),
            esmif1_d=float(sig["ESMIF1"][0]), esmif1_p=float(pv["ESMIF1"]),
            mpnn_d=float(sig["ProteinMPNN"][0]), mpnn_p=float(pv["ProteinMPNN"]))
        F["plm_need"]["worst_pos_d"] = (float(sig[F["plm_need"]["worst_pos_bb"]][0])
                                        if F["plm_need"]["worst_pos_bb"] else None)
        # ⚠ The text names ESM-IF1 and ProteinMPNN as the two that did **not** improve
        #   significantly. ESM-IF1 became significant once the conventions were unified, so the
        #   sentence is built by counting rather than hard-coded.
        F["plm_need"]["not_sig_pos"] = sorted(
            b for b in sig if not (sig[b][0] > 0 and pv[b] < 0.05))
        _d12 = {b: float(sig[b][0]) for b in sig}
        _ten = {b: v for b, v in _d12.items() if b not in IF2}
        F["plm_need"]["ten_mean"] = float(np.mean(list(_ten.values())))
        F["plm_need"]["ten_min"] = float(min(_ten.values()))
        F["plm_need"]["ten_min_bb"] = min(_ten, key=_ten.get)
        # ⚠ The x axis of panel d is the **mean** rho of the backbone alone. Using the median gives
        #   0.89 and makes the figure and the text disagree.
        _r = spearmanr([gb.plm[b] for b in _d12], list(_d12.values()))
        F["plm_need"]["rho_acc"] = float(_r.statistic)
        F["plm_need"]["rho_acc_p"] = float(_r.pvalue)
    except Exception as exc:
        F["plm_need"] = dict(error=str(exc))

    # -- Qualitative claim checks. These are the sentences a number substitution cannot fix.
    #    If any is False, that sentence has to be rewritten by hand.
    cl = {}
    cl["main164_all12_up_function"] = F["function"]["n_up"] == 12
    cl["main164_all12_up_expression"] = F["expression"]["n_up"] == 12
    cl["main164_p_lt_1e6_function"] = F["function"]["max_p"] < 1e-6
    cl["main164_p_lt_001_expression"] = F["expression"]["max_p"] < 0.01
    cl["main164_strongest_is_AIDO"] = F["function"]["strongest"] == "AIDO-RAG 16B"
    cl["main164_lo_is_strongest"] = F["function"]["lo_bb"] == F["function"]["strongest"]
    # "started from the lowest baselines and gained the most": true on the function axis only.
    _bm, _ra = F["function"]["base_med"], F["function"]["rel_all"]
    cl["main164_if2_lowest_base_function"] = set(sorted(_bm, key=_bm.get)[:2]) == set(IF2)
    cl["main164_if2_top_gain_function"] = set(sorted(_ra, key=_ra.get)[-2:]) == set(IF2)
    sw_f, sw_e = F["cross"]["function_swapped"], F["cross"]["expression_swapped"]
    cl["main187_swapped_all24_sig"] = (sw_f["n_sig"] == sw_f["n"] == 12
                                       and sw_e["n_sig"] == sw_e["n"] == 12)
    pl_f = F["cross"]["function_pooled"]
    cl["main189_pooled_11of12"] = pl_f["n_sig"] == 11
    cl["main189_one_exception"] = len(pl_f["exceptions"]) == 1
    s = F["sbi"]
    cl["main242_all8_negative_after"] = s["ProFusion"]["n_neg_aware"] == 8
    cl["main242_mpnn_positive_before"] = s["pLM"]["mpnn"] > 0
    for lab in ("function", "expression"):
        cl[f"main254_all5_positive_{lab}"] = all(v > 0 for v in F["shapley"][lab].values())
        pos12 = F["shapley"].get(lab + "_pos_all12", {})
        top2 = sorted(F["shapley"][lab].items(), key=lambda x: -x[1])[:2]
        cl[f"main254_top2_all12_{lab}"] = all(pos12.get(k, 0) == 1 for k, _ in top2)
    cl["main257_equal5_positive_both"] = (F["ablation"].get("function_equal5", {}).get("lo", -1) > 0
                                          and F["ablation"].get("expression_equal5", {}).get("lo", -1) > 0)
    mb = F["ablation"].get("function_merge both", {})
    cl["main259_merge_both_all_positive"] = mb.get("lo", -1) > 0
    cl["supp192_B2L11_is_largest"] = F["case"].get("split_rank") == 1
    F["claims"] = cl
    # ⚠ The two below are **not blocking**. They are what an older sentence claimed, and that
    #   sentence is now written from the data, so they are no longer claims of the paper. The values
    #   are still reported, because if rescoring flips them again the new sentence needs attention.
    F["claims_info"] = dict(
        supp147_mpnn_below_all=F["supp"].get("odd_below_all_ok"),
        supp148_span_monotone=F["supp"].get("span_with_odd_monotone"))

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(F, open(OUT / "final_numbers.json", "w"), indent=2, ensure_ascii=False)

    L = ["# Final paper numbers (129 function / 18 expression, after unifying the MPNN and IF1 conventions)", ""]
    for lab in ("function", "expression"):
        g = F[lab]
        L += [f"## {lab}", f"- assays {g['n']}, proteins {g['prot']}, {g['types']}",
              f"- median relative gain (12 backbones) **{g['rel12']*100:.1f}%**  (the number used by the abstract and contributions)",
              f"- median relative gain (10 backbones) **{g['rel10']*100:.1f}%**  range {g['r10_min']*100:.1f}% to {g['r10_max']*100:.1f}%",
              f"- lowest {g['lo_bb']} {g['lo']*100:.1f}% ({g['lo_base']:.4f} to {g['lo_fit']:.4f})",
              f"- highest {g['hi_bb']} {g['hi']*100:.1f}% ({g['hi_base']:.4f} to {g['hi_fit']:.4f})",
              f"- ESM-IF1 {g['esmif1']*100:.1f}%, ProteinMPNN {g['mpnn']*100:.1f}% "
              f"({g['mpnn_base']:.4f} to {g['mpnn_fit']:.4f})",
              f"- worst p {g['max_p']:.2e}, improved {g['n_up']}/12, strongest zero-shot {g['strongest']} {g['strongest_base']:.4f}",
              f"- GEMME share {g['share_GEMME']:.1f}%, ddG {g['share_ddG']:.1f}%, RSA {g['share']['a_RSA']:.1f}%",
              f"- ddG groups: structure-aware {g['ddg_group']['aware_med']:.1f}% ({g['ddg_group']['aware_lo']:.1f} to {g['ddg_group']['aware_hi']:.1f})"
              f" vs unaware {g['ddg_group']['un_med']:.1f}% ({g['ddg_group']['un_lo']:.1f} to {g['ddg_group']['un_hi']:.1f})"
              f" p={g['ddg_group']['p']:.4f}", ""]
    L += ["## rho thresholds (median count over 12 backbones)"]
    for t, v in F["thr"].items():
        L.append(f"- rho<{t}: {v['base']:.1f} to {v['fus']:.1f} ({v['red']:.0%})")
    L += ["", "## Correlations", f"- rho(ddG share, acc) {F['corr']['share_acc']:+.3f} (p={F['corr']['share_acc_p']:.2f})",
          f"- rho(structure-aware, acc) {F['corr']['struct_acc']:+.3f} (p={F['corr']['struct_acc_p']:.2f})", ""]
    L += ["## Axis cross-application"]
    for k, v in F["cross"].items():
        L.append(f"- {k}: {v['lo']:.4f} to {v['hi']:.4f}, significant {v['n_sig']}/{v['n']}, "
                 f"max q {v['q_max']:.1e}, exceptions {v['exceptions']}")
    L += ["", "## Shapley (percent of total gain, median over backbones)"]
    for lab in ("function", "expression"):
        L.append(f"- {lab}: " + ", ".join(f"{k} {v:.1f}" for k, v in
                 sorted(F["shapley"][lab].items(), key=lambda x: -x[1])))
    L += ["", "## Ablation (median over backbones of the mean d_rho)"]
    for k, v in F["ablation"].items():
        L.append(f"- {k}: {v['med']:+.4f}  ({v['lo']:+.4f} to {v['hi']:+.4f})")
    pn = F.get("plm_need", {})
    if "solo" in pn:
        L += ["", "## Supplementary app:plm_need",
              f"- channels alone: " + ", ".join(f"{k} {v:.4f}" for k, v in
                                          sorted(pn["solo"].items(), key=lambda x: -x[1])),
              f"- backbones below GEMME alone ({pn['gemme']:.4f}): {pn['n_below_gemme']}/12",
              f"- channel sum {pn['chan']:.4f} -> refitted {pn['chan_refit']:.4f} "
              f"(range {pn['refit_lo']:.4f} to {pn['refit_hi']:.4f})",
              f"- pLM contribution {pn['d_mean']:+.4f}, wins {pn['d_win']}/{pn['d_n']}, p={pn['d_p']:.1e}",
              f"- significant positives {pn['n_sig_pos']}/12, largest {pn['best_bb']} {pn['best_d']:+.4f}"
              + (f", smallest significant {pn['worst_pos_bb']} {pn['worst_pos_d']:+.4f}" if pn['worst_pos_bb'] else ""),
              f"- ESM-IF1 {pn['esmif1_d']:+.4f} (p={pn['esmif1_p']:.2g}), "
              f"ProteinMPNN {pn['mpnn_d']:+.4f} (p={pn['mpnn_p']:.2g})"]
    L += ["", "## Qualitative claim checks (False = the sentence must be rewritten by hand)"]
    for k, v in F["claims"].items():
        L.append(f"- {'OK  ' if v else '**X**'} {k}")
    bad_claims = [k for k, v in F["claims"].items() if not v]
    if bad_claims:
        L.append(f"\n**{len(bad_claims)} broken claims: the sentences need rewriting**")
    sp = F["supp"]
    if "odd_below_all_ok" in sp:
        L += ["", "## ⚠ Supplementary qualitative claim checks",
              f"- ProteinMPNN lowest at every percentile: **{sp['odd_below_all_ok']}** "
              f"({ {k: v for k, v in sp['odd_below_all'].items()} })",
              f"- span including MPNN increases monotonically p10 to p90: **{sp['span_with_odd_monotone']}** "
              f"({ {k: round(v,3) for k, v in sp['span_with_odd'].items()} })",
              "  If either is False, the corresponding supplementary sentence **must be rewritten by hand**."]
    s = F["sbi"]
    L += ["", "## SBI (11 proteins)",
          f"- proteins {s['n_prot']}, shared variants {s['n_var']:,}, positions {s['n_pos']:,}, SBI {s['n_sbi']}",
          f"- before fusion, structure-aware {s['pLM']['aware']:+.4f} vs unaware {s['pLM']['un']:+.4f}, "
          f"gap {s['pLM']['gap']:.4f}, p={s['pLM']['p']:.4f}, positive {s['pLM']['n_pos_aware']}/8",
          f"- after fusion, gap {s['ProFusion']['gap']:.4f}, p={s['ProFusion']['p']:.4f}, negative {s['ProFusion']['n_neg_aware']}/8",
          f"- ProteinMPNN {s['pLM']['mpnn']:+.4f} -> {s['ProFusion']['mpnn']:+.4f}",
          f"- after fusion, WT-like {s['wtlike']:+.4f}, total-loss {s['totalloss']:+.4f}"]
    (OUT / "FINAL.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {OUT/'final_numbers.json'} and {OUT/'FINAL.md'}")


if __name__ == "__main__":
    main()
