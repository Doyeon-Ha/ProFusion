#!/usr/bin/env python
"""Generate the paper numbers. **All weights are fitted on the MaveDB external set only.**

Principles
  fitting (w) = MaveDB external set. Anything identical or homologous to a ProteinGym sequence is
                excluded with MMseqs2 (pg_tier=='none'), as are Domainome and Tsuboyama (the SPURS
                training set). The rule is frozen by sha256.
  evaluation  = ProteinGym. Labels are used **for evaluation only**. There is no fitting on the
                ProteinGym side at all: fit once, evaluate once. No path in this script runs
                coordinate descent on a ProteinGym record, and an assert below pins that down.
  axes        = function and expression, fitted separately. The first and second most useful
                channels swap between axes, so the two are never pooled.
                ⚠ The npz stores the expression axis under its old name "stability". Only the
                display name is expression.
  grid        = uni3. All 7 slots from −3.0 to 3.0 in steps of 0.1, coordinate descent, 3 passes,
                starting from zero.
  backbones   = 12, restricted to those with an llr on both sides (EXTBB).

What it produces
  F1 fusion ladder   pLM alone -> champ3 (pooled) -> champ5 (+substitution channels)
                     -> comp7 (position/residual split)
  F3 channel ablation  comp7 with one channel removed, or with one channel only
  F4 implementation identity  the substitution axis of "only RSA" == the pLM substitution axis;
                     the position axis of "only substitution" == the pLM position axis
                     ⚠ Do not expect exactly 0. The amount a residual channel adds to a position
                     mean is mathematically 0 but about 1e-16 in floating point, and when the
                     baseline has positions whose means are **exactly equal**, that residue breaks
                     the tie and changes the ranking. Measured upper bound 4e-05 (function, ESM2,
                     two tied positions in GDIA_HUMAN). Suspect the decomposition above 1e-4.
  metrics            global rho, position-axis rho, substitution-axis rho, P@10, EF@1%,
                     AUROC (bottom 20% counted as damaging)
output: paper_perdms.csv (per DMS), paper_summary.csv, and the printed tables
run: OMP_NUM_THREADS=6 python scripts/paper_numbers.py   (output goes to results/)
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import roc_auc_score
from plmfusion import common as C
from plmfusion.common import descend_tied
from plmfusion.external import load_ext
from plmfusion.grids import use_grid
from plmfusion.paths import REPO, RESULTS

# ProteinGym npz key <-> external npz key. Only backbones with an llr on both sides are run.
# ⚠ Two names disagree: ProteinGym 'ESM3' (multimodal) <-> external 'ESM3mm', and ProteinGym
#   'SaProt650M' <-> external 'SaProt650M_own'. They are not interchangeable.
EXTBB = {"ESM2": "ESM2", "ESM1v": "ESM1v", "ESM3": "ESM3mm", "ProSST": "ProSST",
         "ESMIF1": "ESMIF1", "ProteinMPNN": "ProteinMPNN",
         "TranceptionLret": "TranceptionLret", "S2F": "S2F", "VespaG": "VespaG",
         "VenusREM": "VenusREM", "SaProt650M": "SaProt650M_own",
         "AIDO_RAG16B": "AIDO_RAG16B"}
BBS = list(EXTBB)
# The axis name inside the npz and the name used for display. The npz stores the expression axis
# under its old name "stability". The rename happens only here, so that the data need not be
# rebuilt (see DATA.md).
AXES = ["function", "stability"]
SHOW = {"function": "function", "stability": "expression"}
GRID = "uni3"
MINPOS = 5
SL = ["a_GEMME", "b_GEMME", "a_ddG", "b_ddG", "a_RSA", "b_BLOSUM", "b_dchg"]
# Arms: name -> how the 7 slots are tied into free parameters (same digit = same value)
TIES = {
    "champ3 (3 pooled channels)":  [0, 0, 1, 1, 2, None, None],
    "champ5 (+2 substitution channels)":     [0, 0, 1, 1, 2, 3, 4],
    "comp7 (position/residual split)":    [0, 1, 2, 3, 4, 5, 6],
}
ABL = {"full": [0, 1, 2, 3, 4, 5, 6], "−GEMME": [2, 3, 4, 5, 6], "−ddG": [0, 1, 4, 5, 6],
       "−RSA": [0, 1, 2, 3, 5, 6], "−BLOSUM": [0, 1, 2, 3, 4, 6], "−dchg": [0, 1, 2, 3, 4, 5],
       "only GEMME": [0, 1], "only ddG": [2, 3], "only RSA": [4], "only substitution": [5, 6]}


def metrics(r, s, pre):
    pid, cnt, npos, big, uq = pre
    y = r.y.astype(float)
    rk = rankdata(s); rk = rk - rk.mean()
    rho = float((rk @ r.ry) / (np.sqrt((rk*rk).sum()) + 1e-12))
    pm = np.bincount(pid, s, minlength=npos) / np.maximum(cnt, 1)
    ym = np.bincount(pid, y, minlength=npos) / np.maximum(cnt, 1)
    rp = float(np.corrcoef(rankdata(pm[uq]), rankdata(ym[uq]))[0, 1]) if len(uq) > 5 else np.nan
    o = []
    for p in big:
        m = pid == p
        if m.sum() >= MINPOS and np.std(s[m]) > 0 and np.std(y[m]) > 0:
            a, b = rankdata(s[m]), rankdata(y[m]); a, b = a - a.mean(), b - b.mean()
            o.append((a @ b) / (np.sqrt((a*a).sum() * (b*b).sum()) + 1e-12))
    rw = float(np.mean(o)) if o else np.nan
    dp, dt = -s, -y; order = np.argsort(-dp)
    p10 = len(set(np.argsort(-dt)[:10].tolist()) & set(order[:10].tolist())) / 10
    lab = (dt >= np.percentile(dt, 80)).astype(int)
    k1 = max(3, int(round(0.01 * len(y))))
    ef = lab[order[:k1]].mean() / lab.mean() if lab.mean() > 0 else np.nan
    au = roc_auc_score(lab, dp) if 0 < lab.sum() < len(lab) else np.nan
    return dict(rho=rho, rho_pos=rp, rho_sub=rw, p10=p10, ef1=ef, auroc=au)


def compute(bbs, axes):
    """Run the (axis, bb) cells and build the per-DMS rows and the weight rows.

    **Cells are completely independent.** Both the external fit and the ProteinGym evaluation see
    only that cell's backbone and axis. Coordinate descent is deterministic, so splitting the work
    into cells and concatenating afterwards gives the same result
    (scripts/run_paper_numbers.sh does exactly that).
    """
    t0 = time.time(); rows = []; wrows = []
    for axis in axes:
        for bb in bbs:
            E = load_ext(EXTBB[bb], axis)
            R = C.load(bb, axis=axis)
            if len(E) < 8:
                print(f"[{SHOW[axis]}/{bb}] external {len(E)} < 8, skipped", flush=True)
                continue
            PRE = []
            for r in R:
                z = np.load(C.CACHE / f"{r.dms}.npz", allow_pickle=True)
                pid = z["pidx"].astype(np.int64); npos = int(pid.max()) + 1
                cnt = np.bincount(pid, minlength=npos)
                PRE.append((pid, cnt, npos, np.flatnonzero(cnt >= MINPOS), np.flatnonzero(cnt > 0)))
            print(f"\n[{SHOW[axis]}/{bb}] fit on {len(E)} external -> evaluate on {len(R)} PG ({time.time()-t0:.0f}s)", flush=True)
            # Weights are fitted on the external set E only. The PG records R enter evaluation only.
            with use_grid(GRID):
                # F1 ladder
                WS = {"pLM alone": np.zeros(7)}
                for nm, tie in TIES.items():
                    WS[nm] = descend_tied(E, list(range(len(E))), tie)
                # F3 ablation (relative to comp7)
                for nm, slots in ABL.items():
                    if nm == "full": continue
                    tie = [i if i in slots else None for i in range(7)]
                    WS[f"abl {nm}"] = descend_tied(E, list(range(len(E))), tie)
            for nm, w in WS.items():
                wrows.append(dict(axis=axis, bb=bb, arm=nm, **dict(zip(SL, np.round(w, 2)))))
                for i, r in enumerate(R):
                    rows.append(dict(axis=axis, bb=bb, arm=nm, dms=r.dms, **metrics(r, r.s(w), PRE[i])))
    return pd.DataFrame(rows), pd.DataFrame(wrows)


def write(D, W, results):
    results = Path(results); results.mkdir(parents=True, exist_ok=True)
    D.to_csv(results / "paper_perdms.csv", index=False)
    W.to_csv(results / "paper_weights.csv", index=False)
    MET = ["rho", "rho_pos", "rho_sub", "p10", "ef1", "auroc"]
    g = D.groupby(["axis", "bb", "arm"])[MET]
    # ⚠ mean() skips NaN, and the denominator can differ per metric, so the count is printed
    #   alongside. rho_sub is undefined on a DMS whose within-position label SD is 0
    #   (CCDB_ECOLI_Adkar_2012 is like that at all 100 positions). Without the denominator there
    #   is no way to see that one table has rho over 129 assays but rho_sub over only 128.
    S = g.mean().reset_index()
    N = g.count().reset_index().rename(columns={m: "n_" + m for m in MET})
    S = S.merge(N, on=["axis", "bb", "arm"])
    S.to_csv(results / "paper_summary.csv", index=False)
    return S


def report(D, S):
    have_ax = [a for a in AXES if a in set(D.axis)]
    have_bb = [b for b in BBS if b in set(D.bb)]
    for axis in have_ax:
        print(f"\n{'='*104}\n### {SHOW[axis]}")
        for bb in have_bb:
            s = S[(S.axis == axis) & (S.bb == bb)]
            if not len(s): continue
            d = D[(D.axis == axis) & (D.bb == bb)]
            base = d[d.arm == "comp7 (position/residual split)"].set_index("dms")["rho"]
            print(f"\n[{bb}]  {'arm':24s} {'rho':>8s} {'rho_pos':>8s} {'rho_sub':>8s} {'P@10':>7s} "
                  f"{'EF@1%':>7s} {'AUROC':>7s} {'vs comp7':>9s} {'p':>8s}")
            for a in ["pLM alone"] + list(TIES) + [f"abl {k}" for k in ABL if k != "full"]:
                r = s[s.arm == a]
                if not len(r): continue
                v = d[d.arm == a].set_index("dms")["rho"]
                dd = (v - base.reindex(v.index)).dropna()
                pv = wilcoxon(dd)[1] if dd.std() > 1e-12 else 1.0
                r = r.iloc[0]
                print(f"      {a:24s} {r.rho:8.4f} {r.rho_pos:8.4f} {r.rho_sub:8.4f} {r.p10:7.4f} "
                      f"{r.ef1:7.3f} {r.auroc:7.4f} {dd.mean():+9.5f} {pv:8.1e}")
    # F4 identities
    print(f"\n{'='*104}\n### Implementation identities (these must hold)")
    for axis in have_ax:
        for bb in have_bb:
            d = D[(D.axis == axis) & (D.bb == bb)]
            if not len(d): continue
            g = lambda a, m: d[d.arm == a].set_index("dms")[m]
            e1 = np.abs(g("abl only RSA", "rho_sub") - g("pLM alone", "rho_sub")).max()
            e2 = np.abs(g("abl only substitution", "rho_pos") - g("pLM alone", "rho_pos")).max()
            print(f"  [{SHOW[axis]}/{bb:16s}] only RSA rho_sub==pLM: {e1:.2e}   only substitution rho_pos==pLM: {e2:.2e}")


def merge(celldir, results):
    """Concatenate results run cell by cell. **A missing cell is never passed over silently.**"""
    cells = sorted(Path(celldir).glob("*/paper_perdms.csv"))
    if not cells:
        raise SystemExit(f"no cell results under {celldir}")
    D = pd.concat([pd.read_csv(f) for f in cells], ignore_index=True)
    W = pd.concat([pd.read_csv(f.parent / "paper_weights.csv") for f in cells], ignore_index=True)
    got = {(a, b) for a, b in zip(D.axis, D.bb)}
    want = {(a, b) for a in AXES for b in BBS}
    if got != want:
        print(f"⚠ {len(got)}/{len(want)} cells. Missing: {sorted(want - got)}", flush=True)
    S = write(D, W, results)
    report(D, S)
    # Never write an absolute path outside the repo into the log (the frozen log is public).
    rel = results.relative_to(REPO) if results.is_relative_to(REPO) else results
    print(f"\nmerged {len(cells)} cells -> {rel}")


def main(bbs=None, axes=None, results=None):
    t0 = time.time()
    D, W = compute(list(bbs or BBS), list(axes or AXES))
    S = write(D, W, Path(results or RESULTS))
    report(D, S)
    print(f"\nwrote paper_perdms.csv, paper_summary.csv, paper_weights.csv ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Paper numbers. Run cell by cell with --bb/--axis, then combine with --merge.")
    ap.add_argument("--bb", help="one backbone only (default: all 12)")
    ap.add_argument("--axis", help="one axis only (default: function,stability)")
    ap.add_argument("--out", help="output directory (default: results/)")
    ap.add_argument("--merge", help="combine the cell results under this directory")
    a = ap.parse_args()
    if a.merge:
        merge(a.merge, Path(a.out or RESULTS))
    else:
        main([a.bb] if a.bb else None, [a.axis] if a.axis else None, a.out)
