"""Does the pLM earn its place over the channels alone? Paired tests over assays, stratified by
MSA depth.

output  plm_need_perdms.csv (per backbone and DMS)
run     OMP_NUM_THREADS=6 python scripts/plm_need.py
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))   # paper_numbers lives in this folder
from plmfusion import common as C, grids as P
from plmfusion.paths import RESULTS, data_root
import paper_numbers as PN
EXTBB = PN.EXTBB                       # the same 12-backbone mapping
from scipy.stats import rankdata, wilcoxon

BB=["AIDO_RAG16B","VenusREM","ProSST","VespaG","TranceptionLret","S2F","ESM1v",
    "SaProt650M","ESM3","ESM2","ESMIF1","ProteinMPNN"]
SL=C.SLOTS
W=pd.read_csv(RESULTS/"extend_backbones.csv").set_index(["axis","bb"])
ref=pd.read_csv(data_root()/"proteingym"/"DMS_substitutions.csv").set_index("DMS_id")

def load_bb(bb,axis):
    miss={f.stem for f in sorted(C.CACHE.glob("*.npz"))
          if f"llr_{bb}" not in np.load(f,allow_pickle=True).files}
    old=set(C.DROP); C.DROP|=miss
    try: return C.load(bb,axis)
    finally: C.DROP.clear(); C.DROP|=old

def rho(x,ry):
    r=rankdata(x); r=r-r.mean(); n=np.sqrt((r*r).sum())
    return float((r@ry)/n) if n else 0.0

def descend_chan(E):
    """**Refit** the 7 slots from the channels alone, with no pLM term.

    Why this is needed: the 7 slots of ProFusion were fitted **together with** the pLM term.
    Reusing them without the pLM puts the channels at a disadvantage and understates the
    baseline. The fair baseline is a refit on the same MaveDB assays with the pLM removed
    (0.496 -> 0.502 in the appendix).

    ⚠ The grid is the same uni3 used for the main fit. Otherwise only the baseline would come
      from a different grid.
    """
    w = np.zeros(7)
    for _ in range(C.NPASS):
        for k in range(7):
            best, bv = w[k], -9.0
            for g in C.GRIDS[k]:
                c = w.copy(); c[k] = g
                m = float(np.mean([rho(r.CH @ c, r.ry) for r in E]))
                if m > bv:
                    best, bv = g, m
            w[k] = best
    return w


rows=[]
for bb in BB:
    Rs=load_bb(bb,"function"); w=W.loc[("function",bb),SL].to_numpy(float)
    E=PN.load_ext(EXTBB[bb], "function")
    with P.use_grid("uni3"):
        wc=descend_chan(E)
    print(f"  {bb:16s} channel-only refit w = " + " ".join(f"{x:.1f}" for x in wc), flush=True)
    for r in Rs:
        ch=rho(r.CH@w, r.ry)                 # channels only, with the fusion weights as they are
        cr=rho(r.CH@wc, r.ry)                # channels only, with the refitted weights
        fu=r.rho(w)                          # fusion
        pl=r.rho(np.zeros(7))                # pLM alone
        rows.append(dict(bb=bb,dms=r.dms,chan=ch,chan_refit=cr,fused=fu,plm=pl,
                         d=fu-ch,d_refit=fu-cr,
                         neff=ref.MSA_Neff_L_category.get(r.dms)))
D=pd.DataFrame(rows)
D.to_csv(RESULTS/"plm_need_perdms.csv",index=False)

NDMS=D.dms.nunique()
print(f"=== fusion minus channels-only (paired over assays, function {NDMS}) ===")
for bb in BB:
    g=D[D.bb==bb]
    p=wilcoxon(g.fused,g.chan).pvalue
    print(f"  {bb:16s} d {g.d.mean():+.4f}  wins {int((g.d>0).sum()):3d}/{NDMS}  p={p:.2g}")
pool=D.groupby("dms").d.mean()
print(f"\n  pooled over 12 backbones: d {pool.mean():+.4f}  wins {int((pool>0).sum())}/{NDMS}  "
      f"p={wilcoxon(pool, np.zeros(len(pool))).pvalue:.2g}")

print(f"\n=== channel-only baseline ===")
print(f"  with the fusion weights : {D.groupby('bb').chan.mean().mean():.4f}")
print(f"  refitted without pLM    : {D.groupby('bb').chan_refit.mean().mean():.4f}")
print(f"  range over 12 backbones : {D.groupby('bb').chan_refit.mean().min():.4f} "
      f"to {D.groupby('bb').chan_refit.mean().max():.4f}")
pr=D.groupby("dms").d_refit.mean()
print(f"  pLM contribution over the refitted baseline: d {pr.mean():+.4f}  wins {int((pr>0).sum())}/{len(pr)}  "
      f"p={wilcoxon(pr, np.zeros(len(pr))).pvalue:.2g}")

print("\n=== d (fusion minus channels-only) by MSA depth ===")
t=D.groupby(["neff","bb"]).d.mean().unstack(0)
print(t.round(4).to_string())
print("\n  column means:"); print(D.groupby("neff").d.mean().round(4).to_string())
print("  assay counts:"); print(D[D.bb=="ESM2"].neff.value_counts().to_string())
