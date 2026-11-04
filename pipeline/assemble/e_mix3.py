#!/usr/bin/env python
"""Per-DMS channel bundle loader for ProteinGym (load). Used by build_features.py and
build_stability.py.

Only load() and the constants it needs are kept (main belonged to a separate experiment), and the
input paths now resolve through paths.data_root().

What load() returns, one dict per DMS
    dms, prot, axis, n, ry (label ranks, centred and unit-norm), pos0, mut
    S = {gemme, ddg, rsa}, variant-level z. ddg is z(-SPURS ddG)
    ESM1v, ESM2, ESM3, ProSST, the raw LLR per backbone
A DMS survives if all four backbone LLRs, the SPURS ddG, the RSA and GEMME are present, it has at
least 100 single substitutions, and at most 5% of its GEMME values are missing.
⚠ ESM1v comes from DETANGO output (collection.csv), which contains no protein over 1022 residues.
  Those DMS drop out for every backbone.
"""
from __future__ import annotations
import numpy as np, pandas as pd, torch
from scipy.stats import rankdata

from plmfusion import paths

PG=paths.data_root()/"proteingym"
RUN=PG/"detango"; DMSD=PG/"spurs/dms"
ZSD=PG/"zeroshot"; NP2=PG/"node_physics_v2"; GM=PG/"gemme"
AXIS={"OrganismalFitness":"function","Activity":"function","Binding":"function","Expression":"stability"}
AA3=dict(A="Ala",R="Arg",N="Asn",D="Asp",C="Cys",Q="Gln",E="Glu",G="Gly",H="His",I="Ile",
         L="Leu",K="Lys",M="Met",F="Phe",P="Pro",S="Ser",T="Thr",W="Trp",Y="Tyr",V="Val")
BB=["ESM1v","ESM2","ESM3","ProSST"]
hgv=lambda m:f"p.{AA3[m[0]]}{m[1:-1]}{AA3[m[-1]]}"
def z(x):
    x=np.asarray(x,float); s=x.std(); return (x-x.mean())/s if s else x*0

def load():
    ref=pd.read_csv(PG/"DMS_substitutions.csv"); ref["axis"]=ref.coarse_selection_type.map(AXIS)
    phys={}
    for d,u in zip(ref.DMS_id,ref.UniProt_ID):
        f=NP2/f"{d}.pt"
        if u not in phys and f.exists(): phys[u]=torch.load(f,weights_only=False)["feat"].numpy()
    have={d.name for d in (RUN/"data").iterdir() if d.is_dir()}
    D=[]
    for _,r in ref.dropna(subset=["axis"]).iterrows():
        if (r.UniProt_ID not in have or not (DMSD/f"{r.DMS_id}.csv").exists()
            or r.UniProt_ID not in phys or not (GM/f"{r.DMS_id}.pt").exists()): continue
        Y=pd.read_csv(DMSD/f"{r.DMS_id}.csv")
        Y=Y[(Y.n_sub==1)&Y.spurs_ddg.notna()&Y.DMS_score.notna()].copy()
        if len(Y)<100: continue
        zn=np.load(PG/f"spurs/matrix/{r.UniProt_ID}.npz",allow_pickle=True); lo=int(zn["pdb_range"][0])
        col=pd.read_csv(RUN/"data"/r.UniProt_ID/"collection.csv",index_col=0)
        col["pos0"]=[int(m[1:-1]) for m in col.mutant]
        col["mutant"]=[f"{m[0]}{q+lo}{m[-1]}" for m,q in zip(col.mutant,col.pos0)]
        M=Y.merge(col[["mutant","pos0","esm1v"]].rename(columns={"esm1v":"ESM1v"}),on="mutant")
        M["hg"]=[hgv(m) for m in M.mutant]; ok=True
        for b in BB[1:]:
            f=ZSD/b/f"{r.DMS_id}.pt"
            if not f.exists(): ok=False; break
            t=torch.load(f,weights_only=False)
            M[b]=pd.Series(dict(zip(t["hgvsp_keys"],t["scores"].numpy()))).reindex(M.hg).to_numpy()
        if not ok: continue
        M=M.dropna(subset=BB)
        if len(M)<100: continue
        P=phys[r.UniProt_ID]; pos=M.pos0.to_numpy()
        if pos.max()>=P.shape[0]: continue
        gm=torch.load(GM/f"{r.DMS_id}.pt",weights_only=False)
        gv=pd.Series(gm["gemme"].numpy(),index=gm["variant_ids"]).reindex(M.hg).to_numpy()
        if np.isnan(gv).mean()>0.05: continue
        S={"ddg":z(-M.spurs_ddg.to_numpy()),"rsa":z(P[pos,0]),
           "gemme":z(np.where(np.isnan(gv),np.nanmedian(gv),gv))}
        if any(np.isnan(v).any() or v.std()==0 for v in S.values()): continue
        y=M.DMS_score.to_numpy(); ry=rankdata(y); ry=ry-ry.mean(); ry=ry/np.sqrt((ry*ry).sum())
        D.append(dict(dms=r.DMS_id,prot=r.UniProt_ID,axis=r.axis,n=len(M),ry=ry,S=S,pos0=pos,
                      mut=M.mutant.to_numpy(),
                      **{b:M[b].to_numpy() for b in BB}))
    return D


