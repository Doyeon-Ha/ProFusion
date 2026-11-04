"""node_physics v2: 20-channel per-residue features (richer than v1's 8).
Output: data/proteingym/node_physics_v2/{dms_id}.pt = {"feat": tensor[N,20]}
 0-7  : rsa, ss_H, ss_E, ss_L, plddt/100, n_count/20, n_hydro/4.5, n_charge/5  (v1)
 8-11 : WT AA props  kd_hydro/4.5, charge, volume/200, flex
 12-17: WT AA-class onehot  aliphatic/aromatic/polar/positive/negative/special
 18   : rel_pos       19: disorder flag(plddt<50)
"""
import os, sys, warnings, glob
import numpy as np, torch, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "biophysical_probe"))
import build_features as BF
import common as C
PG="<PROTEINGYM_DIR>"; TASKS="<REPO>/data/proteingym/tasks_ens"
OUT="<REPO>/data/proteingym/node_physics_v2"; os.makedirs(OUT,exist_ok=True)
meta=pd.read_csv(f"{PG}/DMS_substitutions.csv").set_index("DMS_id")
GROUP={}
for g,m in {"aliphatic":"AVLIM","aromatic":"FWY","polar":"STNQ","positive":"KRH","negative":"DE","special":"GPC"}.items():
    for a in m: GROUP[a]=g
GORDER=["aliphatic","aromatic","polar","positive","negative","special"]
def feat20(N,st):
    aa,plddt,ss3,rsa,nc,nh,ncg=st; f=np.zeros((N,20),dtype=np.float32)
    for i in range(N):
        p=i+1
        rv=rsa.get(p,np.nan); f[i,0]=rv if np.isfinite(rv) else 0.0
        ss=ss3.get(p); f[i,1]=float(ss=="H"); f[i,2]=float(ss=="E"); f[i,3]=float(ss=="L")
        pl=plddt.get(p,0.0); f[i,4]=pl/100.0
        f[i,5]=min(nc.get(p,0.0)/20.0,2.0); f[i,6]=nh.get(p,0.0)/4.5; f[i,7]=ncg.get(p,0.0)/5.0
        a=aa.get(p)
        if a in C.AA:
            f[i,8]=C.KD_HYDRO[a]/4.5; f[i,9]=C.CHARGE[a]; f[i,10]=C.VOLUME[a]/200.0; f[i,11]=C.FLEX[a]
            f[i,12+GORDER.index(GROUP[a])]=1.0
        f[i,18]=p/max(N,1); f[i,19]=float(pl<50.0)
    return torch.from_numpy(f)
cache={}; files=sorted(glob.glob(f"{TASKS}/*.pt")); print(f"{len(files)} tasks",flush=True)
done=fail=0
for i,pt in enumerate(files):
    dms=os.path.basename(pt)[:-3]; outp=f"{OUT}/{dms}.pt"
    if os.path.exists(outp): done+=1; continue
    if dms not in meta.index: fail+=1; continue
    uni=meta.loc[dms,"UniProt_ID"]; pdb=f"{PG}/ProteinGym_AF2_structures/{uni}.pdb"
    if not os.path.exists(pdb): fail+=1; continue
    try:
        N=int(torch.load(pt,map_location="cpu",weights_only=False)["sequence_length"])
        if uni not in cache: cache[uni]=BF.structure_tables(pdb)
        torch.save({"feat":feat20(N,cache[uni])},outp); done+=1
    except Exception as e:
        print(f"  FAIL {dms}: {type(e).__name__}: {e}",flush=True); fail+=1
    if (i+1)%40==0: print(f"  {i+1}/{len(files)} done={done} fail={fail}",flush=True)
print(f"DONE built {done} fail {fail} -> {OUT}",flush=True)
