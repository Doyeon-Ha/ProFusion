#!/usr/bin/env bash
# Step 3b: a3m to a2m conversion and the GEMME work directories (mutations.txt).
# ⚠ This predates the regular expression fixed on 2026-10-07 and therefore did not filter out
# synonymous substitutions.
# This step never existed as a file in the repository. The commands were recovered verbatim from
# the working record (run on 2026-10-02). The paths are those of the original repository; this was
# not adapted to run from this package.

cd "<REPO>"; mkdir -p data/mavedb_external/gemme/work; python3 - <<'EOF'
"""a3m to a2m (match-state) conversion, plus the GEMME work directory layout."""
import json, re, os
from pathlib import Path
import pandas as pd
R=Path('"<REPO>"'); A3M=R/'data/mavedb_external/msa/a3m'; W=R/'data/mavedb_external/gemme/work'
man=json.load(open(R/'data/mavedb_external/seq_manifest.json'))
F=pd.read_csv(R/'data/mavedb_external/frozen.csv')
import hashlib
F['tag']=[hashlib.md5(s.encode()).hexdigest()[:10] for s in F.pseq]
MIS=re.compile(r'^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$')
T3={'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E','Gly':'G','His':'H','Ile':'I',
    'Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P','Ser':'S','Thr':'T','Trp':'W','Tyr':'Y','Val':'V'}
MAVE='"<MAVEDB_DIR>/csv"'
def read_a3m(p):
    out=[];h=None;buf=[]
    for ln in open(p):
        if ln.startswith('>'):
            if h is not None: out.append((h,''.join(buf)))
            h=ln[1:].strip(); buf=[]
        elif not ln.startswith('#'): buf.append(ln.rstrip())
    if h is not None: out.append((h,''.join(buf)))
    return out
ok=0
for tag,info in man.items():
    f=A3M/f'{tag}.a3m'
    if not f.exists(): print('NO a3m',tag); continue
    rows=read_a3m(f)
    ms=[(h,''.join(c for c in s if c!='.' and not c.islower())) for h,s in rows]
    L=len(ms[0][1])
    ms=[(h,s) for h,s in ms if len(s)==L]
    d=W/tag; d.mkdir(parents=True,exist_ok=True)
    with open(d/f'ali{tag}.fasta','w') as o:
        for i,(h,s) in enumerate(ms): o.write(f">{'query' if i==0 else h.split()[0]}\n{s}\n")
    # The union of the variants of every DMS that uses this sequence
    muts=set()
    for u in F[F.tag==tag].urn:
        t=pd.read_csv(f"{MAVE}/{u.replace(':','-')}.scores.csv",usecols=lambda c:c in('hgvs_pro','score'),low_memory=False)
        s=pd.to_numeric(t.score,errors='coerce')
        for h,sc in zip(t.hgvs_pro.astype(str),s):
            m=MIS.match(h)
            if not m or pd.isna(sc): continue
            w,pos,mu=T3.get(m[1]),int(m[2]),T3.get(m[3])
            if w and mu and 1<=pos<=L and ms[0][1][pos-1]==w: muts.add(f"{w}{pos}{mu}")
    (d/'mutations.txt').write_text('\n'.join(sorted(muts,key=lambda x:(int(x[1:-1]),x[-1])))+'\n')
    json.dump(dict(tag=tag,tname=info['tname'],L=L,depth=len(ms),n_mut=len(muts)),open(d/'meta.json','w'))
    ok+=1
    print(f"{info['tname'][:30]:32s} tag={tag} L={L} depth={len(ms):6d} muts={len(muts)}")
print('built',ok)
EOF
