#!/usr/bin/env bash
# Step 2b: the query sequence FASTA and seq_manifest.json (tag = md5(pseq)[:10]).
# This step never existed as a file in the repository. The commands were recovered verbatim from
# the working record (run on 2026-10-02). The paths are those of the original repository; this was
# not adapted to run from this package.

cd "<REPO>"; mkdir -p data/mavedb_external/msa/{query,a3m,a2m,logs}; python3 - <<'EOF'
import pandas as pd, hashlib, json
sel=pd.read_pickle('data/mavedb_external/frozen.pkl')
u=sel.drop_duplicates('pseq')[['tname','pseq','urn']]
m={}
for r in u.itertuples():
    tag=hashlib.md5(r.pseq.encode()).hexdigest()[:10]
    m[tag]=dict(tname=r.tname, L=len(r.pseq), example_urn=r.urn)
    open(f'data/mavedb_external/msa/query/{tag}.fasta','w').write(f">{tag}\n{r.pseq}\n")
json.dump(m,open('data/mavedb_external/seq_manifest.json','w'),indent=1,ensure_ascii=False)
print(len(m),"sequences")
print(pd.DataFrame(m).T.sort_values('L').to_string())
EOF
