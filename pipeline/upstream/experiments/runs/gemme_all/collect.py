#!/usr/bin/env python3
"""Collect GEMME outputs into per-DMS CSVs (mutant, evolCombi, evolEpi, evolInd).

Maps query-coord labels back to original ProteinGym mutant ids via mapping.json.
"""
import os, json, glob, csv

ROOT = '<REPO>'
BASE = f'{ROOT}/experiments/runs/gemme_all'
RES = f'{BASE}/results'
os.makedirs(RES, exist_ok=True)

manifest = json.load(open(f'{BASE}/manifest.json'))

def parse_pred(path):
    d = {}
    with open(path) as f:
        f.readline()  # "x"
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            lab = p[0].strip('"')
            try:
                d[lab] = float(p[1])
            except ValueError:
                d[lab] = float('nan')
    return d

ok, missing, empty = [], [], []
total_rows = 0
for dms, info in manifest.items():
    wdir = info['work']
    combi = glob.glob(os.path.join(wdir, '*_normPred_evolCombi.txt'))
    if not combi:
        missing.append(dms)
        continue
    epi = glob.glob(os.path.join(wdir, '*_normPred_evolEpi.txt'))
    ind = glob.glob(os.path.join(wdir, '*_normPred_evolInd.txt'))
    pc = parse_pred(combi[0])
    pe = parse_pred(epi[0]) if epi else {}
    pi = parse_pred(ind[0]) if ind else {}

    mapping = json.load(open(os.path.join(wdir, 'mapping.json')))
    label2mut = mapping['label2mut']

    out = os.path.join(RES, dms + '.csv')
    n = 0
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mutant', 'gemme_evolCombi', 'gemme_evolEpi', 'gemme_evolInd'])
        for label, mut in label2mut.items():
            if label not in pc:
                continue
            w.writerow([mut, pc.get(label), pe.get(label, ''), pi.get(label, '')])
            n += 1
    if n == 0:
        empty.append(dms)
    else:
        ok.append(dms)
        total_rows += n

print(f"collected: {len(ok)} DMS  | missing output: {len(missing)} | empty: {len(empty)}")
print(f"total mutant rows saved: {total_rows}")
if missing:
    print("MISSING:", missing)
if empty:
    print("EMPTY:", empty)
print(f"\nresults -> {RES}/<DMS_id>.csv")
