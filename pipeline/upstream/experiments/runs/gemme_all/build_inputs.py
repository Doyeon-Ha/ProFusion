#!/usr/bin/env python3
"""Build GEMME inputs for all ProteinGym DMS with an MSA file.

Per DMS: convert a2m -> ali{TAG}.fasta (uppercase, '.'/'-' gaps, sanitized header),
build mutations.txt (SINGLE mutants only, positions remapped to query coords via
MSA_start offset, WT-letter validated). Records a mapping query_label -> original mutant.
"""
import csv, os, json, re, sys

ROOT = '<REPO>'
MSA_DIR = f'{ROOT}/data/proteingym/DMS_msa_files'
SUB_DIR = f'{ROOT}/data/proteingym/DMS_ProteinGym_substitutions'
REF = f'{ROOT}/data/proteingym/DMS_substitutions.csv'
OUT = f'{ROOT}/experiments/runs/gemme_all/work'
os.makedirs(OUT, exist_ok=True)

def read_first_record_seq(path):
    seq = []
    started = False
    with open(path) as f:
        for line in f:
            if line.startswith('>'):
                if started:
                    break
                started = True
            elif started:
                seq.append(line.strip())
    return ''.join(seq)

def sanitize(name):
    return re.sub(r'[^A-Za-z0-9_]', '_', name)

rows = list(csv.DictReader(open(REF)))
summary = []
manifest = {}  # dms_id -> {tag, work, n_mut, n_dropped}

for r in rows:
    dms = r['DMS_id']
    msa = r.get('MSA_filename', '')
    csvf = os.path.join(SUB_DIR, dms + '.csv')
    msaf = os.path.join(MSA_DIR, msa)
    if not (msa and os.path.exists(msaf) and os.path.exists(csvf)):
        continue
    s = int(r['MSA_start'])  # 1-based target pos of first query residue
    e = int(r['MSA_end'])
    target = r['target_seq']

    tag = sanitize(dms)
    wdir = os.path.join(OUT, dms)
    os.makedirs(wdir, exist_ok=True)

    # 1) convert a2m -> ali{tag}.fasta
    ali_path = os.path.join(wdir, f'ali{tag}.fasta')
    first = True
    with open(msaf) as fin, open(ali_path, 'w') as fout:
        for line in fin:
            if line.startswith('>'):
                fout.write(f'>{tag}\n' if first else line)
                first = False
            else:
                fout.write(line.strip().upper().replace('.', '-') + '\n')

    # query seq (ungapped expected)
    qseq = read_first_record_seq(ali_path)
    has_gap = '-' in qseq
    qlen = len(qseq)

    # 2) build mutations (singles only), remap pos -> query coords
    n_total = n_single = n_kept = n_oob = n_wtmismatch = n_multi = 0
    seen = {}  # query_label -> original mutant
    for row in csv.DictReader(open(csvf)):
        n_total += 1
        m = row['mutant']
        if ':' in m:
            n_multi += 1
            continue
        n_single += 1
        mm = re.match(r'^([A-Z])(\d+)([A-Z])$', m)
        if not mm:
            continue
        wt, pos, mt = mm.group(1), int(mm.group(2)), mm.group(3)
        qpos = pos - s + 1
        if qpos < 1 or qpos > qlen:
            n_oob += 1
            continue
        if qseq[qpos - 1] != wt:
            n_wtmismatch += 1
            continue
        label = f'{wt}{qpos}{mt}'
        seen[label] = m
        n_kept += 1

    mut_path = os.path.join(wdir, 'mutations.txt')
    with open(mut_path, 'w') as f:
        f.write('\n'.join(seen.keys()) + ('\n' if seen else ''))
    json.dump({'tag': tag, 'msa_start': s, 'qlen': qlen,
               'label2mut': seen}, open(os.path.join(wdir, 'mapping.json'), 'w'))

    manifest[dms] = {'tag': tag, 'work': wdir, 'n_kept': n_kept}
    summary.append((dms, n_single, n_kept, n_oob, n_wtmismatch, n_multi, qlen, has_gap, s))

json.dump(manifest, open(f'{ROOT}/experiments/runs/gemme_all/manifest.json', 'w'), indent=0)

# report
print(f"DMS built: {len(summary)}")
flagged = [x for x in summary if x[7] or x[2] == 0 or (x[1] and x[2] / x[1] < 0.9)]
print(f"clean: {len(summary)-len(flagged)} | flagged: {len(flagged)}")
print(f"total single mutants kept: {sum(x[2] for x in summary)}")
print("\n== FLAGGED (gap in query / 0 kept / <90% kept) ==")
print(f"{'DMS_id':45s} {'single':>7} {'kept':>6} {'oob':>5} {'wtmm':>5} {'qlen':>6} {'gap':>4} {'start':>6}")
for dms, ns, nk, oob, wtmm, nmu, ql, gap, st in flagged:
    print(f"{dms:45s} {ns:7d} {nk:6d} {oob:5d} {wtmm:5d} {ql:6d} {str(gap):>4} {st:6d}")
