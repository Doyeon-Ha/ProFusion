#!/usr/bin/env python
"""Freeze the MaveDB external tuning set.

This builds the tuning set used to choose the ProteinGym fusion weights **outside** the benchmark.
The rule is fixed before any performance is seen, and the sha256 of the rule string is recorded in
the manifest. Changing the rule afterwards changes the hash, so any post-hoc adjustment shows up in
the record.

    rule = passes the team filter and is an individual study and has at least 100 single missense
           variants and L >= 30 and pg_tier == 'none'

individual study = the 2,680 MaveDB score sets minus two large collections.
  Domainome 1.0  experiment title contains 'domainome 1.0' (521 sets, aPCA domain abundance)
  Tsuboyama 2023 experiment title contains '(chymo)trypsin digestion' (1,623 sets, the SPURS
                 training set)
  A meta-analysis score set inherits the collection of its parent.

pg_tier, computed with MMseqs2 against the 186 ProteinGym sequences
  same    : one sequence contains the other, or >=95% identity with 90% coverage of the shorter
  homolog : >=30% identity with 50% coverage of the shorter and E < 1e-3
  none    : everything else. Only this tier is accepted

run: PYTHONPATH=src python experiments/mavedb_external/freeze_tuneset.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

R0 = Path(__file__).resolve().parents[2]
MAVE = Path("<MAVEDB_DIR>")
MMSEQS = Path("<MMSEQS_DIR>/bin/mmseqs")
PG_CSV = R0 / "data/proteingym/DMS_substitutions.csv"
OUT_DIR = R0 / "data/mavedb_external"
OUT_YAML = R0 / "configs/data/mavedb_external_tuneset.yaml"
SCAN_CACHE = OUT_DIR / "scan_cache.pkl"

EXCLUDED_YAML = Path(__file__).resolve().parent / "excluded.yaml"
RULE = (
    "in_filtered & coll=='individual' & n_single_missense>=100 & seq_len>=30 "
    "& pg_tier=='none' & assay_class in {Activity,Binding,OrganismalFitness,Expression} "
    "& urn not in excluded.yaml{hard,assay}"
)


def _rule_sha() -> str:
    """Hash the rule string together with the exclusion list, so changing either changes the hash."""
    ex = yaml.safe_load(EXCLUDED_YAML.read_text())
    urns = sorted(e["urn"] for g in ("hard", "assay") for e in (ex.get(g) or []))
    return hashlib.sha256((RULE + "|" + ",".join(urns)).encode()).hexdigest()

AXIS = {  # matches the axis vocabulary of the evaluation cache (two_dim/*.npz)
    "Activity": "function",
    "Binding": "function",
    "OrganismalFitness": "function",
    "Expression": "stability",
}
# ⚠ The (?!\1$) and (?!Ter$) parts are essential. Without them a **synonymous substitution such as
# p.Ala30Ala passes as missense**. Some MaveDB records spell synonymous changes out in three-letter
# form, so filtering on "=" and "Ter" alone does not catch them.
# Measured: 5 of the 49 external assays were contaminated (SGCB 304, STIM1 31 and others), which
# makes it 20 substitutions per position instead of 19.
MIS = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)(?!\1$)(?!Ter$)([A-Z][a-z]{2})$")


def load_scoresets() -> pd.DataFrame:
    """main.json to one row per score set, with the collection classification done."""
    d = json.loads((MAVE / "main.json").read_text())
    rows, meta_of, coll = [], {}, {}
    for es in d["experimentSets"]:
        for ex in es["experiments"]:
            t = ex["title"]
            for ss in ex.get("scoreSets", []):
                u = ss["urn"]
                meta_of[u] = ss.get("metaAnalyzesScoreSetUrns") or []
                if re.search(r"domainome 1\.0", t, re.I):
                    coll[u] = "Domainome"
                elif re.search(r"(tryp|chymotryp)sin digestion", t, re.I):
                    coll[u] = "Tsuboyama"
                else:
                    coll[u] = None
                tg = (ss.get("targetGenes") or [{}])[0]
                ids = {}
                for e in tg.get("externalIdentifiers") or []:
                    i = e.get("identifier", {})
                    ids[i.get("dbName")] = i.get("identifier")
                tax = (tg.get("targetSequence") or {}).get("taxonomy") or {}
                rows.append(
                    dict(urn=u, exp_title=t, title=ss.get("title", ""),
                         tname=tg.get("name"), organism=tax.get("organismName"),
                         uniprot=ids.get("UniProt"), refseq=ids.get("RefSeq"))
                )
    # A meta-analysis inherits its parent collection
    for u, par in meta_of.items():
        if coll[u] is None and par:
            pc = {coll.get(p) for p in par} - {None}
            coll[u] = pc.pop() if len(pc) == 1 else "meta_other"
    df = pd.DataFrame(rows)
    df["coll"] = df.urn.map(lambda u: coll[u] or "individual")
    return df


def attach_team_filter(df: pd.DataFrame) -> pd.DataFrame:
    """The team's first-pass filter: protein_coding, has a sequence, not superseded."""
    flt = json.loads((MAVE / "parsed_result/filtered_mave_db.json").read_text())
    df["in_filtered"] = df.urn.isin(flt)
    df["pseq"] = df.urn.map(lambda u: (flt.get(u) or {}).get("sequence"))
    df["seq_len"] = df.pseq.map(lambda s: len(s) if isinstance(s, str) else 0)
    return df


def attach_llm_labels(df: pd.DataFrame) -> pd.DataFrame:
    """assay_class and score_direction from three LLMs. Used for the axis assignment and as a sign fallback."""
    P = MAVE / "parsed_result"
    models = ["gpt-4.1", "gpt-5-2025-08-07", "o3"]
    L = {m: json.loads((P / f"llm_processed_{m}.json").read_text()) for m in models}
    for m in models:
        df[f"cls_{m}"] = df.urn.map(lambda u: (L[m].get(u) or {}).get("assay_class"))
        df[f"dir_{m}"] = df.urn.map(lambda u: (L[m].get(u) or {}).get("score_direction"))
    df["cls_agree"] = df.apply(lambda r: len({r[f"cls_{m}"] for m in models}) == 1, axis=1)
    df["dir_agree"] = df.apply(lambda r: len({r[f"dir_{m}"] for m in models}) == 1, axis=1)
    df["assay_class"] = df["cls_o3"]           # o3 agreed best with the nonsense evidence (97.9%)
    return df


def scan_csvs(df: pd.DataFrame) -> pd.DataFrame:
    """Count single missense variants across every score CSV. Slow (about 15 minutes), so it is cached."""
    if SCAN_CACHE.exists():
        return pd.read_pickle(SCAN_CACHE)
    import numpy as np
    out = []
    for u in df.urn:
        f = MAVE / "csv" / f"{u.replace(':', '-')}.scores.csv"
        if not f.exists():
            out.append(dict(urn=u, miss=np.nan)); continue
        t = pd.read_csv(f, usecols=lambda c: c in ("hgvs_pro", "score"), low_memory=False)
        if "hgvs_pro" not in t or "score" not in t:
            out.append(dict(urn=u, miss=np.nan)); continue
        ok = pd.to_numeric(t.score, errors="coerce").notna()
        out.append(dict(urn=u, miss=int((t.hgvs_pro.astype(str).str.match(MIS) & ok).sum())))
    r = pd.DataFrame(out)
    SCAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    r.to_pickle(SCAN_CACHE)
    return r


def pg_tiers(df: pd.DataFrame) -> dict[str, str]:
    """The leakage tier against the 186 ProteinGym sequences."""
    pg = {}
    for r in pd.read_csv(PG_CSV).itertuples():
        u, s = getattr(r, "UniProt_ID", None), (getattr(r, "target_seq", "") or "").strip()
        if u and s and u not in pg:
            pg[u] = s
    cand = df[df.pseq.notna()].drop_duplicates("pseq")[["pseq"]].reset_index(drop=True)
    cand["sid"] = [f"q{i}" for i in range(len(cand))]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "q.fasta").write_text("".join(f">{r.sid}\n{r.pseq}\n" for r in cand.itertuples()))
        (td / "t.fasta").write_text("".join(f">{k}\n{v}\n" for k, v in pg.items()))
        subprocess.run([str(MMSEQS), "easy-search", str(td / "q.fasta"), str(td / "t.fasta"),
                        str(td / "h.m8"), str(td / "tmp"), "-s", "7.5", "--min-seq-id", "0.3",
                        "--threads", "8", "--format-output",
                        "query,target,pident,alnlen,qlen,tlen,evalue"],
                       check=True, capture_output=True)
        h = pd.read_csv(td / "h.m8", sep="\t", header=None,
                        names="q t pid aln qlen tlen ev".split())
    h["scov"] = h.aln / h[["qlen", "tlen"]].min(axis=1)

    tier = {}
    pgseqs = list(pg.values())
    for r in cand.itertuples():
        s, hh = r.pseq, h[h.q == r.sid]
        if len(s) >= 8 and any(s in p or p in s for p in pgseqs):
            tier[s] = "same"
        elif len(hh[(hh.pid >= 95) & (hh.scov >= 0.9)]):
            tier[s] = "same"
        elif len(hh[(hh.pid >= 30) & (hh.scov >= 0.5) & (hh.ev < 1e-3)]):
            tier[s] = "homolog"
        else:
            tier[s] = "none"
    return tier


def main() -> None:
    df = attach_llm_labels(attach_team_filter(load_scoresets()))
    df = df.merge(scan_csvs(df)[["urn", "miss"]], on="urn", how="left")
    tier = pg_tiers(df)
    df["pg_tier"] = df.pseq.map(lambda s: tier.get(s))

    sel = df[(df.in_filtered) & (df.coll == "individual") & (df.miss >= 100)
             & (df.seq_len >= 30) & (df.pg_tier == "none")
             & (df.assay_class.isin(AXIS))].copy()

    ex = yaml.safe_load(EXCLUDED_YAML.read_text())
    drop = {e["urn"]: g for g in ("hard", "assay") for e in (ex.get(g) or [])}
    n_rule = len(sel)
    sel = sel[~sel.urn.isin(drop)].copy()
    sel["axis"] = sel.assay_class.map(AXIS)

    n_dms, n_prot = len(sel), sel.pseq.nunique()
    print(f"[rule] {RULE}")
    print(f"[sha256] {_rule_sha()}")
    print(f"[excluded] {n_rule} passed the rule, {len(drop)} dropped manually "
          f"(hard {sum(v == 'hard' for v in drop.values())} / "
          f"assay {sum(v == 'assay' for v in drop.values())})")
    print(f"[frozen] {n_dms} DMS / {n_prot} proteins")
    print(sel.groupby("axis").agg(DMS=("urn", "size"), proteins=("pseq", "nunique")).to_string())
    if (n_dms, n_prot) != (49, 23):                                            # EC-001
        sys.exit(f"[FAIL] expected 49 DMS / 23 proteins, got {n_dms} / {n_prot}")

    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    recs = [dict(urn=r.urn, tname=r.tname, organism=r.organism, axis=r.axis,
                 assay_class=r.assay_class, n_missense=int(r.miss), seq_len=int(r.seq_len),
                 uniprot=r.uniprot, cls_unanimous=bool(r.cls_agree),
                 dir_unanimous=bool(r.dir_agree))
            for r in sel.sort_values("urn").itertuples()]
    OUT_YAML.write_text(yaml.safe_dump(
        dict(schema_version=1, rule=RULE, rule_sha256=_rule_sha(),
             source="MaveDB public dump 2025-06-12 (<MAVEDB_DIR>)",
             counts=dict(dms=n_dms, proteins=n_prot,
                         function=int((sel.axis == "function").sum()),
                         stability=int((sel.axis == "stability").sum())),
             datasets=recs), sort_keys=False, allow_unicode=True))
    sel.to_pickle(OUT_DIR / "frozen.pkl")
    # pandas versions differ between environments and the pickle will not open, so a CSV is written alongside
    sel[["urn", "tname", "axis", "assay_class", "miss", "seq_len", "pseq"]].to_csv(
        OUT_DIR / "frozen.csv", index=False)
    print(f"[saved] {OUT_YAML}")


if __name__ == "__main__":
    main()
