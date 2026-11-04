#!/usr/bin/env python
"""LLR for the backbones that need structure: ProSST and ESM3 (multimodal and sequence-only).

Unlike ESM2 and ESM-1v, these two have to wait for the ColabFold structures.

ProSST
    Identical to the official ai4protein/ProSST zero-shot: one forward over the WT sequence plus the
    WT quantised structure tokens (`ss_input_ids`), then wildtype-marginal. ProteinGym reuses the
    tokens it was given, but the external proteins have none, so they are **quantised from our own
    PDB** with `SSTPredictor` (structure_vocab_size=2048).

ESM3 (multimodal)

    Sequence-only is the default, but here the WT context is encoded with **both the structure
    (atom37) and the InterProScan function annotations**. Either way the score comes from the
    residue distribution of the sequence track.
    ⚠️ The `llr_ESM3` of the evaluation cache was built sequence-only. This output is therefore
       stored as a **separate channel**, `llr_ESM3mm`. Overwriting the same name would make the
       tuning set and the evaluation set refer to different modalities.

ESM3 (sequence-only), `esm3seq`
    The **paired control**. Same weights, same tokenizer, with only coordinates and function set to
    None. In the hypothesis that a structure-aware pLM internalises ddG and loses conservation, this
    is the only arm that holds architecture, data and size fixed and turns off the structure input.
    ⚠️ Even though no structure is needed, **the PDB-existence and length gates are kept identical
       to esm3mm**. If the DMS sets differ by even one assay it stops being a paired comparison.

env: prosst and esm3 were put in one environment. The ProSST source is on node-local disk, so the
node was pinned.
run: sbatch experiments/mavedb_external/slurm_llr_struct.sbatch prosst|esm3mm
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

R0 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R0 / "src"))
sys.path.insert(0, str(R0 / "external" / "esm3"))

from cvm.data.proteingym import Variant                                  # noqa: E402
from cvm.labels import zeroshot as ZS                                    # noqa: E402

EXT = R0 / "data/mavedb_external"
FOLD = EXT / "fold/out"
# ⚠ The (?!\1$) and (?!Ter$) parts are essential. Without them a **synonymous substitution such as
# p.Ala30Ala passes as missense**. Some MaveDB records spell synonymous changes out in three-letter
# form, so filtering on "=" and "Ter" alone does not catch them.
# Measured: 5 of the 49 external assays were contaminated (SGCB 304, STIM1 31 and others), which
# makes it 20 substitutions per position instead of 19.
MIS = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)(?!\1$)(?!Ter$)([A-Z][a-z]{2})$")
T3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
      "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
      "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}
MAVE = Path("<MAVEDB_DIR>/csv")
ESM3_MAX = 2046


def pdb_of(tag: str) -> str | None:
    c = sorted(glob.glob(str(FOLD / f"{tag}_unrelaxed_rank_001_*.pdb")))
    return c[0] if c else None


def variants_of(urn: str, wt: str):
    t = pd.read_csv(MAVE / f"{urn.replace(':', '-')}.scores.csv",
                    usecols=lambda c: c in ("hgvs_pro", "score"), low_memory=False)
    s = pd.to_numeric(t.score, errors="coerce")
    out, seen, bad = [], set(), 0
    for h, sc in zip(t.hgvs_pro.astype(str), s):
        m = MIS.match(h)
        if not m or pd.isna(sc):
            continue
        w, pos, mu = T3.get(m[1]), int(m[2]), T3.get(m[3])
        if not w or not mu or pos < 1 or pos > len(wt) or wt[pos - 1] != w:
            bad += 1; continue
        k = f"{w}{pos}{mu}"
        if k in seen:
            continue
        seen.add(k); out.append(Variant(mutant=k, hgvsp=h, sequence="", score=float(sc)))
    return out, bad


def load_coords(pdb: str, L: int, device: str):
    """PDB to atom37 coordinates [L,37,3] (the ESM3 structure-track input)."""
    from esm.utils.structure.protein_chain import ProteinChain
    ch = ProteinChain.from_pdb(pdb)
    assert len(ch.sequence) == L, f"structure length {len(ch.sequence)} != sequence {L}"
    return torch.tensor(ch.atom37_positions, dtype=torch.float32, device=device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["prosst", "esm3mm", "esm3seq"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # For attaching a frozen set outside the 49 (for example frozen_stability_pool.csv). Same convention as build_llr.py.
    ap.add_argument("--frozen", default="frozen.csv")
    A = ap.parse_args()

    man = json.loads((EXT / "seq_manifest.json").read_text())
    seqs = {t: (EXT / "msa/query" / f"{t}.fasta").read_text().split("\n", 1)[1].replace("\n", "").strip()
            for t in man}
    # ⚠ build_llr.py takes --frozen as a path relative to the repository root while this takes a
    # file name relative to EXT. Passing the same argument to both sbatch scripts raised
    # FileNotFoundError, so both forms are accepted here.
    fz = Path(A.frozen)
    sel = pd.read_csv(fz if fz.exists() else EXT / fz.name)
    import hashlib
    sel["tag"] = [hashlib.md5(s.encode()).hexdigest()[:10] for s in sel.pseq]

    outdir = EXT / "llr" / A.backbone
    outdir.mkdir(parents=True, exist_ok=True)

    if A.backbone == "prosst":
        from prosst.structure.get_sst_seq import SSTPredictor
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        name = "AI4Protein/ProSST-2048"
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(name, trust_remote_code=True).to(A.device).eval()
        pred = SSTPredictor(structure_vocab_size=2048, device=A.device)
        sst_cache: dict[str, list[int]] = {}
    else:
        from esm.models.esm3 import ESM3
        name = ZS.DEFAULT_ESM3_MODEL_NAME
        model = ESM3.from_pretrained(name, device=torch.device(A.device)).eval()
        fa_pkl = EXT / "function_annotations.pkl"
        FUNC = pickle.load(open(fa_pkl, "rb")) if fa_pkl.exists() else {}
        print(f"[func] {sum(1 for v in FUNC.values() if v)} / {len(FUNC)} have annotations", flush=True)

    for i, r in enumerate(sel.to_dict("records"), 1):
        urn, tag = r["urn"], r["tag"]
        p = outdir / f"{urn.replace(':', '-')}.pt"
        if p.exists():
            print(f"[{i}/{len(sel)}] skip {urn}", flush=True); continue
        wt = seqs[tag]
        pdb = pdb_of(tag)
        if pdb is None:
            print(f"[{i}/{len(sel)}] NO-STRUCT {urn} ({man[tag]['tname']})", flush=True); continue
        if A.backbone.startswith("esm3") and len(wt) > ESM3_MAX:
            print(f"[{i}/{len(sel)}] SKIP-LONG {urn} L={len(wt)} > {ESM3_MAX}", flush=True); continue
        vs, bad = variants_of(urn, wt)
        if not vs:
            print(f"[{i}/{len(sel)}] EMPTY {urn}", flush=True); continue

        try:
            if A.backbone == "prosst":
                if tag not in sst_cache:
                    res = pred.predict_from_pdb(pdb)
                    toks = res[0]["2048_sst_seq"] if isinstance(res, list) else res["2048_sst_seq"]
                    sst_cache[tag] = [int(x) for x in toks]
                st = sst_cache[tag]
                assert len(st) == len(wt), f"{tag}: {len(st)} structure tokens != sequence {len(wt)}"
                sc = ZS.score_dms_prosst_wildtype_marginal(wt, vs, st, model, tok, device=A.device)
                extra = dict(n_struct_tokens=len(st))
            else:
                if A.backbone == "esm3seq":
                    sc = ZS.score_dms_esm3_wildtype_marginal(
                        wt, vs, model, device=A.device,
                        coordinates=None, function_annotations=None)
                    extra = dict(multimodal=False, n_func_ann=0, coords_shape=None)
                else:
                    coords = load_coords(pdb, len(wt), A.device)
                    fa = FUNC.get(tag) or None
                    sc = ZS.score_dms_esm3_wildtype_marginal(
                        wt, vs, model, device=A.device, coordinates=coords, function_annotations=fa)
                    extra = dict(multimodal=True, n_func_ann=len(fa) if fa else 0,
                                 coords_shape=list(coords.shape))
        except Exception as e:                                  # noqa: BLE001
            print(f"[{i}/{len(sel)}] FAIL {urn} ({man[tag]['tname']}): "
                  f"{type(e).__name__}: {str(e)[:140]}", flush=True)
            continue

        torch.save(dict(mutants=[v.mutant for v in vs], scores=sc.cpu(),
                        dms_scores=torch.tensor([v.score for v in vs]),
                        backbone=A.backbone, model_name=name, n_dropped=bad,
                        computed_at=ZS.utc_now_iso(), **extra), p)
        print(f"[{i}/{len(sel)}] {urn} {man[tag]['tname'][:20]:22s} L={len(wt):5d} "
              f"V={len(vs):6d} {extra}", flush=True)
    print(f"[done] {A.backbone} — {len(list(outdir.glob('*.pt')))} / {len(sel)}")


if __name__ == "__main__":
    main()
