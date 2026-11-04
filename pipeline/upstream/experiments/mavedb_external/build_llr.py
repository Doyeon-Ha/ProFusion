#!/usr/bin/env python
"""pLM LLR for the external tuning set (wildtype-marginal).

Uses the same scoring method as the evaluation cache. The LLR depends on the sequence only, so it
does not have to wait for the structures. ProSST is the one exception.

output: data/mavedb_external/llr/{backbone}/{urn}.pt

run (GPU work is submitted through Slurm):
    sbatch experiments/mavedb_external/slurm_llr.sbatch esm2
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch

R0 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R0 / "src"))

from cvm.data.proteingym import Variant                                  # noqa: E402
from cvm.labels import zeroshot as ZS                                    # noqa: E402

MAVE = Path("<MAVEDB_DIR>")
FROZEN = R0 / "data/mavedb_external/frozen.csv"
OUT = R0 / "data/mavedb_external/llr"
# ⚠ The (?!\1$) and (?!Ter$) parts are essential. Without them a **synonymous substitution such as
# p.Ala30Ala passes as missense**. Some MaveDB records spell synonymous changes out in three-letter
# form, so filtering on "=" and "Ter" alone does not catch them.
# Measured: 5 of the 49 external assays were contaminated (SGCB 304, STIM1 31 and others), which
# makes it 20 substitutions per position instead of 19.
MIS = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)(?!\1$)(?!Ter$)([A-Z][a-z]{2})$")
T3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
      "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
      "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}
# The same composition as the evaluation cache:
#   ESM1v = the **mean** over esm1v_t33_650M_UR90S_1..5 (the standard ProteinGym ensemble)
#   ESM3  = sequence-only
# ESM-1v has a learned positional embedding and therefore caps at 1022 residues (ESM2 uses rotary
# embeddings and has no limit). Longer sequences are recorded as missing and listed as excluded,
# never silently truncated.
MAXLEN = {"esm1v": 1022}
MODELS = {
    "esm2": ["facebook/esm2_t33_650M_UR50D"],
    # The capacity ladder, all **sequence-only**. It is the control that separates "the ddG
    # redundancy follows structure conditioning" from "it simply follows model performance".
    # The official aggregate goes .226 -> .321 -> .387 -> .414 -> .406, so performance spans a
    # factor of 1.8 while none of them sees structure.
    "esm2_8M": ["facebook/esm2_t6_8M_UR50D"],
    "esm2_35M": ["facebook/esm2_t12_35M_UR50D"],
    "esm2_150M": ["facebook/esm2_t30_150M_UR50D"],
    "esm2_3B": ["facebook/esm2_t36_3B_UR50D"],
    "esm1v": [f"facebook/esm1v_t33_650M_UR90S_{i}" for i in range(1, 6)],
    "esm3": ["esm3_sm_open_v1"],
}


def _tok(name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def _load_esm_mlm(name: str):
    """Load an EsmForMaskedLM.

    ESM-1v has no safetensors on the hub, only `.bin`, and transformers 4.52 refuses to load `.bin`
    under torch<2.6 because of CVE-2025-32434. The torch here is 2.5.1, so `from_pretrained` fails
    outright. An empty model is built from the config and the state_dict loaded directly instead.
    (It is read with `weights_only=True`, so the arbitrary-code path the CVE points at is never
    taken.) The only missing keys are `esm.contact_head.*`, which MLM scoring does not use.
    """
    from transformers import EsmConfig, EsmForMaskedLM
    try:
        return EsmForMaskedLM.from_pretrained(name)
    except ValueError as e:
        if "torch.load" not in str(e):
            raise
    from huggingface_hub import hf_hub_download
    cfg = EsmConfig.from_pretrained(name)
    sd = torch.load(hf_hub_download(name, "pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    model = EsmForMaskedLM(cfg)
    missing, _ = model.load_state_dict(sd, strict=False)
    assert all(k.startswith("esm.contact_head.") for k in missing), f"unexpected missing keys: {missing}"
    return model


def variants_of(urn: str, wt: str) -> list[Variant]:
    """hgvs_pro from the CSV to 1-letter. Keeps only rows whose wild-type residue matches the sequence."""
    f = MAVE / "csv" / f"{urn.replace(':', '-')}.scores.csv"
    t = pd.read_csv(f, usecols=lambda c: c in ("hgvs_pro", "score"), low_memory=False)
    s = pd.to_numeric(t.score, errors="coerce")
    out, seen, bad = [], set(), 0
    for h, sc in zip(t.hgvs_pro.astype(str), s):
        m = MIS.match(h)
        if not m or pd.isna(sc):
            continue
        wt_aa, pos, mu_aa = T3.get(m[1]), int(m[2]), T3.get(m[3])
        if not wt_aa or not mu_aa or pos < 1 or pos > len(wt):
            bad += 1; continue
        if wt[pos - 1] != wt_aa:                                   # coordinate mismatch, never passed over silently
            bad += 1; continue
        mut = f"{wt_aa}{pos}{mu_aa}"
        if mut in seen:
            continue
        seen.add(mut)
        out.append(Variant(mutant=mut, hgvsp=h, sequence="", score=float(sc)))
    return out, bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="esm2", choices=list(MODELS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Swaps the frozen set. The Domainome (stability_ddg) arm uses frozen_domainome.csv.
    ap.add_argument("--frozen", default=str(FROZEN),
                    help="the frozen csv holding urn and pseq (default: the 49-assay frozen.csv)")
    A = ap.parse_args()

    names = MODELS[A.backbone]
    if A.backbone == "esm3":
        sys.path.insert(0, str(R0 / "external" / "esm3"))
        from esm.models.esm3 import ESM3
        loaded = [(names[0], ESM3.from_pretrained(names[0], device=torch.device(A.device)).eval(), None)]
    else:
        loaded = [(n, _load_esm_mlm(n).to(A.device).eval(), _tok(n)) for n in names]
    print(f"[model] {A.backbone}: {len(loaded)} loaded: {names}", flush=True)

    sel = pd.read_csv(A.frozen)
    outdir = OUT / A.backbone
    outdir.mkdir(parents=True, exist_ok=True)
    tot_bad = 0
    for i, r in enumerate(sel.to_dict("records"), 1):
        urn, wt = r["urn"], r["pseq"]
        p = outdir / f"{urn.replace(':', '-')}.pt"
        if p.exists():
            print(f"[{i}/{len(sel)}] skip {urn}", flush=True); continue
        lim = MAXLEN.get(A.backbone)
        if lim and len(wt) > lim:
            print(f"[{i}/{len(sel)}] SKIP-LONG {urn} L={len(wt)} > {lim} (EC-007)", flush=True)
            continue
        vs, bad = variants_of(urn, wt)
        tot_bad += bad
        if not vs:
            print(f"[{i}/{len(sel)}] EMPTY {urn} ({bad} coordinate mismatches)", flush=True); continue
        if A.backbone == "esm3":
            sc = ZS.score_dms_esm3_wildtype_marginal(wt, vs, loaded[0][1], device=A.device)
        else:   # mean over the per-seed scores (5 for ESM1v, 1 for ESM2)
            sc = torch.stack([ZS.score_dms_esm2_wildtype_marginal(wt, vs, m, tk, device=A.device)
                              for _, m, tk in loaded]).mean(0)
        torch.save(dict(mutants=[v.mutant for v in vs], scores=sc.cpu(),
                        dms_scores=torch.tensor([v.score for v in vs]),
                        backbone=A.backbone, model_name=names, n_dropped=bad,
                        computed_at=ZS.utc_now_iso()), p)
        print(f"[{i}/{len(sel)}] {urn} L={len(wt)} V={len(vs)} drop={bad}", flush=True)
    print(f"[done] {A.backbone}: {tot_bad} variants dropped for coordinate mismatch")


if __name__ == "__main__":
    main()
