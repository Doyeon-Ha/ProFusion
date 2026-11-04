#!/usr/bin/env python
"""LLR for the **structure-only** backbones: ESM-IF1 and ProteinMPNN.

In the hypothesis that a structure-aware pLM internalises ddG and loses conservation, these two are
**the fixed points at either end of the axis**. They never see the sequence, only the backbone, so
they are the structural extreme, and the official aggregate separates them by a factor of 1.6
(ESM-IF1 .4224 against ProteinMPNN .2578), which breaks the confound between "structure
conditioning" and "model performance". If both move the same way, performance is not the cause.

⚠️ They are inverse-folding models, so their LLR is itself "does this amino acid fit this
   backbone". A high ddG redundancy is almost true by definition, so it **must not be used as
   evidence** for the hypothesis, only as a scale marker at the ends.

The score convention matches the other backbones: LLR(i, mut) = logP(mut | structure) −
logP(wt | structure), larger means closer to wild type.
  ESM-IF1     teacher-forced autoregressive logits (logits[:,:,i] predicts seq[i])
  ProteinMPNN `unconditional_probs`: no sequence context, a pure backbone-to-amino-acid distribution

env: s3f (fair-esm 2.0.0 and biotite 0.40). CPU is enough.
run: PYTHONPATH=src python experiments/mavedb_external/build_llr_structonly.py --backbone esmif1
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

R0 = Path(__file__).resolve().parents[2]
EXT = R0 / "data/mavedb_external"
FOLD = EXT / "fold/out"
MPNN = R0 / "external/ProteinMPNN"
MAVE = Path("<MAVEDB_DIR>/csv")
# ⚠ The (?!\1$) and (?!Ter$) parts are essential. Without them a **synonymous substitution such as
# p.Ala30Ala passes as missense**. Some MaveDB records spell synonymous changes out in three-letter
# form, so filtering on "=" and "Ter" alone does not catch them.
# Measured: 5 of the 49 external assays were contaminated (SGCB 304, STIM1 31 and others), which
# makes it 20 substitutions per position instead of 19.
MIS = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)(?!\1$)(?!Ter$)([A-Z][a-z]{2})$")
T3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
      "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
      "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}
MPNN_AB = "ACDEFGHIKLMNPQRSTVWYX"


def pdb_of(tag: str) -> str | None:
    c = sorted(glob.glob(str(FOLD / f"{tag}_unrelaxed_rank_001_*.pdb")))
    return c[0] if c else None


def variants_of(urn: str, wt: str):
    """The same convention as build_llr_struct.variants_of, copied rather than imported because this runs in another environment."""
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
        seen.add(k); out.append((k, h, float(sc)))
    return out, bad


def logprobs_esmif1(model, alphabet, pdb: str, wt: str, device: str) -> np.ndarray:
    """Returns the full (L, vocab) log-probabilities, not (L, 20) in amino acid order."""
    import esm.inverse_folding as IF
    from esm.inverse_folding.util import CoordBatchConverter
    coords, native = IF.util.load_coords(pdb, "A")
    assert native == wt, f"the PDB sequence differs from the WT ({len(native)} vs {len(wt)})"
    bc = CoordBatchConverter(alphabet)
    cx, conf, _s, tokens, pad = bc([(coords, None, wt)], device=device)
    with torch.no_grad():
        logits, _ = model.forward(cx, pad, conf, tokens[:, :-1])
    # logits[:, :, i] predicts tokens[:, i+1] = wt[i] (the same as util.get_sequence_loss)
    lp = torch.log_softmax(logits, dim=1)[0].T.cpu().numpy()        # (L, vocab)
    assert lp.shape[0] == len(wt), f"logit length {lp.shape[0]} != WT {len(wt)}"
    return lp


def logprobs_mpnn(model, pdb: str, wt: str, device: str) -> np.ndarray:
    sys.path.insert(0, str(MPNN))
    from protein_mpnn_utils import StructureDatasetPDB, parse_PDB, tied_featurize
    pdbd = parse_PDB(pdb, input_chain_list=["A"])
    ds = StructureDatasetPDB(pdbd, truncate=None, max_length=20000)
    b = [ds[0]]
    # tied_featurize returns 20 values, whose order varies between versions, so they are taken by index
    ft = tied_featurize(b, device, {pdbd[0]["name"]: (["A"], [])},
                        None, None, None, None, None, ca_only=False)
    X, mask, cea, ridx = ft[0], ft[2], ft[5], ft[12]
    with torch.no_grad():
        lp = model.unconditional_probs(X, mask, ridx, cea)[0].cpu().numpy()    # (L, 21)
    assert lp.shape[0] == len(wt), f"MPNN length {lp.shape[0]} != WT {len(wt)}"
    return lp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["esmif1", "proteinmpnn"])
    ap.add_argument("--device", default="cpu")
    A = ap.parse_args()

    man = json.loads((EXT / "seq_manifest.json").read_text())
    seqs = {t: (EXT / "msa/query" / f"{t}.fasta").read_text().split("\n", 1)[1].replace("\n", "").strip()
            for t in man}
    sel = pd.read_csv(EXT / "frozen.csv")
    sel["tag"] = [hashlib.md5(s.encode()).hexdigest()[:10] for s in sel.pseq]
    outdir = EXT / "llr" / A.backbone
    outdir.mkdir(parents=True, exist_ok=True)

    if A.backbone == "esmif1":
        import esm
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model = model.eval().to(A.device)
        name = "esm_if1_gvp4_t16_142M_UR50"
        idx_of = {a: alphabet.get_idx(a) for a in "ACDEFGHIKLMNPQRSTVWY"}
    else:
        sys.path.insert(0, str(MPNN))
        from protein_mpnn_utils import ProteinMPNN
        ck = torch.load(MPNN / "vanilla_model_weights/v_48_020.pt", map_location="cpu",
                        weights_only=False)
        model = ProteinMPNN(num_letters=21, node_features=128, edge_features=128,
                            hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                            augment_eps=0.0, k_neighbors=ck["num_edges"])
        model.load_state_dict(ck["model_state_dict"]); model = model.eval().to(A.device)
        name = "ProteinMPNN v_48_020"
        idx_of = {a: MPNN_AB.index(a) for a in "ACDEFGHIKLMNPQRSTVWY"}

    lp_cache: dict[str, np.ndarray] = {}
    for i, r in enumerate(sel.to_dict("records"), 1):
        urn, tag = r["urn"], r["tag"]
        p = outdir / f"{urn.replace(':', '-')}.pt"
        if p.exists():
            print(f"[{i}/{len(sel)}] skip {urn}", flush=True); continue
        wt, pdb = seqs[tag], pdb_of(tag)
        if pdb is None:
            print(f"[{i}/{len(sel)}] NO-STRUCT {urn}", flush=True); continue
        vs, bad = variants_of(urn, wt)
        if not vs:
            print(f"[{i}/{len(sel)}] EMPTY {urn}", flush=True); continue
        try:
            if tag not in lp_cache:
                lp_cache[tag] = (logprobs_esmif1(model, alphabet, pdb, wt, A.device)
                                 if A.backbone == "esmif1"
                                 else logprobs_mpnn(model, pdb, wt, A.device))
            lp = lp_cache[tag]
            sc = torch.tensor([lp[int(k[1:-1]) - 1, idx_of[k[-1]]]
                               - lp[int(k[1:-1]) - 1, idx_of[k[0]]] for k, _h, _s in vs],
                              dtype=torch.float32)
        except Exception as e:                                       # noqa: BLE001
            print(f"[{i}/{len(sel)}] FAIL {urn} ({man[tag]['tname']}): "
                  f"{type(e).__name__}: {str(e)[:150]}", flush=True)
            continue
        torch.save(dict(mutants=[k for k, _h, _s in vs], scores=sc,
                        dms_scores=torch.tensor([s for _k, _h, s in vs]),
                        backbone=A.backbone, model_name=name, n_dropped=bad,
                        structure_only=True), p)
        print(f"[{i}/{len(sel)}] {urn} {man[tag]['tname'][:20]:22s} L={len(wt):5d} "
              f"V={len(vs):6d}", flush=True)
    print(f"[done] {A.backbone} — {len(list(outdir.glob('*.pt')))} / {len(sel)}")


if __name__ == "__main__":
    main()
