#!/usr/bin/env python
"""A **convention control**: rescore ESM2 with masked-marginal.

S2F is structure-conditioned, yet its D = (ddG redundancy − GEMME redundancy) = −0.069 falls in the
sequence-only range, breaking the clean separation between structure conditioning and redundancy.
But S2F scores with the **masked-marginal** convention (the position is covered with <mask> and read
back), while our ESM2, ESM3 and ProSST use **wildtype-marginal** (one forward over the WT). Masking
removes the leak of wild-type identity, which makes the LLR more of a position profile, which in
turn can raise the GEMME redundancy and push D negative.

So **the same ESM2 is scored under both conventions** to measure how far D moves. If it moves as
far as the S2F deviation (about 0.05), S2F is a convention artefact rather than a counterexample.

Masking is done per protein (tag) at the positions that are needed, and cached: the 7 INSR DMS are
computed once.
run: PYTHONPATH=src python experiments/mavedb_external/build_llr_masked.py --shard 0 --nshard 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

R0 = Path(__file__).resolve().parents[2]
EXT = R0 / "data/mavedb_external"
AA20 = "ACDEFGHIKLMNPQRSTVWY"
MAXLEN = 1022


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    A = ap.parse_args()
    from transformers import AutoTokenizer, EsmForMaskedLM
    name = "facebook/esm2_t33_650M_UR50D"
    tok = AutoTokenizer.from_pretrained(name)
    model = EsmForMaskedLM.from_pretrained(name).to(A.device).eval()
    mask_id = tok.mask_token_id

    man = json.loads((EXT / "seq_manifest.json").read_text())
    seqs = {t: (EXT / "msa/query" / f"{t}.fasta").read_text().split("\n", 1)[1].replace("\n", "").strip()
            for t in man}
    F = pd.read_csv(EXT / "frozen.csv")
    F["tag"] = [hashlib.md5(s.encode()).hexdigest()[:10] for s in F.pseq]
    tag_of = dict(zip(F.urn, F.tag))

    # Collect the positions (1-based) each tag needs. Several DMS of the same protein share them
    need: dict[str, set[int]] = {}
    items = []
    for f in sorted((EXT / "two_dim").glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        urn = str(d["urn"]); t = tag_of[urn]
        need.setdefault(t, set()).update(int(p) for p in d["pos1"])
        items.append((urn, t, f))
    tags = sorted(need)
    mine = [t for i, t in enumerate(tags) if i % A.nshard == A.shard]
    out = EXT / "llr/esm2_masked"; out.mkdir(parents=True, exist_ok=True)
    cache = EXT / "llr/esm2_masked/_mat"; cache.mkdir(exist_ok=True)

    for t in mine:
        mp = cache / f"{t}.npy"
        if mp.exists():
            print(f"[{t}] cached", flush=True); continue
        seq = seqs[t]
        if len(seq) > MAXLEN:
            print(f"[{t}] SKIP-LONG L={len(seq)}", flush=True); continue
        pos = sorted(need[t])
        enc = tok(seq, return_tensors="pt")
        ids = enc["input_ids"].to(A.device)
        M = np.full((len(seq), 20), np.nan, dtype=np.float32)
        aa_ids = [tok.convert_tokens_to_ids(a) for a in AA20]
        for i in range(0, len(pos), A.batch):
            chunk = pos[i:i + A.batch]
            b = ids.repeat(len(chunk), 1)
            for j, p in enumerate(chunk):
                b[j, p] = mask_id                      # because of BOS, the token index is the 1-based position
            with torch.no_grad():
                lg = model(input_ids=b, attention_mask=torch.ones_like(b)).logits
            lp = torch.log_softmax(lg, dim=-1)
            for j, p in enumerate(chunk):
                M[p - 1] = lp[j, p, aa_ids].cpu().numpy()
            if i % (A.batch * 20) == 0:
                print(f"  [{t}] {i}/{len(pos)}", flush=True)
        np.save(mp, M)
        print(f"[{t}] L={len(seq)}, {len(pos)} positions written", flush=True)

    # Build the per-DMS .pt for every tag whose matrix is ready
    made = 0
    for urn, t, f in items:
        mp = cache / f"{t}.npy"
        p = out / f"{urn.replace(':', '-')}.pt"
        if p.exists() or not mp.exists():
            continue
        M = np.load(mp)
        R = torch.load(EXT / "llr/esm2" / f"{urn.replace(':', '-')}.pt",
                       map_location="cpu", weights_only=False)
        v = []
        for k in R["mutants"]:
            q = int(k[1:-1]); v.append(M[q - 1, AA20.index(k[-1])] - M[q - 1, AA20.index(k[0])])
        v = torch.tensor(v, dtype=torch.float32)
        if not torch.isfinite(v).all():
            print(f"  {urn}: contains NaN, skipped"); continue
        torch.save(dict(mutants=R["mutants"], scores=v, dms_scores=R["dms_scores"],
                        backbone="esm2_masked", model_name=name + " (masked-marginal)",
                        n_dropped=0), p)
        made += 1
    print(f"[done] shard {A.shard}: {made} .pt written")


if __name__ == "__main__":
    main()
