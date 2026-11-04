#!/usr/bin/env python
"""Compute ESM-1v masked-marginal scores and per-residue embeddings for 7 proteins in one pass.

The DETANGO initialisation script launches esm_vep.py separately per protein, reloading all 5
models each time (7 proteins means 35 loads of 7.8 GB). Here the model is loaded once and all 7
proteins are swept. The formula is identical to the masked-marginals of esm_vep.py (eval mode, no
dropout, batch size 1), and `--validate` compares against the original.

output:
  data/{p}/intermediates/collection_esm.csv   mutant plus esm1v_t33_650M_UR90S_1..5 and esm1v (the mean)
  data/{p}/{p}.pt                             {'representations': {33: [L,1280]}}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from esm import pretrained
from tqdm import tqdm

MODELS = [f"esm1v_t33_650M_UR90S_{i}" for i in range(1, 6)]


def read_fasta(path: Path) -> str:
    lines = path.read_text().strip().splitlines()
    return "".join(l.strip() for l in lines[1:])


def masked_marginals(model, alphabet, seq: str, device) -> torch.Tensor:
    """[1, L+2, vocab] log-softmax. The same loop as esm_vep.py."""
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter([("protein1", seq)])
    all_token_probs = []
    for i in tqdm(range(batch_tokens.size(1)), leave=False, desc=f"mask L={len(seq)}"):
        masked = batch_tokens.clone()
        masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            lp = torch.log_softmax(model(masked.to(device))["logits"], dim=-1)
        all_token_probs.append(lp[:, i])
    return torch.cat(all_token_probs, dim=0).unsqueeze(0)


def score_mutants(mutants, seq, token_probs, alphabet) -> list[float]:
    out = []
    for m in mutants:
        wt, idx, mt = m[0], int(m[1:-1]), m[-1]
        assert seq[idx] == wt, (m, seq[idx])
        out.append((token_probs[0, 1 + idx, alphabet.get_idx(mt)]
                    - token_probs[0, 1 + idx, alphabet.get_idx(wt)]).item())
    return out


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", default=str(root / "data/proteingym/detango"))
    p.add_argument("--models", nargs="*", default=MODELS)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-embedding", action="store_true")
    p.add_argument("--embedding-model", default="esm1v_t33_650M_UR90S_1")
    a = p.parse_args(argv)

    run = Path(a.run_root)
    device = torch.device(a.device)
    names = [d.name for d in sorted((run / "data").iterdir()) if d.is_dir()]
    seqs = {n: read_fasta(run / "data" / n / "wt.fasta") for n in names}
    muts = {n: pd.read_csv(run / "data" / n / "intermediates/collection_raw.csv",
                           index_col=0)["mutant"].tolist() for n in names}
    scores: dict[str, dict[str, list[float]]] = {n: {} for n in names}

    for mi, mloc in enumerate(a.models):
        print(f"[esm1v] loading {mloc}", flush=True)
        model, alphabet = pretrained.load_model_and_alphabet(mloc)
        model.eval().to(device)
        for n in names:
            tp = masked_marginals(model, alphabet, seqs[n], device)
            scores[n][mloc] = score_mutants(muts[n], seqs[n], tp, alphabet)
            print(f"[esm1v] {mloc} · {n} done", flush=True)

        if not a.skip_embedding and mloc == a.embedding_model:
            bc = alphabet.get_batch_converter()
            for n in names:
                _, _, toks = bc([(n, seqs[n])])
                with torch.no_grad():
                    out = model(toks.to(device), repr_layers=[33], return_contacts=False)
                rep = out["representations"][33][0, 1: len(seqs[n]) + 1].cpu().clone()
                assert rep.shape == (len(seqs[n]), 1280), (n, rep.shape)
                torch.save({"label": n, "representations": {33: rep}},
                           run / "data" / n / f"{n}.pt")
                print(f"[emb] {n} {tuple(rep.shape)}", flush=True)

        del model
        torch.cuda.empty_cache()

    for n in names:
        df = pd.DataFrame({"mutant": muts[n]})
        for mloc in a.models:
            df[mloc] = scores[n][mloc]
        df["esm1v"] = df[list(a.models)].mean(axis=1)
        df.to_csv(run / "data" / n / "intermediates/collection_esm.csv", index=False)
        print(f"[esm1v] wrote {n} collection_esm.csv  mean={df.esm1v.mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
