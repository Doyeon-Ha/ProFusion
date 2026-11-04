#!/usr/bin/env python
"""Per-variant pLM LLR. Two conventions are supported.

    masked-marginal    mask the position and read it back. One forward per position.
    wildtype-marginal  one forward over the wild-type sequence, reading the log-softmax at the
                       position.

⚠ **The convention is fixed per backbone.** The fusion weights were fitted under that convention,
   so computing the LLR another way gives perfectly reasonable-looking numbers that are simply
   wrong. Do not change the defaults.

    ESM2   masked-marginal    cache key llr_ESM2mask. This is the official ProteinGym convention,
                              and it reproduces the official per-DMS rho to within 0.001 on
                              121 of 147 assays (wildtype-marginal manages 6 of 147).
    ESM1v  wildtype-marginal  cache key llr_ESM1v. The mean over 5 models (the standard ProteinGym
                              ensemble). Its learned positional embedding caps it at 1022 residues.

The computation matches the evaluation path: the masking loop of the masked-marginal builder, and
the wildtype-marginal scorer. Both were copied here so that the package stands on its own.

A GPU is used when present and it falls back to CPU otherwise. masked runs one forward per
**position that has a variant**, so it is correspondingly slower than wildtype: about 10 minutes on
CPU for 300 positions.

run: python inference/steps/llr.py --seq p.fasta --variants v.txt --backbone ESM2 --out llr.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MODELS = {
    "ESM2": ["facebook/esm2_t33_650M_UR50D"],
    "ESM1v": [f"facebook/esm1v_t33_650M_UR90S_{i}" for i in range(1, 6)],
}
MAXLEN = {"ESM1v": 1022}
# Backbone -> the convention it was fitted under. Changing it makes the fusion weights refer to a different quantity.
CONVENTION = {"ESM2": "masked", "ESM1v": "wildtype"}


def _load_mlm(name: str):
    """Load an EsmForMaskedLM.

    ESM-1v has no safetensors on the hub, only `.bin`, and transformers 4.52 refuses to load `.bin`
    under torch<2.6 because of CVE-2025-32434. In that case an empty model is built from the config
    and the state_dict loaded directly (with weights_only=True, so no arbitrary-code path is taken).
    """
    from transformers import EsmConfig, EsmForMaskedLM
    try:
        return EsmForMaskedLM.from_pretrained(name)
    except ValueError as e:
        if "torch.load" not in str(e):
            raise
    from huggingface_hub import hf_hub_download
    cfg = EsmConfig.from_pretrained(name)
    sd = torch.load(hf_hub_download(name, "pytorch_model.bin"), map_location="cpu",
                    weights_only=True)
    model = EsmForMaskedLM(cfg)
    missing, _ = model.load_state_dict(sd, strict=False)
    assert all(k.startswith("esm.contact_head.") for k in missing), f"unexpected missing keys: {missing}"
    return model


def _score_masked(seq: str, muts: list[str], model, tok, device: str,
                  batch: int = 8) -> np.ndarray:
    """masked-marginal. For each position that has a variant, replace it with <mask> and run one
    forward.

    Substitutions at the same position share a forward, so the cost scales with **the number of
    positions**, not the number of variants. Because of the BOS token, the token index equals the
    1-based position number.
    """
    enc = tok(seq, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    pos = sorted({int(m[1:-1]) for m in muts})
    if pos[0] < 1 or pos[-1] >= ids.shape[1]:
        raise ValueError(f"a variant position is outside the token range (T={ids.shape[1]})")
    lp_at = {}
    for i in range(0, len(pos), batch):
        chunk = pos[i:i + batch]
        b = ids.repeat(len(chunk), 1)
        for j, q in enumerate(chunk):
            b[j, q] = tok.mask_token_id
        with torch.no_grad():
            lg = model(input_ids=b, attention_mask=torch.ones_like(b)).logits
        lp = torch.log_softmax(lg.float(), dim=-1).cpu()
        for j, q in enumerate(chunk):
            lp_at[q] = lp[j, q]
        print(f"    position {min(i + batch, len(pos))}/{len(pos)}", flush=True)
    s = np.array([float(lp_at[int(m[1:-1])][tok.convert_tokens_to_ids(m[-1])]
                        - lp_at[int(m[1:-1])][tok.convert_tokens_to_ids(m[0])]) for m in muts],
                 dtype=np.float32)
    if not np.isfinite(s).all():
        raise ValueError("the LLR contains NaN or Inf")
    return s


def _score_one(seq: str, muts: list[str], model, tok, device: str) -> np.ndarray:
    """wildtype-marginal. One forward over the wild-type, reading the log-softmax at the variant positions only."""
    enc = {k: v.to(device) for k, v in tok(seq, return_tensors="pt").items()}
    with torch.no_grad():
        logits = model(**enc).logits[0].cpu()           # [T, vocab]
    pos = torch.tensor([int(m[1:-1]) for m in muts], dtype=torch.long)   # 1-based
    wt_ids = torch.tensor([tok.convert_tokens_to_ids(m[0]) for m in muts], dtype=torch.long)
    mt_ids = torch.tensor([tok.convert_tokens_to_ids(m[-1]) for m in muts], dtype=torch.long)
    tok_idx = pos                                       # offset=1 (CLS), so the 1-based position is the token index
    if int(tok_idx.min()) < 0 or int(tok_idx.max()) >= logits.shape[0]:
        raise ValueError(f"a variant position is outside the token range (T={logits.shape[0]})")
    lp = torch.log_softmax(logits[tok_idx], dim=-1)
    s = (lp.gather(1, mt_ids.view(-1, 1)) - lp.gather(1, wt_ids.view(-1, 1))).squeeze(1).float()
    if not torch.isfinite(s).all():
        raise ValueError("the LLR contains NaN or Inf")
    return s.numpy()


def compute(seq: str, muts: list[str], backbone: str = "ESM2",
            device: str | None = None, convention: str | None = None,
            batch: int = 8) -> np.ndarray:
    """Per-variant LLR. ESM1v averages over its 5 models.

    Without an explicit convention, the backbone's fitted convention (CONVENTION) is used. There is
    almost never a reason to change it.
    """
    if backbone not in MODELS:
        raise ValueError(f"backbone must be one of {sorted(MODELS)} (structure backbones are separate). Got: {backbone!r}")
    lim = MAXLEN.get(backbone)
    if lim and len(seq) > lim:
        raise ValueError(f"{backbone} caps at {lim} residues but the sequence is {len(seq)} long. "
                         "Use ESM2 or cut the domain out.")
    for m in muts:
        if seq[int(m[1:-1]) - 1] != m[0]:
            raise ValueError(f"the wild-type residue of variant {m} differs from the sequence "
                             f"(position {int(m[1:-1])} is {seq[int(m[1:-1]) - 1]})")
    conv = convention or CONVENTION[backbone]
    if conv not in ("masked", "wildtype"):
        raise ValueError(f"convention must be 'masked' or 'wildtype'. Got: {conv!r}")
    if conv != CONVENTION[backbone]:
        print(f"  ⚠ {backbone} was fitted under {CONVENTION[backbone]} but is being scored as {conv}. "
              "The fused score will be wrong.", flush=True)
    from transformers import AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = []
    for name in MODELS[backbone]:
        model = _load_mlm(name).to(device).eval()
        tok = AutoTokenizer.from_pretrained(name)
        if conv == "masked":
            out.append(_score_masked(seq, muts, model, tok, device, batch))
        else:
            out.append(_score_one(seq, muts, model, tok, device))
        del model
        print(f"  [{backbone}] {name} done ({conv})", flush=True)
    return np.mean(out, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seq", required=True, help="FASTA file")
    ap.add_argument("--variants", required=True, help="one W12A per line")
    ap.add_argument("--backbone", default="ESM2", choices=sorted(MODELS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--convention", default=None, choices=["masked", "wildtype"],
                    help="defaults to the backbone's fitted convention (ESM2=masked, ESM1v=wildtype). Do not change it")
    ap.add_argument("--batch", type=int, default=8, help="how many positions to mask at once under the masked convention")
    ap.add_argument("--out", required=True)
    A = ap.parse_args()
    seq = "".join(l.strip() for l in Path(A.seq).read_text().splitlines() if not l.startswith(">"))
    muts = [l.strip() for l in Path(A.variants).read_text().splitlines() if l.strip()]
    s = compute(seq, muts, A.backbone, A.device, A.convention, A.batch)
    pd.DataFrame(dict(mutant=muts, llr=s)).to_csv(A.out, index=False)
    print(f"[saved] {A.out}  {len(muts):,} variants, median LLR {np.median(s):.3f}")


if __name__ == "__main__":
    main()
