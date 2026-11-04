#!/usr/bin/env python
"""Score one new protein with comp7. The MSA and the structure are supplied, not computed here.

What it does
    1. decide the variant list (default: every position x 19 substitutions, or a --variants file)
    2. build the four channels: GEMME (docker), RSA (DSSP), ddG (SPURS) and LLR (pLM)
    3. map the assay type to an axis and read **the fitted weights** for that axis and backbone
       (no labels needed)
    4. compute s = z(LLR) + CH @ w and write a per-variant table

Each step caches into the --out directory, so rerunning after a failure skips what already
finished.

Only the ddG step needs a separate environment (SPURS). Run that step on its own and pass the
result with --ddg, and this script runs end to end in the main environment.

How to read the output
    A **larger** fused value means closer to wild type (more tolerated). deleterious_rank 1 is the
    most damaging.
    The scores are meaningful only as **a ranking within this protein**. Do not compare across
    proteins.
"""
from __future__ import annotations

import os
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from plmfusion.predict import AA20, Channels, axis_of, load_weights, predict_table

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:      # so that steps is found when imported as a module too
    sys.path.insert(0, str(HERE))


def read_fasta(p: str | Path) -> str:
    seq = "".join(l.strip() for l in Path(p).read_text().splitlines() if not l.startswith(">"))
    bad = sorted(set(seq) - set(AA20))
    if bad:
        raise ValueError(f"the sequence contains characters outside the standard 20 amino acids: {bad}")
    return seq


def all_substitutions(seq: str) -> list[str]:
    """Every position x 19 substitutions."""
    return [f"{wt}{i}{mu}" for i, wt in enumerate(seq, 1) for mu in AA20 if mu != wt]



def _need_ckpt(path):
    """Stop with a recognisable message when the checkpoint is missing. Does not block --help."""
    if not path:
        raise SystemExit(
            "No SPURS checkpoint path. Set the SPURS_CKPT environment variable or pass it as an argument.\n"
            "  for example: export SPURS_CKPT=/path/to/SPURS/data/checkpoints/spurs")
    if not Path(path).exists():
        raise SystemExit(f"No SPURS checkpoint at that path: {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seq", required=True, help="wild-type sequence FASTA")
    ap.add_argument("--a3m", required=True, help="MSA (hhblits a3m). The query is the first sequence")
    ap.add_argument("--pdb", required=True, help="structure. Residue number = 1-based sequence position")
    ap.add_argument("--assay", required=True,
                    help="assay type: Activity, Binding, OrganismalFitness or Expression")
    ap.add_argument("--backbone", default="ESM2", choices=["ESM2", "ESM1v"])
    ap.add_argument("--variants", default=None, help="one W12A per line. Default is every position x 19 substitutions")
    ap.add_argument("--ddg", default=None, help="a precomputed ddG npz (when run in a separate environment)")
    ap.add_argument("--spurs-ckpt", default=os.environ.get("SPURS_CKPT"),
                    help="path to the SPURS checkpoint. Also read from SPURS_CKPT.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    A = ap.parse_args()
    if A.ddg is None:
        _need_ckpt(A.spurs_ckpt)   # only needed when ddG is computed here

    out = Path(A.out)
    out.mkdir(parents=True, exist_ok=True)
    axis = axis_of(A.assay)
    w = load_weights(axis, A.backbone)
    seq = read_fasta(A.seq)
    muts = ([l.strip() for l in Path(A.variants).read_text().splitlines() if l.strip()]
            if A.variants else all_substitutions(seq))
    print(f"[config] assay {A.assay} -> axis {axis}, backbone {A.backbone}, length {len(seq)}, "
          f"{len(muts):,} variants")
    print(f"[weights] {dict(zip(['a_GEMME', 'b_GEMME', 'a_ddG', 'b_ddG', 'a_RSA', 'b_BLOSUM', 'b_dchg'], w))}")
    (out / "variants.txt").write_text("\n".join(muts) + "\n")

    # --- 1. RSA ------------------------------------------------------------
    f_rsa = out / "rsa.json"
    if not f_rsa.exists():
        from steps.rsa import compute as rsa_compute
        f_rsa.write_text(json.dumps({str(k): v for k, v in rsa_compute(A.pdb, seq).items()}))
    rsa_map = {int(k): v for k, v in json.loads(f_rsa.read_text()).items()}
    print(f"[1/4] RSA over {len(rsa_map)} positions")

    # --- 2. GEMME ----------------------------------------------------------
    f_gem = out / "gemme.csv"
    if not f_gem.exists():
        from steps.gemme import compute as gemme_compute
        s = gemme_compute(A.a3m, muts, out / "gemme_work")
        pd.DataFrame(dict(mutant=muts, gemme=s.to_numpy())).to_csv(f_gem, index=False)
    gem = pd.read_csv(f_gem).set_index("mutant").gemme.reindex(muts).to_numpy()
    print(f"[2/4] GEMME missing {int(np.isnan(gem).sum())}/{len(muts)}")

    # --- 3. ΔΔG ------------------------------------------------------------
    f_ddg = Path(A.ddg) if A.ddg else out / "ddg.npz"
    if not f_ddg.exists():
        from steps.ddg import compute as ddg_compute
        mat, alphabet = ddg_compute(A.pdb, seq, A.spurs_ckpt, device=A.device)
        np.savez_compressed(f_ddg, ddg=mat, seq=seq, alphabet=alphabet)
    zn = np.load(f_ddg, allow_pickle=True)
    if str(zn["alphabet"]) != AA20:
        raise ValueError(f"the ddG alphabet differs: {zn['alphabet']}")
    if str(zn["seq"]) != seq:
        raise ValueError("the sequence the ddG was built from differs from --seq")
    DDG = zn["ddg"].astype(np.float32)
    print(f"[3/4] ΔΔG {DDG.shape}")

    # --- 4. LLR ------------------------------------------------------------
    f_llr = out / f"llr_{A.backbone}.csv"
    if not f_llr.exists():
        from steps.llr import compute as llr_compute
        pd.DataFrame(dict(mutant=muts, llr=llr_compute(seq, muts, A.backbone, A.device))
                     ).to_csv(f_llr, index=False)
    llr = pd.read_csv(f_llr).set_index("mutant").llr.reindex(muts).to_numpy()
    print(f"[4/4] median LLR {np.median(llr):.3f}")

    # --- assemble ----------------------------------------------------------
    pos1 = np.array([int(m[1:-1]) for m in muts])
    wt = np.array([m[0] for m in muts])
    mut = np.array([m[-1] for m in muts])
    rsa_v = np.array([rsa_map.get(int(p), np.nan) for p in pos1])
    keep = ~np.isnan(rsa_v) & ~np.isnan(llr)
    if keep.sum() < len(muts):
        print(f"[excluded] {len(muts) - int(keep.sum())} variants with a missing RSA or LLR "
              "(positions absent from the structure). Channel normalisation uses the remaining set")
    if keep.sum() < 100:
        raise ValueError(f"only {int(keep.sum())} variants remain: channel normalisation is unstable")
    ddg_v = DDG[pos1[keep] - 1, [AA20.index(a) for a in mut[keep]]]
    ch = Channels(pos1=pos1[keep], wt=wt[keep], mut=mut[keep], gemme=gem[keep],
                  rsa=rsa_v[keep], ddg=ddg_v, llr=llr[keep])
    T = predict_table(ch, w)
    T.insert(0, "axis", axis)
    T.insert(0, "backbone", A.backbone)
    T.to_csv(out / "fused_scores.csv", index=False)
    print(f"\n[saved] {out / 'fused_scores.csv'}  {len(T):,} variants")
    print("\nthe 10 most damaging:")
    print(T.head(10)[["mutant", "fused", "llr", "gemme", "ddg", "rsa"]].to_string(index=False))


if __name__ == "__main__":
    main()
