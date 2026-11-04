#!/usr/bin/env python
"""Structure plus sequence to a SPURS ddG matrix (L x 20). CPU is enough.

ddG is **positive = destabilising**. The sign flip happens when the channel is built, so the raw
values are stored here.

⚠ This needs a separate environment. Fetch the SPURS repository and checkpoint and run with that
   environment's python. The default paths are overridden with --spurs-root and --ckpt.

It uses the same code path as the evaluation set: `get_SPURS` from the SPURS repository, plus
build_batch and ddg_matrix from the ProteinGym scoring script. A copy of that script is in
pipeline/upstream/experiments/spurs_ddg/.
"""
from __future__ import annotations

import os
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = HERE / "pipeline/upstream/experiments/spurs_ddg"


def compute(pdb: str | Path, seq: str, ckpt: str, runner_dir: str | Path = DEFAULT_RUNNER,
            chain: str = "A", device: str | None = None):
    """An (L, 20) ddG matrix. The column order is AA20 = ACDEFGHIKLMNPQRSTVWY."""
    import torch
    sys.path.insert(0, str(runner_dir))
    try:
        from run_spurs_proteingym import ALPHABET, build_batch, ddg_matrix
    except ImportError as e:
        raise ImportError(
            f"could not import run_spurs_proteingym from {runner_dir}: {e}\n"
            "Pass --runner-dir pointing at the directory that holds that file.") from e
    from spurs.inference import get_SPURS

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = get_SPURS(ckpt, device=device)
    model.eval()                                    # required, since dropout is 0.5
    print(f"  [SPURS] {device}", flush=True)
    mat = ddg_matrix(model, build_batch(str(pdb), "x", chain, cfg, device, seq), chunk=512)
    if mat.shape != (len(seq), 20):
        raise ValueError(f"ddG shape {mat.shape} differs from ({len(seq)}, 20)")
    # The ddG at a wild-type residue is 0 by definition, which catches a coordinate mismatch here
    wt_vals = mat[np.arange(len(seq)), [ALPHABET.index(a) for a in seq]]
    if np.abs(wt_vals).max() >= 1e-4:
        raise ValueError(f"ddG at wild-type positions is not 0 (max {np.abs(wt_vals).max():.2e}). "
                         "The structure and the sequence are most likely out of register.")
    return mat, ALPHABET



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
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--seq", required=True, help="FASTA")
    ap.add_argument("--ckpt", default=os.environ.get("SPURS_CKPT"),
                    help="path to the SPURS checkpoint. Also read from SPURS_CKPT.")
    ap.add_argument("--runner-dir", default=str(DEFAULT_RUNNER))
    ap.add_argument("--chain", default="A")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    A = ap.parse_args()
    seq = "".join(l.strip() for l in Path(A.seq).read_text().splitlines() if not l.startswith(">"))
    mat, alphabet = compute(A.pdb, seq, _need_ckpt(A.ckpt), A.runner_dir, A.chain, A.device)
    np.savez_compressed(A.out, ddg=mat, seq=seq, alphabet=alphabet,
                        pdb_range=np.array([1, len(seq)]))
    print(f"[saved] {A.out}  L={len(seq)}  median ddG {np.median(mat):.3f} "
          f"range [{mat.min():.2f}, {mat.max():.2f}]")


if __name__ == "__main__":
    main()
