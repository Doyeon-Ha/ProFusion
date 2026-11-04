#!/usr/bin/env python
"""Attach SPURS ddG values to the ProteinGym substitution variants.

SPURS (Li and Luo, Nat Commun 2025, https://github.com/luo-group/SPURS) combines a ProteinMPNN
encoder, ESM2-650M with a structural adapter and an MLP head to produce a ddG matrix over every
position x 20 amino acids from one structure in a single forward pass.
Sign convention: **positive = destabilising**, negative = stabilising, the wild-type residue 0.

Stages

matrix : one forward per UniProt_ID, storing the [L,20] ddG matrix as npz (186 of them)
map    : matrix to a per-DMS variant CSV. Single substitutions are looked up; multiples are the
         additive sum of the single ddG values
multi  : (optional) predict multi-variant ddG directly with SPURS-multi, including epistasis

Coordinate mapping (verified beforehand)

The ProteinGym AF2 structures (data/proteingym/ProteinGym_AF2_structures/{UniProt_ID}.pdb) are
chain A with residues numbered 1..L contiguously, and their sequence matches
target_seq[pdb_range] exactly (185 of 186 match completely; only P53_HUMAN differs at the one
P72R polymorphism position, which is overwritten with target_seq).
So row0 = mutant_pos - pdb_range_start.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA2IDX = {a: i for i, a in enumerate(ALPHABET)}


# --------------------------------------------------------------------------- #
# PDB to model input
# --------------------------------------------------------------------------- #
def _move(obj, device):
    if isinstance(obj, dict):
        return {k: _move(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move(v, device) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    return obj


def build_batch(pdb_path: str, name: str, chain: str, cfg, device, seq_override=None):
    """The same as spurs.inference.parse_pdb, but the WT sequence can be overridden.

    seq_override is the same hook the SPURS enzyme dataset uses
    (get_pdb(check_assert=False) adopts pdb['seq'] as the WT as it is).
    """
    from spurs.datamodules.datasets.data_utils import Alphabet
    from spurs.datamodules.datasets.utils import alt_parse_PDB, get_pdb

    rec = alt_parse_PDB(pdb_path, chain)[0]
    if seq_override is not None:
        if len(seq_override) != len(rec["seq"]):
            raise ValueError(
                f"{name}: seq_override len {len(seq_override)} != PDB len {len(rec['seq'])}"
            )
        rec["seq"] = seq_override
        rec[f"seq_chain_{chain}"] = seq_override

    prot = get_pdb(rec, name, name, check_assert=False)
    prot["mut_ids"] = torch.tensor([0])
    prot["ddG"] = torch.tensor([[0]])
    prot["append_tensors"] = torch.tensor([0])
    prot["dataset"] = ["0"]

    batch = Alphabet(**cfg["datamodule"]["alphabet"]).featurize([prot])
    return _move(batch, device)


# --------------------------------------------------------------------------- #
# The ddG matrix (a memory-efficient version of SPURS.forward(return_logist=True))
# --------------------------------------------------------------------------- #
@torch.no_grad()
def ddg_matrix(model, batch, chunk: int = 512) -> np.ndarray:
    """[L, 20] ddG. Numerically identical to the original return_logist path under eval().

    The original replicates the representation 20 times per position to build [1, 20L, H] and then
    picks only its own mt column from each row. Under eval(), with dropout off, all 20 copies give
    the same logit, so passing one row per position gives the same result with 20 times less
    memory. (That difference is decisive on long proteins such as BRCA2 at 2832 aa.)
    """
    feats = model.forward_encoder(batch)[:, :, : model.input_dim]
    encoder_out = {"feats": F.pad(feats, (0, 0, 1, 1))}
    rep = model.decoder(tokens=batch["tokens"], encoder_out=encoder_out)["representations"][-1]
    if model.cfg.mlp.flat_dim > 0:
        rep = F.gelu(model.dp(model.flat_layers(rep)))
    rep = torch.cat([rep, encoder_out["feats"]], dim=-1)  # [1, L+2, H]

    seq = batch["seq"] if isinstance(batch["seq"], str) else batch["seq"][0]
    L = len(seq)
    out = np.empty((L, 20), dtype=np.float32)
    for s in range(0, L, chunk):
        e = min(s + chunk, L)
        logits = model.mlp({"muted_id_representation": rep[:, s + 1 : e + 1]}).squeeze(0)  # [n,21]
        wt = torch.tensor([AA2IDX[a] for a in seq[s:e]], device=logits.device)
        wt_val = logits.gather(-1, wt.view(-1, 1))
        out[s:e] = (logits[:, :20] - wt_val).float().cpu().numpy()
    return out


# --------------------------------------------------------------------------- #
# stage: matrix
# --------------------------------------------------------------------------- #
def stage_matrix(args):
    from spurs.inference import get_SPURS, get_SPURS_from_hub

    ref = pd.read_csv(args.dms_ref)
    proteins = ref.drop_duplicates("UniProt_ID")[["UniProt_ID", "target_seq", "pdb_range"]]
    if args.uniprot_ids:
        keep = set(args.uniprot_ids)
        proteins = proteins[proteins.UniProt_ID.isin(keep)]

    out_dir = Path(args.matrix_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if args.ckpt_path:
        model, cfg = get_SPURS(args.ckpt_path, device=device)
    else:
        model, cfg = get_SPURS_from_hub(device=device)
    model.eval()  # required, since dropout is 0.5

    todo = []
    for _, r in proteins.iterrows():
        dst = out_dir / f"{r.UniProt_ID}.npz"
        if dst.exists() and not args.overwrite:
            continue
        todo.append((r, dst))
    print(f"[matrix] {len(todo)} / {len(proteins)} proteins to run", flush=True)

    failed = []
    for i, (r, dst) in enumerate(todo, 1):
        pdb_path = Path(args.struct_dir) / f"{r.UniProt_ID}.pdb"
        lo, hi = (int(x) for x in str(r.pdb_range).split("-"))
        seq = r.target_seq[lo - 1 : hi]
        try:
            batch = build_batch(pdb_path.as_posix(), r.UniProt_ID, args.chain, cfg, device, seq)
            mat = ddg_matrix(model, batch, chunk=args.chunk)
            np.savez_compressed(
                dst, ddg=mat, seq=seq, pdb_range=np.array([lo, hi]), alphabet=ALPHABET
            )
            print(
                f"[matrix {i}/{len(todo)}] {r.UniProt_ID} L={len(seq)} "
                f"mean={mat.mean():+.3f} → {dst.name}",
                flush=True,
            )
        except Exception as exc:  # skip OOM and parse failures and keep going
            failed.append((r.UniProt_ID, len(seq), repr(exc)))
            print(f"[matrix {i}/{len(todo)}] FAIL {r.UniProt_ID} L={len(seq)}: {exc}", flush=True)
        finally:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    if failed:
        print("\n[matrix] failures:")
        for u, l, e in failed:
            print(f"  {u} (L={l}): {e}")
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# stage: map
# --------------------------------------------------------------------------- #
def _split(mutant: str):
    return [(s[0], int(s[1:-1]), s[-1]) for s in mutant.split(":")]


def stage_map(args):
    ref = pd.read_csv(args.dms_ref)
    if args.dms_ids:
        ref = ref[ref.DMS_id.isin(set(args.dms_ids))]
    mat_dir, out_dir = Path(args.matrix_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[str, tuple[np.ndarray, int, str]] = {}
    summary = []
    for _, r in ref.iterrows():
        dst = out_dir / f"{r.DMS_id}.csv"
        if dst.exists() and not args.overwrite:
            continue
        npz = mat_dir / f"{r.UniProt_ID}.npz"
        if not npz.exists():
            print(f"[map] SKIP {r.DMS_id}: no matrix for {r.UniProt_ID}", flush=True)
            continue
        if r.UniProt_ID not in cache:
            z = np.load(npz, allow_pickle=True)
            cache[r.UniProt_ID] = (z["ddg"], int(z["pdb_range"][0]), str(z["seq"]))
        mat, lo, seq = cache[r.UniProt_ID]

        dms = pd.read_csv(Path(args.dms_dir) / r.DMS_filename)
        vals, nsub, bad = [], [], 0
        for mu in dms.mutant:
            subs = _split(mu)
            tot, ok = 0.0, True
            for wt, pos, mt in subs:
                i = pos - lo
                if not (0 <= i < mat.shape[0]) or seq[i] != wt or mt not in AA2IDX:
                    ok = False
                    break
                tot += float(mat[i, AA2IDX[mt]])
            if ok:
                vals.append(tot)
            else:
                vals.append(np.nan)
                bad += 1
            nsub.append(len(subs))

        out = pd.DataFrame(
            {
                "mutant": dms.mutant,
                "spurs_ddg": vals,
                "n_sub": nsub,
                "DMS_score": dms.DMS_score if "DMS_score" in dms else np.nan,
            }
        )
        out.to_csv(dst, index=False)

        sing = out[(out.n_sub == 1) & out.spurs_ddg.notna()]
        rho = (
            sing.spurs_ddg.corr(sing.DMS_score, method="spearman")
            if len(sing) > 10 and sing.DMS_score.notna().any()
            else np.nan
        )
        summary.append(
            dict(
                DMS_id=r.DMS_id,
                UniProt_ID=r.UniProt_ID,
                n=len(out),
                n_single=int((out.n_sub == 1).sum()),
                n_unmapped=bad,
                spearman_single=rho,
                coarse_selection_type=r.get("coarse_selection_type"),
            )
        )
        print(f"[map] {r.DMS_id}: n={len(out)} unmapped={bad} rho_single={rho:+.3f}", flush=True)

    if summary:
        s = pd.DataFrame(summary)
        s.to_csv(out_dir / "_summary.csv", index=False)
        print("\n[map] Spearman(SPURS ddG, DMS_score), single substitutions only")
        print(s.spearman_single.describe().to_string())
        print(
            "\nSign check: ddG is positive for destabilising, so rho should be negative on most DMS.\n"
            "As a zero-shot predictor, use -spurs_ddg."
        )
        print("\n" + s.groupby("coarse_selection_type").spearman_single.mean().to_string())
    return 0


# --------------------------------------------------------------------------- #
# stage: multi (SPURS-multi, including epistasis)
# --------------------------------------------------------------------------- #
def stage_multi(args):
    from spurs.inference import get_SPURS_multi_from_hub

    ref = pd.read_csv(args.dms_ref)
    ref = ref[ref.DMS_number_multiple_mutants > 0]
    if args.dms_ids:
        ref = ref[ref.DMS_id.isin(set(args.dms_ids))]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    model, cfg = get_SPURS_multi_from_hub(device=device)
    model.eval()

    for _, r in ref.iterrows():
        dst = out_dir / f"{r.DMS_id}.multi.csv"
        if dst.exists() and not args.overwrite:
            continue
        lo, hi = (int(x) for x in str(r.pdb_range).split("-"))
        seq = r.target_seq[lo - 1 : hi]
        batch = build_batch(
            (Path(args.struct_dir) / f"{r.UniProt_ID}.pdb").as_posix(),
            r.UniProt_ID, args.chain, cfg, device, seq,
        )

        dms = pd.read_csv(Path(args.dms_dir) / r.DMS_filename)
        muts = [m for m in dms.mutant if ":" in m]
        kmax = max(len(m.split(":")) for m in muts)
        rows, out_vals = [], []
        for m in muts:
            mi, at = [], []
            for wt, pos, mt in _split(m):
                mi.append(pos - lo)
                at += [AA2IDX[wt], AA2IDX[mt]]
            # Variable lengths are padded with -1; SPURSMulti handles them with a mut_ids > -1 mask
            mi += [-1] * (kmax - len(mi))
            at += [-1, -1] * (kmax - len(_split(m)))
            rows.append(mi)
            out_vals.append(at)

        preds = []
        with torch.no_grad():
            for s in range(0, len(rows), args.multi_batch):
                e = min(s + args.multi_batch, len(rows))
                batch["mut_ids"] = torch.tensor(rows[s:e], device=device)
                batch["append_tensors"] = torch.tensor(out_vals[s:e], device=device)
                preds.append(model(batch).reshape(-1).float().cpu().numpy())
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
        pd.DataFrame({"mutant": muts, "spurs_multi_ddg": np.concatenate(preds)}).to_csv(
            dst, index=False
        )
        print(f"[multi] {r.DMS_id}: {len(muts)} multi-mutants → {dst.name}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("PROJECT_ROOT", root.as_posix())

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["matrix", "map", "multi"], required=True)
    p.add_argument("--dms-ref", default=str(root / "data/proteingym/DMS_substitutions.csv"))
    p.add_argument("--dms-dir", default=str(root / "data/proteingym/DMS_ProteinGym_substitutions"))
    p.add_argument("--struct-dir", default=str(root / "data/proteingym/ProteinGym_AF2_structures"))
    p.add_argument("--matrix-dir", default=str(root / "data/proteingym/spurs/matrix"))
    p.add_argument("--out-dir", default=str(root / "data/proteingym/spurs/dms"))
    p.add_argument("--ckpt-path", default=None, help="a local checkpoint directory (empty means the HF hub)")
    p.add_argument("--chain", default="A")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--chunk", type=int, default=512, help="matrix: the position chunk for the MLP head")
    p.add_argument("--multi-batch", type=int, default=2048)
    p.add_argument("--uniprot-ids", nargs="*", default=None)
    p.add_argument("--dms-ids", nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    return {"matrix": stage_matrix, "map": stage_map, "multi": stage_multi}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
