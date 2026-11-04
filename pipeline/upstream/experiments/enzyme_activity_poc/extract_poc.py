"""Embedding extraction runner.

For the 40 DMS with phenotype == 'enz_activity' in the split (32 train + 8 test), extract
per-residue embeddings with a given backbone and store them in LMDB
(`{lmdb_root}/{backbone}/{dms_id}.lmdb`).

Multi-GPU: `--num-shards K --shard-index i` splits the DMS list into K parts and handles the i-th.

Idempotent: keys that are already stored are skipped, so it is safe to stop and rerun.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from cvm.backbones.base import BackboneAdapter  # noqa: E402
from cvm.extraction.extractor import extract_dms_to_lmdb  # noqa: E402
from cvm.splits.primo import load_primo_split  # noqa: E402

logger = logging.getLogger("extract_poc")

CONFIG_PATH = _REPO_ROOT / "configs" / "001_extraction.yaml"
SPLIT_PATH = _REPO_ROOT / "configs" / "data" / "primo_split.yaml"
PHENOTYPE = "enz_activity"

# Sources for the ESM3 multimodal context (structure = AF2 PDB, function = precomputed InterPro/keyword annotations)
META_CSV = _REPO_ROOT / "data" / "proteingym" / "DMS_substitutions.csv"
PDB_DIR = Path("<PROTEINGYM_DIR>/ProteinGym_AF2_structures")
FUNCTION_PKL = Path("<PROTEINGYM_DIR>/function_annotations_dict.pkl")


def enzyme_dms_ids() -> list[str]:
    """The enz_activity DMS ids of the split (train and test, in a deterministic order)."""
    sp = load_primo_split(SPLIT_PATH)
    train = [e.dms_id for e in sp.train if e.primo == PHENOTYPE]
    test = [e.dms_id for e in sp.test if e.primo == PHENOTYPE]
    return train + test


def balanced_shard(dms_ids: list[str], csv_dir: Path, num_shards: int, shard_index: int) -> list[str]:
    """Greedy LPT assignment by variant count (a proxy for cost), so that large DMS do not pile up
    on one GPU.

    Every process computes it deterministically from the same input, so shards neither overlap nor
    miss anything.
    """
    if num_shards == 1:
        return dms_ids
    costs = []
    for d in dms_ids:
        try:
            n = sum(1 for _ in open(csv_dir / f"{d}.csv")) - 1  # rows excluding the header
        except OSError:
            n = 1
        costs.append((max(1, n), d))
    costs.sort(reverse=True)  # largest first
    buckets: list[list[str]] = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for cost, d in costs:
        j = loads.index(min(loads))  # the least loaded shard
        buckets[j].append(d)
        loads[j] += cost
    return buckets[shard_index]


def build_backbone(name: str, cfg: dict, device: str) -> BackboneAdapter:
    """Build the adapter from the backbone definition in config. name is one of {esm2, prosst, esm3}."""
    bdef = cfg["backbones"][name]
    btype = bdef["type"]
    if btype == "esm3":
        from cvm.backbones.esm3 import ESM3Adapter

        return ESM3Adapter(
            model_name=bdef["model_name"], device=device, embedding_dim=bdef["embedding_dim"]
        )
    if btype == "prosst":
        from cvm.backbones.prosst import ProSSTAdapter

        return ProSSTAdapter(
            model_name=bdef["model_name"], name=name, device=device,
            embedding_dim=bdef.get("embedding_dim"),
        )
    if btype == "huggingface":
        from cvm.backbones.huggingface import HFBackboneAdapter

        return HFBackboneAdapter(
            model_name=bdef["model_name"], name=name, device=device,
            embedding_dim=bdef.get("embedding_dim"),
        )
    raise ValueError(f"unknown backbone type: {btype!r} (name={name!r})")


class ESM3ContextLoader:
    """Loader for the ESM3 multimodal WT context: atom37 coordinates and function annotations, per DMS."""

    def __init__(self) -> None:
        import pickle

        import pandas as pd

        # The esm module must be on the path to deserialise FunctionAnnotation from the precomputed pickle
        sys.path.insert(0, str(_REPO_ROOT / "external" / "esm3"))
        self._uniprot = pd.read_csv(META_CSV).set_index("DMS_id")["UniProt_ID"].to_dict()
        with open(FUNCTION_PKL, "rb") as f:
            self._functions = pickle.load(f)

    def load(self, dms_id: str) -> dict:
        """Returns `{coordinates: [L,37,3] float32, function_annotations: [...]}`."""
        import torch
        sys.path.insert(0, str(_REPO_ROOT / "external" / "esm3"))
        from esm.utils.structure.protein_chain import ProteinChain

        uni = self._uniprot[dms_id]
        pdb = PDB_DIR / f"{uni}.pdb"
        chain = ProteinChain.from_pdb(str(pdb))
        coords = torch.tensor(chain.atom37_positions, dtype=torch.float32)
        anns = self._functions.get(dms_id)
        if anns is None:
            logger.warning("DMS %s: no function annotation, using structure only", dms_id)
        return {"coordinates": coords, "function_annotations": anns}


class ProSSTContextLoader:
    """Loader for the WT quantised structure tokens used by structure-aware ProSST extraction, per DMS.

    Reads the pre-quantised output of the ProSST ProteinGym benchmark
    (`structure_sequence/2048/{dms_id}.fasta`, comma-separated integers, one structure token per
    residue) and returns the raw token list. Every variant of a DMS shares the WT structure (single
    SAV), so it is loaded once per DMS.
    """

    def __init__(self, structure_sequence_root: Path) -> None:
        self._root = Path(structure_sequence_root)
        if not self._root.is_dir():
            raise FileNotFoundError(f"no ProSST structure-token root: {self._root}")

    def load(self, dms_id: str) -> dict:
        """Returns `{structure_tokens: [int, ...]}` (passed through extract_kwargs)."""
        fasta = self._root / f"{dms_id}.fasta"
        if not fasta.exists():
            raise FileNotFoundError(f"no structure-token fasta for DMS {dms_id}: {fasta}")
        body = "".join(
            line.strip() for line in fasta.read_text().splitlines() if not line.startswith(">")
        )
        tokens = [int(t) for t in body.split(",") if t != ""]
        return {"structure_tokens": tokens}


def estimate_map_size_gb(csv_path: Path, embedding_dim: int) -> tuple[float, int]:
    """Estimate the initial LMDB map_size (GB) and the maximum sequence length L from the DMS CSV.

    map_size is about n_rows x L x d x 2 bytes x 1.4 (at least 2 GB). The writer doubles it
    automatically on MapFullError, so being conservative is safe; only over-estimation is avoided.
    L is reused for the length-aware batch size (`effective_batch_size`).
    """
    df = pd.read_csv(csv_path, usecols=["mutated_sequence"])
    n = len(df) + 1  # +WT
    seq_len = int(df["mutated_sequence"].str.len().max())
    nbytes = n * seq_len * embedding_dim * 2 * 1.4
    return max(2.0, nbytes / (1024**3)), seq_len


# ESM3 (1.4B, multitrack) is dominated by attention at B x L^2, so a fixed batch_size of 8 runs out
# of memory on large proteins. Measured on an A100-80GB: multimodal 2185aa bs2 = 51GB OK, bs4 OOM;
# sequence 3423aa bs1 = 61GB OK, bs2 OOM. The safe threshold is B x L^2 <= about 1.1e7. esm2 and
# prosst still have room at bs8 (at most 28GB) even at 3423aa, so this does not apply to them.
ESM3_BL2_BUDGET = 1.1e7


def effective_batch_size(backbone_name: str, seq_len: int, base_bs: int) -> int:
    """A safe batch size for the backbone and sequence length. Only ESM3 is scaled down by length; the others keep base."""
    if backbone_name != "esm3":
        return base_bs
    bs = int(ESM3_BL2_BUDGET // (seq_len * seq_len))
    return max(1, min(base_bs, bs))


def main() -> int:
    ap = argparse.ArgumentParser(description="Embedding extraction")
    ap.add_argument("--backbone", required=True, choices=["esm2", "prosst", "esm3"])
    ap.add_argument("--device", default="cuda:0", help="torch device (together with CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N DMS (for smoke tests)")
    ap.add_argument("--dms-ids", nargs="*", default=None, help="an explicit list of DMS ids (ignores the split)")
    ap.add_argument("--lmdb-root", default=None,
                    help="override the LMDB output root (default: lmdb_root from the extraction config). "
                         "Used to extract a few proteins locally when the default root is a broken link.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{args.backbone} sh{args.shard_index}/{args.num_shards}] %(message)s",
    )

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    lmdb_root = Path(args.lmdb_root) if args.lmdb_root else _REPO_ROOT / cfg["lmdb_root"]
    if not lmdb_root.is_absolute():
        lmdb_root = _REPO_ROOT / lmdb_root
    csv_dir = _REPO_ROOT / cfg["proteingym_root"]

    all_ids = args.dms_ids if args.dms_ids else enzyme_dms_ids()
    shard_ids = balanced_shard(all_ids, csv_dir, args.num_shards, args.shard_index)
    if args.limit:
        shard_ids = shard_ids[: args.limit]

    bdef = cfg["backbones"][args.backbone]
    emb_dim = bdef.get("embedding_dim") or 1280
    logger.info("processing %d DMS (of %d, shard %d/%d): %s",
                len(shard_ids), len(all_ids), args.shard_index, args.num_shards, shard_ids)

    logger.info("loading backbone: %s (%s) on %s", args.backbone, bdef["model_name"], args.device)
    t0 = time.time()
    backbone = build_backbone(args.backbone, cfg, args.device)
    logger.info("backbone loaded (%.1fs, embedding_dim=%d)", time.time() - t0, backbone.embedding_dim)

    ctx_loader = None
    if args.backbone == "esm3":
        ctx_loader = ESM3ContextLoader()
        logger.info("ESM3 multimodal mode: injecting the WT structure (atom37) and function annotations")
    elif args.backbone == "prosst":
        ctx_loader = ProSSTContextLoader(_REPO_ROOT / bdef["structure_sequence_root"])
        logger.info("ProSST structure-aware mode: injecting WT quantised structure tokens (ss_input_ids)")

    total_keys = 0
    skipped_dms: list[str] = []
    for i, dms_id in enumerate(shard_ids, 1):
        csv_path = csv_dir / f"{dms_id}.csv"
        if not csv_path.exists():
            logger.error("[%d/%d] no CSV, skipping: %s", i, len(shard_ids), csv_path)
            continue
        lmdb_path = lmdb_root / args.backbone / f"{dms_id}.lmdb"
        # Isolate a failure on one DMS (missing context, OOM and so on) so the whole shard does not stop.
        # For example a DMS with no ProSST structure-token fasta is skipped, since those tokens are required.
        try:
            map_gb, seq_len = estimate_map_size_gb(csv_path, backbone.embedding_dim)
            eff_bs = effective_batch_size(args.backbone, seq_len, args.batch_size)
            extract_kwargs = ctx_loader.load(dms_id) if ctx_loader else None
            t1 = time.time()
            logger.info("[%d/%d] %s starting (L=%d, map_size about %.1fGB, batch_size=%d)",
                        i, len(shard_ids), dms_id, seq_len, map_gb, eff_bs)
            n = extract_dms_to_lmdb(
                csv_path, dms_id, backbone, lmdb_path,
                batch_size=eff_bs, map_size_gb=map_gb,
                extract_kwargs=extract_kwargs,
            )
            total_keys += n
            logger.info("[%d/%d] %s done: %d new keys (%.1fs)", i, len(shard_ids), dms_id, n, time.time() - t1)
        except Exception:
            logger.exception("[%d/%d] %s failed, skipping", i, len(shard_ids), dms_id)
            skipped_dms.append(dms_id)
            continue

    logger.info("shard done: %d DMS, %d new keys, %.1f minutes total", len(shard_ids), total_keys, (time.time() - t0) / 60)
    if skipped_dms:
        logger.warning("%d DMS skipped: %s", len(skipped_dms), skipped_dms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
