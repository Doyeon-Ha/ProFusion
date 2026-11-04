"""Runner for the zero-shot pseudo-label computation.

For the 40 enz_activity DMS, compute the wildtype-marginal zero-shot scores of a given backbone and
store them under `data/proteingym/zeroshot/{Backbone}/{dms_id}.pt` (the cvm.labels.zeroshot schema).

Method (the same for every backbone, the ProteinGym standard):
    score_i = log P(mut_i | WT context) − log P(wt_i | WT context)   (one forward over the WT)

- ProSST: ProSSTForMaskedLM, sequence plus WT quantised structure tokens (ss_input_ids), the same as
          the official benchmark.
- ESM3  : sequence-track logits. With --esm3-multimodal the WT structure (atom37) and function are
          encoded as context (the same modality as the embedding extraction); the default is
          sequence-only.

Idempotent: an existing (backbone, dms_id) cache is skipped.
Multi-GPU: --num-shards K --shard-index i splits the DMS into K parts.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch  # noqa: E402

from cvm.data.proteingym import collect_variants, derive_wt_sequence, load_dms_csv  # noqa: E402
from cvm.labels import zeroshot as zs  # noqa: E402

# Reuses the helpers of extract_poc (DMS list, balanced sharding, context loaders)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_poc import (  # noqa: E402
    ESM3ContextLoader,
    ProSSTContextLoader,
    balanced_shard,
    enzyme_dms_ids,
)

logger = logging.getLogger("zeroshot_poc")

CSV_DIR = _REPO_ROOT / "data" / "proteingym" / "DMS_ProteinGym_substitutions"
CACHE_ROOT = _REPO_ROOT / "data" / "proteingym" / "zeroshot"
PROSST_STRUCT_ROOT = _REPO_ROOT / "data/proteingym/prosst_proteingym_benchmark/structure_sequence/2048"

# Runner backbone name to the cvm.labels.zeroshot cache backbone name
BACKBONE_LABEL = {"esm2": "ESM2", "esm3": "ESM3", "prosst": "ProSST"}


def load_esm2(device: str):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    name = zs.DEFAULT_ESM2_MODEL_NAME
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForMaskedLM.from_pretrained(name).to(device).eval()
    return model, tok, name


def load_prosst(device: str):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    name = zs.DEFAULT_PROSST_MODEL_NAME
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(name, trust_remote_code=True).to(device).eval()
    return model, tok, name


def load_esm3(device: str):
    sys.path.insert(0, str(_REPO_ROOT / "external" / "esm3"))
    from esm.models.esm3 import ESM3

    name = zs.DEFAULT_ESM3_MODEL_NAME
    model = ESM3.from_pretrained(name, device=torch.device(device)).eval()
    return model, None, name


def main() -> int:
    ap = argparse.ArgumentParser(description="zero-shot score computation")
    ap.add_argument("--backbone", required=True, choices=["esm2", "esm3", "prosst"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--esm3-multimodal", action="store_true",
                    help="encode the ESM3 zero-shot with structure and function context (default is sequence-only)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids-file", default=None,
                    help="a file with one DMS id per line. When given, this list is used instead of the enzyme split.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [zs/{args.backbone} sh{args.shard_index}/{args.num_shards}] %(message)s",
    )

    if args.ids_file:
        all_ids = [l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()]
    else:
        all_ids = enzyme_dms_ids()
    shard_ids = balanced_shard(all_ids, CSV_DIR, args.num_shards, args.shard_index)
    if args.limit:
        shard_ids = shard_ids[: args.limit]

    label = BACKBONE_LABEL[args.backbone]
    logger.info("%d DMS (of %d) -> zero-shot backbone=%s", len(shard_ids), len(all_ids), label)

    t0 = time.time()
    if args.backbone == "esm2":
        model, tok, model_name = load_esm2(args.device)
        context_label = "sequence"
    elif args.backbone == "prosst":
        model, tok, model_name = load_prosst(args.device)
        struct_loader = ProSSTContextLoader(PROSST_STRUCT_ROOT)
        context_label = "sequence+structure_tokens"
    else:
        model, tok, model_name = load_esm3(args.device)
        ctx_loader = ESM3ContextLoader() if args.esm3_multimodal else None
        context_label = "sequence+structure+function" if args.esm3_multimodal else "sequence"
        if args.esm3_multimodal:
            logger.info("ESM3 multimodal zero-shot: injecting WT structure (atom37) and function context")
    logger.info("input modality (recorded in the cache): context=%s", context_label)
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info("model loaded (%.1fs): %s", time.time() - t0, model_name)

    done = skipped = 0
    for i, dms_id in enumerate(shard_ids, 1):
        if zs.cache_path(label, dms_id, cache_root=CACHE_ROOT).exists():
            skipped += 1
            continue
        csv = CSV_DIR / f"{dms_id}.csv"
        if not csv.exists():
            logger.error("[%d/%d] no CSV, skipping: %s", i, len(shard_ids), csv)
            continue
        df = load_dms_csv(csv)
        wt = derive_wt_sequence(df)
        variants, bad = collect_variants(df)
        if bad:
            logger.warning("DMS %s: excluded %d non-SAV rows", dms_id, len(bad))

        t1 = time.time()
        if args.backbone == "esm2":
            scores = zs.score_dms_esm2_wildtype_marginal(wt, variants, model, tok, device=args.device)
        elif args.backbone == "prosst":
            struct = struct_loader.load(dms_id)["structure_tokens"]
            scores = zs.score_dms_prosst_wildtype_marginal(
                wt, variants, struct, model, tok, device=args.device
            )
        else:
            ctx = ctx_loader.load(dms_id) if ctx_loader else {}
            scores = zs.score_dms_esm3_wildtype_marginal(
                wt, variants, model, device=args.device,
                coordinates=ctx.get("coordinates"),
                function_annotations=ctx.get("function_annotations"),
            )

        record = zs.ZeroshotCacheRecord(
            backbone=label,
            dms_id=dms_id,
            hgvsp_keys=tuple(v.hgvsp for v in variants),
            scores=scores.to(torch.float16),
            scoring_method=zs.SCORING_WILDTYPE_MARGINAL,
            model_name=model_name,
            computed_at=zs.utc_now_iso(),
            context=context_label,
        )
        zs.save_zeroshot_cache(record, cache_root=CACHE_ROOT)
        done += 1
        logger.info("[%d/%d] %s done: V=%d (%.1fs)", i, len(shard_ids), dms_id, len(variants), time.time() - t1)

    logger.info("shard done: %d new, %d skipped, %.1f minutes total", done, skipped, (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
