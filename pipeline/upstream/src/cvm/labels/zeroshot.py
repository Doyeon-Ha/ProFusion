"""Zero-shot pseudo-labels: wildtype-marginal scoring with a (backbone, DMS) cache.

Scope
- scoring method: ``wildtype_marginal`` only (the ProteinGym standard, one forward pass per DMS)
- cache location: ``data/proteingym/zeroshot/{backbone}/{dms_id}.pt`` (overridable by config)

Separation of concerns:
- ``wildtype_marginal_score``: a *pure* function (logits, positions, token ids -> Tensor[V]). This
  is what the unit tests target.
- ``score_dms_esm2_wildtype_marginal``: runs one HF model forward and passes the logits to it.
- ``save_zeroshot_cache`` / ``load_zeroshot_cache``: the `.pt` storage schema.
- ``compute_y_pseudo_label``: the entry point, dispatching on the cache.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import torch

from cvm.data.proteingym import Variant

logger = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = Path("data/proteingym/zeroshot")

# Pseudo-label forms
LABEL_FORM_LOWEST_SENSITIVITY_ZEROSHOT = "lowest_sensitivity_zeroshot"  # implemented
LABEL_FORM_INVERSE_STD_WEIGHT = "inverse_std_weight"                    # not implemented (NotImplementedError)

# scoring method: wildtype-marginal only
SCORING_WILDTYPE_MARGINAL = "wildtype_marginal"

# Implemented backbones (must match the y_pseudo_source values)
_IMPLEMENTED_BACKBONES_FOR_ZEROSHOT: frozenset[str] = frozenset({"ESM2", "ESM3", "ProSST"})

# Default models, overridable by config
DEFAULT_ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DEFAULT_ESM3_MODEL_NAME = "esm3_sm_open_v1"
DEFAULT_PROSST_MODEL_NAME = "AI4Protein/ProSST-2048"

# ProSST structure tokenisation convention (the official ai4protein/ProSST)
_PROSST_SS_SHIFT = 3
_PROSST_SS_BOS = 1
_PROSST_SS_EOS = 2


class _TokenizerLike(Protocol):
    def __call__(self, text: str, return_tensors: str = ...) -> dict: ...
    def convert_tokens_to_ids(self, token: str) -> int: ...


class _ModelLike(Protocol):
    def __call__(self, **kwargs: object) -> object: ...
    def to(self, device: torch.device) -> "_ModelLike": ...
    def eval(self) -> "_ModelLike": ...


@dataclass(frozen=True)
class ZeroshotCacheRecord:
    backbone: str
    dms_id: str
    hgvsp_keys: tuple[str, ...]
    scores: torch.Tensor          # [V]. The on-disk cache (single backbone) is float16; an in-memory ensemble is float32
    scoring_method: str
    model_name: str
    computed_at: str              # ISO-8601 UTC
    # The input modality. For the same backbone and scoring method, the scores still differ
    # depending on whether structure went into the wild-type context (ESM3), and the earlier cache
    # did not record it, which made it impossible to tell without the logs.
    # Example values: "sequence", "sequence+structure", "sequence+structure+function".
    # Reading an older file gives "unknown". Do not paper over that with a default.
    context: str = "sequence"


# ---------------------------------------------------------------------------
# The pure algorithm: logits to scores (the core of the unit tests)
# ---------------------------------------------------------------------------

def wildtype_marginal_score(
    logits: torch.Tensor,
    positions_1based: torch.Tensor,
    wt_token_ids: torch.Tensor,
    mut_token_ids: torch.Tensor,
    *,
    offset: int = 1,
) -> torch.Tensor:
    """Wildtype-marginal log-likelihood ratio (the ProteinGym standard zero-shot score).

    score_i = log P(mut_i | full WT context) − log P(wt_i | full WT context)

        logits:           [T, vocab]. The logits of one forward pass over the WT sequence (batch
                          dimension squeezed).
        positions_1based: [V]. Variant positions (1-based, 1..L).
        wt_token_ids:     [V]. The token id of each variant's wild-type residue.
        mut_token_ids:    [V]. The token id of each variant's mutant residue.
        offset:           how many special tokens the tokenizer prepends (CLS = 1). Default 1.

    Returns:
        [V] float32 tensor. NaN or Inf raises ValueError (nothing is skipped silently).
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be [T, vocab]: {tuple(logits.shape)}")
    if not (positions_1based.shape == wt_token_ids.shape == mut_token_ids.shape):
        raise ValueError(
            f"shape mismatch between positions, wt and mut: "
            f"{positions_1based.shape}, {wt_token_ids.shape}, {mut_token_ids.shape}"
        )

    token_idx = positions_1based.long() + (offset - 1)  # 1-based pos + (offset-1) = token index
    if int(token_idx.min()) < 0 or int(token_idx.max()) >= logits.shape[0]:
        raise ValueError(
            f"a variant position is outside the token range: token_idx in [{int(token_idx.min())}, "
            f"{int(token_idx.max())}], logits T={logits.shape[0]}"
        )

    log_probs = torch.log_softmax(logits[token_idx], dim=-1)  # [V, vocab]
    log_p_mut = log_probs.gather(1, mut_token_ids.long().view(-1, 1)).squeeze(1)
    log_p_wt = log_probs.gather(1, wt_token_ids.long().view(-1, 1)).squeeze(1)
    scores = (log_p_mut - log_p_wt).float()

    if not torch.isfinite(scores).all():
        bad = (~torch.isfinite(scores)).nonzero().squeeze(-1).tolist()
        raise ValueError(f"NaN or Inf in the wildtype-marginal score (variant indices: {bad})")
    return scores


# ---------------------------------------------------------------------------
# ESM-2 driver (needs HF transformers; used in the GPU stage)
# ---------------------------------------------------------------------------

def score_dms_esm2_wildtype_marginal(
    wt_sequence: str,
    variants: list[Variant],
    model: _ModelLike,
    tokenizer: _TokenizerLike,
    *,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute the wildtype-marginal scores of one DMS with an HF ``EsmForMaskedLM``.

        wt_sequence: the wild-type protein sequence (1-letter).
        variants:    a list of `Variant` (including the 1-letter mutant notation).
        model:       an ``EsmForMaskedLM`` instance (eval and frozen recommended).

    Returns
        a [V] float32 score tensor, in the order of `variants`.
    """
    enc = tokenizer(wt_sequence, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    logits = out.logits[0].cpu()  # type: ignore[union-attr]  # [T, vocab]

    positions = torch.tensor([int(v.mutant[1:-1]) for v in variants], dtype=torch.long)
    wt_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(v.mutant[0]) for v in variants], dtype=torch.long
    )
    mut_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(v.mutant[-1]) for v in variants], dtype=torch.long
    )
    return wildtype_marginal_score(logits, positions, wt_ids, mut_ids, offset=1)


def _variant_token_tensors(
    variants: list[Variant], convert: "callable[[str], int]"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """variants to (positions_1based, wt_token_ids, mut_token_ids). The tokenizer-independent part."""
    positions = torch.tensor([int(v.mutant[1:-1]) for v in variants], dtype=torch.long)
    wt_ids = torch.tensor([convert(v.mutant[0]) for v in variants], dtype=torch.long)
    mut_ids = torch.tensor([convert(v.mutant[-1]) for v in variants], dtype=torch.long)
    return positions, wt_ids, mut_ids


# ---------------------------------------------------------------------------
# ProSST driver: structure-aware wildtype-marginal (the same logits as the official benchmark script)
# ---------------------------------------------------------------------------

def score_dms_prosst_wildtype_marginal(
    wt_sequence: str,
    variants: list[Variant],
    structure_tokens: list[int],
    model: _ModelLike,
    tokenizer: _TokenizerLike,
    *,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute the structure-aware wildtype-marginal scores of one DMS with ``ProSSTForMaskedLM``.

    Identical to the official ai4protein/ProSST zero-shot: one forward over the WT sequence plus the
    WT quantised structure tokens (ss_input_ids), then wildtype-marginal on the logits. There is one
    structure token per residue (so as many as the WT length).
    """
    if len(structure_tokens) != len(wt_sequence):
        raise ValueError(
            f"the number of structure tokens ({len(structure_tokens)}) differs from the WT sequence length ({len(wt_sequence)})"
        )
    ss = [_PROSST_SS_BOS, *(int(t) + _PROSST_SS_SHIFT for t in structure_tokens), _PROSST_SS_EOS]
    ss_input_ids = torch.tensor([ss], dtype=torch.long, device=device)
    enc = tokenizer([wt_sequence], return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(  # type: ignore[call-arg]
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            ss_input_ids=ss_input_ids,
        )
    logits = out.logits[0].cpu()  # type: ignore[union-attr]  # [T, vocab]
    positions, wt_ids, mut_ids = _variant_token_tensors(variants, tokenizer.convert_tokens_to_ids)
    return wildtype_marginal_score(logits, positions, wt_ids, mut_ids, offset=1)


# ---------------------------------------------------------------------------
# ESM3 driver: sequence-track wildtype-marginal (optionally with structure and function context)
# ---------------------------------------------------------------------------

def score_dms_esm3_wildtype_marginal(
    wt_sequence: str,
    variants: list[Variant],
    model: object,  # esm.models.esm3.ESM3
    *,
    device: str = "cpu",
    coordinates: torch.Tensor | None = None,
    function_annotations: object | None = None,
) -> torch.Tensor:
    """ESM3 sequence-track wildtype-marginal scores.

    When ``coordinates`` (`[L,37,3]` atom37) or ``function_annotations`` are given, the WT context is
    encoded with structure and function as well (multimodal), producing sequence logits under the
    same modality as the embedding extraction. With both None it is sequence-only. Either way the
    score is computed from the residue distribution of the sequence track.
    """
    import attr

    from esm.sdk.api import ESMProtein, ESMProteinTensor, LogitsConfig
    from esm.utils import encoding

    seq_tokenizer = model.tokenizers.sequence  # type: ignore[attr-defined]
    seq_tok = encoding.tokenize_sequence(
        wt_sequence, seq_tokenizer, add_special_tokens=True
    ).to(device)

    multimodal = coordinates is not None or function_annotations is not None
    if multimodal:
        base = model.encode(  # type: ignore[attr-defined]
            ESMProtein(
                sequence=wt_sequence,
                coordinates=coordinates,
                function_annotations=function_annotations,
            )
        )
        tokens = attr.evolve(base, sequence=seq_tok)
    else:
        tokens = ESMProteinTensor(sequence=seq_tok)

    with torch.no_grad():
        out = model.logits(tokens, LogitsConfig(sequence=True))  # type: ignore[attr-defined]
    logits = out.logits.sequence.squeeze(0).cpu()  # [T, vocab]
    positions, wt_ids, mut_ids = _variant_token_tensors(
        variants, seq_tokenizer.convert_tokens_to_ids
    )
    return wildtype_marginal_score(logits, positions, wt_ids, mut_ids, offset=1)


# ---------------------------------------------------------------------------
# `.pt` cache I/O: `data/proteingym/zeroshot/{backbone}/{dms_id}.pt`
# ---------------------------------------------------------------------------

def cache_path(backbone: str, dms_id: str, *, cache_root: Path | None = None) -> Path:
    root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT
    return root / backbone / f"{dms_id}.pt"


def save_zeroshot_cache(record: ZeroshotCacheRecord, *, cache_root: Path | None = None) -> Path:
    """Atomic write (tmp then rename). Idempotent: if it already exists it raises ``FileExistsError`` rather than overwriting."""
    path = cache_path(record.backbone, record.dms_id, cache_root=cache_root)
    if path.exists():
        raise FileExistsError(
            f"zero-shot cache already exists (idempotent, not overwritten): {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "backbone": record.backbone,
        "dms_id": record.dms_id,
        "hgvsp_keys": list(record.hgvsp_keys),
        "scores": record.scores.to(torch.float16),
        "scoring_method": record.scoring_method,
        "model_name": record.model_name,
        "computed_at": record.computed_at,
        "context": record.context,
    }
    torch.save(payload, tmp)
    tmp.rename(path)
    logger.info("wrote zero-shot cache: %s (V=%d)", path, len(record.hgvsp_keys))
    return path


def load_zeroshot_cache(
    backbone: str, dms_id: str, *, cache_root: Path | None = None
) -> ZeroshotCacheRecord:
    path = cache_path(backbone, dms_id, cache_root=cache_root)
    if not path.exists():
        raise FileNotFoundError(
            f"no zero-shot cache: backbone={backbone!r}, dms_id={dms_id!r}, path={path}. "
            "Check that the extraction step was run first."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return ZeroshotCacheRecord(
        backbone=payload["backbone"],
        dms_id=payload["dms_id"],
        hgvsp_keys=tuple(payload["hgvsp_keys"]),
        scores=payload["scores"],
        scoring_method=payload["scoring_method"],
        model_name=payload["model_name"],
        computed_at=payload["computed_at"],
        # Older caches do not have it. Filling in "sequence" would assert a modality, so it is unknown.
        context=payload.get("context", "unknown"),
    )


# The default ensemble backbones (3-backbone z-score mean). Must match the cache directory names.
DEFAULT_ENSEMBLE_BACKBONES: tuple[str, ...] = ("ESM2", "ESM3", "ProSST")
ENSEMBLE_MODEL_NAME = "mean_zscore_ensemble"


def load_ensemble_zeroshot(
    dms_id: str,
    *,
    backbones: tuple[str, ...] | list[str] = DEFAULT_ENSEMBLE_BACKBONES,
    cache_root: Path | None = None,
    eps: float = 1e-8,
) -> ZeroshotCacheRecord:
    """A multi-backbone zero-shot ensemble label: **per-DMS z-score, then mean**.

    Rationale: alpha-weighted gating collapsed to uniform and its signal was unrelated to accuracy,
    so it was dropped. The uniform-mean ensemble (0.516) beats the best single backbone
    (ProSST 0.466) and the per-DMS oracle (0.500). The zero-shot scale differs per backbone, so the
    mean is taken **after z-normalisation**, otherwise one backbone dominates through its variance.

    Every backbone cache must have the same set of variant keys (a missing one raises KeyError,
    nothing is skipped silently). The hgvsp_keys order of the returned record follows the first
    backbone.
    """
    if len(backbones) < 2:
        raise ValueError(f"an ensemble needs at least 2 backbones: {list(backbones)}")
    records = [load_zeroshot_cache(bb, dms_id, cache_root=cache_root) for bb in backbones]
    canonical = records[0].hgvsp_keys
    canon_set = set(canonical)
    z_scores = []
    for bb, rec in zip(backbones, records):
        if rec.scoring_method != SCORING_WILDTYPE_MARGINAL:
            raise ValueError(
                f"ensemble backbone {bb} has scoring_method={rec.scoring_method!r} "
                f"!= {SCORING_WILDTYPE_MARGINAL!r}"
            )
        if set(rec.hgvsp_keys) != canon_set:
            raise KeyError(
                f"the variant keys of ensemble backbone {bb} differ from {backbones[0]} "
                f"(DMS={dms_id!r}): {len(rec.hgvsp_keys)} vs {len(canonical)}"
            )
        kmap = {k: float(v) for k, v in zip(rec.hgvsp_keys, rec.scores.float().tolist())}
        aligned = torch.tensor([kmap[k] for k in canonical], dtype=torch.float32)
        z_scores.append((aligned - aligned.mean()) / (aligned.std(unbiased=True) + eps))
    ensemble = torch.stack(z_scores, dim=0).mean(dim=0)  # [V]
    return ZeroshotCacheRecord(
        backbone="ensemble:" + ",".join(backbones),
        dms_id=dms_id,
        hgvsp_keys=canonical,
        scores=ensemble,  # kept float32, to preserve rank and pairwise-margin precision in the z-score training target
        scoring_method=SCORING_WILDTYPE_MARGINAL,
        model_name=ENSEMBLE_MODEL_NAME,
        computed_at=utc_now_iso(),
        # The modality is inherited from the constituent backbones: if any is unknown, so is the ensemble.
        context="ensemble(" + ",".join(r.context for r in records) + ")",
    )


# ---------------------------------------------------------------------------
# Dispatch: the entry point the LMDB builder calls
# ---------------------------------------------------------------------------

def compute_y_pseudo_label(
    dms_id: str,
    y_pseudo_source_backbone: str,
    *,
    label_form: str = LABEL_FORM_LOWEST_SENSITIVITY_ZEROSHOT,
    scoring_method: str = SCORING_WILDTYPE_MARGINAL,
    cache_root: Path | None = None,
) -> ZeroshotCacheRecord:
    """Dispatch the pseudo-label from the y_pseudo_source of a Pair.

    Raises
        NotImplementedError: ``label_form``, ``scoring_method`` or the backbone is outside what is
                             implemented here.
        FileNotFoundError:   the cache is missing (run the zero-shot extraction first).
    """
    if label_form == LABEL_FORM_INVERSE_STD_WEIGHT:
        raise NotImplementedError(
            "label_form='inverse_std_weight' is not implemented here. "
            "Only 'lowest_sensitivity_zeroshot' is supported."
        )
    if label_form != LABEL_FORM_LOWEST_SENSITIVITY_ZEROSHOT:
        raise ValueError(f"unknown label_form: {label_form!r}")

    if scoring_method != SCORING_WILDTYPE_MARGINAL:
        raise NotImplementedError(
            f"scoring_method={scoring_method!r} is not implemented here. "
            f"Only 'wildtype_marginal' is supported."
        )

    if y_pseudo_source_backbone not in _IMPLEMENTED_BACKBONES_FOR_ZEROSHOT:
        raise NotImplementedError(
            f"zero-shot scoring for backbone={y_pseudo_source_backbone!r} is not implemented. "
            f"Only {sorted(_IMPLEMENTED_BACKBONES_FOR_ZEROSHOT)} are supported. "
            "Add a driver to zeroshot.py if another is needed."
        )

    return load_zeroshot_cache(y_pseudo_source_backbone, dms_id, cache_root=cache_root)


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp (for the cache ``computed_at`` field)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
