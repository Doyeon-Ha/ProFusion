"""Score a new protein with comp7, without labels.

The only difference from the evaluation path (common.Rec) is that **no labels are needed**. The
convention for building the channels is character-for-character the same as build_channels.py; if
they differed, the fixed weights would refer to different quantities. tests/test_predict.py
checks that equivalence against a cached npz.

    CH has 7 slots = (a_GEMME, b_GEMME, a_ddG, b_ddG, a_RSA, b_BLOSUM, b_dchg)
    a_* is the position-mean component, b_* the within-position residual

A **larger** s means closer to wild type (more tolerated). Sort by -s to list the damaging
variants first.

⚠ Three things
  - Channels are z-normalised **within the variant set**. Change the set and the scores change.
    Do not pass a single variant.
  - If a position has only one variant, its residual components (b_*) are 0 and comp7 behaves
    like champ3.
  - Scores are not comparable across proteins. Use them as a ranking within one protein.

⚠ With ESM2 the LLR must be computed **masked-marginal**, because the weights were fitted on that
   convention. A wildtype-marginal LLR gives perfectly reasonable-looking numbers that are simply
   wrong. On the evaluation side, common.LLR_ALIAS redirects the npz key to llr_ESM2mask so that
   both paths use the same convention.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import SLOTS, zs
from .paths import REFERENCE, RESULTS

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AAI = {a: i for i, a in enumerate(AA20)}
# The same table as build_channels.py
CHG = dict(D=-1.0, E=-1.0, K=1.0, R=1.0, H=0.1)
_CHG = np.array([CHG.get(a, 0.0) for a in AA20], np.float32)

AXIS_OF_ASSAY = {
    "Activity": "function",
    "Binding": "function",
    "OrganismalFitness": "function",
    "Expression": "stability",
}


def _blosum20() -> np.ndarray:
    from Bio.Align import substitution_matrices
    m = substitution_matrices.load("BLOSUM62")
    return np.array([[float(m[a, b]) for b in AA20] for a in AA20], np.float32)


def load_weights(axis: str, backbone: str, path=None) -> np.ndarray:
    """The 7 fitted comp7 weights. No labels are needed: they are already fitted on the MaveDB
    external set.

    The default is reference/paper_weights.csv (the paper weights). Pass path to use results/.
    """
    if path is None:
        path = REFERENCE / "paper_weights.csv"
        if not path.exists():
            path = RESULTS / "paper_weights.csv"
    W = pd.read_csv(path)
    hit = W[(W.axis == axis) & (W.bb == backbone) & W.arm.str.startswith("comp7")]
    if len(hit) != 1:
        raise ValueError(
            f"Found {len(hit)} comp7 weight rows for (axis={axis!r}, backbone={backbone!r}). "
            f"Choose an axis from {sorted(set(W.axis))} and a backbone from {sorted(set(W.bb))}.")
    return hit.iloc[0][list(SLOTS)].to_numpy(float)


def axis_of(assay_type: str) -> str:
    """Assay type -> axis. An unknown type is never passed over silently."""
    if assay_type not in AXIS_OF_ASSAY:
        raise ValueError(
            f"Unknown assay type {assay_type!r}. Valid values: {sorted(AXIS_OF_ASSAY)}.\n"
            "Folding-ΔG assays (the Tsuboyama family) have never been validated with these "
            "weights: the ΔΔG channel comes from a model trained on that data, so scoring "
            "them reads the training data back.")
    return AXIS_OF_ASSAY[assay_type]


@dataclass
class Channels:
    """One protein's variant set. Every field is a per-variant array, all the same length.

    pos1   1-based residue position
    wt, mut  1-letter amino acids (array of strings, or a list)
    gemme  GEMME score (raw). Missing values are filled with the median
    rsa    relative solvent accessibility (raw, 0 to 1)
    ddg    SPURS ΔΔG, **raw**. Positive means destabilising. The sign flip happens inside
    llr    pLM zero-shot LLR (raw)
    """
    pos1: np.ndarray
    wt: np.ndarray
    mut: np.ndarray
    gemme: np.ndarray
    rsa: np.ndarray
    ddg: np.ndarray
    llr: np.ndarray

    def __post_init__(self):
        n = len(self.pos1)
        for f in ("wt", "mut", "gemme", "rsa", "ddg", "llr"):
            v = np.asarray(getattr(self, f))
            if len(v) != n:
                raise ValueError(f"{f} has length {len(v)}, but pos1 has length {n}")
            setattr(self, f, v)
        self.pos1 = np.asarray(self.pos1, np.int64)


def build_matrix(ch: Channels) -> np.ndarray:
    """Channels -> an (n_variants, 7) matrix, under the build_channels.py and common.Rec convention."""
    pid = ch.pos1
    _, inv = np.unique(pid, return_inverse=True)
    npos = int(inv.max()) + 1
    cnt = np.bincount(inv, minlength=npos)

    def split(v):
        v = zs(np.asarray(v, float))
        pm = (np.bincount(inv, v, minlength=npos) / np.maximum(cnt, 1))[inv]
        return pm, v - pm

    gem = np.asarray(ch.gemme, float)
    if np.isnan(gem).any():
        if np.isnan(gem).all():
            raise ValueError("every GEMME score is missing")
        gem = np.where(np.isnan(gem), np.nanmedian(gem), gem)
    wt_i = np.array([AAI[a] for a in ch.wt], np.int8)
    mt_i = np.array([AAI[a] for a in ch.mut], np.int8)

    g_p, g_r = split(gem)
    d_p, d_r = split(-np.asarray(ch.ddg, float))      # sign flip: positive means destabilising
    r_p, _ = split(np.asarray(ch.rsa, float))
    _, bl_r = split(_blosum20()[wt_i, mt_i])
    _, ac_r = split(-np.abs(_CHG[mt_i] - _CHG[wt_i]))
    return np.ascontiguousarray(np.stack([g_p, g_r, d_p, d_r, r_p, bl_r, ac_r]).T)


def predict(ch: Channels, w: np.ndarray) -> np.ndarray:
    """The fused score. Larger means closer to wild type (more tolerated)."""
    w = np.asarray(w, float)
    if w.shape != (7,):
        raise ValueError(f"expected 7 weights, got {w.shape}")
    return zs(np.asarray(ch.llr, float)) + build_matrix(ch) @ w


def predict_table(ch: Channels, w: np.ndarray) -> pd.DataFrame:
    """Per-variant score table.

    A **smaller** fused value is more damaging. The evaluation metrics (paper_numbers.metrics)
    treat the smallest s as the most damaging in the same way, so deleterious_rank 1 = smallest
    fused = most damaging.
    """
    s = predict(ch, w)
    df = pd.DataFrame(dict(
        mutant=[f"{a}{p}{b}" for a, p, b in zip(ch.wt, ch.pos1, ch.mut)],
        pos1=ch.pos1, wt=ch.wt, mut=ch.mut,
        llr=np.asarray(ch.llr, float), gemme=np.asarray(ch.gemme, float),
        ddg=np.asarray(ch.ddg, float), rsa=np.asarray(ch.rsa, float),
        fused=s))
    df["deleterious_rank"] = df.fused.rank(method="min").astype(int)
    return df.sort_values("deleterious_rank").reset_index(drop=True)
