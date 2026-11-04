"""Shared utilities for the biophysical-feature probing study.

Goal: assemble, per ProteinGym DMS, a per-variant feature table covering the
classic protein-engineering axes (chemistry / sterics / structural environment /
evolution / function / neighborhood / family), then probe how much of the DMS
score they explain with tree (HistGradientBoosting) and linear models.

This module only holds *cheap, dependency-free* lookups + ID conversion so that
the heavy extraction (DSSP, neighborhood, embeddings) can import it freely.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

# ----------------------------------------------------------------------------
# Amino-acid property tables (single-letter keys)
# ----------------------------------------------------------------------------
AA = "ACDEFGHIKLMNPQRSTVWY"

# Kyte & Doolittle (1982) hydropathy index
KD_HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Net side-chain charge at pH 7 (His partial)
CHARGE = {a: 0.0 for a in AA}
CHARGE.update({"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1})

# Residue volume (A^3), Zamyatnin (1972)
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
    "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
    "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
    "Y": 193.6, "V": 140.0,
}

# Vihinen et al. (1994) average flexibility index
FLEX = {
    "A": 0.357, "R": 0.529, "N": 0.463, "D": 0.511, "C": 0.346, "Q": 0.493,
    "E": 0.497, "G": 0.544, "H": 0.323, "I": 0.462, "L": 0.365, "K": 0.466,
    "M": 0.295, "F": 0.314, "P": 0.509, "S": 0.507, "T": 0.444, "W": 0.305,
    "Y": 0.420, "V": 0.386,
}

# Side-chain hydrogen-bond donor/acceptor counts (rough polarity proxy)
HBOND = {
    "A": 0, "R": 5, "N": 4, "D": 4, "C": 1, "Q": 4, "E": 4, "G": 0, "H": 3,
    "I": 0, "L": 0, "K": 2, "M": 0, "F": 0, "P": 0, "S": 2, "T": 2, "W": 1,
    "Y": 2, "V": 0,
}

IS_AROMATIC = {a: float(a in "FWY") for a in AA}
IS_CHARGED = {a: float(abs(CHARGE[a]) >= 0.9) for a in AA}


@lru_cache(maxsize=1)
def blosum62():
    """BLOSUM62 substitution score lookup as {(wt,mut): score}."""
    from Bio.Align import substitution_matrices

    m = substitution_matrices.load("BLOSUM62")
    out = {}
    for a in AA:
        for b in AA:
            try:
                out[(a, b)] = float(m[a, b])
            except Exception:
                out[(a, b)] = np.nan
    return out


# ----------------------------------------------------------------------------
# Variant ID conversion: HGVS three-letter <-> single-letter short form
# ----------------------------------------------------------------------------
THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}

_HGVS_RE = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$")
_SHORT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def parse_variant(vid: str):
    """Parse a single-substitution id (HGVS 'p.Ala30Gln' or short 'A30Q').

    Returns (wt_aa, pos_1based, mut_aa) or None if not a parseable SAV.
    """
    m = _HGVS_RE.match(vid)
    if m:
        wt = THREE_TO_ONE.get(m.group(1))
        mut = THREE_TO_ONE.get(m.group(3))
        if wt and mut:
            return wt, int(m.group(2)), mut
        return None
    m = _SHORT_RE.match(vid)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None


def to_short(vid: str):
    p = parse_variant(vid)
    return None if p is None else f"{p[0]}{p[1]}{p[2]}"


# Variant-level chemical / steric deltas (mut - wt), plus context + identity.
def variant_features(wt: str, mut: str) -> dict:
    b = blosum62()
    return {
        "d_hydro": KD_HYDRO[mut] - KD_HYDRO[wt],
        "d_charge": CHARGE[mut] - CHARGE[wt],
        "d_volume": VOLUME[mut] - VOLUME[wt],
        "d_flex": FLEX[mut] - FLEX[wt],
        "d_hbond": HBOND[mut] - HBOND[wt],
        "abs_d_hydro": abs(KD_HYDRO[mut] - KD_HYDRO[wt]),
        "abs_d_volume": abs(VOLUME[mut] - VOLUME[wt]),
        "abs_d_charge": abs(CHARGE[mut] - CHARGE[wt]),
        "blosum62": b[(wt, mut)],
        "wt_hydro": KD_HYDRO[wt],
        "wt_charge": CHARGE[wt],
        "wt_volume": VOLUME[wt],
        "to_proline": float(mut == "P"),
        "from_glycine": float(wt == "G"),
        "to_charged": IS_CHARGED[mut],
        "lose_aromatic": float(IS_AROMATIC[wt] and not IS_AROMATIC[mut]),
    }


VARIANT_FEATURE_COLS = list(variant_features("A", "V").keys())
