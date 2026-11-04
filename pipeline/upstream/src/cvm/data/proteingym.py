"""Load a ProteinGym DMS substitution CSV, derive the wild-type sequence and collect the variants.

Required CSV columns: `mutant` (1-letter, for example M1H) and `mutated_sequence` (the full
sequence with the substitution applied). Optional: `DMS_score`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cvm.data.hgvsp import mutant_to_hgvsp


@dataclass(frozen=True)
class Variant:
    mutant: str          # 1-letter notation (for example "M1H")
    hgvsp: str           # the LMDB key (for example "p.Met1His")
    sequence: str        # the full sequence with the substitution applied
    score: float | None  # DMS_score, when present


def load_dms_csv(csv_path: Path | str) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"no DMS CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = {"mutant", "mutated_sequence"} - set(df.columns)
    if missing:
        raise ValueError(f"required DMS CSV columns are missing: {sorted(missing)}")
    return df


def derive_wt_sequence(df: pd.DataFrame) -> str:
    """Recover the wild-type sequence from the single-missense rows. Any inconsistency raises ValueError."""
    if df.empty:
        raise ValueError("cannot derive a wild type from an empty DMS DataFrame")
    wt: str | None = None
    for _, row in df.iterrows():
        mutant = str(row["mutant"])
        try:
            mutant_to_hgvsp(mutant)
        except ValueError:
            continue  # rows that are not SAVs are not used to derive the wild type
        ref_aa, pos_str, alt_aa = mutant[0], mutant[1:-1], mutant[-1]
        pos = int(pos_str)
        seq = str(row["mutated_sequence"])
        if pos < 1 or pos > len(seq) or seq[pos - 1] != alt_aa:
            raise ValueError(
                f"integrity violation: with mutant={mutant!r}, mutated_sequence[{pos - 1}] should be {alt_aa!r}"
            )
        candidate = seq[: pos - 1] + ref_aa + seq[pos:]
        if wt is None:
            wt = candidate
        elif candidate != wt:
            raise ValueError(f"wild-type mismatch: mutant={mutant!r} implies a different wild type from an earlier row")
    if wt is None:
        raise ValueError("no single-missense rows, so the wild type cannot be derived")
    return wt


def collect_variants(df: pd.DataFrame) -> tuple[list[Variant], list[str]]:
    """Returns the single-missense variants together with the notations that were excluded (non-SAV), so nothing is skipped silently."""
    variants: list[Variant] = []
    skipped: list[str] = []
    has_score = "DMS_score" in df.columns
    for _, row in df.iterrows():
        mutant = str(row["mutant"])
        try:
            hgvsp = mutant_to_hgvsp(mutant)
        except ValueError:
            skipped.append(mutant)
            continue
        score = float(row["DMS_score"]) if has_score and pd.notna(row["DMS_score"]) else None
        variants.append(Variant(mutant, hgvsp, str(row["mutated_sequence"]), score))
    return variants, skipped
