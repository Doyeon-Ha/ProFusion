"""Variant notation to HGVSp keys. Single missense (SAV) only.

Key convention: variants use 3-letter HGVSp (for example F1I becomes p.Phe1Ile), and the wild-type
reference is `p.={dms_id}`.
"""

from __future__ import annotations

import re

_AA1_TO_AA3 = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
}
_AA3_TO_AA1 = {v: k for k, v in _AA1_TO_AA3.items()}

# Single missense p. notation: p.Arg31Gln (3-letter ref, position, 3-letter alt)
_HGVSP_SAV_RE = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$")


def mutant_to_hgvsp(mutant: str) -> str:
    """Convert a single missense in `M1H` form to `p.Met1His`. Multiple, nonsense or malformed raise ValueError."""
    if ":" in mutant:
        raise ValueError(f"multiple variants are not supported (single SAV only): {mutant!r}")
    if "*" in mutant:
        raise ValueError(f"nonsense variants are not supported: {mutant!r}")
    if len(mutant) < 3:
        raise ValueError(f"malformed variant: {mutant!r}")
    ref_aa, pos_str, alt_aa = mutant[0], mutant[1:-1], mutant[-1]
    if not pos_str.isdigit():
        raise ValueError(f"could not parse the variant position: {mutant!r}")
    if ref_aa not in _AA1_TO_AA3 or alt_aa not in _AA1_TO_AA3:
        raise ValueError(f"unknown amino acid: {mutant!r}")
    return f"p.{_AA1_TO_AA3[ref_aa]}{int(pos_str)}{_AA1_TO_AA3[alt_aa]}"


def hgvsp_to_mutant(hgvsp: str) -> str:
    """`p.Arg31Gln` to `R31Q`, the inverse of mutant_to_hgvsp.

    Single missense only. Synonymous (`p.Ile1855=`, `p.Leu5Leu`), nonsense (`p.Arg31Ter`, `*`),
    in-frame indel or delins (`p.Lys601_Glu602delinsGluVal`) and malformed input raise ValueError.
    (Used to convert the `(p.XXX)` notation of some sources into the 1-letter `mutant` form.)
    """
    s = str(hgvsp).strip()
    if "=" in s:
        raise ValueError(f"synonymous variants are not supported: {hgvsp!r}")
    if s.endswith("Ter") or "*" in s:
        raise ValueError(f"nonsense variants are not supported: {hgvsp!r}")
    m = _HGVSP_SAV_RE.match(s)
    if not m:
        raise ValueError(f"not a single missense p. notation (delins, fs or malformed): {hgvsp!r}")
    ref3, pos_str, alt3 = m.group(1), m.group(2), m.group(3)
    if ref3 not in _AA3_TO_AA1 or alt3 not in _AA3_TO_AA1:
        raise ValueError(f"unknown 3-letter amino acid: {hgvsp!r}")
    if ref3 == alt3:
        raise ValueError(f"synonymous (ref == alt) variants are not supported: {hgvsp!r}")
    return f"{_AA3_TO_AA1[ref3]}{int(pos_str)}{_AA3_TO_AA1[alt3]}"


def gene_ref_key(dms_id: str) -> str:
    """The wild-type reference embedding key for one DMS."""
    return f"p.={dms_id}"
