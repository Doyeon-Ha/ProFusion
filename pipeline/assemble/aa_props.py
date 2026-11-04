"""Amino acid property constants, shared by the assembly scripts.

Only what the assembly needs is kept.
"""
from __future__ import annotations

AA20 = "ACDEFGHIKLMNPQRSTVWY"


# Grantham 1974 table: (composition, polarity, volume). The assembly uses volume only (dvol)
_GRANTHAM_CPV = {
    "S": (1.42, 9.2, 32.0), "R": (0.65, 10.5, 124.0), "L": (0.00, 4.9, 111.0),
    "P": (0.39, 8.0, 32.5), "T": (0.71, 8.6, 61.0), "A": (0.00, 8.1, 31.0),
    "V": (0.00, 5.9, 84.0), "G": (0.74, 9.0, 3.0), "I": (0.00, 5.2, 111.0),
    "F": (0.00, 5.2, 132.0), "Y": (0.20, 6.2, 136.0), "C": (2.75, 5.5, 55.0),
    "H": (0.58, 10.4, 96.0), "Q": (0.89, 10.5, 85.0), "N": (1.33, 11.6, 56.0),
    "K": (0.33, 11.3, 119.0), "D": (1.38, 13.0, 54.0), "E": (0.92, 12.3, 83.0),
    "M": (0.00, 5.7, 105.0), "W": (0.13, 5.4, 170.0),
}


def blosum62():
    """BLOSUM62 as dict[(a, b)] -> int. Requires biopython."""
    from Bio.Align import substitution_matrices
    m = substitution_matrices.load("BLOSUM62")
    return {(a, b): float(m[a, b]) for a in AA20 for b in AA20}
