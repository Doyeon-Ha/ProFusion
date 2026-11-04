"""Data paths are defined only in this file.

The data root is resolved in this order:
  1. the PLMFUSION_DATA environment variable
  2. <repo>/data  (usually a symbolic link made by scripts/link_data.sh)

DATA.md describes the layout below that root. No absolute path appears anywhere in the code.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Where output goes. It can be moved with an environment variable, which is useful for comparing
# two runs in one checkout without mixing them (for example, validating a ported figure script
# without touching the official results/).
RESULTS = Path(os.environ["PLMFUSION_RESULTS"]) if os.environ.get("PLMFUSION_RESULTS") else REPO / "results"
REFERENCE = REPO / "reference"


def data_root() -> Path:
    env = os.environ.get("PLMFUSION_DATA")
    root = Path(env) if env else REPO / "data"
    if not root.exists():
        raise FileNotFoundError(
            f"The data root {root} does not exist. Run `bash scripts/link_data.sh <path>` or set "
            "the PLMFUSION_DATA environment variable (see DATA.md).")
    return root


def pg_cache() -> Path:
    """Per-DMS channel cache for ProteinGym (one npz per DMS)."""
    return data_root() / "proteingym" / "two_dim"


def ext_cache() -> Path:
    """Per-DMS channel cache for the MaveDB external set (npz). Used for weight fitting only."""
    return data_root() / "mavedb_external" / "two_dim"


def ext_direction() -> Path:
    """The external DMS whose score direction is resolved (urn column). Only these are fitted on."""
    return data_root() / "mavedb_external" / "direction_manual.csv"
def ext_cache_named(name: str) -> Path:
    """A named external cache. The appendix Stability axis uses the Tsuboyama natural pool `two_dim_tsu129`."""
    return data_root() / "mavedb_external" / name


def ext_table(name: str) -> Path:
    """Tables that sit next to the external set (direction_*.csv, frozen_*.csv and so on)."""
    return data_root() / "mavedb_external" / name


def figure_assets(name: str = "") -> Path:
    """Assets a figure embeds as-is (rendered structure PNGs, coordinate CSVs, mmCIF).

    These are **frozen assets**, not recomputed. How they were made is in pipeline/README.md.
      sbi_gck/  the GCK panel of Figure 5 (panel_data.npz, panel_c12.csv, r_c5.png, 1v4s/1v4t.cif)
      struct/   the B2L11 panel of Supp Fig S6 (fig_struct_*.png, *_anchor.csv, 4 PDB files)
    """
    root = data_root() / "figure_assets"
    return root / name if name else root


def frozen(name: str = "") -> Path:
    """Intermediate output frozen because it cannot be rebuilt, or rebuilding it is excessive.

    paneld_perdms.csv, FINAL_perdms.csv and v1recipe_ens_perdms.csv hold derived values only, so
    that the 152 MB of raw scores from the competing method VenusREM2 need not be rerun.
    """
    root = data_root() / "frozen"
    return root / name if name else root
