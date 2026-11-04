"""Check that a run (results/) matches the frozen values (reference/).

The full pipeline takes about 2.5 hours on CPU, so it is not run inside the tests. Run
`bash scripts/run_all.sh` (or the individual scripts) first, then pytest. Anything with no
result file or no frozen value is skipped.

The weights lie on a grid and must match exactly. The metrics are allowed a floating-point
difference of 1e-9.
"""
from __future__ import annotations

import pandas as pd
import pytest

from plmfusion.paths import REFERENCE, RESULTS

# File -> sort key. A key of None compares the rows in their existing order.
PAPER = {
    "paper_weights.csv": ["axis", "bb", "arm"],
    "paper_summary.csv": ["axis", "bb", "arm"],
    "paper_perdms.csv": ["axis", "bb", "arm", "dms"],
}
# The evaluation cohort. If these numbers change, the data snapshot or common.DROP changed, and
# every metric comparison below will break. Report what moved here first.
COHORT = {"function": 129, "stability": 18}
N_BACKBONES = 12
# Sizes of the weight-fitting set: the 49 external npz filtered by axis and llr availability.
# ESM-1v has 34 on function because one external npz has no llr_ESM1v.
TUNE = {("function", "ESM1v"): 34, "function": 35, "stability": 14}

# The numbers the figures actually drew. Unlike the input CSVs (fig_data/), these land next to
# the figures (figures/). The frozen copies are collected in reference/fig_data/, in paper order.
FIG_DRAWN = {
    "fig1_ext_v3_data.csv": None,                   # Figure 2
    "fig_axis_weights_data.csv": None,              # Figure 3
    "fig_model_needs_2ax_bymodel_data.csv": None,   # Figure 4
    "sbi_figure_data.csv": None,                    # Figure 5
    "figS_breakdown_147_data.csv": None,            # Supp decomposition ECDF
    "fig_plm_need_data.csv": None,                  # Supp how much pLM is needed
    "internal_tune_data.csv": None,                 # Supp ProteinGym-internal fitting control
    "stability_13bb_three_arm_data.csv": None,      # Supp appendix Stability
}
FIG_DATA = {
    "panelc_comp7_weights.csv": None,
    "panelb_eta2_verified.csv": None,
}


def _pair(rel, keys, ref_rel=None):
    got = RESULTS / rel
    want = REFERENCE / (ref_rel or rel)
    if not got.exists():
        pytest.skip(f"{got} is missing. Run the script that generates it first.")
    if not want.exists():
        pytest.skip(f"no frozen value: {want}")
    ref, new = pd.read_csv(want), pd.read_csv(got)
    if keys:
        ref = ref.sort_values(keys).reset_index(drop=True)
        new = new.sort_values(keys).reset_index(drop=True)
    return ref, new


def test_weights_exact():
    ref, new = _pair("paper_weights.csv", PAPER["paper_weights.csv"])
    pd.testing.assert_frame_equal(ref, new, check_exact=True)


@pytest.mark.parametrize("name", ["paper_summary.csv", "paper_perdms.csv"])
def test_metrics_match(name):
    ref, new = _pair(name, PAPER[name])
    pd.testing.assert_frame_equal(ref, new, check_exact=False, rtol=0, atol=1e-9)


@pytest.mark.parametrize("name", sorted(FIG_DATA))
def test_figure_data_match(name):
    ref, new = _pair(f"fig_data/{name}", FIG_DATA[name])
    pd.testing.assert_frame_equal(ref, new, check_exact=False, rtol=0, atol=1e-9)


@pytest.mark.parametrize("name", sorted(FIG_DRAWN))
def test_figure_drawn_match(name):
    """Do the numbers the figure drew match the frozen values? A skip means it was never drawn."""
    ref, new = _pair(f"figures/{name}", FIG_DRAWN[name], ref_rel=f"fig_data/{name}")
    pd.testing.assert_frame_equal(ref, new, check_exact=False, rtol=0, atol=1e-9)


def test_cohort_sizes():
    """Is the evaluation cohort the one in the paper (function 129, expression 18, 12 backbones)?

    Check **the cohort before the metrics**. Pointing at the wrong snapshot leaves the code
    running normally while DMS or backbones quietly disappear, and a metric comparison alone
    does not reveal why.
    """
    got = RESULTS / "paper_perdms.csv"
    if not got.exists():
        pytest.skip(f"{got} is missing. Run scripts/run_paper_numbers.sh first.")
    P = pd.read_csv(got)
    bad = []
    for axis, n in COHORT.items():
        sub = P[P.axis == axis]
        if sub.dms.nunique() != n:
            bad.append(f"{axis}: {sub.dms.nunique()} DMS (expected {n})")
        if sub.bb.nunique() != N_BACKBONES:
            bad.append(f"{axis}: {sub.bb.nunique()} backbones (expected {N_BACKBONES})")
    assert not bad, (
        "The evaluation cohort differs from the paper. Check the data snapshot (see DATA.md) and\n"
        "common.DROP.\n  " + "\n  ".join(bad))


def test_tune_sizes():
    """Is the fitting set the size in the paper (function 35, 34 for ESM-1v, expression 14)?"""
    got = RESULTS / "extend_backbones.csv"
    if not got.exists():
        pytest.skip(f"{got} is missing. Run scripts/extend_backbones.py first.")
    D = pd.read_csv(got)
    bad = []
    for _, r in D.iterrows():
        want = TUNE.get((r.axis, r.bb), TUNE.get(r.axis))
        if want is not None and int(r.n_ext) != want:
            bad.append(f"{r.axis}/{r.bb}: fitted on {int(r.n_ext)} (expected {want})")
    assert not bad, "The fitting set sizes differ.\n  " + "\n  ".join(bad)
