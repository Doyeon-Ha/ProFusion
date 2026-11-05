#!/usr/bin/env bash
# Build the paper numbers, then the figure inputs, then the figures. CPU only.
#   bash scripts/run_all.sh            # everything
#   SKIP_PLOTS=1 bash scripts/run_all.sh
# Each step logs to results/logs/<step>.log and records success in results/logs/STATUS.
# When it finishes, run `pytest tests/` to compare against reference/.
#
# The stages run in dependency order A to D. Steps inside a stage are independent and run together.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PYTHON:-python3}"
export PYTHONPATH="$REPO/src:$REPO/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"

# Check the packages first. Otherwise the run dies two hours later, all at once, in the figures.
if ! "$PY" -c "import numpy, pandas, scipy, sklearn" 2>/dev/null; then
    echo "Numeric packages are missing. Run pip install -e . first (PYTHON= selects an interpreter)." >&2
    exit 1
fi
if [ "${SKIP_PLOTS:-0}" != 1 ] && ! "$PY" -c "import matplotlib" 2>/dev/null; then
    echo "matplotlib is missing. Run pip install -e \".[figures]\" or set SKIP_PLOTS=1." >&2
    exit 1
fi

mkdir -p results/logs
: > results/logs/STATUS
step() {
    local name=$1; shift
    if "$PY" "$@" > "results/logs/$name.log" 2>&1; then
        echo "$name ok" >> results/logs/STATUS
    else
        echo "$name FAILED (results/logs/$name.log)" >> results/logs/STATUS
    fi
}

# ---------------------------------------------------------- A. fitting and evaluation
# Independent of each other. Only paper_numbers is parallelised cell by cell (run_paper_numbers.sh).
( bash scripts/run_paper_numbers.sh > results/logs/paper_numbers.log 2>&1 \
    && echo "paper_numbers ok" >> results/logs/STATUS \
    || echo "paper_numbers FAILED (results/logs/paper_numbers.log)" >> results/logs/STATUS ) &
( step extend_backbones    scripts/extend_backbones.py ) &
( step ablation_12bb       scripts/ablation_12bb.py ) &
( step ablation_lattice    scripts/ablation_lattice.py ) &
( step axis_crossapply     scripts/axis_crossapply_12bb.py ) &
( step control_random_uni3 scripts/control_random_uni3.py ) &
( step msa_depth           scripts/msa_depth.py ) &
( step eta2                figures/data/eta2.py ) &
# Appendix Stability. Two fitting conventions (pre/off) are run to make the three arms.
# ⚠ --extra is required, not optional. It adds the ESM2mask row, which is the ESM2 the
#   appendix reports: masked-marginal on BOTH the tuning and the evaluation side. Without it
#   the row is absent, plot_stability_three_arm.py drops the mixed-convention "ESM2" row it
#   does have, and the figure comes out with 11 backbones instead of the 12 the text states.
( step stability_pre scripts/stability_protocol.py --extra --tag _pre \
  && step stability_off scripts/stability_protocol.py --extra --official --tag _off ) &
wait

# ---------------------------------------------------------- B. analyses that consume A
# The first three read A's output. Putting them inside A makes them race and raise FileNotFoundError.
( step shapley_channels     scripts/shapley_channels.py ) &      # ← ablation_lattice.csv
( step plm_vs_channel       scripts/plm_vs_channel.py ) &        # ← extend_backbones.csv
( step comp7_weights        figures/data/comp7_weights.py ) &    # ← paper_weights.csv
( step add_equal5           scripts/add_equal5.py ) &
( step plm_need             scripts/plm_need.py ) &
( step internal_tune_12bb   scripts/internal_tune_12bb.py ) &
( step sbi_positions        scripts/sbi_positions.py ) &
( step build_case156        scripts/build_case156.py ) &
( step backbone_variability scripts/backbone_variability.py ) &
( step dump_three_arm       scripts/dump_perdms_three_arm.py ) &
wait

# ---------------------------------------------------------- C. tables and in-text numbers
( step ablation_table  scripts/ablation_table.py ) &
( step final_numbers   scripts/final_numbers.py ) &
( step paneld_numbers  scripts/paneld_numbers.py ) &
wait

# ---------------------------------------------------------- D. figures
# ⚠ plot_model_needs_2ax.py defaults to mode="stack" with no argument, which is not the paper
#   version. Paper Figure 4 is --bymodel.
if [ "${SKIP_PLOTS:-0}" != 1 ]; then
    step fig1_ext_v3        figures/plot_fig1_ext_v3.py
    step axis_weights       figures/plot_axis_weights.py
    step model_needs_2ax    figures/plot_model_needs_2ax.py --bymodel
    step sbi_figure         figures/plot_sbi_figure.py
    step eta2_criterion     figures/plot_eta2_criterion.py
    # ⚠ What appears in the paper is panel b (the ECDF) alone, over both axes (147).
    #   With no argument it produces the two-panel composite over function 129 only, which is
    #   not the paper version.
    step supp_breakdown     figures/plot_supp_breakdown.py --both --ecdf-b
    # --ecdf-b draws the figure but writes no numeric CSV (it returns before that). For regression
    # checks, the same two axes are run again as the two-panel composite to leave
    # figS_breakdown_147_data.csv.
    step supp_breakdown_data figures/plot_supp_breakdown.py --both
    step plm_need_fig       figures/plot_plm_need.py
    step internal_tune_fig  figures/plot_internal_tune.py
    step axis_crossapply_fig figures/plot_axis_crossapply.py
    step fig5_case          figures/plot_fig5_case.py
    step stability_fig      figures/plot_stability_three_arm.py
fi

cat results/logs/STATUS
! grep -q FAILED results/logs/STATUS
