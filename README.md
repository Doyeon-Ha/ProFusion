# plmfusion: reproduction code for ProFusion (comp7)

ProFusion adds five auxiliary channels (GEMME, ΔΔG, RSA, BLOSUM62, Δcharge) to a protein language
model's zero-shot score. Each channel is split into a position-mean component and a within-position
residual component, and the seven resulting terms are mixed with fitted weights. That seven-weight
model is `comp7`, which the paper calls ProFusion.

Weights are fitted on an external MaveDB tuning set only. ProteinGym labels are used for evaluation
only. **No part of the fitting path touches ProteinGym.**

The code runs 12 backbones across 2 assay axes (function, 129 DMS; expression, 18 DMS).

⚠ The `axis` column of the output CSVs and the file names under `results/` still use the older name
`stability` for the expression axis. It is the same axis the paper calls expression. The appendix
ΔΔG stability axis is a different set of 65 assays, named `stability_ddg`.

## Quick start

```bash
git clone https://github.com/Doyeon-Ha/ProFusion.git && cd ProFusion
export PLMFUSION_DATA=/path/to/plmfusion_data      # channel cache, see DATA.md
python -m venv .venv && source .venv/bin/activate  # conda works too (python >= 3.11)
pip install -e ".[test,figures]"
bash scripts/run_all.sh                            # paper numbers and figures. CPU only, about 2.5 h
pytest tests/                                      # checks results/ against reference/
```

- No GPU is used.
- For the paper numbers alone, run `bash scripts/run_paper_numbers.sh` (about 6 minutes).
- Most of the 2.5 hours is a single step, `scripts/ablation_lattice.py` (32 channel subsets x 24
  cells, about 2 hours). Table 1 and the Shapley values read its output. Everything else together
  takes under 30 minutes.
- The channel cache is not in this repository. [DATA.md](DATA.md) describes what it contains and how
  to point the code at it; [pipeline/README.md](pipeline/README.md) describes how it was built.
- Each step logs to `results/logs/`, and `results/logs/STATUS` records success or failure. A correct
  run has no `FAILED` line in `STATUS`.
- Four Python files are not called directly by `run_all.sh` and are not dead code.
  `scripts/paper_numbers.py` is invoked cell by cell by `run_paper_numbers.sh`, and
  `figures/{backbone_key,fig_common,plot_fig1_ext}.py` are shared modules the figure scripts import.

## Layout

```
ProFusion/
├── scripts/
│   ├── run_all.sh               full run (numbers, then figure inputs, then figures)
│   ├── run_paper_numbers.sh     paper numbers over 24 cells in parallel (same result as serial)
│   └── paper_numbers.py         paper numbers. --bb/--axis for one cell, --merge to combine
├── src/plmfusion/
│   ├── paths.py                 data and output paths (the only place paths are defined)
│   ├── common.py                7-slot fusion record (Rec), coordinate descent, protein-wise K-fold
│   ├── grids.py                 weight grids. All paper weights use the "uni3" grid
│   ├── external.py              MaveDB external-set record (ERec) and loader
│   └── predict.py               scoring without labels (for inference)
├── figures/
│   ├── data/                    scripts that build figure inputs
│   ├── legacy/                  older figures no longer in the paper (see legacy/README.md)
│   ├── backbone_key.py          backbone names, colours, structure-conditioning flag (shared)
│   ├── fig_common.py            fonts, sizes, save conventions (shared)
│   ├── plot_fig1_ext.py         panel functions used by plot_fig1_ext_v3 (shared)
│   └── plot_*.py                the 11 paper figures
├── inference/                   scoring a new protein from MSA and structure (inference/README.md)
├── pipeline/                    how the channel cache npz files were built (pipeline/README.md)
│   ├── assemble/                intermediate files to npz. Reproducible on CPU
│   ├── upstream/                the code that produced the intermediate files (GPU, docker)
│   ├── run_assemble.sh          assemble, then compare against the cache
│   └── check_synonymous.py      measures whether 842 synonymous rows change the numbers
├── reference/                   frozen values (paper CSVs; figure data under reference/fig_data/)
├── results/                     run output (not tracked by git)
└── tests/
    ├── test_reproduce.py        compares results/ against reference/
    └── test_predict.py          checks the inference path scores the same as the evaluation path
```

## Protocol

| | |
|---|---|
| Fitting | MaveDB external set. Anything identical or homologous to a ProteinGym sequence is excluded with MMseqs2 (`pg_tier=='none'`), and the rule is frozen by sha256 |
| Evaluation | ProteinGym. **Labels are used for evaluation only.** There is no fitting loop on the ProteinGym side |
| Axes | function and expression are fitted separately. The first and second most useful channels swap between axes, so the two are never pooled |
| Grid | **uni3**: all 7 slots from −3.0 to 3.0 in steps of 0.1 (61 points), coordinate descent, 3 passes, starting from zero |
| Backbones | 12, restricted to those with an LLR on both the ProteinGym and the external side (`paper_numbers.EXTBB`) |

⚠ **State the grid whenever you argue a mechanism from an individual slot value.** The earlier
`wide` grid had a lower bound of 0 on the GEMME and ΔΔG slots, so those slots could not go negative.
Under `wide`, the a_ddG value for ESM3 and ProSST was exactly 0; under `uni3` it is −0.1 for ESM3 and
0.2 for ProSST. The ordering of the arms by performance is the same under both grids.

## What paper_numbers.py produces

| Group | Contents |
|---|---|
| F1 fusion ladder | pLM alone, then champ3 (3 pooled channels), champ5 (+2 substitution channels), comp7 (position/residual split) |
| F3 channel ablation | comp7 with one channel removed, and with one channel only |
| F4 implementation identity | the substitution axis of "only RSA" equals the pLM substitution axis; the position axis of "only substitution" equals the pLM position axis |

F4 should not be expected to be exactly zero. The amount a residual channel adds to a position mean
is mathematically zero but about 1e-16 in floating point, and when the baseline has positions whose
means are *exactly* equal, that residue breaks the tie and changes the ranking. The measured upper
bound is 4e-05. Suspect the decomposition only if it exceeds 1e-4.

### Metrics

Every metric is computed within one DMS and then averaged over DMS.

| Column | Definition |
|---|---|
| `rho` | Spearman correlation between the fused score and the DMS label. Unitless, [-1, 1], higher is better |
| `rho_pos` | Position axis. Spearman between per-position mean score and per-position mean label |
| `rho_sub` | Substitution axis. Mean of the Spearman values measured within positions that have at least 5 variants |
| `p10` | Fraction of the 10 most damaging variants recovered (overlap between the top 10 predicted and the top 10 labelled) |
| `ef1` | Fraction of damaging variants in the top 1% of predictions (at least 3) divided by the overall fraction. Damaging means the bottom 20% of labels |
| `auroc` | AUROC under the same definition of damaging |

### Output (results/)

- `paper_perdms.csv`: raw values per DMS, backbone and arm
- `paper_summary.csv`: means per axis, backbone and arm
- `paper_weights.csv`: the 7 fitted weights. Column order is a_GEMME, b_GEMME, a_ddG, b_ddG, a_RSA, b_BLOSUM, b_dchg
- `logs/paper_numbers.log`: the tables and the F4 identity output

## Figures

Figures (PDF, PNG) and the numbers they draw (`*_data.csv`) land in `results/figures/`; figure inputs
land in `results/fig_data/`.

The figures and tables in the paper map to the generators below. The file names on the paper side
were changed when the files were uploaded to Overleaf, so they may differ from the package output.

| Paper | Generator | Package output (`results/`) |
|---|---|---|
| Figure 1 `main_arch.pdf` | none (hand-drawn schematic) | — |
| Figure 2 `fig1_ext_v3.pdf` | `figures/plot_fig1_ext_v3.py` | `figures/fig1_ext_v3.pdf` |
| Figure 3 `fig_axis_weights.pdf` | `figures/plot_axis_weights.py` | `figures/fig_axis_weights.pdf` |
| Figure 4 `fig_model_needs_2.pdf` | `figures/plot_model_needs_2ax.py --bymodel` | `figures/fig_model_needs_2ax_bymodel.pdf` |
| Figure 5 `sbi_figure_add.pdf` | `figures/plot_sbi_figure.py` | `figures/sbi_figure.pdf` |
| Table 1 | `scripts/ablation_table.py` | `ablation_table.tex`, `.csv` |
| In-text numbers | `scripts/final_numbers.py`, `scripts/paneld_numbers.py` | `logs/final_numbers.log` and others |
| Supp `figS_breakdown_ecdf_b_147.pdf` | `figures/plot_supp_breakdown.py` | `figures/figS_breakdown_ecdf_b_147.pdf` |
| Supp `eta2_criterion.pdf` | `figures/plot_eta2_criterion.py` | `figures/eta2_criterion.pdf` |
| Supp `fig_plm_need.pdf` | `figures/plot_plm_need.py` | `figures/fig_plm_need.pdf` |
| Supp `fig_pg_tuned.pdf` | `figures/plot_internal_tune.py` | `figures/internal_tune.pdf` |
| Supp `axis_crossapply.pdf` | `figures/plot_axis_crossapply.py` | `figures/axis_crossapply.pdf` |
| Supp `fig5_case.pdf` | `figures/plot_fig5_case.py` | `figures/fig5_case.pdf` |
| Supp `fig_stability.pdf` | `figures/plot_stability_three_arm.py` | `figures/stability_13bb_three_arm.pdf` |

⚠ Running `plot_model_needs_2ax.py` with no arguments produces the `stack` variant. The paper version
is `--bymodel`, and `run_all.sh` passes that argument.

- The `_147` in `figS_breakdown_ecdf_b` is the number of DMS across both axes (129 + 18). The file
  name changes with the cohort.
- `figures/legacy/` holds three older figures that are no longer in the paper, with the reason in
  `figures/legacy/README.md`.
- `figures/plot_fig1_ext.py` is not a paper figure. It is a **shared module** whose panel functions
  `plot_fig1_ext_v3.py` imports. Do not run it on its own.
- `plot_supp_breakdown.py` runs twice. `--both --ecdf-b` produces the figure that appears in the
  paper, and `--both` leaves the numeric CSV (`figS_breakdown_147_data.csv`) used for regression
  checks.
- The figure font is Source Sans Pro. DejaVu Sans is used when it is unavailable.

## Where to start reading

- `common.Rec`: the position-mean and residual decomposition (`split`)
- `common.descend_tied`: the coordinate descent
- `scripts/paper_numbers.py`: the metric definitions in `metrics`
- `scripts/run_paper_numbers.sh`: why splitting the work into cells gives the same result

## Reference values (reference/)

Produced with the `uni3` grid, 12 backbones and the **2026-11-03** data snapshot (function 129,
expression 18). The tests require the weights to match exactly and allow a difference of up to 1e-9
on the metrics.

⚠ To regenerate the reference values you must point at the same data snapshot. A different snapshot
can change the cohort silently, which changes every number downstream.

Environment used: Python 3.12, numpy 2.4.3, pandas 3.0.1, scipy 1.17.1, scikit-learn 1.8.0,
matplotlib 3.10.0 (pinned in `pyproject.toml`).

## Known limitations

1. **The steps that build the channel intermediate files** (LLR, GEMME, ΔΔG, structure prediction)
   are not runnable from this package. They need GPUs, docker and external repositories. The code is
   kept under `pipeline/upstream/` as a record of how each file was produced, with machine-specific
   paths replaced by placeholders. The assembly step that turns those intermediate files into npz is
   reproducible on CPU ([pipeline/README.md](pipeline/README.md)).
   ⚠ Assembly reproduces the original 8 backbones. For the 4 added later (VenusREM, VespaG,
   SaProt650M, AIDO RAG 16B) the upstream output is outside the data snapshot.
2. **The channel cache is distributed separately.** It is 966 MB and is not in this repository. See
   [DATA.md](DATA.md).
3. **AIDO RAG 16B is released under a non-commercial licence.** Whether §1.2.2 of that licence
   (which forbids using outputs to train a similar model) applies to fusion outputs is a question for
   legal review, not something this repository settles.
