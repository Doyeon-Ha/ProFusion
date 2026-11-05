# ProFusion

Source code for the Research in Computational Molecular Biology (RECOMB) 2027
submission. ProFusion raises the accuracy of a protein language model (pLM) on
variant effect prediction by fusing its zero-shot score with five auxiliary
channels, using weights fitted entirely outside the evaluation benchmark.

The five channels are GEMME, a stability change on folding (ΔΔG) predicted by
SPURS, relative solvent accessibility (RSA), BLOSUM62 and the change in residue
charge. Each channel is split into a position-mean component and a
within-position residual component, giving seven weighted terms. That seven-term
model is `comp7`, which the manuscript calls ProFusion. Fitting is a coordinate
descent over a fixed grid; there is no gradient training and no learned
parameters beyond those seven numbers.

The code covers the variant effect prediction results of the manuscript and the supplementary
material: weight fitting, evaluation across 12 backbones and 2 assay axes, channel ablations
and Shapley attribution, the axis cross-application control, the ProteinGym-internal tuning
control, the appendix Stability axis, variant scoring for a new protein, and figure rendering.
The [figure and table map](#figures-and-tables) identifies each reproduction path.

⚠ **The active-residue identification results are not in this release.** Everything the
manuscript reports on identifying ligand-binding residues is excluded: Section 4.5 and its
figure, the dataset section that describes the PDBbind-derived proteins, and the two
supplementary sections on position-level tuning and on the threshold sweep. See
[Known limitations](#known-limitations) item 1.

## Installation

Run commands from this repository's root. The environment targets Linux and
Python 3.12 or later. No graphics processing unit (GPU) is required for anything
in this repository.

```bash
git clone https://github.com/Doyeon-Ha/ProFusion.git
cd ProFusion
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test,figures]"
.venv/bin/python -c 'import numpy, pandas, scipy, sklearn, matplotlib; print(numpy.__version__)'
```

`pyproject.toml` pins the versions that produced the frozen values in
`reference/`: numpy 2.4.3, pandas 3.0.1, scipy 1.17.1, scikit-learn 1.8.0 and
matplotlib 3.10.0. Another version may shift the last digits of a metric; the
weights lie on a grid and should not move.

Two optional extras cover work beyond reproducing the numbers.
`pip install -e ".[pipeline]"` adds what the channel-cache assembly needs, and
`pip install -e ".[inference]"` adds what scoring a new protein needs.

## Data downloads and access

The raw sources are public. ProteinGym supplies the evaluation assays and MaveDB
supplies the external tuning assays. This repository does not redistribute either
one, and it does not contain the derived channel cache.

**The channel cache is distributed separately.** It holds 2,761 files totalling
966 MB, which is too large to track in git. Point the code at it with an
environment variable, or place it (or a symbolic link to it) at `data/`:

```bash
export PLMFUSION_DATA=/path/to/plmfusion_data
```

| Input | Source | Required local input |
| --- | --- | --- |
| Evaluation assays and labels | [ProteinGym](https://proteingym.org/) | folded into the channel cache under `proteingym/two_dim/` |
| External tuning assays | [MaveDB](https://www.mavedb.org/) | folded into the channel cache under `mavedb_external/two_dim/` |
| Channel cache (966 MB) | distributed separately; contact the authors | `PLMFUSION_DATA`, or a `data/` symbolic link |

[DATA.md](DATA.md) lists what the cache contains, how the code reads it, every
npz key the paper numbers use, and the integrity check
(`sha256sum -c MANIFEST.sha256`). [pipeline/README.md](pipeline/README.md)
documents how the cache was built from the public sources.

The cache directory may be called anything. Nothing in the code reads its name.

⚠ Check which snapshot you have. Earlier snapshots exist and differ in which
backbones and which assays they contain. Pointing at the wrong one leaves the
code running normally while backbones disappear silently. The directory name
does not identify a snapshot; the sha256 of `MANIFEST.sha256` does, and it is
`0f1fd5588c6e...` for the **2026-11-03** snapshot that the frozen values in
`reference/` were produced with. `scripts/link_data.sh` checks this and warns on
a mismatch.

**Trained model weights are not needed.** ProFusion has no weights beyond the
seven fitted numbers, which are in `reference/paper_weights.csv`. The backbone
scores they are fused with are already in the channel cache.

## Hardware resources

CPU means central processing unit and RAM is host random-access memory. The
table records reference measurements and provisioning guidance, **not measured
minimum hardware requirements**.

| Stage | GPU | CPUs | Host RAM | Wall time |
| --- | --- | --- | --- | --- |
| Paper numbers, one cell (one backbone, one axis) | None | 4 threads | 210 MiB peak resident | 4 min 45 s measured |
| Paper numbers, all 24 cells in parallel | None | 24 x 4 threads | Provision about 6 GiB | about 6 min |
| Full pipeline (`run_all.sh`) | None | 6 threads default | Provision about 4 GiB | about 2.5 h |
| Channel-cache assembly (`pipeline/run_assemble.sh`) | None | 1 | Provision about 4 GiB | about 3 min |
| Scoring one new protein (`inference/`) | Optional | 1 to 4 | Provision about 8 GiB | minutes, see below |

The one-cell figures are a measurement on a single backbone and axis
(ESM2, function); they are a reference point, not a bound for every cell. The
parallel figure is derived from it and from the default 24-way split, not
measured as a whole.

**About 80% of the full pipeline is one step.** `scripts/ablation_lattice.py`
fits 32 channel subsets across 24 cells and runs for roughly two hours on a
single core. Everything else together finishes inside 30 minutes. The cells are
independent, so that step parallelises the same way
`scripts/run_paper_numbers.sh` already does, but it is shipped serial because
that is how the published numbers were produced.

Scoring a new protein is the one place with external dependencies: GEMME runs in
docker (`elodielaine/gemme`), ΔΔG needs the SPURS repository and checkpoint in
their own environment, and RSA needs `mkdssp`. A GPU only speeds up the pLM
log-likelihood ratio (LLR) step. See [inference/README.md](inference/README.md).

## Check the implementation

The tests read the channel cache and compare against the frozen values in
`reference/`. They cover the evaluation cohort sizes, the fitting-set sizes, the
fitted weights, every published metric, the numbers each figure drew, and the
equivalence between the inference path and the evaluation path.

```bash
.venv/bin/python -m pytest -q tests
```

The weights are compared exactly, because they lie on a grid. The metrics are
allowed a floating-point difference of 1e-9. Tests skip rather than fail when a
result file has not been generated yet, so run the pipeline first for a complete
check. These tests verify the implementation against frozen values; they do not
recompute the benchmark from the raw sources.

## Reproduce the paper numbers

```bash
bash scripts/run_all.sh     # numbers, then figure inputs, then figures
pytest tests/               # compare results/ against reference/
```

Each step logs to `results/logs/` and records success in `results/logs/STATUS`.
A correct run has no `FAILED` line there. For the paper numbers alone, without
the ablations or the figures, run `bash scripts/run_paper_numbers.sh`.

The protocol is the claim of the paper, so it is stated exactly:

| | |
| --- | --- |
| Fitting | MaveDB external set only. Anything identical or homologous to a ProteinGym sequence is excluded with MMseqs2, and the rule is frozen by sha256 |
| Evaluation | ProteinGym only. **No part of the fitting path touches it** |
| Axes | function (129 DMS) and expression (18 DMS), fitted separately. The two most useful channels swap between axes, so the axes are never pooled |
| Grid | all 7 slots from −3.0 to 3.0 in steps of 0.1 (61 points), coordinate descent, 3 passes, starting from zero |
| Fitting set | 35 deep mutational scanning (DMS) assays on function (34 for ESM-1v, which lacks one score) and 14 on expression |
| Backbones | 12, restricted to those with an LLR on both sides |

⚠ The `axis` column of the output and the file names under `results/` use the
older name `stability` for the expression axis. The appendix ΔΔG stability axis
is a different set of 65 assays, named `stability_ddg`.

⚠ State the grid whenever you argue a mechanism from an individual weight. An
earlier grid had a lower bound of 0 on two slots, which made a fitted value of
exactly 0 an artefact of the boundary rather than a result. The ordering of the
arms by performance is the same under both grids.

## Score a new protein

Given a multiple sequence alignment and a structure, `inference/` scores the
variants of one protein with the fitted weights. **No weights are refitted and no
labels are needed.**

```bash
python inference/predict_protein.py \
  --seq p.fasta --a3m p.a3m --pdb p.pdb \
  --assay Binding --backbone ESM2 --out out/
```

A larger fused score means closer to wild type. The scores are a ranking **within
one protein** and are not comparable across proteins.
[inference/README.md](inference/README.md) covers the inputs, the per-step cost,
the validation against the cached channels, and six conditions under which the
output should not be trusted.

## Figures and tables

Figures and the numbers they drew land in `results/figures/`; figure inputs land
in `results/fig_data/`. The file names on the manuscript side differ, because
they were renamed on upload.

| Manuscript | Generator | Output under `results/` |
| --- | --- | --- |
| Figure 1 | none, a hand-drawn schematic | — |
| Figure 2 | `figures/plot_fig1_ext_v3.py` | `figures/fig1_ext_v3.pdf` |
| Figure 3 | `figures/plot_axis_weights.py` | `figures/fig_axis_weights.pdf` |
| Figure 4 | `figures/plot_model_needs_2ax.py --bymodel` | `figures/fig_model_needs_2ax_bymodel.pdf` |
| Figure 5 | `figures/plot_sbi_figure.py` | `figures/sbi_figure.pdf` |
| Table 1 | `scripts/ablation_table.py` | `ablation_table.tex`, `.json` |
| In-text numbers | `scripts/final_numbers.py`, `scripts/paneld_numbers.py` | `final_numbers.json`, `FINAL.md` |
| Supplementary, tail decomposition | `figures/plot_supp_breakdown.py --both --ecdf-b` | `figures/figS_breakdown_ecdf_b_147.pdf` |
| Supplementary, eta-squared criterion | `figures/plot_eta2_criterion.py` | `figures/eta2_criterion.pdf` |
| Supplementary, how much pLM is needed | `figures/plot_plm_need.py` | `figures/fig_plm_need.pdf` |
| Supplementary, internal tuning control | `figures/plot_internal_tune.py` | `figures/internal_tune.pdf` |
| Supplementary, axis cross-application | `figures/plot_axis_crossapply.py` | `figures/axis_crossapply.pdf` |
| Supplementary, case study | `figures/plot_fig5_case.py` | `figures/fig5_case.pdf` |
| Supplementary, appendix Stability | `figures/plot_stability_three_arm.py` | `figures/stability_13bb_three_arm.pdf` |

⚠ `plot_model_needs_2ax.py` produces a different layout with no arguments. The
manuscript version is `--bymodel`, and `run_all.sh` passes it.

The figure font is Source Sans Pro, with DejaVu Sans as the fallback.
`figures/legacy/` holds three figures from an earlier version of the manuscript,
with the reason in `figures/legacy/README.md`.

## Reference values

`reference/` holds the frozen values the tests compare against: the fitted
weights, every per-assay metric, the numbers behind each figure, and two logs.
They were produced with the 2026-11-03 snapshot, 12 backbones and the grid above.
Regenerating them requires the same snapshot, since a different one can change
the cohort silently. `scripts/link_data.sh` reports whether the cache you have is
that snapshot.

## Source provenance and third-party code

The code was ported from the authors' research repository. **No research git
history is included.** Machine-specific values in `pipeline/upstream/` —
absolute paths, cluster node names and environment directories — were replaced
by placeholders such as `<REPO>` and `<MAVEDB_DIR>`; those files document how
each intermediate file was produced and are not runnable as they stand.

Channel and backbone sources carry their own terms. In particular, **AIDO RAG
16B is released under a non-commercial licence**, and whether its clause on using
outputs to train a similar model reaches fusion outputs is a question for legal
review rather than one this repository settles.

## Known limitations

1. **The active-residue identification branch is not released.** The manuscript applies the
   same fitted-weight idea to a second task, identifying ligand-binding residues on
   PDBbind-derived human proteins, and reports it in the abstract, in Section 4.5 with its
   figure, in the corresponding dataset section, and in two supplementary sections (the
   position-level weight fitting and the sweep over the critical-position threshold). None of
   that code, and none of its inputs or outputs, is in this repository, and nothing here
   reproduces those numbers. It is a separate benchmark with its own data dependencies, held
   back by the authors for now. Everything else the manuscript reports is covered.
2. **The steps that build the channel intermediate files** (LLR, GEMME, ΔΔG,
   structure prediction) are not runnable from this package; they need GPUs,
   docker and external repositories. The assembly step that turns those files
   into the cache is reproducible on CPU.
   ⚠ Assembly reproduces the original 8 backbones. For the 4 added later
   (VenusREM, VespaG, SaProt650M, AIDO RAG 16B) the upstream output lies outside
   the distributed snapshot.
3. **GEMME is not deterministic.** Rerunning it on an identical alignment and
   variant list gives different values. The effect on a fused score is small
   (Spearman 0.998 between two runs) but it is not zero.
4. **Folding-ΔG assays are not supported** by the inference path. The ΔΔG channel
   comes from a model trained on that data, so scoring them would read the
   training data back.

## License

The project's original code is licensed under the Apache License, Version 2.0.
The full text is in [LICENSE](LICENSE).

That covers this repository's own code and nothing else. The datasets and the
backbone models the code builds on carry their own terms, which apply
independently, and AIDO RAG 16B in particular is released under a
non-commercial licence. The section "Source provenance and third-party code"
above has the detail.
