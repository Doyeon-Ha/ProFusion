# How the channel cache was built

The paper numbers start from one npz per DMS (`data/*/two_dim/*.npz`). This directory records
where those npz came from.

| | Contents | Reproducible |
|---|---|---|
| `assemble/` | intermediate files **assembled** into npz | yes, about 3 minutes on CPU |
| `upstream/` | the code that produced the intermediate files (LLR, GEMME, ddG, RSA, structures) | no. Kept as a record, with machine-specific paths replaced by placeholders |

## Rerunning the assembly step

```bash
bash pipeline/run_assemble.sh
```

This writes fresh npz into `results/rebuild/` and compares them against the existing cache, key by
key. `data/` is never touched. It needs `pip install -e ".[pipeline]"` for torch (a CPU build is
enough) and biopython.

**Checked on 2026-10-14:** all 194 ProteinGym and 49 external npz matched the cache of the time
exactly, over 5,820 and 1,713 (DMS, key) pairs respectively, with zero mismatches.

⚠ That check was made **when there were 8 backbones**. The snapshot has since grown to 212
ProteinGym npz (129 + 18 in the main text, 65 in the appendix), and for the four backbones added
later (VenusREM, VespaG, SaProt650M, AIDO RAG 16B) the upstream output lies outside the snapshot,
so they are outside what assembly can reproduce.

| File | What it does |
|---|---|
| `assemble/e_mix3.py` | per-DMS channel bundle loader for ProteinGym |
| `assemble/build_features.py` | assembles the two main ProteinGym axes (function 129 + expression 18) |
| `assemble/build_stability.py` | assembles the 65 ProteinGym stability_ddg DMS (**appendix only**, a different axis from the two main ones) |
| `assemble/build_channels.py` | assembles the 49 MaveDB external DMS |
| `assemble/aa_props.py` | BLOSUM62 and volume constants |
| `assemble/verify.py` | compares a rebuild against the cache |

The only changes from the originals are the input and output paths, plus one check added to
`build_channels.py`: an `assert` that every backbone has the same variant list, in the same order,
as ESM2. The original applied the ESM2 row mask to every backbone without checking. On the current
data they all match.

## Intermediate files (distributed with the data)

| Used for | Path (under `data/`) | Size |
|---|---|---|
| labels, single-substitution lists, SPURS ddG | `proteingym/spurs/{dms,matrix}` | 126M |
| ESM-1v scores, deciding DMS inclusion | `proteingym/detango/data/*/collection.csv` | 165M |
| ESM2, ESM3 and ProSST scores | `proteingym/zeroshot/{ESM2,ESM3,ProSST}` | 48M |
| GEMME, RSA | `proteingym/{gemme,node_physics_v2}` | 23M |
| position embeddings (for the `hwt` key only) | `proteingym/esm3_seqonly_embeddings.pkl` | 343M |
| external-set LLR for 12 backbones | `mavedb_external/llr/<backbone>` | 51M |
| external-set GEMME, RSA and ddG | `mavedb_external/{gemme,rsa,ddg}` | 13M |
| external-set DMS list and score direction | `mavedb_external/{frozen.csv,direction_manual.csv}` | 34K |

`esm3_seqonly_embeddings.pkl` is used only for the `hwt` key and for deciding DMS inclusion. It
enters none of the keys the paper numbers use.

## The steps that produced the intermediate files (upstream/, not rerun)

These need GPUs, docker, external repositories and node-local disks, so they were not adapted to
run from this package. The original files are kept with their structure intact, and
machine-specific values (absolute paths, cluster node names, environment directories) replaced by
placeholders such as `<REPO>` and `<MAVEDB_DIR>`. They document how each file was produced; they
are not runnable as they stand.

### ProteinGym (the evaluation set)

| Step | Code | Environment |
|---|---|---|
| SPURS ddG | `spurs_ddg/run_spurs_proteingym.py` | spurs env, A100 |
| ESM-1v (5 models) | `detango_spurs/{prepare_data,run_esm1v,merge_stability}.py` | DETANGO venv, A100, several hours |
| ESM2, ESM3, ProSST | `enzyme_activity_poc/zeroshot_poc.py` | GPU |
| GEMME | `runs/gemme_all/*` then `amortized/build_gemme_aligned.py` | docker, CPU |
| RSA | `build_node_physics_v2.py` | CPU (DSSP) |

### MaveDB external set (the fitting set)

| Step | Code | Environment |
|---|---|---|
| 1. freeze the DMS set (49/23) | `mavedb_external/freeze_tuneset.py` + `excluded.yaml` | CPU, MMseqs2 |
| 2. score direction | `mavedb_external/resolve_direction.py` + `manual_overrides.yaml` | CPU |
| 2b. query FASTA and manifest | `_from_session/step2b_query_fasta_manifest.sh` | CPU |
| 3. MSA | `mavedb_external/run_hhblits.sh` | hhsuite, UniRef30 |
| 3b. GEMME inputs | `_from_session/step3b_gemme_workdirs.sh` | CPU |
| 4. GEMME | `mavedb_external/run_gemme.sh` | rootless docker |
| 5a. structure prediction inputs | `_from_session/step5a_fold_inputs.sh` | CPU |
| 5. structures | `mavedb_external/slurm_fold.sbatch` | ColabFold, A100 |
| 6. RSA | `mavedb_external/build_rsa.py` | mkdssp, CPU |
| 7. ddG | `mavedb_external/build_ddg.py` | spurs env, CPU |
| 8. ESM2 and ESM-1v | `mavedb_external/build_llr.py` | CPU, about 50 minutes |
| 8b. ProSST and ESM3mm | `mavedb_external/build_llr_struct.py` (+ `build_function_annotations.py`) | A100 |
| 8c. figure-only backbones | `build_llr_structonly.py`, `build_llr_masked.py`, `run_s2f*.py`, `ingest_s2f.py`, `slurm_llr_ladder.sbatch` | mixed |

The three `_from_session/` steps never existed as files in the repository. The commands were
recovered verbatim from the working record, and each file states its origin at the top.

## Things to know

**1. The fitting set contains 842 synonymous substitutions.**
Five external DMS (SGCB 304, TECR 294, NCS1 183, STIM1 30+31) have rows where `wt == mut`, 0.34%
of all 248,272 rows. The evaluation set (ProteinGym) has none.

The regular expression used in step 3b, `^p\.(Xxx)(\d+)(Xxx)$`, did not filter out synonymous or
nonsense variants, and that variant list went straight into the GEMME and LLR intermediate files.
The regular expression in the other steps was fixed on 2026-10-07 and now blocks both with
`(?!\1$)(?!Ter$)`. Rerunning step 3b or step 8 with the current code therefore drops those rows
and produces different intermediate files.

The set of 49 DMS used for fitting does not change. Counted with the new regular expression, the
minimum missense count is still 131, above the threshold of 100.

**The effect was measured (2026-11-03, 12 backbones, the paper grid uni3).** Those rows were
removed, comp7 refitted, and the result applied to the evaluation set.

| | Result |
|---|---|
| weights | of 24 cells (2 axes x 12 backbones), only 5 were unchanged and 19 moved |
| evaluation | the d_rho of those 19 cells ranges from −0.0032 to +0.0033 (11 positive, 7 negative, 1 zero) |

**The direction is mixed.** Cells that improve when the rows are removed (function SaProt650M
+0.0033, function Tranception L +0.0026) sit alongside cells that get worse (function VenusREM
−0.0032, function ESM2 −0.0017). Nothing exceeds 0.0034, so no conclusion in the paper changes,
but it cannot be said that removing them is better.

⚠ The first measurement, on 2026-10-14, used 4 backbones and the wide grid. There, only 2 of 8
cells moved and both improved, which is why it was once written that removing them was slightly
better. Matching the backbone roster and the grid to the paper protocol breaks that statement. The
table above is the correct one.

The measurement can be rerun with `pipeline/check_synonymous.py`, and its output is kept in
`reference/check_synonymous.reference.log`.

Method: the rows are dropped from the npz and the z-normalisation, position means and label ranks
recomputed on the remaining subset. GEMME and LLR are computed independently per variant, so the
raw scores of the remaining variants are unchanged by a shorter list. This therefore gives the same
result as rerunning steps 3b and 8.

**2. ESM-1v is scored differently in the fitting set and the evaluation set.**
The evaluation set uses masked-marginal (`run_esm1v.py`) and the fitting set wildtype-marginal
(`build_llr.py`). Only the averaging over 5 models is shared.

**2b. ESM2 is masked-marginal, and two keys for it exist side by side.**
Every npz carries both `llr_ESM2` (wildtype-marginal, the original) and `llr_ESM2mask`
(masked-marginal, the official ProteinGym convention). The paper uses the masked one:
`common.LLR_ALIAS = {"ESM2": "ESM2mask"}` redirects the evaluation side, so asking for the
backbone `ESM2` reads `llr_ESM2mask`. ⚠ Reading `llr_ESM2` directly gives the wildtype-marginal
quantity instead, with no error and plausible-looking numbers. Of the remaining backbones,
ESM3 and ProSST are wildtype-marginal on both sides.

The two keys also differ on the **fitting** side, and there the choice is explicit rather than
aliased. `scripts/stability_protocol.py` keeps both: the backbone `ESM2` means masked on
evaluation but wildtype on tuning, and `ESM2mask` (added by `--extra`) means masked on both.
The appendix Stability axis reports `ESM2mask`.

**3. ESM3 is multimodal on both sides.**
`build_llr.py`, `build_llr_struct.py`, `build_channels.py` and the comments in the original README
described the evaluation-set `llr_ESM3` as sequence-only, which was wrong. A verification job
recomputed the same 171 DMS as `sequence+structure+function` and matched the existing cache on
171 of 171.

**4. The fitting set and the evaluation set are built from different materials.** This is by
design, not a defect.

| | Fitting set (MaveDB) | Evaluation set (ProteinGym) |
|---|---|---|
| MSA | hhblits with UniRef30 | the benchmark's own MSA |
| structure | ColabFold | the benchmark's AF2 structures |
| ProSST structure tokens | computed from our structures | the official benchmark tokens |
| missing RSA | the row is dropped | filled with 0.0 |

**5. The evaluation set now includes long proteins (unlike earlier versions).**
ESM-1v scores used to come from DETANGO output, which caps at 1022 residues, so proteins longer
than that dropped out across every backbone at once (function 113 / expression 16). Long proteins
have since been rescored and added back.

| | Earlier (2026-10-18 snapshot) | Now (2026-11-03 snapshot) |
|---|---|---|
| function | 113 | **129** (15 added, none lost) |
| expression | 16 | **18** (2 added, none lost) |
| variants across the two main axes | 568,510 | **627,230** |
| longest protein | at most 1,022 residues | 1,368 residues (`CAS9_STRP1_Spencer_2017_positive`) |

All 12 backbones have an LLR on all 147 DMS (none missing). Everything that changed was an
addition; no DMS was lost. Part of why function is 129 rather than 114 is that `common.DROP` was
emptied: `CCDB_ECOLI_Adkar_2012` used to be excluded.

**6. GEMME is not deterministic.**
Rerunning it on the same alignment and variant list gives different values (measured on SOD1:
Spearman 0.972, 0.65% identical values). The conservation computation uses random sampling and has
no seed option. The cached values are therefore canonical for the `ch_gemme` channel, and a fresh
GEMME run will not match them. Reproducing the assembly step (`run_assemble.sh`) takes the GEMME
output as input, so it is unaffected. The numbers are in
[inference/README.md](../inference/README.md).

**7. The original scripts wrote straight into the shared cache.**
The original output path of `build_features.py` and `build_channels.py` was `data/.../two_dim`.
Several scripts read that directory by unconditional glob, so a single added or overwritten file
changes results silently. In this package the output goes to `results/rebuild/` instead, and the
data directory is kept read-only.
