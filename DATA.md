# Data

The code is in this repository; the data is distributed separately. The channel cache is about
966 MB over 2,758 files, which is too large to track in git.

Point the code at it with an environment variable:

```bash
export PLMFUSION_DATA=/path/to/plmfusion_data_2026-11-03
```

Alternatively, place the cache (or a symbolic link to it) at `data/` inside the repository.
`data/` is listed in `.gitignore`, so it is never committed. `PLMFUSION_DATA` takes priority over
`data/` when both are present.

⚠ Keep the date suffix in the directory name. Earlier snapshots exist, and they differ in which
backbones and which assays they contain. Pointing at the wrong one leaves the code running normally
while backbones disappear silently and the paper numbers fail to reproduce. The reference values in
`reference/` correspond to the **2026-11-03** snapshot.

## Contents

```
plmfusion_data_2026-11-03/
├── MANIFEST.sha256                        sha256 checksums (2,758 files)
├── proteingym/
│   ├── two_dim/<DMS_id>.npz               212 files. Evaluation set
│   │                                      (function 129, expression 18, appendix stability_ddg 65)
│   └── {detango,gemme,spurs,node_physics_v2,zeroshot}/   intermediate files for pipeline/assemble
├── mavedb_external/
│   ├── direction_manual.csv               external DMS used for fitting (score direction resolved)
│   ├── two_dim/<urn>.npz                  49 files. The weight-fitting set
│   ├── two_dim_tsu129/<urn>.npz           125 files. Tsuboyama natural-protein tuning set,
│   │                                      used only for the appendix Stability axis
│   ├── {direction,frozen}_tsu129.csv      direction and freeze lists for that set
│   ├── frozen.csv, neff.json, hhblits_run.log   freeze rule and MSA depths
│   └── {llr,gemme,ddg,rsa}/               intermediate files for reassembly
├── figure_assets/
│   ├── sbi_gck/                           rendered images for the GCK T168 panel of Figure 5
│   └── struct/                            structure panel images for fig5_case
└── frozen/                                three frozen per-DMS CSVs kept as comparison baselines
```

## How the code uses the data

- **Weight fitting (MaveDB external set).** Only DMS listed in `direction_manual.csv` whose axis
  matches and which carry an `llr_<backbone>` key for the backbone in question. The code is
  `external.ERec` and `load_ext` in `scripts/paper_numbers.py`.
- **Evaluation (ProteinGym).** DMS whose `axis` is `function` or `stability`, giving
  **function 129 and expression 18**. The code is `common.load`.
  - `common.DROP` is empty. An earlier version excluded `CCDB_ECOLI_Adkar_2012` because its
    `rho_sub` is NaN; it is now included. The reasoning is in the `common.py` comments.
  - The 65 `stability_ddg` assays are a **separate appendix axis**, not one of the two main axes.
    Only `scripts/stability_protocol.py` reads them; the main path never does.
- **Fitting set sizes.** function 35 DMS (34 for ESM-1v), expression 14 DMS. These are what remains
  of the 49 external npz files after filtering by axis and by LLR availability.
- **Axis naming.** The npz files store the expression axis under its older name, `stability`. Rather
  than rebuild the data, the rename to `expression` happens **in the display layer only**
  (`paper_numbers.SHOW`). Inside the code and the npz files the name stays `stability`.
- ProteinGym labels are used for evaluation only. They never enter weight fitting.

## npz keys

Only the keys used to produce the paper numbers are listed.

| Key | Contents |
|---|---|
| `pidx` | Position index of the variant (0-based). Used for the position-mean and residual split |
| `ry` | Normalised rank of the DMS label. Monotone, so it is used directly for Spearman |
| `llr_<backbone>` | pLM zero-shot score. The 12 backbones used in the paper are listed in `paper_numbers.EXTBB` |
| `ch_gemme`, `ch_ddg`, `ch_rsa` | GEMME, ΔΔG and RSA channels, per variant |
| `blosum`, `dchg` | BLOSUM62 score and change in charge |
| `prot`, `axis` | Protein identifier and axis |
| `urn` | MaveDB urn (present only in the external set) |

Three things to watch:

- ⚠ Two keys differ between the two sides: ProteinGym `ESM3` (multimodal) corresponds to external
  `ESM3mm`, and ProteinGym `SaProt650M` corresponds to external `SaProt650M_own`. They are not
  interchangeable.
- ⚠ One external npz has no `llr_ESM1v`, which is why ESM-1v fits on 34 function DMS where the other
  backbones fit on 35. The loader filters it out and writes a warning.
- ⚠ The 65 `stability_ddg` assays carry no LLR for S2F, VespaG, VenusREM or AIDO. Because
  `common.load` filters by axis first, the main numbers are unaffected, and the appendix Stability
  results cover only the backbones that have an LLR there.

Other keys (`llr_S2F`, `dhyd`, `wt20` and so on) are used by other analyses and by some figures, but
not by the paper numbers.

## Provenance

The raw sources are the public ProteinGym and MaveDB datasets. The code that turned them into these
channels is kept under `pipeline/upstream/`, and the step order is documented in
[pipeline/README.md](pipeline/README.md).

## Integrity

- Verify with `cd "$PLMFUSION_DATA" && sha256sum -c MANIFEST.sha256`.
- **Do not add files to the data directory.** The code reads `two_dim/*.npz` by glob, so a single
  extra npz changes the results silently. Keeping the directory read-only prevents this.
- To update the data: unlock with `chmod -R u+w`, replace the files, regenerate `MANIFEST.sha256`
  and `reference/`, then lock it read-only again.
