# Scoring a new protein

Given an MSA and a structure, this scores the variants of one protein with comp7.

```bash
pip install -e ".[inference]"

python inference/predict_protein.py \
    --seq p.fasta --a3m p.a3m --pdb p.pdb \
    --assay Binding --backbone ESM2 --out out/
```

**No weights are refitted.** They were fitted on the MaveDB external set and live in
`reference/paper_weights.csv`; choosing an axis and a backbone is all that happens. That is why no
labels are needed.

## Inputs

| Argument | Meaning |
|---|---|
| `--seq` | wild-type sequence FASTA, standard 20 amino acids only |
| `--a3m` | hhblits MSA. The query must be the first sequence |
| `--pdb` | structure. **Residue numbers must be 1-based sequence positions** |
| `--assay` | `Activity`, `Binding`, `OrganismalFitness` map to function; `Expression` maps to stability |
| `--backbone` | `ESM2` (default) or `ESM1v`. ESM1v caps at 1022 residues |
| `--variants` | omitted means every position x 19 substitutions |
| `--ddg` | a precomputed ddG npz, when run in a separate environment |

Structures can be fetched from the AlphaFold database. The default `curl` user agent gets a 403,
so it has to be changed.

```bash
curl -A "Mozilla/5.0" https://alphafold.ebi.ac.uk/files/AF-<UniProt>-F1-model_v6.pdb -o p.pdb
```

## Steps and cost

| Step | Uses | Time (for 154 residues) |
|---|---|---|
| 1. RSA | DSSP (`mkdssp`), CPU | seconds |
| 2. GEMME | **docker** (`elodielaine/gemme`), CPU | 1 to 2 minutes |
| 3. ddG | SPURS, CPU | 1 to 2 minutes |
| 4. LLR | transformers, CPU or GPU | minutes for ESM2; ESM1v is 5 times that |
| 5. assemble and score | CPU | seconds |

**No GPU is required.** Everything falls back to CPU. A GPU only speeds up the LLR step.

Steps 1 to 4 are independent and cache their results in `--out`, so rerunning after a failure
skips whatever finished.

**Only ddG needs a separate environment**, because it uses the SPURS repository and checkpoint.
Run it separately and pass the result with `--ddg`, and the rest runs in the main environment.

```bash
<spurs-env>/bin/python inference/steps/ddg.py --pdb p.pdb --seq p.fasta --out ddg.npz
python inference/predict_protein.py ... --ddg ddg.npz
```

## Output

`out/fused_scores.csv`:

| Column | Meaning |
|---|---|
| `fused` | the fused score. **Smaller is more damaging** |
| `deleterious_rank` | 1 is the most damaging |
| `llr`, `gemme`, `ddg`, `rsa` | the raw channels (`ddg` positive = destabilising) |

## Validation (SOD1, 154 residues, 2,605 variants)

The whole path was run on a protein from the fitting set and compared against the cached channels.

| Channel | Max difference | Correlation |
|---|---|---|
| LLR | 6e-6 | 1.000000 |
| RSA | 9e-8 | 1.000000 |
| ddG | 1e-7 | 1.000000 |
| GEMME | 0.71 | 0.989 |

ddG matches the pipeline cache **bit for bit**. LLR and RSA agree to within floating-point error.
Only GEMME differs, for the reason below.

The scoring itself produces **the same values** as the evaluation path. `tests/test_predict.py`
checks that against a cached npz and pins the sign convention to the actual labels.

## Things to know

**1. GEMME is not deterministic.**
Rerunning it on a character-for-character identical alignment and variant list gives different
values. Over 2,926 SOD1 variants the maximum difference was 1.35, the Spearman 0.972, and only
0.65% of values were identical. The conservation computation uses random sampling and has no seed
option.

The effect shrinks by the time it reaches the final score.

| | Final Spearman | Top-50 overlap | Mean rank shift |
|---|---|---|---|
| default | 0.9977 | 47/50 | 36 places out of 2,605 |
| `--n-iter 5` | 0.9989 | 48/50 | 25 places out of 2,605 |

`--n-iter 5` halves the spread and costs 23 seconds. **The default specifies no iteration count**,
because that is how the channels in the paper were built. Change it deliberately or not at all.

In practice, **do not delete the work directory**. When its output is present nothing is rerun, so
the values stay the same.

**2. Scoring a single variant is meaningless.**
The channels are z-normalised within the variant set. Change the set and the scores change.

**3. comp7 needs several variants per position.**
With one variant per position every residual component (`b_*`) is 0 and only the position
components remain.

**4. Scores cannot be compared across proteins.**
Everything is normalised within a protein. Use them as a ranking inside one protein.

**5. Folding-ΔG assays are not supported.**
The `stability` axis of the paper is Expression (protein abundance). Folding ΔG (the Tsuboyama
family) is a separate axis, because the ddG channel comes from a model trained on that data and
scoring it would read the training data back. These weights have never been validated there.
Passing `--assay Stability` raises an error.

**6. The structure's residue numbering must match the sequence.**
AlphaFold database and ColabFold output do. Experimental structures from the PDB often do not. The
RSA step checks this, and the ddG step checks it again by requiring the ddG at wild-type positions
to be 0.

## Not included

- **The ProSST and ESM3 backbones.** They need structure tokens and InterProScan functional
  annotations in addition, which means adapting the structure-backbone LLR builder under
  `pipeline/upstream/`.
- **Building the MSA and the structure.** hhblits needs UniRef30 (328 GB), which is not shipped
  here. Structures can be taken from the AlphaFold database instead.
