#!/usr/bin/env python
"""InterProScan functional annotations for the 23 external tuning-set sequences.

They are the function-track input of the ESM3 multimodal encoding, in **the same label space** as
the other sets.

⚠️ Do not use the batch entry point (`from_fasta_file`): one irrecoverable failure in the FunFam
(CATH) stage kills the whole job and leaves zero output. Running **one sequence at a time** isolates
a failure to that sequence. Successes and failures are both recorded by key, so it can resume.

No structure is needed, so this can run **in parallel** with the folding jobs.
output: data/mavedb_external/function_annotations.pkl  dict[tag, list[FunctionAnnotation]]
"""
from __future__ import annotations

import json
import pickle
import sys
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
R0 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R0 / "external" / "esm3"))
sys.path.insert(0, str(R0 / "src"))

EXT = R0 / "data/mavedb_external"
OUT_PKL = EXT / "function_annotations.pkl"
ESM3_MAX = 2046


def main() -> None:
    from esm.utils.function.interpro_search import InterProSearch

    man = json.loads((EXT / "seq_manifest.json").read_text())
    seqs = {}
    for tag in man:
        f = EXT / "msa/query" / f"{tag}.fasta"
        seqs[tag] = f.read_text().split("\n", 1)[1].replace("\n", "").strip()

    d: dict[str, list] = {}
    if OUT_PKL.exists():
        d = pickle.load(open(OUT_PKL, "rb"))
        print(f"resume: skipping {len(d)} existing", flush=True)

    todo = [(t, s) for t, s in seqs.items() if t not in d]
    print(f"running InterProScan on {len(todo)} / {len(seqs)} (one per sequence; may take hours)\n", flush=True)

    failed = []
    for i, (tag, s) in enumerate(todo, 1):
        name = man[tag]["tname"]
        if len(s) > ESM3_MAX:
            print(f"[{i}/{len(todo)}] SKIP-LONG {name} ({len(s)}aa > {ESM3_MAX})", flush=True)
            d[tag] = []                      # beyond the ESM3 input limit, recorded as an empty annotation
            continue
        print(f"[{i}/{len(todo)}] {name} ({len(s)}aa) ...", flush=True)
        try:
            res = InterProSearch.from_sequence(s, tag)
            d[tag] = res.get(tag, [])
            print(f"    → {len(d[tag])} annotation", flush=True)
        except Exception:                     # noqa: BLE001  isolate the failure to this sequence
            traceback.print_exc()
            d[tag] = []
            failed.append(tag)
        pickle.dump(d, open(OUT_PKL, "wb"))   # incremental save, for resume

    n_ann = sum(1 for v in d.values() if v)
    print(f"\n[done] {len(d)} sequences, {n_ann} with annotations, {len(failed)} failed {failed}")
    print(f"[saved] {OUT_PKL}")


if __name__ == "__main__":
    main()
