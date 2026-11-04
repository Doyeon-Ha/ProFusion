#!/usr/bin/env python
"""MSA (a3m) to per-variant GEMME scores. The only step that needs docker.

It does three things.
  1. a3m to a match-state alignment (lower-case letters and dots removed). The query is the first
     sequence and is named `query`. Rows whose length differs from the query are dropped.
  2. writes mutations.txt (W12A format, sorted by position then substitution).
  3. runs GEMME through docker and parses `*_normPred_evolCombi.txt`.

The convention is the same as the one used to build the fitting set (see the step scripts under
pipeline/upstream/). Any difference would make the a_GEMME and b_GEMME weights refer to a different
quantity.

⚠ **GEMME is not deterministic.** Rerunning it on a character-for-character identical alignment and
  variant list gives different values (measured on SOD1, 2,926 variants: max absolute difference
  1.35, Spearman 0.972, 0.65% identical values). The conservation computation uses random sampling
  and has no seed option. Raising the iteration count -n reduces the spread (--n-iter).
  By the time it reaches the fused score the effect shrinks: the final Spearman between two runs is
  0.998 (measured).

  So **do not delete the work directory.** When its output is present nothing is rerun, and the
  values stay the same.

By default every position x 19 substitutions is queried regardless of the variant list
(scan_all=True). Not because the values are stable, but so that one work directory can be reused
for any subset of variants.

docker image: elodielaine/gemme:gemme (about 1.24 GB). Under rootless docker, DOCKER_HOST is set
automatically.

run: python inference/steps/gemme.py --a3m p.a3m --variants v.txt --workdir work/ --out gemme.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

IMAGE = "elodielaine/gemme:gemme"
TIMEOUT = 10800


def read_a3m(path: str | Path) -> list[tuple[str, str]]:
    out, h, buf = [], None, []
    for ln in Path(path).read_text().splitlines():
        if ln.startswith(">"):
            if h is not None:
                out.append((h, "".join(buf)))
            h, buf = ln[1:].strip(), []
        elif not ln.startswith("#"):
            buf.append(ln.rstrip())
    if h is not None:
        out.append((h, "".join(buf)))
    return out


def to_match_state(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Strip lower-case (insertion) columns and dots, and keep only rows the same length as the query."""
    ms = [(h, "".join(c for c in s if c != "." and not c.islower())) for h, s in rows]
    L = len(ms[0][1])
    return [(h, s) for h, s in ms if len(s) == L]


def _docker_env() -> dict:
    env = dict(os.environ)
    if "DOCKER_HOST" not in env:
        sock = f"/run/user/{os.getuid()}/docker.sock"
        if Path(sock).exists():                       # rootless docker
            env["DOCKER_HOST"] = f"unix://{sock}"
            env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return env


def all_substitutions(seq: str) -> list[str]:
    AA20 = "ACDEFGHIKLMNPQRSTVWY"
    return [f"{wt}{i}{mu}" for i, wt in enumerate(seq, 1) for mu in AA20 if mu != wt]


def compute(a3m: str | Path, muts: list[str], workdir: str | Path,
            n_seq: int = 20000, scan_all: bool = True, n_iter: int | None = None) -> pd.Series:
    """{mutant: GEMME score}. If output already exists, docker is not rerun.

    With scan_all=True (the default), every position x 19 substitutions is queried regardless of
    muts, and only muts is returned. This is so that one work directory can be reused for any
    subset of variants (see the module docstring).
    """
    w = Path(workdir)
    w.mkdir(parents=True, exist_ok=True)
    tag = w.name
    ali = w / f"ali{tag}.fasta"

    rows = to_match_state(read_a3m(a3m))
    query = rows[0][1]
    for m in muts:
        p = int(m[1:-1])
        if not (1 <= p <= len(query)) or query[p - 1] != m[0]:
            raise ValueError(f"variant {m} does not match the MSA query sequence "
                             f"(length {len(query)}, position {p} is "
                             f"{query[p - 1] if 1 <= p <= len(query) else 'out of range'})")
    with open(ali, "w") as o:
        for i, (h, s) in enumerate(rows):
            o.write(f">{'query' if i == 0 else h.split()[0]}\n{s}\n")
    ask = all_substitutions(query) if scan_all else list(muts)
    (w / "mutations.txt").write_text(
        "\n".join(sorted(set(ask), key=lambda x: (int(x[1:-1]), x[-1]))) + "\n")
    print(f"  [GEMME] alignment depth {len(rows):,}, length {len(query)}, queried {len(set(ask)):,} "
          f"({'all 19 substitutions' if scan_all else 'requested only'}) -> returning {len(muts):,}", flush=True)

    hit = glob.glob(str(w / "*_normPred_evolCombi.txt"))
    if not hit:
        cmd = ["docker", "run", "--rm", "-v", f"{w.resolve()}:/project", "-w", "/project",
               "--entrypoint", "/bin/bash", IMAGE, "-c",
               f"python2.7 $GEMME_PATH/gemme.py {ali.name} -r input -f {ali.name} "
               f"-m mutations.txt -N {n_seq}"
               + (f" -n {n_iter}" if n_iter else "")]
        r = subprocess.run(cmd, env=_docker_env(), capture_output=True, text=True, timeout=TIMEOUT)
        (w / "run.log").write_text(r.stdout + r.stderr)
        hit = glob.glob(str(w / "*_normPred_evolCombi.txt"))
        if not hit:
            raise RuntimeError(f"GEMME produced no output (rc={r.returncode}). "
                               f"log: {w / 'run.log'}")
    s = pd.read_csv(hit[0], sep=" ", index_col=0).iloc[:, 0]
    return s.reindex(muts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a3m", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--workdir", required=True, help="GEMME work directory (caches the output)")
    ap.add_argument("--n-seq", type=int, default=20000)
    ap.add_argument("--n-iter", type=int, default=None,
                    help="conservation iteration count. Raising it reduces run-to-run spread (and is slower)")
    ap.add_argument("--only-requested", action="store_true",
                    help="query only the requested variants. Not recommended, since normPred depends on the set")
    ap.add_argument("--out", required=True)
    A = ap.parse_args()
    muts = [l.strip() for l in Path(A.variants).read_text().splitlines() if l.strip()]
    s = compute(A.a3m, muts, A.workdir, A.n_seq, scan_all=not A.only_requested,
                n_iter=A.n_iter)
    pd.DataFrame(dict(mutant=muts, gemme=s.to_numpy())).to_csv(A.out, index=False)
    miss = int(np.isnan(s.to_numpy()).sum())
    print(f"[saved] {A.out}  {len(muts):,} variants ({miss} missing)  "
          f"median GEMME {np.nanmedian(s.to_numpy()):.3f}")


if __name__ == "__main__":
    main()
