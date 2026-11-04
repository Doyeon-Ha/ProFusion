"""Align GEMME evolCombi to the variant ids of a task and store it as a sidecar.

For each task, build a GEMME evolCombi[V] tensor in the order of its variant_ids.
The GEMME mutant is converted from short form ("A14C") to hgvsp ("p.Ala14Cys") for matching.
Missing variants are filled with the mean GEMME of that DMS (z-score 0, a neutral contribution).

output: data/proteingym/gemme/{dms_id}.pt = {"variant_ids": [...], "gemme": Tensor[V] float32}
run:    PYTHONPATH=src python experiments/amortized/build_gemme_aligned.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")

TASKS = Path("data/proteingym/tasks_ens")
GEMME_DIR = Path("experiments/runs/gemme_all/results")
OUT = Path("data/proteingym/gemme")
AA1to3 = {"A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln", "E": "Glu",
          "G": "Gly", "H": "His", "I": "Ile", "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe",
          "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val"}
_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def g2h(m: str) -> str | None:
    g = _RE.match(m)
    if not g or g.group(1) not in AA1to3 or g.group(3) not in AA1to3:
        return None
    return f"p.{AA1to3[g.group(1)]}{g.group(2)}{AA1to3[g.group(3)]}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = sorted(TASKS.glob("*.pt"))
    done = miss_dms = 0
    cov = []
    for p in tasks:
        dms = p.stem
        task = torch.load(p, map_location="cpu", weights_only=False)
        vids = list(task["variant_ids"])
        gf = GEMME_DIR / f"{dms}.csv"
        if not gf.exists():
            miss_dms += 1
            print(f"[no-gemme] {dms}")
            continue
        gm = {}
        for ln in gf.read_text().splitlines()[1:]:
            c = ln.split(",")
            h = g2h(c[0]) if len(c) >= 2 else None
            if h:
                try:
                    gm[h] = float(c[1])  # evolCombi
                except ValueError:
                    pass
        present = [gm[k] for k in vids if k in gm]
        if not present:
            miss_dms += 1
            print(f"[empty-align] {dms}")
            continue
        fill = sum(present) / len(present)  # the fill value for missing entries is the mean
        gem = torch.tensor([gm.get(k, fill) for k in vids], dtype=torch.float32)
        n_present = sum(1 for k in vids if k in gm)
        cov.append(n_present / len(vids))
        torch.save({"variant_ids": vids, "gemme": gem,
                    "coverage": n_present / len(vids)}, OUT / f"{dms}.pt")
        done += 1
        if done % 40 == 0:
            print(f"  {done}/{len(tasks)} ...")

    import numpy as np
    print(f"\ndone: {done} written, {miss_dms} DMS missing. Mean variant coverage {np.mean(cov):.3f}, "
          f"minimum {np.min(cov):.3f}, {sum(c<0.9 for c in cov)} DMS below 0.9")
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
