#!/usr/bin/env python
"""Resolve the score direction of the external tuning set without labels.

MaveDB does not supply the direction (2 of 2,680 score sets have `scoreRanges` and none have
`scoreCalibrations`). An LLM label (`score_direction`) agrees with the nonsense-variant evidence
92.3% of the time, so 8% are wrong. A flipped sign flips the whole Spearman, so the label cannot be
taken at face value.

Decision order
  1. nonsense       p.*Ter is unambiguous loss of function. The direction comes from the
                    standardised gap between the nonsense mean and the missense mean,
                    gap = (mean_mis - mean_ter) / sd_mis. gap > 0 means higher_is_wt. Accepted only
                    with at least 3 nonsense variants and |gap| > 0.3.
  2. llm_unanimous  if 1 does not apply and three LLMs agree unanimously, that value is used.
  3. manual         if neither applies, manual_overrides.yaml is consulted, and the assay is
                    excluded if it is not listed there.

The output is `sign_flip`: True means the score must be multiplied by −1 to become higher_is_wt.

run: PYTHONPATH=src python experiments/mavedb_external/resolve_direction.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

R0 = Path(__file__).resolve().parents[2]
MAVE = Path("<MAVEDB_DIR>")
FROZEN = R0 / "data/mavedb_external/frozen.pkl"
OVERRIDES = Path(__file__).resolve().parent / "manual_overrides.yaml"
OUT_PKL = R0 / "data/mavedb_external/direction.pkl"
OUT_YAML = R0 / "configs/data/mavedb_external_tuneset.yaml"

# Conflict-handling variants (both were produced in parallel)
#   manual    resolve a conflict from the documented evidence in manual_overrides.yaml (49 DMS kept)
#   exclude   drop the conflicting assay from the tuning set (49 -> 46 DMS)
#   nobinding on top of manual, drop the whole Binding class (function axis 35 -> 24 DMS,
#             18 -> 13 proteins).
#             Note: not pre-registered. It was added after tuning-set QC showed a mean rho of 0.099
#             for Binding. ProteinGym was never touched, so "the weights never saw the benchmark"
#             still holds. Rationale: the fusion channels (conservation, ddG, RSA) model global
#             constraints, not the interface affinity towards one designated partner.
VARIANTS = ("manual", "exclude", "nobinding")

# ⚠ The (?!\1$) and (?!Ter$) parts are essential. Without them a **synonymous substitution such as
# p.Ala30Ala passes as missense**. Some MaveDB records spell synonymous changes out in three-letter
# form, so filtering on "=" and "Ter" alone does not catch them.
# Measured: 5 of the 49 external assays were contaminated (SGCB 304, STIM1 31 and others), which
# makes it 20 substitutions per position instead of 19.
MIS = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)(?!\1$)(?!Ter$)([A-Z][a-z]{2})$")
TER = re.compile(r"^p\.[A-Z][a-z]{2}\d+(Ter|\*)$")
MIN_TER, MIN_GAP = 3, 0.3
MODELS = ["gpt-4.1", "gpt-5-2025-08-07", "o3"]


def read_scores(urn: str) -> tuple[pd.Series, pd.Series]:
    """(single-missense scores, nonsense scores)."""
    f = MAVE / "csv" / f"{urn.replace(':', '-')}.scores.csv"
    t = pd.read_csv(f, usecols=lambda c: c in ("hgvs_pro", "score"), low_memory=False)
    h, s = t.hgvs_pro.astype(str), pd.to_numeric(t.score, errors="coerce")
    ok = s.notna()
    return s[h.str.match(MIS) & ok], s[h.str.match(TER) & ok]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="manual", choices=VARIANTS)
    A = ap.parse_args()
    sel = pd.read_pickle(FROZEN)
    ov = yaml.safe_load(OVERRIDES.read_text()) if OVERRIDES.exists() else {}
    ov = {e["urn"]: e for e in (ov.get("direction") or [])}

    rows = []
    for r in sel.to_dict("records"):
        mis, ter = read_scores(r["urn"])
        sd = mis.std()
        gap = (mis.mean() - ter.mean()) / sd if (len(ter) >= MIN_TER and sd > 0) else np.nan

        llm = {r[f"dir_{m}"] for m in MODELS}
        llm_dir = llm.pop() if len(llm) == 1 else None

        # If the **sign** from the nonsense variants opposes the LLM, that is a conflict, and it is never decided automatically, whatever |gap| is.
        ns_dir = None if np.isnan(gap) else ("higher_is_wt" if gap > 0 else "higher_is_lof")
        conflict = bool(ns_dir and llm_dir and ns_dir != llm_dir)

        if conflict:
            # The exclude variant is a sensitivity check that **does not use** any manual judgement, so it drops the assay even when an override exists.
            if A.variant == "exclude":
                direction, src = None, "EXCLUDED_CONFLICT"
            elif r["urn"] in ov:
                direction, src = ov[r["urn"]]["decision"], "manual"
            else:
                direction, src = None, "UNRESOLVED"                            # EC-002
        elif r["urn"] in ov:                     # explicitly pinned, for example for consistency with sibling assays
            direction, src = ov[r["urn"]]["decision"], "manual"
        elif ns_dir and abs(gap) > MIN_GAP:
            direction, src = ns_dir, "nonsense"
        elif llm_dir:
            direction, src = llm_dir, "llm_unanimous"
        else:
            direction, src = None, "UNRESOLVED"                                # EC-002

        rows.append(dict(urn=r["urn"], tname=r["tname"], axis=r["axis"], n_mis=len(mis), n_ter=len(ter),
                         gap=gap, direction=direction, sign_source=src, llm_dir=llm_dir,
                         ns_dir=ns_dir, llm_conflict=conflict))
    D = pd.DataFrame(rows)

    print(D.sign_source.value_counts().to_string())
    bad = D[D.sign_source == "UNRESOLVED"]
    if len(bad):
        print("\n[unresolved] the sign could not be determined. Record it in manual_overrides.yaml or exclude it:")
        print(bad[["urn", "tname", "n_ter", "gap"]].to_string(index=False))

    conf = D[D.llm_conflict]
    print(f"\n[conflict] assays where the nonsense evidence overruled the LLM: {len(conf)}")
    if len(conf):
        print(conf[["urn", "tname", "gap", "direction", "llm_dir"]].round(3).to_string(index=False))

    D = D[D.direction.notna()].copy()            # drop EXCLUDED_CONFLICT and UNRESOLVED
    if A.variant == "nobinding":
        cls = sel.set_index("urn").assay_class
        D = D[D.urn.map(cls) != "Binding"].copy()
    D["sign_flip"] = D.direction == "higher_is_lof"
    print(f"\n[{A.variant}] final {len(D)} DMS, {int(D.sign_flip.sum())} to be sign-flipped")
    out_pkl = OUT_PKL.with_name(f"direction_{A.variant}.pkl")
    D.to_pickle(out_pkl)
    D.to_csv(out_pkl.with_suffix(".csv"), index=False)

    # merge into the manifest
    man = yaml.safe_load(OUT_YAML.read_text())
    man["variant"] = A.variant
    m = D.set_index("urn")
    man["datasets"] = [d for d in man["datasets"] if d["urn"] in m.index]
    for d in man["datasets"]:
        if True:
            row = m.loc[d["urn"]]
            d["sign_source"] = row.sign_source
            d["sign_flip"] = bool(row.sign_flip)
            d["nonsense_gap"] = None if pd.isna(row.gap) else round(float(row.gap), 3)
            d["n_nonsense"] = int(row.n_ter)
    man["counts"]["sign_source"] = D.sign_source.value_counts().to_dict()
    man["counts"]["dms"] = len(D)
    out_yaml = OUT_YAML.with_name(f"mavedb_external_tuneset_{A.variant}.yaml")
    out_yaml.write_text(yaml.safe_dump(man, sort_keys=False, allow_unicode=True))
    print(f"[saved] {out_pkl} · {out_yaml}")


if __name__ == "__main__":
    main()
