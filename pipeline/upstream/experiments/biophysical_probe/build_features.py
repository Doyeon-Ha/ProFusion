"""Build the per-variant feature table for each selected DMS.

Feature blocks (column prefix):
  v_   variant chemistry/sterics (common.variant_features)
  p_   position structural env: RSA, SS3 one-hot, pLDDT, rel position
  n_   neighborhood (<=8A, excl. sequence neighbors): count, hydro, charge
  e_   evolution: MSA conservation, GEMME-epistatic
  f_   function: in-domain flag, #domains overlapping the site
  z_   zero-shot ESM3 wildtype-marginal score (baseline / hybrid input)
target: DMS_score (+ y_z within-DMS z-score)
meta : dms_id, uniprot, assay, family, pos, wt, mut

Writes one parquet per DMS under data/biophysical_probe/features/{dms_id}.parquet
and a combined data/biophysical_probe/all_features.parquet
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

PG = "<PROTEINGYM_DIR>"
ROOT = "<REPO>"
OUT = f"{ROOT}/data/biophysical_probe"
FEAT_DIR = f"{OUT}/features"
DSSP_EXE = "<CONDA_BIN>/mkdssp"
NEIGH_RADIUS = 8.0

THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
SS_MAP = {"H": "H", "G": "H", "I": "H", "E": "E", "B": "E"}  # else -> Loop


def structure_tables(pdb_path: str):
    """Return per-resnum dicts: aa, plddt, ss3, rsa, and neighborhood stats."""
    from Bio.PDB import PDBParser
    from Bio.PDB.DSSP import DSSP
    from scipy.spatial import cKDTree

    parser = PDBParser(QUIET=True)
    model = parser.get_structure("x", pdb_path)[0]
    chain = next(iter(model))  # single chain for AF2 monomers

    resnums, aas, coords, plddt = [], [], [], {}
    for res in chain:
        if res.id[0] != " ":
            continue
        rn = res.id[1]
        aa = THREE2ONE.get(res.resname)
        if aa is None:
            continue
        # CB (CA for Gly) for neighborhood; pLDDT = CA b-factor
        atom = res["CB"] if "CB" in res else (res["CA"] if "CA" in res else None)
        if atom is None:
            continue
        resnums.append(rn)
        aas.append(aa)
        coords.append(atom.get_coord())
        ca = res["CA"] if "CA" in res else atom
        plddt[rn] = float(ca.get_bfactor())

    coords = np.asarray(coords, dtype=float)
    aa_by_rn = dict(zip(resnums, aas))
    tree = cKDTree(coords)
    pairs = tree.query_ball_tree(tree, r=NEIGH_RADIUS)
    n_count, n_hydro, n_charge = {}, {}, {}
    for i, rn in enumerate(resnums):
        nbr = [j for j in pairs[i]
               if j != i and abs(resnums[j] - rn) > 1]  # spatial, not sequence
        n_count[rn] = len(nbr)
        if nbr:
            hy = [C.KD_HYDRO[aas[j]] for j in nbr]
            ch = [C.CHARGE[aas[j]] for j in nbr]
            n_hydro[rn] = float(np.mean(hy))
            n_charge[rn] = float(np.sum(ch))
        else:
            n_hydro[rn] = 0.0
            n_charge[rn] = 0.0

    # DSSP for SS3 + RSA
    ss3, rsa = {}, {}
    try:
        d = DSSP(model, pdb_path, dssp=DSSP_EXE)
        for key in d.keys():
            rn = key[1][1]
            ss3[rn] = SS_MAP.get(d[key][2], "L")
            val = d[key][3]
            rsa[rn] = float(val) if isinstance(val, (int, float)) and val == val else np.nan
    except Exception as e:  # noqa: BLE001
        print(f"    DSSP failed: {type(e).__name__}: {e}")
    return aa_by_rn, plddt, ss3, rsa, n_count, n_hydro, n_charge


def aligned_lookup(pt_path: str, value_key: str):
    """Build {short_variant_id: value} from a per-variant .pt sidecar."""
    o = torch.load(pt_path, map_location="cpu", weights_only=False)
    ids = o.get("variant_ids") or o.get("hgvsp_keys")
    vals = o[value_key]
    if hasattr(vals, "numpy"):
        vals = vals.numpy()
    out = {}
    for vid, v in zip(ids, np.asarray(vals).reshape(len(ids), -1)):
        s = C.to_short(vid)
        if s is not None:
            out[s] = v if v.size > 1 else float(v[0])
    return out


def domain_lookup(pt_path: str):
    o = torch.load(pt_path, map_location="cpu", weights_only=False)
    mh = o["multihot"].numpy()
    out = {}
    for vid, row in zip(o["variant_ids"], mh):
        s = C.to_short(vid)
        if s is not None:
            out[s] = (float(row.max()), float(row.sum()))
    return out


def build_one(dms_id: str, uniprot: str, assay: str, family: str):
    df = pd.read_csv(f"{PG}/DMS_ProteinGym_substitutions/{dms_id}.csv")
    cons = aligned_lookup(f"{PG}/conservation/{dms_id}.pt", "cons")
    gemme = aligned_lookup(f"{PG}/gemme_epi/{dms_id}.pt", "gemme")
    zs = aligned_lookup(f"{PG}/zeroshot/ESM3/{dms_id}.pt", "scores")
    dom = domain_lookup(f"{PG}/domain_multihot/{dms_id}.pt")
    aa_by_rn, plddt, ss3, rsa, n_count, n_hydro, n_charge = structure_tables(
        f"{PG}/ProteinGym_AF2_structures/{uniprot}.pdb")
    seq_len = max(aa_by_rn) if aa_by_rn else 1

    rows, wt_match, wt_total = [], 0, 0
    for _, r in df.iterrows():
        p = C.parse_variant(str(r["mutant"]))
        if p is None:
            continue
        wt, pos, mut = p
        if wt == mut or wt not in C.AA or mut not in C.AA:
            continue
        sid = f"{wt}{pos}{mut}"
        rec = {"dms_id": dms_id, "uniprot": uniprot, "assay": assay,
               "family": family, "pos": pos, "wt": wt, "mut": mut,
               "DMS_score": float(r["DMS_score"])}
        rec.update({f"v_{k}": v for k, v in C.variant_features(wt, mut).items()})
        # position / structure (NaN if resnum absent)
        wt_total += 1
        struct_aa = aa_by_rn.get(pos)
        if struct_aa is not None:
            wt_match += int(struct_aa == wt)
        rec["p_plddt"] = plddt.get(pos, np.nan)
        rec["p_rsa"] = rsa.get(pos, np.nan)
        ss = ss3.get(pos)
        rec["p_ss_helix"] = float(ss == "H") if ss else np.nan
        rec["p_ss_sheet"] = float(ss == "E") if ss else np.nan
        rec["p_ss_loop"] = float(ss == "L") if ss else np.nan
        rec["p_relpos"] = pos / seq_len
        rec["n_count"] = n_count.get(pos, np.nan)
        rec["n_hydro"] = n_hydro.get(pos, np.nan)
        rec["n_charge"] = n_charge.get(pos, np.nan)
        # evolution / function / zero-shot
        rec["e_cons"] = cons.get(sid, np.nan)
        rec["e_gemme_epi"] = gemme.get(sid, np.nan)
        d = dom.get(sid, (np.nan, np.nan))
        rec["f_in_domain"], rec["f_n_domain"] = d
        rec["z_esm3"] = zs.get(sid, np.nan)
        rows.append(rec)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return None, 0.0
    out["y_z"] = (out["DMS_score"] - out["DMS_score"].mean()) / (out["DMS_score"].std() + 1e-9)
    match_rate = wt_match / max(wt_total, 1)
    return out, match_rate


def main():
    os.makedirs(FEAT_DIR, exist_ok=True)
    sel = pd.read_csv(f"{OUT}/selected_dms.csv")
    allparts, summary = [], []
    for i, row in sel.iterrows():
        dms = row["dms_id"]
        print(f"[{i+1}/{len(sel)}] {dms} ({row['assay']})")
        try:
            tbl, mr = build_one(dms, row["uniprot"], row["assay"], row["family"])
        except Exception as e:  # noqa: BLE001
            print(f"    ERROR: {type(e).__name__}: {e}")
            continue
        if tbl is None:
            print("    no variants; skip")
            continue
        tbl.to_parquet(f"{FEAT_DIR}/{dms}.parquet", index=False)
        allparts.append(tbl)
        cov = tbl[["p_rsa", "e_cons", "z_esm3"]].notna().mean()
        print(f"    V={len(tbl):5d}  wt_match={mr:.2f}  "
              f"rsa_cov={cov['p_rsa']:.2f} cons_cov={cov['e_cons']:.2f} "
              f"zs_cov={cov['z_esm3']:.2f}")
        summary.append({"dms_id": dms, "assay": row["assay"], "family": row["family"],
                        "V": len(tbl), "wt_match": mr,
                        "rsa_cov": cov["p_rsa"], "cons_cov": cov["e_cons"],
                        "zs_cov": cov["z_esm3"]})
    combined = pd.concat(allparts, ignore_index=True)
    combined.to_parquet(f"{OUT}/all_features.parquet", index=False)
    pd.DataFrame(summary).to_csv(f"{OUT}/feature_build_summary.csv", index=False)
    print(f"\nDONE: {len(allparts)} DMS, {len(combined):,} variants")
    print(f"combined -> {OUT}/all_features.parquet")
    print(f"mean wt_match={np.mean([s['wt_match'] for s in summary]):.3f}")


if __name__ == "__main__":
    main()
