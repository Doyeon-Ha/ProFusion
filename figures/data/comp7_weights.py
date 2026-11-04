#!/usr/bin/env python
"""Input for the comp7 weight figures: the 24 comp7 weight rows (2 axes x 12 backbones).

Selects the comp7 rows from the paper_weights.csv produced by scripts/paper_numbers.py and drops
the arm column.

input:  results/paper_weights.csv
output: results/fig_data/panelc_comp7_weights.csv
"""
import pandas as pd

from plmfusion.paths import RESULTS

OUT = RESULTS / "fig_data"
OUT.mkdir(parents=True, exist_ok=True)
W = pd.read_csv(RESULTS / "paper_weights.csv")
W = W[W.arm.str.startswith("comp7")].drop(columns="arm")
W.to_csv(OUT / "panelc_comp7_weights.csv", index=False)
print(W.to_string(index=False))
