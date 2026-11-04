# Earlier versions of the figures

These three are **the figures used for the earlier submission (function 113 / expression 16)**.
No figure of that name appears in the current paper or appendix. They are kept here in case those
numbers ever need to be redrawn.

| File | Role at the time | Current equivalent |
|---|---|---|
| `plot_zeroshot_vs_fusion.py` | box plot of pLM alone vs ProFusion per backbone | replaced by Figure 2 (`plot_fig1_ext_v3.py`) |
| `plot_comp7_design.py` | 3 panels: (a) component split, (b) eta-squared, (c) weight heatmap | split into the supplementary (`plot_eta2_criterion.py`) for (b) and Figure 3 (`plot_axis_weights.py`) for (c) |
| `plot_comp7_weights.py` | the weight allocation heatmap on its own | replaced by Figure 3 (`plot_axis_weights.py`) |

`scripts/run_all.sh` does not run these three.

The frozen values from that time are in `reference/legacy/`. Since the cohort grew from 113/16 to
129/18, rerunning these scripts on the current data no longer produces those values. They are for
looking at the earlier numbers only.
