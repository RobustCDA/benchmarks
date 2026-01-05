# CDA - Benchmarks & Plots

This repository contains Python scripts and LaTeX files to compute parameters, generate CSVs, and plot figures for the _CDA_ paper.

## Main contents

- `cda_estimate_params.py`: compute CDA parameters/costs, write CSVs to `results/cda/`.
- `rda_estimate_params.py`: compute RDA parameters/costs, write CSVs to `results/rda/` (used for CDA vs RDA comparisons).
- `commitment_size_ratio.py`: commitment-size ratio benchmark, writes `results/commitment_size/commitment_benchmark.csv`.
- `grid_core.py`: grid join/leave (churn) simulation and statistics.
- `honest_nodes_per_column.py`: runs the simulation in `grid_core.py`, writes `results/honest_nodes_per_column/protocol_results.csv`.
- `constants.py`: shared constants (block sizes, time parameters, etc.).
- `estimate_params.tex`: plots CDA vs RDA comparisons from CSVs.
- `rda_simulate_params.tex`: RDA k1/k2 plots (eps 0.05, 0.10).
- `cda_simulate_params_eps5.tex`, `cda_simulate_params_eps10.tex`: CDA plots by epsilon.
- `commitment_size_ratio.tex`, `honest_nodes_per_column.tex`: plots for their respective benchmarks/simulations.

## How to run

Requires: Python 3.10+ (no external packages).

Generate CSVs:

```bash
python cda_estimate_params.py
python rda_estimate_params.py
python commitment_size_ratio.py
python honest_nodes_per_column.py
```

Main results PDFs from LaTeX (requires `pdflatex` + `pgfplots`), other is for simulation plots:

```bash
pdflatex prod_estimate_params.tex
pdflatex commitment_size_ratio.tex
pdflatex honest_nodes_per_column.tex
```

## Outputs

- CSVs are stored under `results/` (CDA/RDA/commitment/simulation).
- PDFs are generated alongside the corresponding `.tex` files.
