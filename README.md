# From Mean to Extreme: Formal Differential Privacy Bounds on the Success of Real-World Data Reconstruction Attacks

Accompanying code for the paper "From Mean to Extreme: Formal Differential Privacy Bounds on the Success of Real-World Data Reconstruction Attacks". 

## Setup
 1. Install [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html).
 2. `conda env create -f environment.yml`
 3. `conda activate frommeantoextreme`

## Main-text figures
[figures.py](figures.py) reproduces the main-text figures (Figures 1-5). 

We recommend using the [python interactive window](https://code.visualstudio.com/docs/python/jupyter-support-py) provided by vscode, which allows to execute and inspect the output for each figure at a time. 
However, running `python figures.py` will also save all figures to the working directory. 

## Appendix experiments

The [experiments/](experiments/) directory contains the empirical evaluation reported in the appendix (Figures 6-14 and Table 1): parameter sweeps (clipping norm, architecture, dimensionality, batch size), cross-dataset validation, perceptual metrics, and the attack comparison. The summary statistics quoted in the main text (Section 5.4, Parameter Sensitivity) are computed here as well. It uses the same `frommeantoextreme` environment.

**Datasets.** CIFAR-10, CIFAR-100, and PathMNIST download automatically; CelebA must be placed manually (torchvision cannot download it). By default datasets live under `data/` in the repository root — set the `FMTE_DATA_ROOT` environment variable to point elsewhere.
```bash
export FMTE_DATA_ROOT=/path/to/datasets
```

**Reproducibility.** Every experiment pins its RNGs to seed 42 (`common.set_seed`), so a rerun reproduces the archived numbers.

**Run the experiments** (each writes JSON and per-config outputs under `results/`):
```bash
cd experiments
for e in exp01_cifar10_full_metrics exp02b_sigma_epsilon_realistic exp03_celeba exp04_medmnist \
         exp05_cifar100 exp06_clipping_norm_sweep exp07_architecture_sweep exp08_dimensionality_sweep \
         exp09_batch_size exp10_prior_attacks exp12_natural_norms exp14_perceptual_metrics_natural; do
    python "$e.py"
done
```

**Regenerate the paper's figures and table** from the saved JSON (plotting only, no recompute):
```bash
python replot_figures.py
python replot_exp10.py
python generate_reconstruction_examples.py
python make_tables.py
```

The table below maps every figure and table in the paper to the script that produces it and its output file. Paths are relative to the repository root: `figures.py` writes to the working directory, while the experiment generators write under `results/` (git-ignored; regenerable).

| Script | Output file | In the paper |
| --- | --- | --- |
| `figures.py` | `figure2a.pdf`, `figure2b.pdf` | Figure 2 |
| `figures.py` | `figure3.pdf` | Figure 3 |
| `figures.py` | `figure4.pdf` | Figure 4 |
| `figures.py` | `figure5.pdf` | Figure 5 |
| `experiments/replot_figures.py` | `results/summary_figures/cross_dataset_coverage.pdf` | Figure 6 |
| `experiments/replot_figures.py` | `results/exp14/coverage_comparison.pdf` | Figure 7 |
| `experiments/replot_figures.py` | `results/exp07/architecture_sweep.pdf` | Figure 8 |
| `experiments/replot_figures.py` | `results/exp08/concentration_of_measure.pdf` | Figure 9 |
| `experiments/replot_figures.py` | `results/exp06/clipping_norm_sweep.pdf` | Figure 10 |
| `experiments/replot_figures.py` | `results/exp09/batch_size_effect.pdf` | Figure 11 |
| `experiments/replot_exp10.py` | `results/exp10/attack_comparison.pdf` | Figure 12 |
| `experiments/generate_reconstruction_examples.py` | `results/summary_figures/reconstruction_examples.pdf` | Figure 13 |
| `experiments/replot_figures.py` | `results/exp12/cifar10_natural_norms/natural_norms_analysis.pdf` | Figure 14 |
| `experiments/make_tables.py` | `results/tables/perceptual_table.tex` | Table 1 |

Figure 1 (the paper overview) is a schematic and is not produced by a script.
