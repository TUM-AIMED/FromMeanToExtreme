"""Regenerate all appendix figures from saved JSON with the unified publication style.

Reads only saved summary JSON (no experiment reruns), so numbers are unchanged.
All figures are produced at exactly the paper text width (5.5in) with Linux
Libertine and the colorblind-safe Wong palette.
"""
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotstyle import apply_style, TEXTWIDTH, PALETTE, CYCLE, C_THEORY, C_EMPIRICAL, C_TARGET

apply_style()
RES = Path(__file__).parent.parent / "results"


def loadj(path):
    with open(RES / path) as f:
        return json.load(f)


def save(fig, name):
    (RES / name).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(RES / name, dpi=200)
    fig.savefig(RES / name.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print(f"  saved {name}")


# ---------------------------------------------------------------------------
def clipping_norm_sweep():
    """exp06: coverage vs clipping norm C, lines per sigma; phase transition at C=||X||."""
    d = loadj("exp06/clipping_norm_summary.json")
    ckey = {float(k): k for k in d}
    Cs = sorted(ckey)
    sigmas = sorted({e["sigma"] for e in next(iter(d.values()))})
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.4), layout="constrained")
    for i, sg in enumerate(sigmas):
        cov = [next(e["coverage_95"] for e in d[ckey[C]] if e["sigma"] == sg) for C in Cs]
        ax.plot(Cs, cov, marker="o", color=CYCLE[i], label=fr"$\sigma={sg:g}$")
    ax.axvline(1.01, color="red", linestyle=":", linewidth=1.0)
    ax.text(1.01, 0.90, r" $C=\|X\|$", color="red", fontsize=7, ha="left", va="bottom")
    ax.axhline(0.95, color=C_TARGET, linestyle="--", linewidth=0.8, label="95% target")
    ax.set_xscale("log")
    ax.set_xlabel(r"clipping norm $C$")
    ax.set_ylabel("coverage rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", ncol=2, fontsize=7)
    save(fig, "exp06/clipping_norm_sweep.pdf")


# ---------------------------------------------------------------------------
def architecture_sweep():
    """exp07: grouped bars of coverage per architecture, grouped by sigma."""
    d = loadj("exp07/architecture_summary.json")
    arch_order = ["optimal", "linear_1M", "linear_10M", "resnet18", "resnet50", "resnet101", "vgg16"]
    arch_labels = ["Optimal", "Lin 1M", "Lin 10M", "RN-18", "RN-50", "RN-101", "VGG-16"]
    sigmas = sorted({e["sigma"] for e in d["optimal"]["per_sigma"]})
    x = np.arange(len(arch_order))
    width = 0.8 / len(sigmas)
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.4), layout="constrained")
    for i, sg in enumerate(sigmas):
        cov = [next(e["coverage_95"] for e in d[a]["per_sigma"] if e["sigma"] == sg) for a in arch_order]
        ax.bar(x + i * width - (len(sigmas) - 1) * width / 2, cov, width,
               color=CYCLE[i], label=fr"$\sigma={sg:g}$", edgecolor="white", linewidth=0.4)
    ax.axhline(0.95, color="red", linestyle="--", linewidth=1.0, alpha=0.8, label="95% target")
    ax.set_xticks(x)
    ax.set_xticklabels(arch_labels, rotation=20, ha="right")
    ax.set_ylabel("coverage rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower center", ncol=4, fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "exp07/architecture_sweep.pdf")


# ---------------------------------------------------------------------------
def concentration_of_measure():
    """exp08: CV vs N (with 1/sqrt(N/2) reference) and coverage vs N."""
    d = loadj("exp08/dimensionality_summary.json")
    Ns = sorted(int(k) for k in d.keys())
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.3), layout="constrained")
    # Left: CV vs N (mean over sigma)
    ax = axes[0]
    cv_mean = [np.mean([e["mse_cv"] for e in d[str(N)]]) for N in Ns]
    ax.plot(Ns, cv_mean, marker="o", color=C_EMPIRICAL, label="empirical CV")
    ax.plot(Ns, [np.sqrt(2.0 / N) for N in Ns], linestyle="--", color=C_THEORY,
            label=r"theory $\sqrt{2/N}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"dimensionality $N$")
    ax.set_ylabel("coefficient of variation")
    ax.legend(fontsize=7)
    # Right: coverage vs N per sigma
    ax = axes[1]
    sigmas = sorted({e["sigma"] for e in d[str(Ns[0])]})
    for i, sg in enumerate(sigmas):
        cov = [next(e["coverage_95"] for e in d[str(N)] if e["sigma"] == sg) for N in Ns]
        ax.plot(Ns, cov, marker="o", color=CYCLE[i], label=fr"$\sigma={sg:g}$")
    ax.axhline(0.95, color=C_TARGET, linestyle="--", linewidth=0.8, label="95% target")
    ax.set_xscale("log")
    ax.set_xlabel(r"dimensionality $N$")
    ax.set_ylabel("coverage rate")
    ax.set_ylim(0.88, 1.02)
    ax.legend(fontsize=7, ncol=2)
    save(fig, "exp08/concentration_of_measure.pdf")


# ---------------------------------------------------------------------------
def batch_size_effect():
    """exp09: MSE/bound vs B (3 panels by sigma); normalized to B=1 bound."""
    d = loadj("exp09/batch_size_summary.json")
    Bs = sorted(int(k) for k in d.keys())
    skey_map = {float(k): k for k in d[str(Bs[0])].keys()}
    sigmas = sorted(skey_map)
    fig, axes = plt.subplots(1, len(sigmas), figsize=(TEXTWIDTH, 2.1), layout="constrained")
    for ax, sg in zip(axes, sigmas):
        skey = skey_map[sg]
        th_max = d[str(Bs[0])][skey]["theoretical_mse_B1_max_norm"]
        mmean = [d[str(B)][skey]["mse_to_mean_mean"] / th_max for B in Bs]
        mind = [d[str(B)][skey]["mse_to_individual_mean"] / th_max for B in Bs]
        ax.plot(Bs, mmean, marker="o", color=PALETTE["blue"], label="to batch mean")
        ax.plot(Bs, mind, marker="s", color=PALETTE["green"], label="to individual")
        ax.axhline(1.0, color="red", linestyle=":", linewidth=1.0, label=r"$B{=}1$ bound")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel(r"batch size $B$")
        ax.set_title(fr"$\sigma={sg:g}$")
        ax.set_ylim(5e-4, 2.0)
    axes[0].set_ylabel(r"MSE / $B{=}1$ bound")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=3, fontsize=7)
    save(fig, "exp09/batch_size_effect.pdf")


# ---------------------------------------------------------------------------
def coverage_comparison():
    """exp14: per-sample coverage vs sigma across 3 natural-norm datasets."""
    data = loadj("exp14/summary.json")
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.4), layout="constrained")
    for i, ds in enumerate(data):
        sg = [e["sigma"] for e in ds["per_sigma"]]
        cov = [e["coverage_95_per_sample"] for e in ds["per_sigma"]]
        ax.plot(sg, cov, marker="o", color=CYCLE[i], label=ds["label"])
    ax.axhline(0.95, color=C_TARGET, linestyle="--", linewidth=0.8, label="95% target")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma$ (noise multiplier)")
    ax.set_ylabel("per-sample coverage rate")
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=7)
    save(fig, "exp14/coverage_comparison.pdf")


# ---------------------------------------------------------------------------
def cross_dataset_coverage():
    """summary: grouped bars, coverage across enforced-norm datasets (optimal attack)."""
    datasets = {
        "CIFAR-10": ("exp01", "cifar10_32x32_optimal"),
        "CIFAR-100": ("exp05", "cifar100_optimal"),
        "CelebA 64": ("exp03", "celeba_64_optimal"),
        "CelebA 128": ("exp03", "celeba_128_optimal"),
        "PathMNIST": ("exp04", "pathmnist_optimal"),
    }
    sigmas = [0.01, 0.1, 0.5]  # 3 straddling the wall; subset of the dataset-rep grid

    def cov(exp, cfg, sg):
        raw = loadj(f"{exp}/{cfg}/summary.json")
        for e in raw["per_sigma_summary"]:
            if e["sigma"] == sg:
                return e["coverage_95"]
        return np.nan

    x = np.arange(len(datasets))
    width = 0.8 / len(sigmas)
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.4), layout="constrained")
    for i, sg in enumerate(sigmas):
        covs = [cov(exp, cfg, sg) for exp, cfg in datasets.values()]
        ax.bar(x + i * width - (len(sigmas) - 1) * width / 2, covs, width,
               color=CYCLE[i], label=fr"$\sigma={sg:g}$", edgecolor="white", linewidth=0.4)
    ax.axhline(0.95, color="red", linestyle="--", linewidth=1.0, alpha=0.8, label="95% target")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets.keys(), rotation=20, ha="right")
    ax.set_ylabel("coverage rate")
    ax.set_ylim(0.85, 1.02)
    ax.legend(loc="lower center", ncol=3, fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "summary_figures/cross_dataset_coverage.pdf")


# ---------------------------------------------------------------------------
def natural_norms_analysis():
    """exp12: MSE with norm-dependent theory band; coverage under norm choices."""
    d = loadj("exp12/cifar10_natural_norms/summary.json")
    ps = d["per_sigma_summary"]
    sg = [e["sigma"] for e in ps]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.3), layout="constrained")
    # Left: MSE band
    ax = axes[0]
    th_min = [e["theoretical_mse_min_norm"] for e in ps]
    th_mean = [e["theoretical_mse_mean_norm"] for e in ps]
    th_max = [e["theoretical_mse_max_norm"] for e in ps]
    emp = [e["mse_mean"] for e in ps]
    ax.fill_between(sg, th_min, th_max, alpha=0.2, color=C_THEORY,
                    label=r"theory range ($\|X\|_{\min}..\|X\|_{\max}$)")
    ax.plot(sg, th_mean, "--", color=C_THEORY, label=r"theory (mean $\|X\|$)")
    ax.plot(sg, emp, marker="o", color=C_EMPIRICAL, label="empirical")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma$"); ax.set_ylabel("MSE")
    ax.legend(fontsize=6.5)
    # Right: coverage under norm choices
    ax = axes[1]
    series = [("coverage_95_per_sample", "per-sample", PALETTE["green"], "s"),
              ("coverage_95_mean_norm", "mean norm", PALETTE["blue"], "o"),
              ("coverage_95_max_norm", "max norm (cons.)", PALETTE["orange"], "^"),
              ("coverage_95_min_norm", "min norm", PALETTE["vermillion"], "v")]
    for key, lab, col, mk in series:
        ax.plot(sg, [e[key] for e in ps], marker=mk, color=col, label=lab)
    ax.axhline(0.95, color=C_TARGET, linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma$"); ax.set_ylabel("coverage rate")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=6.5)
    save(fig, "exp12/cifar10_natural_norms/natural_norms_analysis.pdf")


if __name__ == "__main__":
    print("Regenerating appendix figures with unified style...")
    clipping_norm_sweep()
    architecture_sweep()
    concentration_of_measure()
    batch_size_effect()
    coverage_comparison()
    cross_dataset_coverage()
    natural_norms_analysis()
    print("Done.")
