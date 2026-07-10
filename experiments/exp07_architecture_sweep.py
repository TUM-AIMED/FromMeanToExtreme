"""Experiment 07: Architecture depth/width sweep.

Addresses AE point 2: how does architecture (norm sink) affect reconstruction?
Tests: optimal (no arch), linear_1M, linear_10M, ResNet-18, ResNet-50, ResNet-101, VGG-16.
All on CIFAR-10 32x32 RGB.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import json
from pathlib import Path
from common import (
    load_dataset, run_full_experiment, save_results,
    plot_metrics_vs_sigma, plot_success_probability_curves, plot_bound_tightness,
    DEVICE,
)
from architectures import get_architecture, count_params

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "exp07"

# 3 sigmas spanning the reconstruction transition (low / high / practical-DP) for legible plots.
NOISE_MULTIPLIERS_EVAL = np.array([0.01, 0.1, 0.5])
NOISE_MULTIPLIERS_THEORY = np.logspace(-2, 2, 1000)
C = 1.0
M = 1
NUM_SAMPLES = 200

CONFIGS = [
    {"name": "optimal", "arch": None, "desc": "Optimal (no architecture)"},
    {"name": "linear_1M", "arch": "linear_1000000", "desc": "Linear ~1M params"},
    {"name": "linear_10M", "arch": "linear_10000000", "desc": "Linear ~10M params"},
    {"name": "resnet18", "arch": "resnet18", "desc": "ResNet-18"},
    {"name": "resnet50", "arch": "resnet50", "desc": "ResNet-50"},
    {"name": "resnet101", "arch": "resnet101", "desc": "ResNet-101"},
    {"name": "vgg16", "arch": "vgg16", "desc": "VGG-16"},
]


def main():
    from common import set_seed
    set_seed(42)  # reproducibility: pin python/numpy/torch/CUDA RNGs
    print("=" * 60)
    print("Experiment 07: Architecture Sweep")
    print(f"Device: {DEVICE}, Samples: {NUM_SAMPLES}")
    print("=" * 60)

    data = load_dataset("cifar10", resolution=None, num_samples=NUM_SAMPLES, enforced_norm=1.01, flatten=True, grayscale=False)
    print(f"Data shape: {data.shape}, N={data.shape[1]}")
    input_shape = (3, 32, 32)

    all_summaries = {}

    for cfg in CONFIGS:
        desc = f"CIFAR-10 32x32 — {cfg['desc']}"
        print(f"\n--- {desc} ---")

        additional_layers = []
        if cfg["arch"]:
            transform, module = get_architecture(cfg["arch"], input_shape, flatten_dim=data.shape[1])
            module = module.to(DEVICE)
            n_params = count_params(module)
            print(f"Architecture: {cfg['arch']} ({n_params:,} params)")
            additional_layers = [(transform, module)]  # BUGFIX: was never attached
        else:
            n_params = 0

        results = run_full_experiment(data, NOISE_MULTIPLIERS_THEORY, NOISE_MULTIPLIERS_EVAL, C=C, M=M, additional_layers=additional_layers)
        out_dir = OUTPUT_DIR / cfg["name"]
        save_results(results, out_dir)
        plot_metrics_vs_sigma(results, out_dir, title_prefix=desc)
        plot_success_probability_curves(results, out_dir, title_prefix=desc)
        plot_bound_tightness(results, out_dir, title_prefix=desc)

        all_summaries[cfg["name"]] = {"n_params": n_params, "per_sigma": []}
        for sr in results["per_sigma"]:
            s = sr["summary"]
            print(f"  σ={sr['sigma']:.2g}: MSE={s['mse_mean']:.4f}, PSNR={s['psnr_mean']:.1f}, Coverage={sr['coverage_95']:.3f}")
            all_summaries[cfg["name"]]["per_sigma"].append({
                "sigma": sr["sigma"],
                "mse_mean": s["mse_mean"],
                "psnr_mean": s["psnr_mean"],
                "coverage_95": sr["coverage_95"],
            })

    with open(OUTPUT_DIR / "architecture_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    arch_names = list(all_summaries.keys())
    x_pos = np.arange(len(arch_names))

    for i, sigma in enumerate(NOISE_MULTIPLIERS_EVAL):
        ax = axes[i]
        coverages = [next(s["coverage_95"] for s in all_summaries[a]["per_sigma"] if abs(s["sigma"] - sigma) < 0.01) for a in arch_names]
        mses = [next(s["mse_mean"] for s in all_summaries[a]["per_sigma"] if abs(s["sigma"] - sigma) < 0.01) for a in arch_names]

        bars = ax.bar(x_pos, coverages, color="steelblue", alpha=0.7)
        ax.axhline(0.95, color="red", linestyle=":", alpha=0.5, label="95% target")
        ax.set_ylabel("Coverage Rate", fontsize=9)
        ax.set_title(f"σ = {sigma}", fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([a.replace("_", "\n") for a in arch_names], fontsize=7, rotation=45, ha="right")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=7)

    fig.suptitle("Architecture Effect on Bound Coverage (Norm Sink)", fontsize=11)
    fig.savefig(OUTPUT_DIR / "architecture_sweep.pdf", dpi=150)
    plt.close(fig)

    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
