"""Experiment 08: Dimensionality sweep.

Addresses AE point 2: bound tightness across dimensionalities.
Shows concentration of measure: as N grows, empirical distribution tightens.
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

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "exp08"

# 3 sigmas spanning the reconstruction transition (low / high / practical-DP) for legible plots.
NOISE_MULTIPLIERS_EVAL = np.array([0.01, 0.1, 0.5])
NOISE_MULTIPLIERS_THEORY = np.logspace(-2, 2, 1000)
C = 1.0
M = 1
NUM_SAMPLES = 200

RESOLUTIONS = [2, 4, 8, 16, 32]


def main():
    from common import set_seed
    set_seed(42)  # reproducibility: pin python/numpy/torch/CUDA RNGs
    print("=" * 60)
    print("Experiment 08: Dimensionality Sweep")
    print(f"Device: {DEVICE}, Samples: {NUM_SAMPLES}")
    print(f"Resolutions: {RESOLUTIONS}")
    print("=" * 60)

    all_summaries = {}

    for res in RESOLUTIONS:
        N = res * res
        desc = f"CIFAR-10 {res}x{res} gray (N={N})"
        print(f"\n--- {desc} ---")

        data = load_dataset("cifar10", resolution=res, num_samples=NUM_SAMPLES, enforced_norm=1.01, flatten=True, grayscale=True)
        print(f"Data shape: {data.shape}")

        results = run_full_experiment(data, NOISE_MULTIPLIERS_THEORY, NOISE_MULTIPLIERS_EVAL, C=C, M=M)
        out_dir = OUTPUT_DIR / f"N_{N}"
        save_results(results, out_dir)
        plot_metrics_vs_sigma(results, out_dir, title_prefix=desc)
        plot_success_probability_curves(results, out_dir, title_prefix=desc)
        plot_bound_tightness(results, out_dir, title_prefix=desc)

        all_summaries[N] = []
        for sr in results["per_sigma"]:
            s = sr["summary"]
            mse_std = float(np.std(sr["metrics"]["mse"]))
            mse_cv = mse_std / s["mse_mean"] if s["mse_mean"] > 0 else float("inf")
            print(f"  σ={sr['sigma']:.2g}: MSE={s['mse_mean']:.4f} (CV={mse_cv:.3f}), Coverage={sr['coverage_95']:.3f}")
            all_summaries[N].append({
                "sigma": sr["sigma"],
                "mse_mean": s["mse_mean"],
                "mse_cv": mse_cv,
                "coverage_95": sr["coverage_95"],
            })

    with open(OUTPUT_DIR / "dimensionality_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for i, sigma in enumerate(NOISE_MULTIPLIERS_EVAL):
        ax = axes[i]
        dims = sorted(all_summaries.keys())
        cvs = [next(s["mse_cv"] for s in all_summaries[d] if abs(s["sigma"] - sigma) < 0.01) for d in dims]
        coverages = [next(s["coverage_95"] for s in all_summaries[d] if abs(s["sigma"] - sigma) < 0.01) for d in dims]

        ax.plot(dims, cvs, "o-", color="blue", label="MSE CV")
        ax.set_xlabel("N (dimensionality)", fontsize=9)
        ax.set_ylabel("Coefficient of Variation", fontsize=9, color="blue")
        ax.set_xscale("log")
        ax.set_title(f"σ = {sigma}", fontsize=10)
        ax.tick_params(axis="y", labelcolor="blue")

        ax2 = ax.twinx()
        ax2.plot(dims, coverages, "s--", color="red", label="95% Coverage")
        ax2.set_ylabel("Coverage Rate", fontsize=9, color="red")
        ax2.set_ylim(0, 1.1)
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.axhline(0.95, color="red", linestyle=":", alpha=0.3)

    fig.suptitle("Concentration of Measure: MSE Variability vs Dimensionality", fontsize=11)
    fig.savefig(OUTPUT_DIR / "concentration_of_measure.pdf", dpi=150)
    plt.close(fig)

    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
