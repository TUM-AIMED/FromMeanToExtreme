"""Experiment 06: Clipping norm sweep.

Addresses AE point 2: how does clipping norm C affect bounds?
Sweeps C in {0.1, 0.5, 1.0, 2.0, 5.0, 10.0} on CIFAR-10 32x32 grayscale.
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

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "exp06"

# 3 sigmas spanning the reconstruction transition (low / high / practical-DP) for legible plots.
NOISE_MULTIPLIERS_EVAL = np.array([0.01, 0.1, 0.5])
NOISE_MULTIPLIERS_THEORY = np.logspace(-2, 2, 1000)
M = 1
NUM_SAMPLES = 200

CLIPPING_NORMS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def main():
    from common import set_seed
    set_seed(42)  # reproducibility: pin python/numpy/torch/CUDA RNGs
    print("=" * 60)
    print("Experiment 06: Clipping Norm Sweep")
    print(f"Device: {DEVICE}, Samples: {NUM_SAMPLES}")
    print(f"Clipping norms: {CLIPPING_NORMS}")
    print("=" * 60)

    data = load_dataset("cifar10", resolution=32, num_samples=NUM_SAMPLES, enforced_norm=1.01, flatten=True, grayscale=True)
    print(f"Data shape: {data.shape}, N={data.shape[1]}")

    all_summaries = {}

    for C in CLIPPING_NORMS:
        desc = f"CIFAR-10 32x32 gray, C={C}"
        print(f"\n--- {desc} ---")

        results = run_full_experiment(data, NOISE_MULTIPLIERS_THEORY, NOISE_MULTIPLIERS_EVAL, C=C, M=M)
        out_dir = OUTPUT_DIR / f"C_{C}"
        save_results(results, out_dir)
        plot_metrics_vs_sigma(results, out_dir, title_prefix=desc)
        plot_success_probability_curves(results, out_dir, title_prefix=desc)
        plot_bound_tightness(results, out_dir, title_prefix=desc)

        all_summaries[str(C)] = []
        for sr in results["per_sigma"]:
            s = sr["summary"]
            print(f"  σ={sr['sigma']:.2g}: MSE={s['mse_mean']:.4f}, PSNR={s['psnr_mean']:.1f}, Coverage={sr['coverage_95']:.3f}")
            all_summaries[str(C)].append({
                "sigma": sr["sigma"],
                "mse_mean": s["mse_mean"],
                "psnr_mean": s["psnr_mean"],
                "coverage_95": sr["coverage_95"],
            })

    with open(OUTPUT_DIR / "clipping_norm_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for i, sigma in enumerate(NOISE_MULTIPLIERS_EVAL):
        ax = axes[i]
        cs = [float(c) for c in sorted(all_summaries.keys(), key=float)]
        mses = [next(s["mse_mean"] for s in all_summaries[str(c)] if abs(s["sigma"] - sigma) < 0.01) for c in cs]
        coverages = [next(s["coverage_95"] for s in all_summaries[str(c)] if abs(s["sigma"] - sigma) < 0.01) for c in cs]

        ax.plot(cs, mses, "o-", color="blue", label="MSE mean")
        ax.set_xlabel("Clipping norm C", fontsize=9)
        ax.set_ylabel("MSE", fontsize=9, color="blue")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"σ = {sigma}", fontsize=10)
        ax.tick_params(axis="y", labelcolor="blue")

        ax2 = ax.twinx()
        ax2.plot(cs, coverages, "s--", color="red", label="95% Coverage")
        ax2.set_ylabel("Coverage Rate", fontsize=9, color="red")
        ax2.set_ylim(0, 1.1)
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.axhline(0.95, color="red", linestyle=":", alpha=0.3)

    fig.suptitle("Effect of Clipping Norm on Reconstruction Quality", fontsize=11)
    fig.savefig(OUTPUT_DIR / "clipping_norm_sweep.pdf", dpi=150)
    plt.close(fig)

    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
