"""Experiment 12: CIFAR-10 with natural (unnormalized) data norms.

Addresses editorial concern: enforced_norm=1.01 makes universality tautological.
This experiment uses the natural data norms to validate that the bound remains
valid (conservative) when ||X|| varies across samples.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import json
import torch
from pathlib import Path
from common import (
    load_dataset, run_full_experiment, save_results, DEVICE,
    plot_metrics_vs_sigma, plot_success_probability_curves, plot_bound_tightness,
    compute_all_metrics, perform_optimal_recon, compute_success_probability_curve,
    compute_theoretical_success_prob, compute_coverage_rate, compute_bound_tightness,
    get_mse_dist,
)

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "exp12"

# sigmas span the reconstruction transition (faithful -> destroyed), not only the
# noise floor, to validate coverage across the full range.
SIGMAS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.5]
NUM_SAMPLES = 500
ETAS = np.logspace(-3, 3, 200)


def main():
    from common import set_seed
    set_seed(42)  # reproducibility: pin python/numpy/torch/CUDA RNGs
    print("=" * 60)
    print("Experiment 12: Natural Norms (no enforced_norm)")
    print(f"Device: {DEVICE}, Samples: {NUM_SAMPLES}")
    print("=" * 60)

    # Load CIFAR-10 32x32 RGB WITHOUT norm enforcement
    data = load_dataset("cifar10", resolution=32, num_samples=NUM_SAMPLES,
                        enforced_norm=None, flatten=True, grayscale=False)
    N = data.shape[1]
    print(f"Data shape: {data.shape}, N={N}")

    # Compute and report natural norm statistics
    norms = torch.norm(data, dim=1).numpy()
    print(f"\nNatural ||X|| statistics:")
    print(f"  Min:    {norms.min():.4f}")
    print(f"  Mean:   {norms.mean():.4f}")
    print(f"  Median: {np.median(norms):.4f}")
    print(f"  Max:    {norms.max():.4f}")
    print(f"  Std:    {norms.std():.4f}")

    config_dir = OUTPUT_DIR / "cifar10_natural_norms"
    config_dir.mkdir(parents=True, exist_ok=True)

    per_sigma_summary = []
    all_sigma_data = {}

    for sigma in SIGMAS:
        print(f"\n--- sigma={sigma} ---")

        originals = []
        reconstructions = []

        for i in range(NUM_SAMPLES):
            x = data[i:i+1]
            recon = perform_optimal_recon(x, sigma, C=1.0, M=1, device=DEVICE)
            originals.append(x.numpy().flatten())
            reconstructions.append(recon.numpy().flatten())

        originals = np.array(originals)
        reconstructions = np.array(reconstructions)

        mses = np.array([np.mean((o - r)**2) for o, r in zip(originals, reconstructions)])

        import torch as _torch
        origs_t = _torch.stack([_torch.as_tensor(o) for o in originals])
        recs_t = _torch.stack([_torch.as_tensor(r) for r in reconstructions])
        metrics = compute_all_metrics(origs_t, recs_t)

        # Coverage using different norm choices
        coverage_min_norm = np.mean(mses <= get_mse_dist(N, sigma, norms.min()).ppf(0.95))
        coverage_mean_norm = np.mean(mses <= get_mse_dist(N, sigma, norms.mean()).ppf(0.95))
        coverage_max_norm = np.mean(mses <= get_mse_dist(N, sigma, norms.max()).ppf(0.95))

        per_sample_thresholds = np.array([
            get_mse_dist(N, sigma, norms[i]).ppf(0.95) for i in range(NUM_SAMPLES)
        ])
        coverage_per_sample = np.mean(mses <= per_sample_thresholds)

        theoretical_mse_mean_norm = sigma**2 * norms.mean()**2
        theoretical_mse_min_norm = sigma**2 * norms.min()**2
        theoretical_mse_max_norm = sigma**2 * norms.max()**2

        entry = {
            "sigma": sigma,
            "mse_mean": float(np.mean(metrics["mse"])),
            "mse_median": float(np.median(mses)),
            "mse_std": float(np.std(mses)),
            "psnr_mean": float(np.mean(metrics["psnr"])),
            "ssim_mean": float(np.nanmean(metrics["ssim"])),
            "lpips_mean": float(np.nanmean(metrics["lpips"])),
            "coverage_95_min_norm": float(coverage_min_norm),
            "coverage_95_mean_norm": float(coverage_mean_norm),
            "coverage_95_max_norm": float(coverage_max_norm),
            "coverage_95_per_sample": float(coverage_per_sample),
            "theoretical_mse_min_norm": float(theoretical_mse_min_norm),
            "theoretical_mse_mean_norm": float(theoretical_mse_mean_norm),
            "theoretical_mse_max_norm": float(theoretical_mse_max_norm),
        }
        per_sigma_summary.append(entry)
        all_sigma_data[sigma] = {"mses": mses, "metrics": metrics}

        print(f"  MSE: {entry['mse_mean']:.4f} (theory@mean_norm: {theoretical_mse_mean_norm:.4f})")
        print(f"  Coverage (min_norm): {coverage_min_norm:.3f}")
        print(f"  Coverage (mean_norm): {coverage_mean_norm:.3f}")
        print(f"  Coverage (max_norm): {coverage_max_norm:.3f}")
        print(f"  Coverage (per_sample): {coverage_per_sample:.3f}")

    summary = {
        "N": N,
        "C": 1.0,
        "enforced_norm": None,
        "norm_min": float(norms.min()),
        "norm_mean": float(norms.mean()),
        "norm_median": float(np.median(norms)),
        "norm_max": float(norms.max()),
        "norm_std": float(norms.std()),
        "per_sigma_summary": per_sigma_summary,
    }

    with open(config_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: MSE with norm-dependent theoretical bands
    ax = axes[0]
    empirical_mse = [entry["mse_mean"] for entry in per_sigma_summary]
    theory_min = [entry["theoretical_mse_min_norm"] for entry in per_sigma_summary]
    theory_mean = [entry["theoretical_mse_mean_norm"] for entry in per_sigma_summary]
    theory_max = [entry["theoretical_mse_max_norm"] for entry in per_sigma_summary]

    ax.fill_between(SIGMAS, theory_min, theory_max, alpha=0.2, color='blue',
                    label=f'Theory range (||X||∈[{norms.min():.1f}, {norms.max():.1f}])')
    ax.plot(SIGMAS, theory_mean, 'b--', linewidth=2, label=f'Theory (mean ||X||={norms.mean():.1f})')
    ax.plot(SIGMAS, empirical_mse, 'ro-', markersize=8, linewidth=2, label='Empirical mean MSE')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('σ (noise multiplier)')
    ax.set_ylabel('MSE')
    ax.set_title('MSE: Natural Norms')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Coverage rates with different norm choices
    ax = axes[1]
    cov_min = [entry["coverage_95_min_norm"] for entry in per_sigma_summary]
    cov_mean = [entry["coverage_95_mean_norm"] for entry in per_sigma_summary]
    cov_max = [entry["coverage_95_max_norm"] for entry in per_sigma_summary]
    cov_per = [entry["coverage_95_per_sample"] for entry in per_sigma_summary]

    ax.plot(SIGMAS, cov_per, 's-', markersize=8, linewidth=2, label='Per-sample ||X||', color='green')
    ax.plot(SIGMAS, cov_mean, 'o-', markersize=8, linewidth=2, label='Mean ||X||', color='blue')
    ax.plot(SIGMAS, cov_max, '^-', markersize=8, linewidth=2, label='Max ||X|| (conservative)', color='orange')
    ax.plot(SIGMAS, cov_min, 'v-', markersize=8, linewidth=2, label='Min ||X|| (anti-conservative)', color='red')
    ax.axhline(0.95, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='95% target')
    ax.set_xscale('log')
    ax.set_xlabel('σ (noise multiplier)')
    ax.set_ylabel('Coverage Rate')
    ax.set_title('Coverage Under Natural Data Norms')
    ax.legend(fontsize=8)
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)

    fig.suptitle('CIFAR-10 32×32 RGB — Natural Norms (no enforced_norm)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(config_dir / "natural_norms_analysis.pdf", dpi=150)
    fig.savefig(config_dir / "natural_norms_analysis.png", dpi=150)
    plt.close(fig)

    print(f"\nResults saved to {config_dir}")


if __name__ == "__main__":
    main()
