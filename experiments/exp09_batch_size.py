"""Experiment 09: Batch size B>1 analysis with correct DP-SGD noise model.

Uses natural norms (no enforced_norm) so that per-sample clipping is
heterogeneous and batch averaging produces a nontrivial mixture of signals.

DP-SGD noise model for B>1:
  1. Per-sample clipping: x̃_i = x_i · min(1, C/||x_i||)
  2. Sum clipped samples: s = Σ x̃_i
  3. Add noise: s_noisy = s + N(0, σ²C² I)
  4. Average: x̄_noisy = s_noisy / B
  5. Reconstruct: x̂ = x̄_noisy (MVUE of the batch mean)
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import json
import torch
from pathlib import Path
from tqdm import tqdm
from common import load_dataset, DEVICE, get_mse_dist

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "exp09"

# 3 sigmas spanning the reconstruction transition (low / high / practical-DP) for legible plots.
NOISE_MULTIPLIERS_EVAL = np.array([0.01, 0.1, 0.5])
NUM_TRIALS = 200
BATCH_SIZES = [1, 2, 4, 8, 16, 32]


def dp_sgd_reconstruct(batch, sigma, C, device=DEVICE):
    """Correct DP-SGD noise model: per-sample clip, sum, noise, average."""
    batch = batch.to(device)
    B = batch.shape[0]

    clipped = []
    for i in range(B):
        x_i = batch[i]
        norm_i = torch.norm(x_i)
        clip_factor = min(1.0, C / (norm_i.item() + 1e-10))
        clipped.append(x_i * clip_factor)

    clipped_sum = torch.stack(clipped).sum(dim=0)
    noise = torch.randn_like(clipped_sum) * sigma * C
    noisy_sum = clipped_sum + noise
    recon = noisy_sum / B

    batch_mean = batch.mean(dim=0)
    clipped_mean = torch.stack(clipped).mean(dim=0)

    return recon.cpu(), batch_mean.cpu(), clipped_mean.cpu()


def main():
    from common import set_seed
    set_seed(42)  # reproducibility: pin python/numpy/torch/CUDA RNGs
    print("=" * 60)
    print("Experiment 09: Batch Size Analysis (Natural Norms, Correct DP-SGD)")
    print(f"Device: {DEVICE}, Trials: {NUM_TRIALS}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print("=" * 60)

    data = load_dataset("cifar10", resolution=32,
                        num_samples=max(BATCH_SIZES) * NUM_TRIALS,
                        enforced_norm=None, flatten=True, grayscale=False)
    N = data.shape[1]
    norms = torch.norm(data, dim=1).numpy()
    C = float(np.median(norms))

    print(f"Data shape: {data.shape}, N={N}")
    print(f"Natural ||X|| stats: min={norms.min():.1f} mean={norms.mean():.1f} "
          f"median={np.median(norms):.1f} max={norms.max():.1f}")
    print(f"Clipping norm C = {C:.1f} (median ||X||)")

    all_results = {}

    for B in BATCH_SIZES:
        print(f"\n--- Batch size B={B} ---")
        all_results[B] = {}

        for sigma in NOISE_MULTIPLIERS_EVAL:
            mses_to_mean = []
            mses_to_individual = []
            mses_to_clipped_mean = []

            for trial in tqdm(range(NUM_TRIALS), desc=f"B={B}, σ={sigma:.2g}", leave=False):
                indices = np.random.randint(0, len(data), size=B)
                batch = data[indices]

                recon, true_mean, clipped_mean = dp_sgd_reconstruct(batch, sigma, C)

                mse_to_mean = torch.mean((recon - true_mean) ** 2).item()
                mses_to_mean.append(mse_to_mean)

                mse_to_clipped = torch.mean((recon - clipped_mean) ** 2).item()
                mses_to_clipped_mean.append(mse_to_clipped)

                individual_mses = [torch.mean((recon - batch[i]) ** 2).item() for i in range(B)]
                mses_to_individual.append(np.mean(individual_mses))

            # B=1 theoretical bound using max norm (conservative)
            theory_mse_b1_max = float(get_mse_dist(N, sigma, norms.max()).mean())
            # B=1 theoretical bound using median norm
            theory_mse_b1_median = float(get_mse_dist(N, sigma, C).mean())

            all_results[B][float(sigma)] = {
                "mse_to_mean_mean": float(np.mean(mses_to_mean)),
                "mse_to_mean_std": float(np.std(mses_to_mean)),
                "mse_to_clipped_mean_mean": float(np.mean(mses_to_clipped_mean)),
                "mse_to_individual_mean": float(np.mean(mses_to_individual)),
                "mse_to_individual_std": float(np.std(mses_to_individual)),
                "theoretical_mse_B1_max_norm": theory_mse_b1_max,
                "theoretical_mse_B1_median_norm": theory_mse_b1_median,
            }

            r = all_results[B][float(sigma)]
            print(f"  σ={sigma:.2g}: MSE(mean)={r['mse_to_mean_mean']:.4f}, "
                  f"MSE(indiv)={r['mse_to_individual_mean']:.4f}, "
                  f"Theory(B=1,max)={theory_mse_b1_max:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "batch_size_summary.json", "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for i, sigma in enumerate(NOISE_MULTIPLIERS_EVAL):
        ax = axes[i]
        bs = BATCH_SIZES
        mse_mean = [all_results[b][float(sigma)]["mse_to_mean_mean"] for b in bs]
        mse_indiv = [all_results[b][float(sigma)]["mse_to_individual_mean"] for b in bs]
        theory_max = all_results[1][float(sigma)]["theoretical_mse_B1_max_norm"]
        theory_med = all_results[1][float(sigma)]["theoretical_mse_B1_median_norm"]

        ax.plot(bs, mse_mean, "o-", label="MSE to batch mean", color="steelblue", linewidth=2)
        ax.plot(bs, mse_indiv, "s-", label="MSE to individual", color="seagreen", linewidth=2)
        ax.axhline(theory_max, color="red", linestyle=":", linewidth=1.5,
                   label=f"B=1 bound (max ‖X‖)", alpha=0.8)
        ax.axhline(theory_med, color="orange", linestyle="--", linewidth=1.5,
                   label=f"B=1 bound (median ‖X‖)", alpha=0.8)
        ax.set_xlabel("Batch size B", fontsize=9)
        ax.set_ylabel("MSE", fontsize=9)
        ax.set_title(f"σ = {sigma}", fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Batch Size vs Reconstruction Quality — Natural Norms, C={C:.0f} (median ‖X‖)\n"
                 "Correct DP-SGD: per-sample clip → sum → noise(σ²C²) → average",
                 fontsize=11, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "batch_size_effect.pdf", dpi=150)
    fig.savefig(OUTPUT_DIR / "batch_size_effect.png", dpi=150)
    plt.close(fig)

    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
