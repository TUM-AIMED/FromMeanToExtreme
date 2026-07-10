"""Experiment 04: MedMNIST PathMNIST — medical imaging benchmark.

Addresses AE point 4: sensitive-domain scenarios, medical image dataset.
Uses PathMNIST (pathology, 28x28 RGB, N=2352).
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
from pathlib import Path
from common import (
    load_dataset, run_full_experiment, save_results,
    plot_metrics_vs_sigma, plot_success_probability_curves, plot_bound_tightness,
    DEVICE,
)
from architectures import get_architecture, count_params

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "exp04"

# sigmas span the reconstruction transition (faithful -> destroyed), not only the
# noise floor, to validate coverage across the full range.
NOISE_MULTIPLIERS_EVAL = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.5])
NOISE_MULTIPLIERS_THEORY = np.logspace(-2, 2, 1000)
C = 1.0
M = 1
NUM_SAMPLES = 500

CONFIGS = [
    {"name": "pathmnist_optimal", "arch": None, "desc": "PathMNIST 28x28 RGB — Optimal"},
    {"name": "pathmnist_resnet101", "arch": "resnet101", "desc": "PathMNIST 28x28 RGB — ResNet-101"},
]


def main():
    from common import set_seed
    set_seed(42)  # reproducibility: pin python/numpy/torch/CUDA RNGs
    print("=" * 60)
    print("Experiment 04: MedMNIST (PathMNIST)")
    print(f"Device: {DEVICE}, Samples: {NUM_SAMPLES}")
    print("=" * 60)

    data = load_dataset("medmnist_path", resolution=28, num_samples=NUM_SAMPLES, enforced_norm=1.01, flatten=True, grayscale=False)
    print(f"Data shape: {data.shape}, N={data.shape[1]}")
    input_shape = (3, 28, 28)

    for cfg in CONFIGS:
        print(f"\n--- {cfg['desc']} ---")
        additional_layers = []
        if cfg["arch"]:
            transform, module = get_architecture(cfg["arch"], input_shape, flatten_dim=data.shape[1])
            module = module.to(DEVICE)
            print(f"Architecture: {cfg['arch']} ({count_params(module):,} params)")
            additional_layers = [(transform, module)]

        results = run_full_experiment(data, NOISE_MULTIPLIERS_THEORY, NOISE_MULTIPLIERS_EVAL, C=C, M=M, additional_layers=additional_layers)
        out_dir = OUTPUT_DIR / cfg["name"]
        save_results(results, out_dir)
        plot_metrics_vs_sigma(results, out_dir, title_prefix=cfg["desc"])
        plot_success_probability_curves(results, out_dir, title_prefix=cfg["desc"])
        plot_bound_tightness(results, out_dir, title_prefix=cfg["desc"])

        for sr in results["per_sigma"]:
            s = sr["summary"]
            print(f"  σ={sr['sigma']:.2g}: MSE={s['mse_mean']:.4f}, PSNR={s['psnr_mean']:.1f}, Coverage={sr['coverage_95']:.3f}")

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
