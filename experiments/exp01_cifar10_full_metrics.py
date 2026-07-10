"""Experiment 01: Full metrics on CIFAR-10 (2x2 and 32x32).

Addresses AE point 2: SSIM, LPIPS, success probability curves, bound tightness, coverage rate.
Reruns the existing CIFAR-10 experiments with all metrics and new analysis.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import torch
from pathlib import Path
from common import (
    load_dataset, run_full_experiment, save_results,
    plot_metrics_vs_sigma, plot_success_probability_curves, plot_bound_tightness,
    DEVICE,
)
from architectures import get_architecture, count_params

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "exp01"

NOISE_MULTIPLIERS_THEORY = np.logspace(-2, 2, 1000)
# sigmas span the reconstruction transition (faithful -> destroyed), not only the
# noise floor, to validate coverage across the full range.
NOISE_MULTIPLIERS_EVAL = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.5])
C = 1.0
M = 1
NUM_SAMPLES = 500

CONFIGS = [
    {
        "name": "cifar10_2x2_optimal",
        "dataset": "cifar10",
        "resolution": 2,
        "grayscale": True,
        "arch": None,
        "desc": "CIFAR-10 2x2 grayscale — Optimal attack",
    },
    {
        "name": "cifar10_2x2_linear1M",
        "dataset": "cifar10",
        "resolution": 2,
        "grayscale": True,
        "arch": "linear_1000000",
        "desc": "CIFAR-10 2x2 grayscale — Linear 1M params",
    },
    {
        "name": "cifar10_2x2_resnet101",
        "dataset": "cifar10",
        "resolution": 2,
        "grayscale": True,
        "arch": "resnet101",
        "desc": "CIFAR-10 2x2 grayscale — ResNet-101",
    },
    {
        "name": "cifar10_32x32_optimal",
        "dataset": "cifar10",
        "resolution": None,
        "grayscale": False,
        "arch": None,
        "desc": "CIFAR-10 32x32 RGB — Optimal attack",
    },
    {
        "name": "cifar10_32x32_linear1M",
        "dataset": "cifar10",
        "resolution": None,
        "grayscale": False,
        "arch": "linear_1000000",
        "desc": "CIFAR-10 32x32 RGB — Linear 1M params",
    },
    {
        "name": "cifar10_32x32_resnet101",
        "dataset": "cifar10",
        "resolution": None,
        "grayscale": False,
        "arch": "resnet101",
        "desc": "CIFAR-10 32x32 RGB — ResNet-101",
    },
]


def main():
    from common import set_seed
    set_seed(42)  # reproducibility: pin python/numpy/torch/CUDA RNGs
    print("=" * 60)
    print("Experiment 01: CIFAR-10 Full Metrics")
    print(f"Device: {DEVICE}")
    print(f"Samples: {NUM_SAMPLES}, C={C}, M={M}")
    print(f"Sigmas to evaluate: {NOISE_MULTIPLIERS_EVAL}")
    print("=" * 60)

    all_results = {}

    for cfg in CONFIGS:
        print(f"\n{'='*50}")
        print(f"Running: {cfg['desc']}")
        print(f"{'='*50}")

        data = load_dataset(
            cfg["dataset"],
            resolution=cfg["resolution"],
            num_samples=NUM_SAMPLES,
            enforced_norm=1.01,
            flatten=True,
            grayscale=cfg["grayscale"],
        )
        print(f"Data shape: {data.shape}, N={data.shape[1]}")

        additional_layers = []
        if cfg["arch"] is not None:
            if cfg["grayscale"]:
                if cfg["resolution"]:
                    input_shape = (1, cfg["resolution"], cfg["resolution"])
                else:
                    input_shape = (1, 32, 32)
            else:
                if cfg["resolution"]:
                    input_shape = (3, cfg["resolution"], cfg["resolution"])
                else:
                    input_shape = (3, 32, 32)
            transform, module = get_architecture(cfg["arch"], input_shape, flatten_dim=data.shape[1])
            module = module.to(DEVICE)
            print(f"Architecture: {cfg['arch']} ({count_params(module):,} params)")
            additional_layers = [(transform, module)]

        results = run_full_experiment(
            data,
            NOISE_MULTIPLIERS_THEORY,
            NOISE_MULTIPLIERS_EVAL,
            C=C,
            M=M,
            additional_layers=additional_layers,
        )

        out_dir = OUTPUT_DIR / cfg["name"]
        save_results(results, out_dir)
        plot_metrics_vs_sigma(results, out_dir, title_prefix=cfg["desc"])
        plot_success_probability_curves(results, out_dir, title_prefix=cfg["desc"])
        plot_bound_tightness(results, out_dir, title_prefix=cfg["desc"])

        all_results[cfg["name"]] = results

        for sr in results["per_sigma"]:
            s = sr["summary"]
            print(f"  σ={sr['sigma']:.2g}: MSE={s['mse_mean']:.4f}, "
                  f"PSNR={s['psnr_mean']:.1f}, SSIM={s['ssim_mean']:.4f}, "
                  f"LPIPS={s['lpips_mean']:.4f}, Coverage={sr['coverage_95']:.3f}")

    print(f"\nAll results saved to {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
