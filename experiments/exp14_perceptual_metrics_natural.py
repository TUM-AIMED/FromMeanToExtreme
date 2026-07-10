"""Experiment 14: Natural-norm perceptual metrics + multi-dataset validation.

Addresses reviewer concerns about:
1. SSIM/LPIPS being meaningless under enforced norms (data_range mismatch)
2. Natural-norm validation limited to CIFAR-10
3. Enforced-norm creating self-fulfilling validation

Runs optimal reconstruction on 3 datasets WITHOUT enforced_norm, computing
all metrics with correct data_range=1.0.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import json
import torch
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    load_dataset,
    compute_all_metrics,
    perform_optimal_recon,
    get_mse_dist,
    DEVICE,
)

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "exp14"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASETS = {
    "cifar10": {"resolution": 32, "grayscale": False, "label": "CIFAR-10 (32x32 RGB)"},
    "celeba":  {"resolution": 64, "grayscale": False, "label": "CelebA (64x64 RGB)"},
    "medmnist_path": {"resolution": 28, "grayscale": False, "label": "PathMNIST (28x28 RGB)"},
}

NUM_SAMPLES = 200
# Span the MVUE's own SNR transition: per-pixel SNR = 1/(N*sigma^2). These sigmas run
# SNR from ~80 (faithful) down to ~0.03 (destroyed), so SSIM goes ~0.9 -> 0.
SIGMAS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
C = 1.0


def run_dataset(ds_name, ds_cfg):
    """Run the experiment for a single dataset. Returns a results dict."""
    print(f"\n{'=' * 60}")
    print(f"Dataset: {ds_cfg['label']}")
    print(f"{'=' * 60}")

    data = load_dataset(
        ds_name,
        resolution=ds_cfg["resolution"],
        num_samples=NUM_SAMPLES,
        enforced_norm=None,
        flatten=True,
        grayscale=ds_cfg["grayscale"],
    )
    N = data.shape[1]
    norms = torch.norm(data, dim=1).numpy()

    print(f"  Data shape: {data.shape}, N={N}")
    print(f"  Natural ||X|| stats:  min={norms.min():.4f}  mean={norms.mean():.4f}"
          f"  median={np.median(norms):.4f}  max={norms.max():.4f}  std={norms.std():.4f}")

    ds_results = {
        "dataset": ds_name,
        "label": ds_cfg["label"],
        "resolution": ds_cfg["resolution"],
        "N": int(N),
        "num_samples": NUM_SAMPLES,
        "norm_stats": {
            "min": float(norms.min()),
            "mean": float(norms.mean()),
            "median": float(np.median(norms)),
            "max": float(norms.max()),
            "std": float(norms.std()),
        },
        "per_sigma": [],
    }

    for sigma in SIGMAS:
        print(f"\n  --- sigma={sigma} ---")

        recons = perform_optimal_recon(data, sigma, C=C, M=1, device=DEVICE)

        # MSE/PSNR on the raw MVUE estimate: matches the theoretical Gamma (sigma^2||X||^2)
        # and drives the coverage check. SSIM/LPIPS on the clipped [0,1] estimate: the
        # perceptual quality of the actually-displayable reconstruction (out-of-range pixels
        # would otherwise distort the structural metrics).
        recons_clipped = np.clip(np.asarray(recons), 0.0, 1.0)
        m_raw = compute_all_metrics(data, recons, data_range=1.0)
        m_clip = compute_all_metrics(data, recons_clipped, data_range=1.0)
        metrics = {
            "mse": m_raw["mse"], "psnr": m_raw["psnr"],
            "ssim": m_clip["ssim"], "lpips": m_clip["lpips"],
        }

        # Per-sample coverage using each sample's own norm
        mses = metrics["mse"]
        per_sample_thresholds = np.array([
            get_mse_dist(N, sigma, norms[i]).ppf(0.95) for i in range(NUM_SAMPLES)
        ])
        coverage_per_sample = float(np.mean(mses <= per_sample_thresholds))

        # Coverage using global norm choices for comparison
        coverage_min_norm = float(np.mean(mses <= get_mse_dist(N, sigma, norms.min()).ppf(0.95)))
        coverage_max_norm = float(np.mean(mses <= get_mse_dist(N, sigma, norms.max()).ppf(0.95)))

        # Theoretical expected MSE = sigma^2 * ||X||^2  (mean over samples)
        theoretical_mse_mean = float(sigma ** 2 * np.mean(norms ** 2))

        sigma_entry = {
            "sigma": sigma,
            "mse_mean": float(np.mean(mses)),
            "mse_median": float(np.median(mses)),
            "mse_std": float(np.std(mses)),
            "psnr_mean": float(np.nanmean(metrics["psnr"][np.isfinite(metrics["psnr"])])),
            "ssim_mean": float(np.nanmean(metrics["ssim"])),
            "lpips_mean": float(np.nanmean(metrics["lpips"])),
            "coverage_95_per_sample": coverage_per_sample,
            "coverage_95_min_norm": coverage_min_norm,
            "coverage_95_max_norm": coverage_max_norm,
            "theoretical_mse_mean": theoretical_mse_mean,
            "metrics_raw": {k: v.tolist() for k, v in metrics.items()},
        }
        ds_results["per_sigma"].append(sigma_entry)

        print(f"    MSE:  {sigma_entry['mse_mean']:.6f}  (theory: {theoretical_mse_mean:.6f})")
        print(f"    PSNR: {sigma_entry['psnr_mean']:.2f} dB")
        print(f"    SSIM: {sigma_entry['ssim_mean']:.4f}")
        print(f"    LPIPS: {sigma_entry['lpips_mean']:.4f}")
        print(f"    Coverage (per-sample): {coverage_per_sample:.3f}")

    return ds_results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_per_dataset_grid(ds_results, out_dir):
    """2x2 grid: MSE, PSNR, SSIM, LPIPS vs sigma for one dataset."""
    sigmas = [e["sigma"] for e in ds_results["per_sigma"]]
    label = ds_results["label"]

    fig, axs = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    metric_cfgs = [
        ("mse",  "MSE",       True,  True),   # key, ylabel, logy, has_theory
        ("psnr", "PSNR (dB)", False, False),
        ("ssim", "SSIM",      False, False),
        ("lpips", "LPIPS",    False, False),
    ]

    for idx, (mkey, ylabel, logy, has_theory) in enumerate(metric_cfgs):
        ax = axs[idx // 2][idx % 2]
        raw_key = mkey
        means = []
        q25s = []
        q75s = []
        for entry in ds_results["per_sigma"]:
            vals = np.array(entry["metrics_raw"][raw_key])
            vals = vals[np.isfinite(vals)]
            means.append(np.mean(vals) if len(vals) > 0 else np.nan)
            q25s.append(np.percentile(vals, 25) if len(vals) > 0 else np.nan)
            q75s.append(np.percentile(vals, 75) if len(vals) > 0 else np.nan)

        ax.fill_between(sigmas, q25s, q75s, alpha=0.25, color="steelblue", label="IQR")
        ax.plot(sigmas, means, "o-", color="steelblue", linewidth=2, markersize=6, label="Empirical mean")

        if has_theory:
            theory_vals = [e["theoretical_mse_mean"] for e in ds_results["per_sigma"]]
            ax.plot(sigmas, theory_vals, "s--", color="crimson", linewidth=2, markersize=5, label="Theory (mean norm)")

        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(r"$\sigma$")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{label} -- Natural Norms, data_range=1.0", fontsize=12, fontweight="bold")
    fig.savefig(out_dir / f"{ds_results['dataset']}_metrics_grid.pdf", dpi=150)
    fig.savefig(out_dir / f"{ds_results['dataset']}_metrics_grid.png", dpi=150)
    plt.close(fig)
    print(f"  Saved metrics grid: {out_dir / (ds_results['dataset'] + '_metrics_grid.pdf')}")


def plot_coverage_comparison(all_results, out_dir):
    """Coverage comparison across datasets (per-sample coverage at each sigma)."""
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    markers = ["o", "s", "^", "D", "v"]
    colors = plt.cm.tab10(np.arange(len(all_results)))

    for i, ds_res in enumerate(all_results):
        sigmas = [e["sigma"] for e in ds_res["per_sigma"]]
        cov = [e["coverage_95_per_sample"] for e in ds_res["per_sigma"]]
        ax.plot(sigmas, cov, f"{markers[i % len(markers)]}-", color=colors[i],
                linewidth=2, markersize=8, label=ds_res["label"])

    ax.axhline(0.95, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="95% target")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma$ (noise multiplier)", fontsize=11)
    ax.set_ylabel("Per-sample Coverage Rate (95%)", fontsize=11)
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Per-sample Coverage Across Datasets (Natural Norms)", fontsize=12, fontweight="bold")

    fig.savefig(out_dir / "coverage_comparison.pdf", dpi=150)
    fig.savefig(out_dir / "coverage_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved coverage comparison: {out_dir / 'coverage_comparison.pdf'}")


def build_combined_table(all_results, out_dir):
    """Write a combined LaTeX-ready summary table and JSON."""
    rows = []
    for ds_res in all_results:
        for entry in ds_res["per_sigma"]:
            rows.append({
                "dataset": ds_res["label"],
                "N": ds_res["N"],
                "sigma": entry["sigma"],
                "mse_mean": entry["mse_mean"],
                "psnr_mean": entry["psnr_mean"],
                "ssim_mean": entry["ssim_mean"],
                "lpips_mean": entry["lpips_mean"],
                "coverage_95": entry["coverage_95_per_sample"],
                "theoretical_mse": entry["theoretical_mse_mean"],
            })

    with open(out_dir / "combined_table.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Render as text table for quick inspection
    header = f"{'Dataset':<26} {'sigma':>6} {'N':>6} {'MSE':>10} {'PSNR':>8} {'SSIM':>7} {'LPIPS':>7} {'Cov95':>7}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"{r['dataset']:<26} {r['sigma']:>6g} {r['N']:>6d} "
            f"{r['mse_mean']:>10.6f} {r['psnr_mean']:>8.2f} "
            f"{r['ssim_mean']:>7.4f} {r['lpips_mean']:>7.4f} "
            f"{r['coverage_95']:>7.3f}"
        )
    table_str = "\n".join(lines)
    with open(out_dir / "combined_table.txt", "w") as f:
        f.write(table_str + "\n")

    print(f"\n{table_str}")
    print(f"\nSaved combined table: {out_dir / 'combined_table.json'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from common import set_seed
    set_seed(42)  # reproducibility: pin python/numpy/torch/CUDA RNGs
    print("=" * 60)
    print("Experiment 14: Natural-norm Perceptual Metrics (Multi-dataset)")
    print(f"Device: {DEVICE}")
    print(f"Samples per dataset: {NUM_SAMPLES}")
    print(f"Sigmas: {SIGMAS}")
    print(f"C = {C}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for ds_name, ds_cfg in DATASETS.items():
        try:
            ds_res = run_dataset(ds_name, ds_cfg)
            all_results.append(ds_res)
            plot_per_dataset_grid(ds_res, OUTPUT_DIR)
        except Exception as e:
            print(f"\n  WARNING: Failed to process {ds_cfg['label']}: {e}")
            print(f"  Skipping dataset and continuing.\n")
            continue

    if len(all_results) == 0:
        print("ERROR: No datasets completed successfully. Exiting.")
        return

    # Cross-dataset figures and table
    plot_coverage_comparison(all_results, OUTPUT_DIR)
    build_combined_table(all_results, OUTPUT_DIR)

    # Save full results JSON
    # Strip raw metrics arrays for the summary (keep them in per-dataset files)
    for ds_res in all_results:
        ds_out = OUTPUT_DIR / f"{ds_res['dataset']}_results.json"
        with open(ds_out, "w") as f:
            json.dump(ds_res, f, indent=2)
        print(f"Saved full results: {ds_out}")

    summary = []
    for ds_res in all_results:
        ds_summary = {k: v for k, v in ds_res.items() if k != "per_sigma"}
        ds_summary["per_sigma"] = [
            {k: v for k, v in entry.items() if k != "metrics_raw"}
            for entry in ds_res["per_sigma"]
        ]
        summary.append(ds_summary)

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {OUTPUT_DIR}")
    print("Experiment 14 complete.")


if __name__ == "__main__":
    main()
