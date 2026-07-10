import sys
import os
import json
import pickle
import random
import numpy as np
import torch
import torchvision
from pathlib import Path
from warnings import warn
from tqdm import tqdm


def set_seed(seed=42):
    """Pin all RNGs (python, numpy, torch, CUDA) for reproducible experiments.

    The DP noise and attack initialisations are drawn via torch, so torch/CUDA
    seeding is what actually makes a rerun bit-reproducible; python/numpy are
    pinned too for completeness (e.g. sample selection).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from skimage.metrics import (
    mean_squared_error as skimage_mse,
    peak_signal_noise_ratio as skimage_psnr,
    structural_similarity as skimage_ssim,
)
import lpips

PROJECT_ROOT = Path(__file__).parent.parent
# fmte lives at the repo root (this file is repo_root/experiments/common.py)
CODE_ROOT = PROJECT_ROOT
sys.path.insert(0, str(CODE_ROOT))

from fmte.distributions import mse_cdf, mse_pdf, get_mse_dist, get_psnr_dist, psnr_pdf
from fmte.utils import torch_to_plt

from opacus import GradSampleModule
from opacus.optimizers import DPOptimizer
from opacus.validators import ModuleValidator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Dataset root. Defaults to the repo-local data/ directory; override with the
# FMTE_DATA_ROOT environment variable.
DATA_ROOT = Path(os.environ.get("FMTE_DATA_ROOT", CODE_ROOT / "data"))

_lpips_model = None

def get_lpips_model():
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex").to(DEVICE)
    return _lpips_model


class LinearNet(torch.nn.Module):
    def __init__(self, res, bias, M=1, additional_layers=None):
        super().__init__()
        if additional_layers is None:
            additional_layers = []
        self.linear = torch.nn.Linear(np.prod(res), M, bias=bias)
        self.additional_layers = len(additional_layers)
        self.preprocessing_layers = []
        for i, (preprocess, layer) in enumerate(additional_layers):
            self.preprocessing_layers.append(preprocess)
            self.register_module(f"add_{i}", layer)

    def forward(self, x):
        out = []
        out.append(self.linear(x.reshape(1, -1)))
        for i, preprocess in zip(range(self.additional_layers), self.preprocessing_layers):
            out.append(self.get_submodule(f"add_{i}")(preprocess(x)))
        return torch.stack(out)


def perform_optimal_recon(input_data_batch, sigma, C, M=1, additional_layers=None, device=DEVICE):
    if additional_layers is None:
        additional_layers = []
    datanorm = torch.linalg.norm(input_data_batch.flatten(1), 2, axis=1)
    if torch.any(M < ((C / datanorm) ** 2)):
        warn("M < (C/|X|)^2")
    net = LinearNet(input_data_batch.shape[1:], False, M, additional_layers=additional_layers).to(device)
    optim = torch.optim.SGD(net.parameters(), lr=1)

    class OptimalReconLoss(torch.nn.Module):
        def forward(self, pred):
            return pred.sum()

    lossfn = OptimalReconLoss().to(device)
    privnet = GradSampleModule(net)
    privoptim = DPOptimizer(
        optimizer=optim, noise_multiplier=sigma, max_grad_norm=C, expected_batch_size=1
    )
    input_data_batch = input_data_batch.to(device)
    recons = []
    for input_data, dn in tqdm(
        zip(input_data_batch, datanorm),
        total=input_data_batch.shape[0],
        desc=f"Recon σ={sigma:.3g}",
        leave=False,
    ):
        privoptim.zero_grad()
        pred = privnet(input_data.reshape(1, -1))
        scaling_factor = max(1, dn * np.sqrt(M) / C)
        loss = lossfn(pred)
        loss.backward()
        privoptim.pre_step()
        grad_w = net.linear.weight.grad.detach().cpu()
        grad_w = grad_w.mean(dim=0) * scaling_factor
        recons.append(grad_w.reshape(input_data.shape).detach().cpu())
    return torch.stack(recons)


def compute_all_metrics(originals, reconstructions, data_range=None):
    """Compute MSE, PSNR, SSIM, LPIPS for a batch of original/reconstruction pairs."""
    mses, psnrs, ssims, lpips_vals = [], [], [], []
    lp_model = get_lpips_model()

    if data_range is None:
        all_orig = np.stack([o.numpy() if hasattr(o, 'numpy') else np.asarray(o) for o in originals])
        data_range = float(all_orig.max() - all_orig.min())
        if data_range < 1e-10:
            data_range = 1.0

    for orig, recon in zip(originals, reconstructions):
        orig_np = orig.numpy() if hasattr(orig, 'numpy') else np.asarray(orig)
        recon_np = recon.numpy() if hasattr(recon, 'numpy') else np.asarray(recon)

        mse_val = skimage_mse(orig_np, recon_np)
        mses.append(mse_val)

        if mse_val > 0:
            psnr_val = skimage_psnr(orig_np, recon_np, data_range=data_range)
        else:
            psnr_val = float("inf")
        psnrs.append(psnr_val)

        if orig_np.ndim == 1:
            n = orig_np.shape[0]
            side_sq = int(np.sqrt(n))
            side_rgb = int(np.sqrt(n / 3))
            if side_sq * side_sq == n and side_sq >= 3:
                o2d = orig_np.reshape(side_sq, side_sq)
                r2d = recon_np.reshape(side_sq, side_sq)
                win_size = min(7, side_sq)
                if win_size % 2 == 0:
                    win_size -= 1
                if win_size >= 3:
                    ssim_val = skimage_ssim(o2d, r2d, data_range=data_range, win_size=win_size)
                else:
                    ssim_val = float("nan")
            elif side_rgb * side_rgb * 3 == n and side_rgb >= 3:
                o3d = orig_np.reshape(3, side_rgb, side_rgb)
                r3d = recon_np.reshape(3, side_rgb, side_rgb)
                win_size = min(7, side_rgb)
                if win_size % 2 == 0:
                    win_size -= 1
                if win_size >= 3:
                    ssim_val = skimage_ssim(o3d, r3d, data_range=data_range, win_size=win_size, channel_axis=0)
                else:
                    ssim_val = float("nan")
            else:
                ssim_val = float("nan")
        elif orig_np.ndim == 3:
            min_dim = min(orig_np.shape[1], orig_np.shape[2])
            win_size = min(7, min_dim) if min_dim >= 3 else None
            if win_size and win_size % 2 == 0:
                win_size -= 1
            if win_size and win_size >= 3:
                ssim_val = skimage_ssim(
                    orig_np, recon_np, data_range=data_range,
                    win_size=win_size, channel_axis=0,
                )
            else:
                ssim_val = float("nan")
        else:
            ssim_val = float("nan")
        ssims.append(ssim_val)

        import torch as _torch
        orig_t = _torch.as_tensor(orig_np).float() if not isinstance(orig, _torch.Tensor) else orig.float()
        recon_t = _torch.as_tensor(recon_np).float() if not isinstance(recon, _torch.Tensor) else recon.float()

        if orig_t.ndim == 1:
            n = orig_t.shape[0]
            side = int(np.sqrt(n / 3))
            if side * side * 3 == n and side >= 16:
                o_img = orig_t.reshape(3, side, side).unsqueeze(0).to(DEVICE)
                r_img = recon_t.reshape(3, side, side).unsqueeze(0).to(DEVICE)
            elif side * side == n and side >= 16:
                o_img = orig_t.reshape(1, side, side).repeat(3, 1, 1).unsqueeze(0).to(DEVICE)
                r_img = recon_t.reshape(1, side, side).repeat(3, 1, 1).unsqueeze(0).to(DEVICE)
            else:
                lpips_vals.append(float("nan"))
                continue
        elif orig_t.ndim == 3:
            if orig_t.shape[0] == 3 and orig_t.shape[1] >= 16:
                o_img = orig_t.unsqueeze(0).to(DEVICE)
                r_img = recon_t.unsqueeze(0).to(DEVICE)
            elif orig_t.shape[0] == 1 and orig_t.shape[1] >= 16:
                o_img = orig_t.repeat(3, 1, 1).unsqueeze(0).to(DEVICE)
                r_img = recon_t.repeat(3, 1, 1).unsqueeze(0).to(DEVICE)
            else:
                lpips_vals.append(float("nan"))
                continue
        else:
            lpips_vals.append(float("nan"))
            continue

        with torch.no_grad():
            # LPIPS (AlexNet) needs at least 64x64 spatial resolution
            if o_img.shape[-1] < 64 or o_img.shape[-2] < 64:
                o_img = torch.nn.functional.interpolate(o_img, size=(64, 64), mode="bilinear", align_corners=False)
                r_img = torch.nn.functional.interpolate(r_img, size=(64, 64), mode="bilinear", align_corners=False)
            o_scaled = 2 * (o_img / max(data_range, 1e-10)) - 1
            r_scaled = 2 * (r_img / max(data_range, 1e-10)) - 1
            lp_val = lp_model(o_scaled, r_scaled).item()
        lpips_vals.append(lp_val)

    return {
        "mse": np.array(mses),
        "psnr": np.array(psnrs),
        "ssim": np.array(ssims),
        "lpips": np.array(lpips_vals),
    }


def compute_success_probability_curve(mses, etas):
    """Compute P[MSE <= eta] for a grid of eta values."""
    mses = np.array(mses)
    return np.array([np.mean(mses <= eta) for eta in etas])


def compute_theoretical_success_prob(etas, N, sigma, max_data_norm):
    """Compute theoretical P[MSE <= eta] from the gamma CDF."""
    return np.array([mse_cdf(eta, N, sigma, max_data_norm) for eta in etas])


def compute_coverage_rate(mses, sigma, N, max_data_norm, alpha=0.95):
    """Fraction of empirical MSE values that fall below the theoretical alpha-quantile."""
    dist = get_mse_dist(N, sigma, max_data_norm)
    theoretical_quantile = dist.ppf(alpha)
    return np.mean(np.array(mses) <= theoretical_quantile)


def compute_bound_tightness(empirical_probs, theoretical_probs):
    """Ratio of empirical success probability to theoretical bound."""
    mask = theoretical_probs > 1e-10
    ratios = np.full_like(theoretical_probs, np.nan)
    ratios[mask] = empirical_probs[mask] / theoretical_probs[mask]
    return ratios


def sigma_to_epsilon(sigma, sample_rate, steps, delta=1e-5):
    from opacus.accountants import RDPAccountant
    accountant = RDPAccountant()
    for _ in range(steps):
        accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
    return accountant.get_epsilon(delta=delta)


def epsilon_to_sigma(target_epsilon, sample_rate, steps, delta=1e-5):
    from opacus.accountants.utils import get_noise_multiplier
    return get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        steps=steps,
    )


def load_dataset(name, resolution=None, num_samples=500, enforced_norm=1.01, flatten=True, grayscale=False):
    transforms_list = []
    if grayscale:
        transforms_list.append(torchvision.transforms.Grayscale())
    if resolution is not None:
        transforms_list.append(torchvision.transforms.Resize(resolution))
        transforms_list.append(torchvision.transforms.CenterCrop(resolution))
    transforms_list.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(transforms_list)

    if name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=str(DATA_ROOT), download=True, transform=transform,
        )
    elif name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            root=str(DATA_ROOT), download=True, transform=transform,
        )
    elif name == "celeba":
        dataset = torchvision.datasets.CelebA(
            root=str(DATA_ROOT / "CelebA"), split="train", transform=transform,
        )
    elif name == "medmnist_path":
        import medmnist
        info = medmnist.INFO["pathmnist"]
        DataClass = getattr(medmnist, info["python_class"])
        dataset = DataClass(split="train", transform=transform, download=True)
    elif name == "mnist":
        dataset = torchvision.datasets.MNIST(
            root=str(DATA_ROOT / "MNIST"), download=True, transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    np.random.seed(30989)
    indices = np.random.randint(0, len(dataset), size=num_samples)
    samples = [dataset[i][0] for i in indices]

    if flatten:
        datasamples = torch.stack([s.flatten() for s in samples])
    else:
        datasamples = torch.stack(samples)

    if enforced_norm is not None:
        flat_view = datasamples.reshape(datasamples.shape[0], -1)
        norm = torch.linalg.norm(flat_view, 2, dim=1, keepdim=True)
        scale = enforced_norm / norm
        if flatten:
            datasamples = flat_view * scale
        else:
            datasamples = datasamples * scale.reshape(-1, *([1] * (datasamples.ndim - 1)))

    return datasamples


def run_full_experiment(
    datasamples,
    noise_multipliers_theory,
    noise_multipliers_eval,
    C,
    M=1,
    additional_layers=None,
    data_range=1.0,
    eta_grid_size=200,
):
    if additional_layers is None:
        additional_layers = []

    N = datasamples.shape[1] if datasamples.ndim == 2 else np.prod(datasamples.shape[1:])
    data_norms = torch.linalg.norm(datasamples.flatten(1), 2, dim=1)
    min_norm = data_norms.min().item()

    results = {
        "N": N,
        "C": C,
        "M": M,
        "min_data_norm": min_norm,
        "noise_multipliers_eval": noise_multipliers_eval.tolist(),
        "per_sigma": [],
    }

    mse_dist_low = get_mse_dist(N, noise_multipliers_eval.min(), min_norm)
    mse_dist_high = get_mse_dist(N, noise_multipliers_eval.max(), min_norm)
    eta_min = max(1e-10, mse_dist_low.ppf(0.001))
    eta_max = mse_dist_high.ppf(0.999)
    eta_grid = np.logspace(np.log10(eta_min), np.log10(eta_max), eta_grid_size)

    for sigma in tqdm(noise_multipliers_eval, desc="Sigma sweep"):
        recons = perform_optimal_recon(datasamples, sigma, C, M, additional_layers)
        metrics = compute_all_metrics(datasamples, recons, data_range=data_range)

        empirical_success_curve = compute_success_probability_curve(metrics["mse"], eta_grid)
        theoretical_success_curve = compute_theoretical_success_prob(eta_grid, N, sigma, min_norm)
        tightness = compute_bound_tightness(empirical_success_curve, theoretical_success_curve)
        coverage_95 = compute_coverage_rate(metrics["mse"], sigma, N, min_norm, alpha=0.95)

        sigma_results = {
            "sigma": sigma,
            "metrics": {k: v.tolist() for k, v in metrics.items()},
            "eta_grid": eta_grid.tolist(),
            "empirical_success_curve": empirical_success_curve.tolist(),
            "theoretical_success_curve": theoretical_success_curve.tolist(),
            "bound_tightness": tightness.tolist(),
            "coverage_95": coverage_95,
            "summary": {
                "mse_mean": float(np.mean(metrics["mse"])),
                "mse_median": float(np.median(metrics["mse"])),
                "psnr_mean": float(np.nanmean(metrics["psnr"][np.isfinite(metrics["psnr"])])),
                "ssim_mean": float(np.nanmean(metrics["ssim"])),
                "lpips_mean": float(np.nanmean(metrics["lpips"])),
            },
        }
        results["per_sigma"].append(sigma_results)

    return results


def save_results(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "raw_results.pkl", "wb") as f:
        pickle.dump(results, f)

    summary = {
        "N": results["N"],
        "C": results["C"],
        "M": results["M"],
        "min_data_norm": results["min_data_norm"],
        "per_sigma_summary": [],
    }
    for sr in results["per_sigma"]:
        summary["per_sigma_summary"].append({
            "sigma": sr["sigma"],
            "coverage_95": sr["coverage_95"],
            **sr["summary"],
        })

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def plot_metrics_vs_sigma(results, output_dir, title_prefix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sn
    sn.set_theme(context="notebook", style="white", font="Arial")

    output_dir = Path(output_dir)
    sigmas_eval = [sr["sigma"] for sr in results["per_sigma"]]
    N = results["N"]
    min_norm = results["min_data_norm"]

    noise_multipliers_theory = np.logspace(
        np.log10(min(sigmas_eval)) - 0.5,
        np.log10(max(sigmas_eval)) + 0.5,
        500,
    )

    fig, axs = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")

    metric_configs = [
        ("mse", "MSE", True),
        ("psnr", "PSNR (dB)", False),
        ("ssim", "SSIM", False),
        ("lpips", "LPIPS", False),
    ]

    for idx, (metric_key, label, log_y) in enumerate(metric_configs):
        ax = axs[idx // 2][idx % 2]

        if metric_key == "mse":
            mse_etas_low = get_mse_dist(N, min(sigmas_eval), min_norm).ppf(0.05)
            mse_etas_high = get_mse_dist(N, max(sigmas_eval), min_norm).ppf(0.95)
            mse_etaspace = np.logspace(np.log10(max(1e-10, mse_etas_low)), np.log10(mse_etas_high), 300)
            mseX, mseY = np.meshgrid(noise_multipliers_theory, mse_etaspace)
            mse_pdfs = np.array([
                mse_pdf(mse_etaspace, N, s, min_norm) for s in noise_multipliers_theory
            ])
            mse_pdfs /= np.maximum(mse_pdfs.max(axis=1, keepdims=True), 1e-20)
            ax.pcolormesh(
                np.log10(mseX), mseY, mse_pdfs.T,
                cmap="hot_r", shading="auto", alpha=0.7, rasterized=True,
            )

        for sr in results["per_sigma"]:
            vals = np.array(sr["metrics"][metric_key])
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            ax.boxplot(
                vals, positions=[np.log10(sr["sigma"])], widths=0.3,
                patch_artist=True,
                boxprops=dict(facecolor="limegreen", alpha=0.5),
                flierprops=dict(marker="x", markersize=0.5),
                meanline=True,
            )

        ax.set_ylabel(label, fontsize=9)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("$\\sigma$", fontsize=9)

        def update_ticks(x, pos):
            return f"$10^{{{int(x)}}}$"
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))
        ax.tick_params(axis="both", which="major", labelsize=8)

    fig.suptitle(f"{title_prefix}", fontsize=11)
    fig.savefig(output_dir / "metrics_vs_sigma.pdf", dpi=150)
    plt.close(fig)


def plot_success_probability_curves(results, output_dir, title_prefix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    N = results["N"]
    min_norm = results["min_data_norm"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout="constrained")
    colors = plt.cm.viridis(np.linspace(0, 1, len(results["per_sigma"])))

    for i, sr in enumerate(results["per_sigma"]):
        etas = np.array(sr["eta_grid"])
        empirical = np.array(sr["empirical_success_curve"])
        theoretical = np.array(sr["theoretical_success_curve"])

        ax.plot(etas, theoretical, color=colors[i], linestyle="--", alpha=0.7,
                label=f"Theory σ={sr['sigma']:.2g}")
        ax.plot(etas, empirical, color=colors[i], linestyle="-", alpha=0.9,
                label=f"Empirical σ={sr['sigma']:.2g}")

    ax.set_xscale("log")
    ax.set_xlabel("η (MSE threshold)", fontsize=9)
    ax.set_ylabel("P[MSE ≤ η]", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"{title_prefix} Success Probability Curves", fontsize=10)
    fig.savefig(output_dir / "success_probability_curves.pdf", dpi=150)
    plt.close(fig)


def plot_bound_tightness(results, output_dir, title_prefix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), layout="constrained")
    colors = plt.cm.viridis(np.linspace(0, 1, len(results["per_sigma"])))

    for i, sr in enumerate(results["per_sigma"]):
        etas = np.array(sr["eta_grid"])
        tightness = np.array(sr["bound_tightness"])
        mask = np.isfinite(tightness) & (tightness > 0)
        if mask.any():
            ax.plot(etas[mask], tightness[mask], color=colors[i],
                    label=f"σ={sr['sigma']:.2g} (cov={sr['coverage_95']:.2f})")

    ax.axhline(y=1.0, color="red", linestyle=":", alpha=0.5, label="Perfect tightness")
    ax.set_xscale("log")
    ax.set_xlabel("η (MSE threshold)", fontsize=9)
    ax.set_ylabel("Empirical / Theoretical", fontsize=9)
    ax.set_ylim(0, 2.0)
    ax.legend(fontsize=7)
    ax.set_title(f"{title_prefix} Bound Tightness", fontsize=10)
    fig.savefig(output_dir / "bound_tightness.pdf", dpi=150)
    plt.close(fig)
