"""Regenerate figure5.pdf (Appendix B, fig:tightness_high_n) at a shorter height.

Reuses the exact data-generation logic from figures.py (same seeds, same
noise grid, same 500 CIFAR-10 samples at N=3072) so the underlying numbers
are unchanged; only the figure's physical height is reduced (5.5x3.09in ->
5.5x2.15in) so it can be included at width=\\linewidth in the manuscript
instead of the old scale=0.8 hack, keeping fonts at their coded point size.
"""
import pickle
from pathlib import Path
import torch
import torchvision
import numpy as np
from warnings import warn
from typing import Callable

from opacus import GradSampleModule
from opacus.optimizers import DPOptimizer
from opacus.validators import ModuleValidator

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

from fmte.distributions import mse_pdf, psnr_pdf, get_mse_dist, get_psnr_dist
from skimage.metrics import (
    mean_squared_error as mse,
    peak_signal_noise_ratio as psnr,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LinearNet(torch.nn.Module):
    def __init__(self, res: tuple, bias: bool, M: int = 1, additional_layers=[]):
        super().__init__()
        self.linear = torch.nn.Linear(np.prod(res), M, bias=bias)
        self.additional_layers = len(additional_layers)
        self.preprocessing_layers = []
        for i, (preprocess, layer) in enumerate(additional_layers):
            self.preprocessing_layers.append(preprocess)
            self.register_module(f"add_{i}", layer)

    def forward(self, x):
        out = [self.linear(x.reshape(1, -1))]
        for i, preprocess in zip(range(self.additional_layers), self.preprocessing_layers):
            out.append(self.get_submodule(f"add_{i}")(preprocess(x)))
        return torch.stack(out)


def performOptimalRecon(input_data_batch, sigma, C, M=1, additional_layers=[], device=DEVICE):
    assert isinstance(M, int)
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
    privoptim = DPOptimizer(optimizer=optim, noise_multiplier=sigma, max_grad_norm=C, expected_batch_size=1)
    input_data_batch = input_data_batch.to(device)
    recons = []
    for input_data, dn in tqdm(zip(input_data_batch, datanorm), total=input_data_batch.shape[0], desc="Processing batch"):
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


def find_value_ranges(sigmas, min_norm, N_dimensions, coverage=0.95):
    mse_min_cov, mse_max_cov, psnr_min_cov, psnr_max_cov = 1.0 - coverage, coverage, 1.0 - coverage, coverage
    min_sigma, max_sigma = min(sigmas), max(sigmas)
    min_mse_dist = get_mse_dist(N_dimensions, min_sigma, min_norm)
    max_mse_dist = get_mse_dist(N_dimensions, max_sigma, min_norm)
    min_psnr_dist = get_psnr_dist(N_dimensions, min_sigma, 1.0, min_norm)
    max_psnr_dist = get_psnr_dist(N_dimensions, max_sigma, 1.0, min_norm)
    mse_ranges = np.array([min_mse_dist.ppf(mse_min_cov), max_mse_dist.ppf(mse_max_cov)])
    psnr_ranges = np.array([max_psnr_dist.ppf(psnr_min_cov), min_psnr_dist.ppf(psnr_max_cov)])
    return mse_ranges, psnr_ranges


def calc_recon_scores(data_samples, noise_multipliers, num_sigmas_eval, C, M, additional_layers):
    torch.random.manual_seed(120496)
    np.random.seed(0)
    N_dimensions = data_samples.shape[1]
    noise_multipliers_eval = np.logspace(
        np.log10(noise_multipliers[0]), np.log10(noise_multipliers[-1]), num_sigmas_eval + 2
    )[1:-1]

    data_norms = [torch.linalg.norm(img, 2).item() for img in data_samples]
    if min(data_norms) * np.sqrt(M) < C:
        warn("Clipping threshold not exceeded! Empirical results are not as good as they could be!")

    mse_etas, psnr_etas = find_value_ranges(noise_multipliers, min(data_norms), N_dimensions, 0.95)
    mse_etas_log = np.log(mse_etas)

    mse_eta_min, mse_eta_max = mse_etas_log
    mse_etaspace = np.logspace(mse_eta_min, mse_eta_max, 500)
    psnr_etaspace = np.linspace(psnr_etas[0], psnr_etas[1], 500)
    mseX, mseY = np.meshgrid(noise_multipliers, mse_etaspace)
    psnrX, psnrY = np.meshgrid(noise_multipliers, psnr_etaspace)
    mse_pdfs = np.array([mse_pdf(mse_etaspace, N_dimensions, sigma, min(data_norms)) for sigma in noise_multipliers])
    mse_pdfs /= mse_pdfs.max(axis=1, keepdims=True)
    psnr_pdfs = np.array([psnr_pdf(psnr_etaspace, N_dimensions, sigma, 1, min(data_norms)) for sigma in noise_multipliers])

    mses_per_sigma, psnrs_per_sigma = [], []
    for j, sigma in tqdm(enumerate(noise_multipliers_eval), leave=False):
        mses, psnrs = [], []
        recons = performOptimalRecon(data_samples, sigma, C, M, additional_layers=additional_layers)
        for img, recon in tqdm(zip(data_samples, recons), leave=False, desc="metric calc"):
            mses.append(mse(img.numpy(), recon.numpy()))
            psnrs.append(psnr(img.numpy(), recon.numpy(), data_range=1.0))
        mses_per_sigma.append(mses)
        psnrs_per_sigma.append(psnrs)
    return (mse_etas, mseX, mseY, psnrX, psnrY, mse_pdfs, psnr_pdfs, mses_per_sigma, psnrs_per_sigma, noise_multipliers_eval)


def make_dist_overlap_figure_clipped(axs, mse_etas, mseX, mseY, psnrX, psnrY, mse_pdfs, psnr_pdfs, mses_per_sigma, psnrs_per_sigma, noise_multipliers_eval):
    axs[0].set_ylim(*mse_etas)
    axs[0].pcolormesh(np.log10(mseX), mseY, mse_pdfs.T, cmap="hot_r", shading="auto", alpha=0.7, rasterized=True)
    axs[1].pcolormesh(np.log10(psnrX), psnrY, psnr_pdfs.T, cmap="hot_r", shading="auto", alpha=0.7, rasterized=True)
    for j, sigma in tqdm(enumerate(noise_multipliers_eval), leave=False):
        axs[0].boxplot(mses_per_sigma[j], positions=[np.log10(sigma)], widths=0.3, patch_artist=True,
                        boxprops=dict(facecolor="limegreen", alpha=0.5), flierprops=dict(marker="x", markersize=0.2), meanline=True)
        axs[1].boxplot(psnrs_per_sigma[j], positions=[np.log10(sigma)], widths=0.3, patch_artist=True,
                        boxprops=dict(facecolor="limegreen", alpha=0.5), flierprops=dict(marker="x", markersize=0.2), meanline=True)
    for i, a in enumerate(axs):
        if i == 0:
            a.set_yscale("log")

    def update_ticks(x, pos):
        return f"$10^{{{int(x)}}}$"

    for ax in np.ravel(axs):
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.tick_params(axis="both", which="minor", labelsize=6)
        ax.tick_params(axis="x", labelrotation=45)


# --- exact data generation from figures.py (N=3072 CIFAR-10 32x32 block) ---
# Cached to disk: this is the expensive part (ResNet-101 forward/backward over
# 500 samples x 3 archs x 3 sigmas, ~7 min); plot-only tweaks (figsize, axis
# sharing) should not need to rerun it.
CACHE = Path(__file__).parent / "figure5_recon_scores.pkl"
if CACHE.exists():
    with open(CACHE, "rb") as f:
        recon_scores = pickle.load(f)
    print(f"Loaded cached recon_scores from {CACHE}")
else:
    dataset = torchvision.datasets.CIFAR10(
        root="./data/", download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
        target_transform=lambda x: torch.Tensor(x),
    )
    np.random.seed(30989)
    reduceddataset = [dataset[i] for i in np.random.randint(0, len(dataset), size=500)]
    datasamples = torch.stack([d[0].flatten() for d in reduceddataset])
    norm = torch.linalg.norm(datasamples, 2, axis=1, keepdims=True)
    enforced_norm = 1.01
    datasamples = (enforced_norm / norm) * datasamples

    recon_scores = {}
    recon_scores["optimal"] = calc_recon_scores(
        datasamples, noise_multipliers=np.logspace(-2, 2, 1000), num_sigmas_eval=3, C=1, M=1, additional_layers=[]
    )

    linear_net = torch.nn.Sequential(
        torch.nn.Linear(datasamples.shape[-1], int(1000000 / datasamples.shape[-1]), bias=False),
        torch.nn.AvgPool1d(int(1000000 / datasamples.shape[-1])),
    )
    recon_scores["linear"] = calc_recon_scores(
        datasamples, noise_multipliers=np.logspace(-2, 2, 1000), num_sigmas_eval=3, C=1, M=1,
        additional_layers=[(torch.nn.Identity(), linear_net)],
    )

    transform_tensor = lambda x: x.reshape(1, 3, 32, 32)
    resnet = torchvision.models.resnet101(weights=None)
    resnet = ModuleValidator.fix(resnet)
    recon_scores["resnet"] = calc_recon_scores(
        datasamples, noise_multipliers=np.logspace(-2, 2, 1000), num_sigmas_eval=3, C=1, M=1,
        additional_layers=[(transform_tensor, torch.nn.Sequential(resnet, torch.nn.AvgPool1d(1000)))],
    )
    with open(CACHE, "wb") as f:
        pickle.dump(recon_scores, f)

# --- plot at a shorter height, sharing the x-axis across MSE and PSNR rows
# (both index the same sigma grid) so tick labels aren't drawn twice ---
fig, axs = plt.subplots(2, 3, figsize=(5.5, 1.75), sharex=True, sharey="row", layout="constrained")

make_dist_overlap_figure_clipped(axs[:, 0], *recon_scores["optimal"])
make_dist_overlap_figure_clipped(axs[:, 1], *recon_scores["linear"])
make_dist_overlap_figure_clipped(axs[:, 2], *recon_scores["resnet"])

axs[0][0].set_ylabel("MSE", fontsize=8)
axs[1][0].set_ylabel("PSNR", fontsize=8)
axs[0][0].set_title("Optimal", fontsize=8)
axs[0][1].set_title("Linear (1M params)", fontsize=8)
axs[0][2].set_title("ResNet-101", fontsize=8)
axs[1][0].set_xlabel("$\\sigma$", fontsize=8)
axs[1][1].set_xlabel("$\\sigma$", fontsize=8)
axs[1][2].set_xlabel("$\\sigma$", fontsize=8)

fig.savefig("figure5.pdf")
print("Saved figure5.pdf")
