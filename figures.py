# %%
import torch
import torchvision
import numpy as np
import seaborn as sn
from tqdm import tqdm
from warnings import warn

from opacus import GradSampleModule
from opacus.optimizers import DPOptimizer
import matplotlib.ticker as mticker


from matplotlib import pyplot as plt, patches as mpatches

from fmte.utils import (
    ncc,
    torch_to_plt,
    image_prior_numpy,
)
from fmte.ncc import ncc_bound
from fmte.distributions import (
    psnr_cdf,
    mse_pdf,
    psnr_pdf,
    get_mse_dist,
    get_psnr_dist,
)

from lpips import LPIPS
from skimage.metrics import (
    mean_squared_error as mse,
    normalized_mutual_information as nmi,
    peak_signal_noise_ratio as psnr,
    structural_similarity as ssim,
)
from num2tex import num2tex, configure as num2tex_configure

num2tex_configure(exp_format="cdot")
showimg = lambda img: image_prior_numpy(torch_to_plt(img)).squeeze()


sn.set_theme(
    context="notebook",
    font="Times New Roman",
    # palette="viridis",
    rc={
        "text.usetex": True,
        "lines.linewidth": 2,
    },
    font_scale=2,
    style="whitegrid",
)
sn.despine()
colors = {
    "green": "forestgreen",
    "green2": "limegreen",
    "turquoise": "turquoise",
    "orange": "darkorange",
    "purple": "darkorchid",
    "brown": "xkcd:light brown",
    "blue3": "slateblue",
    "blue2": "royalblue",
    "red": "firebrick",
    "gray": "lightslategray",
    "blue": "steelblue",
    "black": "black",
}


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LinearNet(torch.nn.Module):

    def __init__(self, res: tuple[int], bias: bool, M: int = 1):
        super().__init__()
        self.linear = torch.nn.Linear(np.prod(res), M, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.reshape(1, -1))


def performReconFowl(input_data, target, sigma, C, num_fitting_steps=0, M=1):
    net = LinearNet(input_data.shape, True, M)
    optim = torch.optim.Adam(net.parameters(), lr=1e-5)
    lossfn = torch.nn.MSELoss()  # doesnt really matter which one

    for _ in range(num_fitting_steps):
        optim.zero_grad()
        pred = net(input_data)
        loss = pred
        loss.backward()
        optim.step()

    privnet = GradSampleModule(net)
    privoptim = DPOptimizer(
        optimizer=optim, noise_multiplier=sigma, max_grad_norm=C, expected_batch_size=1
    )
    privoptim.zero_grad()
    pred = privnet(input_data)
    loss = lossfn(pred, target.float())
    loss.backward()
    privoptim.pre_step()
    grad_w = net.linear.weight.grad.detach().cpu()
    grad_b = net.linear.bias.grad.detach().cpu()
    recon = (grad_w / grad_b).reshape(M, *input_data.shape)
    return recon


def performOptimalRecon(input_data, sigma, C, M=1):
    assert isinstance(M, int)
    datanorm = torch.linalg.norm(input_data.flatten(), 2)
    if M < ((C / datanorm) ** 2):
        warn("M < (C/|X|)²")
    net = LinearNet(input_data.shape, False, M)
    optim = torch.optim.SGD(net.parameters(), lr=1)

    class OptimalReconLoss(torch.nn.Module):

        def forward(self, pred):
            return pred.sum()

    lossfn = OptimalReconLoss()

    privnet = GradSampleModule(net)
    privoptim = DPOptimizer(
        optimizer=optim, noise_multiplier=sigma, max_grad_norm=C, expected_batch_size=1
    )
    privoptim.zero_grad()
    pred = privnet(input_data)
    scaling_factor = max(1, datanorm * np.sqrt(M) / C)
    loss = lossfn(pred)
    loss.backward()
    privoptim.pre_step()
    grad_w = net.linear.weight.grad.detach().cpu()
    grad_w = grad_w.mean(dim=0) * scaling_factor
    return grad_w.reshape(input_data.shape)


def find_value_ranges(sigmas, max_norm, N_dimensions, max_data_norm, coverage=0.95):
    if isinstance(coverage, float):
        mse_min_cov, mse_max_cov, psnr_min_cov, psnr_max_cov = (
            1.0 - coverage,
            coverage,
            1.0 - coverage,
            coverage,
        )
    elif isinstance(coverage, tuple):
        if len(coverage) == 2:
            mse_min_cov, mse_max_cov, psnr_min_cov, psnr_max_cov = (
                1.0 - coverage[0],
                coverage[0],
                1.0 - coverage[1],
                coverage[1],
            )
        elif len(coverage) == 4:
            mse_min_cov, mse_max_cov, psnr_min_cov, psnr_max_cov = coverage
        else:
            raise ValueError(
                f"Coverage must be of length 1, 2, or 4 but is {len(coverage)}"
            )
    else:
        raise ValueError(f"Coverage must be tuple or float but is {type(coverage)}")
    min_sigma, max_sigma = min(sigmas), max(sigmas)
    min_mse_dist = get_mse_dist(N_dimensions, min_sigma, max_norm)
    max_mse_dist = get_mse_dist(N_dimensions, max_sigma, max_norm)
    min_psnr_dist = get_psnr_dist(N_dimensions, min_sigma, 1.0, max_data_norm)
    max_psnr_dist = get_psnr_dist(N_dimensions, max_sigma, 1.0, max_data_norm)
    mse_ranges = np.array(
        [min_mse_dist.ppf(mse_min_cov), max_mse_dist.ppf(mse_max_cov)]
    )
    psnr_ranges = np.array(
        [
            max_psnr_dist.ppf(psnr_min_cov),
            min_psnr_dist.ppf(psnr_max_cov),
        ]
    )
    return mse_ranges, psnr_ranges


# %%
dataset = torchvision.datasets.ImageNet(
    root="./data/ILSVRC2012/",
    transform=torchvision.transforms.Compose(
        [
            # torchvision.transforms.Grayscale(1),
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
        ]
    ),
)
sigma = 5e-4
largeM = 1000

imgs = [dataset[i][0] for i in [1, 544263, 586966, 539063]]
datanorms = [torch.linalg.norm(img.flatten(), 2) for img in imgs]
Cs = [np.sqrt(largeM) * datanorm for datanorm in datanorms]
C = 5000
print(C)
print([(C / dn) ** 2 for dn in datanorms])

badrecons, goodrecons = [], []
for img in imgs:
    badrecons.append(performOptimalRecon(img, sigma, C, M=1))
    goodrecons.append(performOptimalRecon(img, sigma, C, M=largeM))
# print(f"MSE: {mse(recon.numpy(), img.numpy())}")
# print(f"Recon range: {recon.min()} {recon.max()}")
# print(f"Img range: {img.min()} {img.max()}")
fig, axs = plt.subplots(1, 3, figsize=(15, 10))
for ax in axs:
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2)
convert_img_list_to_grid = lambda imgs: torch_to_plt(
    torchvision.utils.make_grid(torch.stack(imgs), nrow=2)
).squeeze()
axs[0].imshow(convert_img_list_to_grid(imgs))  # , cmap="gray")
axs[1].imshow(convert_img_list_to_grid(badrecons))  # , cmap="gray")
axs[2].imshow(convert_img_list_to_grid(goodrecons))  # , cmap="gray")
axs[0].set_title("Original")
axs[1].set_title("$M<\\left(\\frac{C}{\Vert X\Vert_2}\\right)^2$")
axs[2].set_title("$M\\geq\\left(\\frac{C}{\Vert X\Vert_2}\\right)^2$")

fig.savefig("figure1.pdf", bbox_inches="tight", dpi=600)


def make_dist_overlap_figure_multi_C(
    noise_multipliers, N_dimensions, num_data_samples, num_sigmas_eval, M
):
    torch.random.manual_seed(120496)
    np.random.seed(0)
    max_data_value = 1
    data_samples = (
        torch.rand((num_data_samples, N_dimensions), dtype=torch.float) * max_data_value
    )
    # img_idcs = np.random.randint(0, len(dataset), num_images)
    # imgs = [dataset[i][0].flatten()[:N_dimensions] for i in img_idcs]
    noise_multipliers_eval = np.logspace(
        np.log10(noise_multipliers[0]),
        np.log10(noise_multipliers[-1]),
        num_sigmas_eval + 2,
    )[1:-1]

    data_norms = [torch.linalg.norm(img, 2).item() for img in data_samples]
    max_grad_norms = np.array(
        [
            min(data_norms),
            np.sqrt(M) * min(data_norms),
            10 * np.sqrt(M) * max(data_norms),
        ]
    )
    row_titles = [
        # "$C=\\frac{1}{10}\min_{X \in \mathcal{X}}\Vert X\Vert_2$",
        "$C=\min_{X \in \mathcal{D}}\Vert X\Vert_2$",
        "$C=\sqrt{M}\min_{X \in \mathcal{D}}\Vert X\Vert_2$",
        "$C=10\sqrt{M}\max_{X \in \mathcal{D}}\Vert X\Vert_2$",
    ]

    mse_etas, psnr_etas = find_value_ranges(
        noise_multipliers, max(data_norms), N_dimensions, min(data_norms), 0.95
    )
    mse_etas_log = np.log(mse_etas)
    ncc_bounds = ncc_bound(noise_multipliers, N_dimensions)

    # Visualization setup
    fig, axs = plt.subplots(
        max_grad_norms.shape[0], 3, figsize=(15, 10), sharex="col", sharey="col"
    )
    mse_eta_min, mse_eta_max = mse_etas_log
    mse_etaspace = np.logspace(mse_eta_min, mse_eta_max, 500)
    psnr_etaspace = np.linspace(psnr_etas[0], psnr_etas[1], 500)
    mseX, mseY = np.meshgrid(noise_multipliers, mse_etaspace)
    psnrX, psnrY = np.meshgrid(noise_multipliers, psnr_etaspace)

    for i, C in enumerate(max_grad_norms):
        # Expand the range of etaspace for better coverage
        axs[i, 0].set_ylim(*mse_etas)
        axs[i, 1].set_ylim(*psnr_etas)

        # Compute PDFs
        mse_pdfs = np.array(
            [
                mse_pdf(mse_etaspace, N_dimensions, sigma, max(data_norms))
                for sigma in noise_multipliers
            ]
        )
        mse_pdfs /= mse_pdfs.max(
            axis=1, keepdims=True
        )  # Normalize per noise multiplier
        # mse_pdfs *= mse_cdf(mse_etaspace.max(), N_dimensions, noise_multipliers, C).reshape(
        #     -1, 1
        # )
        psnr_pdfs = np.array(
            [
                psnr_pdf(psnr_etaspace, N_dimensions, sigma, 1, max(data_norms))
                for sigma in noise_multipliers
            ]
        )
        # psnr_pdfs[np.isnan(psnr_pdfs)] = 0
        psnr_pdfs /= psnr_pdfs.max(
            axis=1, keepdims=True
        )  # Normalize per noise multiplier
        psnr_pdfs *= psnr_cdf(
            psnr_etaspace.max(), N_dimensions, noise_multipliers, 1, max(data_norms)
        ).reshape(-1, 1)

        # Plot with pcolormesh
        axs[i, 0].pcolormesh(
            np.log10(mseX),
            mseY,
            mse_pdfs.T,
            cmap="hot_r",
            shading="auto",
            alpha=0.7,
            rasterized=True,
        )
        axs[i, 1].pcolormesh(
            np.log10(psnrX),
            psnrY,
            psnr_pdfs.T,
            cmap="hot_r",
            shading="auto",
            alpha=0.7,
            rasterized=True,
        )
        axs[i, 2].plot(
            np.log10(noise_multipliers), ncc_bounds, color="black", linestyle="--"
        )
        for j, sigma in enumerate(noise_multipliers_eval):
            mses, psnrs, nccs = [], [], []
            for img in data_samples:
                recon = performOptimalRecon(img, sigma, C, M)
                mses.append(mse(img.numpy(), recon.numpy()))
                psnrs.append(psnr(img.numpy(), recon.numpy(), data_range=1.0))
                nccs.append(ncc(img.numpy(), recon.numpy()))
            nccs = np.array(nccs)
            nccs = nccs[~np.isnan(nccs)]
            axs[i, 0].boxplot(
                mses,
                positions=[np.log10(sigma)],
                widths=0.3,
                patch_artist=True,
                boxprops=dict(facecolor="limegreen", alpha=0.5),
            )
            axs[i, 1].boxplot(
                psnrs,
                positions=[np.log10(sigma)],
                widths=0.3,
                patch_artist=True,
                boxprops=dict(facecolor="limegreen", alpha=0.5),
            )
            axs[i, 2].boxplot(
                nccs,
                positions=[np.log10(sigma)],
                widths=0.3,
                patch_artist=True,
                boxprops=dict(facecolor="limegreen", alpha=0.5),
            )
    for i, ax in enumerate(axs.T):
        for a in ax:
            if i == 0:
                a.set_ylabel("MSE")
                a.set_yscale("log")
            if i == 1:
                a.set_ylabel("PSNR")
            if i == 2:
                a.set_ylabel("NCC")
    for ax in axs[-1]:
        ax.set_xlabel("$\\sigma$")

    def update_ticks(x, pos):
        return f"$10^{{{int(x)}}}$"

    for ax in np.ravel(axs):
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))

    for i, ax in enumerate(axs):
        ax[len(ax) // 2].set_title(row_titles[i])

    plt.tight_layout()
    plt.savefig("figure2.pdf")


make_dist_overlap_figure_multi_C(
    noise_multipliers=np.logspace(-3, 2, 1000),
    N_dimensions=10,
    num_data_samples=100,
    num_sigmas_eval=4,
    M=100,
)


# %%
def make_dist_overlap_figure_clipped(
    noise_multipliers, N_dimensions, num_data_samples, num_sigmas_eval, C, M
):
    torch.random.manual_seed(120496)
    np.random.seed(0)
    max_data_value = 1
    data_samples = (
        torch.rand((num_data_samples, N_dimensions), dtype=torch.float) * max_data_value
    )
    # img_idcs = np.random.randint(0, len(dataset), num_images)
    # imgs = [dataset[i][0].flatten()[:N_dimensions] for i in img_idcs]
    noise_multipliers_eval = np.logspace(
        np.log10(noise_multipliers[0]),
        np.log10(noise_multipliers[-1]),
        num_sigmas_eval + 2,
    )[1:-1]

    data_norms = [torch.linalg.norm(img, 2).item() for img in data_samples]
    if min(data_norms) * np.sqrt(M) < C:
        warn(
            f"Clipping threshold not exceeded! Empirical results are not as good as they could be!"
        )

    mse_etas, psnr_etas = find_value_ranges(
        noise_multipliers, max(data_norms), N_dimensions, min(data_norms), 0.95
    )
    mse_etas_log = np.log(mse_etas)
    ncc_bounds = ncc_bound(noise_multipliers, N_dimensions)

    # Visualization setup
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex="col", sharey="col")
    mse_eta_min, mse_eta_max = mse_etas_log
    mse_etaspace = np.logspace(mse_eta_min, mse_eta_max, 500)
    psnr_etaspace = np.linspace(psnr_etas[0], psnr_etas[1], 500)
    mseX, mseY = np.meshgrid(noise_multipliers, mse_etaspace)
    psnrX, psnrY = np.meshgrid(noise_multipliers, psnr_etaspace)

    # for i, C in enumerate(max_grad_norms):
    # Expand the range of etaspace for better coverage
    axs[0].set_ylim(*mse_etas)

    # Compute PDFs
    mse_pdfs = np.array(
        [
            mse_pdf(mse_etaspace, N_dimensions, sigma, max(data_norms))
            for sigma in noise_multipliers
        ]
    )
    mse_pdfs /= mse_pdfs.max(axis=1, keepdims=True)
    psnr_pdfs = np.array(
        [
            psnr_pdf(psnr_etaspace, N_dimensions, sigma, 1, max(data_norms))
            for sigma in noise_multipliers
        ]
    )
    axs[0].pcolormesh(
        np.log10(mseX),
        mseY,
        mse_pdfs.T,
        cmap="hot_r",
        shading="auto",
        alpha=0.7,
        rasterized=True,
    )
    axs[1].pcolormesh(
        np.log10(psnrX),
        psnrY,
        psnr_pdfs.T,
        cmap="hot_r",
        shading="auto",
        alpha=0.7,
        rasterized=True,
    )
    axs[2].plot(np.log10(noise_multipliers), ncc_bounds, color="black", linestyle="--")
    for j, sigma in enumerate(noise_multipliers_eval):
        mses, psnrs, nccs = [], [], []
        for img in data_samples:
            recon = performOptimalRecon(img, sigma, C, M)
            mses.append(mse(img.numpy(), recon.numpy()))
            psnrs.append(psnr(img.numpy(), recon.numpy(), data_range=1.0))
            nccs.append(ncc(img.numpy(), recon.numpy()))
        nccs = np.array(nccs)
        nccs = nccs[~np.isnan(nccs)]
        axs[0].boxplot(
            mses,
            positions=[np.log10(sigma)],
            widths=0.3,
            patch_artist=True,
            boxprops=dict(facecolor="limegreen", alpha=0.5),
        )
        axs[1].boxplot(
            psnrs,
            positions=[np.log10(sigma)],
            widths=0.3,
            patch_artist=True,
            boxprops=dict(facecolor="limegreen", alpha=0.5),
        )
        axs[2].boxplot(
            nccs,
            positions=[np.log10(sigma)],
            widths=0.3,
            patch_artist=True,
            boxprops=dict(facecolor="limegreen", alpha=0.5),
        )
    for i, a in enumerate(axs):
        if i == 0:
            a.set_ylabel("MSE")
            a.set_yscale("log")
        if i == 1:
            a.set_ylabel("PSNR")
        if i == 2:
            a.set_ylabel("NCC")
        a.set_xlabel("$\\sigma$")

    def update_ticks(x, pos):
        return f"$10^{{{int(x)}}}$"

    for ax in np.ravel(axs):
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))

    plt.tight_layout()
    plt.savefig("figure2_clipped.pdf")


make_dist_overlap_figure_clipped(
    noise_multipliers=np.logspace(-3, 2, 1000),
    N_dimensions=4,
    num_data_samples=100,
    num_sigmas_eval=4,
    C=1,
    M=100,
)


# %%
def convert_psnr_eta_to_mse_eta(psnr_eta: float, data_range):
    return (data_range**2) * (10 ** (psnr_eta / 10))


def convert_mse_eta_to_psnr_eta(mse_eta: float, data_range):
    return 10 * np.log10(mse_eta / data_range**2)


def rerofig(
    sigmas_varying: np.array,
    N_dimensions_varying: np.array,
    min_data_norms: np.array,
    sigma_fixed: float,
    N_dimensions_fixed: int,
    min_data_norm_fixed: float,
    threshold=0.999,
):
    max_num_plots = max(
        [
            sigmas_varying.shape[0],
            N_dimensions_varying.shape[0],
            min_data_norms.shape[0],
        ]
    )
    cmap = plt.get_cmap("viridis", max_num_plots + 2)
    colors = [cmap(i) for i in range(1, max_num_plots + 1)]
    scaling_factor = 3
    aspect_ratio = 16 / 9
    fig, axs = plt.subplots(
        3,
        2,
        figsize=(3 * scaling_factor * aspect_ratio, 3 * scaling_factor),
        sharey=True,
        sharex="col",
    )
    min_mse_val, max_mse_val = 1e9, 0
    min_psnr_val, max_psnr_val = 1e9, 0
    mse_distributions, psnr_distributions = [], []
    for sigma in sigmas_varying:
        # lambda_C = max(1, max_grad_norm / min_data_norm_fixed)
        mse_distributions.append(
            get_mse_dist(N_dimensions_fixed, sigma, min_data_norm_fixed)
        )
        psnr_distributions.append(
            get_psnr_dist(N_dimensions_fixed, sigma, 1, min_data_norm_fixed)
        )
    for N_dimensions in N_dimensions_varying:
        # lambda_C = max(1, max_grad_norm / min_data_norm_fixed)
        mse_distributions.append(
            get_mse_dist(N_dimensions, sigma_fixed, min_data_norm_fixed)
        )
        psnr_distributions.append(
            get_psnr_dist(N_dimensions, sigma_fixed, 1, min_data_norm_fixed)
        )
    for min_data_norm in min_data_norms:
        mse_distributions.append(
            get_mse_dist(N_dimensions_fixed, sigma_fixed, min_data_norm)
        )
        psnr_distributions.append(
            get_psnr_dist(N_dimensions_fixed, sigma_fixed, 1, min_data_norm)
        )
    for mse_dist, psnr_dist in zip(mse_distributions, psnr_distributions):
        if mse_dist.ppf(1.0 - threshold) < min_mse_val:
            min_mse_val = mse_dist.ppf(1.0 - threshold)
        if mse_dist.ppf(threshold) > max_mse_val:
            max_mse_val = mse_dist.ppf(threshold)
        if psnr_dist.ppf(1.0 - threshold) < min_psnr_val:
            min_psnr_val = psnr_dist.ppf(1.0 - threshold)
        if psnr_dist.ppf(threshold) > max_psnr_val:
            max_psnr_val = psnr_dist.ppf(threshold)
    mse_eta = np.logspace(np.log10(min_mse_val), np.log10(max_mse_val), 1000)
    psnr_eta = np.linspace(min_psnr_val, max_psnr_val, 1000)
    for i, sigma in enumerate(sigmas_varying):
        mse_dist = get_mse_dist(N_dimensions_fixed, sigma, min_data_norm_fixed)
        psnr_dist = get_psnr_dist(N_dimensions_fixed, sigma, 1, min_data_norm_fixed)
        mse_gamma = mse_dist.cdf(mse_eta)
        psnr_gamma = psnr_dist.sf(psnr_eta)
        axs[0, 0].axvline(x=mse_dist.moment(1), linestyle="--", alpha=0.8, c=colors[i])
        axs[0, 1].axvline(
            x=psnr_dist.moment(1),
            linestyle="--",
            alpha=0.8,
            c=colors[i],
        )
        axs[0, 0].plot(mse_eta, mse_gamma, c=colors[i])
        axs[0, 1].plot(psnr_eta, psnr_gamma, c=colors[i])
    axs[0, 0].set_title("MSE")
    axs[0, 1].set_title("PSNR")
    axs[0, 0].set_xlim(min_mse_val, max_mse_val)

    mse_eta = np.logspace(np.log10(min_mse_val), np.log10(max_mse_val), 1000)
    psnr_eta = np.linspace(min_psnr_val, max_psnr_val, 1000)
    for i, N_dimensions in enumerate(N_dimensions_varying):
        mse_dist = get_mse_dist(N_dimensions, sigma_fixed, min_data_norm_fixed)
        psnr_dist = get_psnr_dist(N_dimensions, sigma_fixed, 1, min_data_norm_fixed)
        mse_gamma = mse_dist.cdf(mse_eta)
        psnr_gamma = psnr_dist.sf(psnr_eta)
        axs[1, 0].axvline(
            x=mse_dist.moment(1),
            linestyle="--",
            alpha=0.8,
            c=colors[i],
        )
        axs[1, 1].axvline(
            x=psnr_dist.moment(1),
            linestyle="--",
            alpha=0.8,
            c=colors[i],
        )
        axs[1, 0].plot(mse_eta, mse_gamma, c=colors[i])
        axs[1, 1].plot(psnr_eta, psnr_gamma, c=colors[i])
    for i, min_data_norm in enumerate(min_data_norms):
        mse_dist = get_mse_dist(N_dimensions_fixed, sigma_fixed, min_data_norm)
        psnr_dist = get_psnr_dist(N_dimensions_fixed, sigma_fixed, 1, min_data_norm)
        mse_gamma = mse_dist.cdf(mse_eta)
        psnr_gamma = psnr_dist.sf(psnr_eta)
        axs[2, 0].axvline(
            x=mse_dist.moment(1),
            linestyle="--",
            alpha=0.8,
            c=colors[i],
        )
        axs[2, 1].axvline(
            x=psnr_dist.moment(1),
            linestyle="--",
            alpha=0.8,
            c=colors[i],
        )
        linestyle = "-"
        # match i:
        #     case 0:
        #         linestyle= (0, (10, 20))
        #     case 1:
        #         linestyle = (10, (10, 20))
        #     case 2:
        #         linestyle = (20, (10, 20))
        axs[2, 0].plot(mse_eta, mse_gamma, c=colors[i], linestyle=linestyle)
        axs[2, 1].plot(psnr_eta, psnr_gamma, c=colors[i])

    for ax in axs[:, 0]:
        ax.set_xscale("log")

    for ax in np.ravel(axs):
        ax.set_ylim([-0.1, 1.1])
        ax.set_ylabel("$\gamma(\eta)$")
    for ax in np.ravel(axs[-1]):
        ax.set_xlabel("$\eta$")
    legend_handles = [
        mpatches.Patch(color=colors[i], label=f"$\\sigma={sigmas_varying[i]:.1f}$")
        for i in range(sigmas_varying.shape[0])
    ]
    axs[0, -1].legend(
        handles=legend_handles, loc="upper left", bbox_to_anchor=(1, 1.06)
    )
    legend_handles = [
        mpatches.Patch(color=colors[i], label=f"$N={N_dimensions_varying[i]}$")
        for i in range(N_dimensions_varying.shape[0])
    ]
    axs[1, -1].legend(
        handles=legend_handles, loc="upper left", bbox_to_anchor=(1, 1.06)
    )
    legend_handles = [
        mpatches.Patch(
            color=colors[i],
            label=f"$\Vert X\Vert_2 = {min_data_norms[i]:.1f}$",
        )
        for i in range(min_data_norms.shape[0])
    ]
    axs[2, -1].legend(
        handles=legend_handles, loc="upper left", bbox_to_anchor=(1, 1.06)
    )
    fig.tight_layout()
    fig.savefig("figure3.pdf", bbox_inches="tight")

    plt.show()


rerofig(
    np.logspace(-1, 1, 3),
    np.array([1, 10, 1000]),
    np.logspace(-1, 1, 3),
    1,
    1,
    1,
    threshold=0.99,
)

# %%
use_imgnet = True
N_samples = 50

if use_imgnet:
    res = 224
    dataset = torchvision.datasets.ImageNet(
        root="./data/ILSVRC2012/",
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(res),
                torchvision.transforms.CenterCrop(res),
                torchvision.transforms.ToTensor(),
            ]
        ),
    )
    noise_mult = np.logspace(-4, -1, N_samples)
else:
    dataset = torchvision.datasets.CIFAR10(
        root="./data/",
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    noise_mult = np.logspace(-3, -0, N_samples)
# %%
perceptual_loss_fn = LPIPS(net="vgg").to(DEVICE)


def make_comparison_figure(
    dataset: torch.utils.data.Dataset,
    img_dimension: int,
    noise_multipliers: np.array,
    N_images=100,
    figsize_factor=4,
    only_bounded_metrics: bool = True,
    bound_colors: list[str] = ["#d95f02", "#7570b3"],
):
    N_metrics = 3 if only_bounded_metrics else 6
    fig, all_axs = plt.subplots(
        2,
        N_metrics // 2,
        figsize=(figsize_factor * N_metrics / 2, figsize_factor * 2),
        sharex=True,
        sharey=False,
    )
    all_axs = np.ravel(all_axs)

    make_column(
        dataset,
        noise_multipliers,
        N_images,
        all_axs,
        1,
        only_bounded_metrics,
    )
    for ax in all_axs[N_metrics // 2 :]:
        ax.set_xlabel("$\sigma$")
    all_axs[0].set_ylabel("MSE")
    all_axs[1].set_ylabel("PSNR")
    all_axs[2].set_ylabel("NCC")
    if not only_bounded_metrics:
        all_axs[3].set_ylabel("NMI")
        all_axs[4].set_ylabel("SSIM")
        all_axs[5].set_ylabel("Perceptual")
    all_axs[0].set_yscale("log")

    fig.tight_layout()
    fig.savefig(
        "figure4.pdf",
        bbox_inches="tight",
    )
    return fig


def make_column(
    dataset,
    noise_multipliers,
    N_images,
    axs,
    clip_norm,
    only_bounded_metrics,
    alpha=0.5,
):

    for i, (img, _) in tqdm(enumerate(dataset), total=N_images, leave=False):
        img_numpy = torch_to_plt(img)
        (
            noised_imgs,
            empirical_nccs,
            percept_loss,
            empirical_mses,
            empirical_nmis,
            empirical_psnr,
            empirical_ssim,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for nm in noise_multipliers:
            noisy_img = performOptimalRecon(img, nm, clip_norm, 10)
            noisy_img_numpy = torch_to_plt(noisy_img)
            noised_imgs.append(noisy_img_numpy)
            empirical_nccs.append(ncc(img_numpy, noisy_img_numpy))
            empirical_mses.append(mse(img_numpy, noisy_img_numpy).item())
            empirical_psnr.append(psnr(img_numpy, noisy_img_numpy, data_range=1))
            if not only_bounded_metrics:
                empirical_nmis.append(nmi(img_numpy, noisy_img_numpy).item())
                empirical_ssim.append(
                    ssim(
                        img_numpy,
                        noisy_img_numpy,
                        win_size=7,
                        channel_axis=2,
                        data_range=1.0,
                    ).item()
                )
                percept_loss.append(
                    perceptual_loss_fn(img.to(DEVICE), noisy_img.to(DEVICE)).item()
                )
        axs[0].plot(noise_multipliers, empirical_mses, alpha=alpha, linestyle=":")
        axs[1].plot(noise_multipliers, empirical_psnr, alpha=alpha, linestyle=":")
        axs[2].plot(noise_multipliers, empirical_nccs, alpha=alpha, linestyle=":")
        if not only_bounded_metrics:
            axs[3].plot(noise_multipliers, empirical_nmis, alpha=alpha, linestyle=":")
            axs[4].plot(noise_multipliers, empirical_ssim, alpha=alpha, linestyle=":")
            axs[5].plot(noise_multipliers, percept_loss, alpha=alpha, linestyle=":")
        if i == N_images:
            break
    for ax in axs:
        ax.set_xscale("log")


fig = make_comparison_figure(
    dataset,
    np.prod(img.shape),
    noise_mult,
    N_images=100,
    only_bounded_metrics=False,
    figsize_factor=5,
)

# %%
dataset = torchvision.datasets.ImageNet(
    root="./data/ILSVRC2012/",
    transform=torchvision.transforms.Compose(
        [
            # torchvision.transforms.Grayscale(1),
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
        ]
    ),
)
img = dataset[1][0]
sigma = 1e-3
factor = 100
img_normal = img.clone()
img_small_norm = img.clone() * factor
datanorm1 = torch.linalg.norm(img_normal.flatten(), 2).item()
datanorm2 = torch.linalg.norm(img_small_norm.flatten(), 2).item()
print(datanorm1)
print(datanorm2)
recon1 = performOptimalRecon(img_normal, sigma, datanorm1, M=1)
recon2 = performOptimalRecon(img_small_norm, sigma, datanorm2, M=1)
mse1 = mse(img_normal.numpy().flatten(), recon1.numpy().flatten())
mse2 = mse(img_small_norm.numpy().flatten(), recon2.numpy().flatten())
print(f"Norm ratio: {datanorm1/ datanorm2:2f}")
print(f"MSE ratio: {mse1/ mse2:2f}")

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ax in axs:
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2)
axs[0].imshow(torch_to_plt(img).squeeze())  # , cmap="gray")
axs[1].imshow(image_prior_numpy(torch_to_plt(recon1)).squeeze())  # , cmap="gray")
axs[2].imshow(image_prior_numpy(torch_to_plt(recon2)).squeeze())  # , cmap="gray")
axs[0].set_title("Original")
axs[1].set_title(
    f"$\Vert X\Vert_2={num2tex(datanorm1):.2e}$ \n $\mathrm{{MSE}}={num2tex(mse1):.2e}$"
)
axs[2].set_title(
    f"$\Vert X\Vert_2={num2tex(datanorm2):.2e}$ \n $\mathrm{{MSE}}={num2tex(mse2):.2e}$"
)


fig.savefig("figure5.pdf", bbox_inches="tight")
# %%
num_scenarios = 3
dataset_idcs = [
    1,
    # 544263,
    586966,
    539063,
    # 1235789,
    1193117,
    1138195,
    # 1102427,
]
C, M, sigmas = 1, 1, np.logspace(-3, -2, 4)
cmap = plt.get_cmap("viridis", 2 * len(dataset_idcs))
colors = [cmap(i * 2) for i in range(2 * len(dataset_idcs))]


figinches = 1.2
add_yspace = 0.2 * figinches
size_metric_plots = 2
assert size_metric_plots < sigmas.shape[0]
figsize = figsize = (
    (len(dataset_idcs) + 4 + 3 * size_metric_plots) * figinches,
    (1 + num_scenarios) * sigmas.shape[0] * (figinches + add_yspace),
)
# print(f"Figure size: {figsize[0]:.1f} x {figsize[1]:.1f} in")
fig, axs = plt.subplots(
    num_scenarios * (1 + sigmas.shape[0]),
    len(dataset_idcs) + 4 + 3 * size_metric_plots,
    figsize=figsize,
)
for ax in np.ravel(axs[:, : len(dataset_idcs)]):
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.set_frame_on(True)
for i, ax in enumerate(axs[:, : len(dataset_idcs)].T):
    for a in ax:
        for spine in a.spines.values():
            spine.set_edgecolor(colors[i])
            spine.set_linewidth(2)
gss_mse = [
    axs[i * (1 + sigmas.shape[0]), 1 + len(dataset_idcs)].get_gridspec()
    for i in range(num_scenarios)
]
gss_psnr = [
    axs[
        i * (1 + sigmas.shape[0]), len(dataset_idcs) + (size_metric_plots + 1)
    ].get_gridspec()
    for i in range(num_scenarios)
]
gss_ncc = [
    axs[
        i * (1 + sigmas.shape[0]), len(dataset_idcs) + 2 * (size_metric_plots + 1)
    ].get_gridspec()
    for i in range(num_scenarios)
]
for ax in np.ravel(axs[:, len(dataset_idcs) :]):
    ax.remove()
mseaxes = [
    fig.add_subplot(
        gss_mse[i][
            i * (1 + sigmas.shape[0]) : (i + 1) * (1 + sigmas.shape[0]) - 1,
            len(dataset_idcs) + 1 : len(dataset_idcs) + 1 + size_metric_plots,
        ]
    )
    for i in range(num_scenarios)
]
psnraxes = [
    fig.add_subplot(
        gss_ncc[i][
            i * (1 + sigmas.shape[0]) : (i + 1) * (1 + sigmas.shape[0]) - 1,
            len(dataset_idcs)
            + 2
            + size_metric_plots : len(dataset_idcs)
            + 2 * (size_metric_plots + 1),
        ]
    )
    for i in range(num_scenarios)
]
nccaxes = [
    fig.add_subplot(
        gss_psnr[i][
            i * (1 + sigmas.shape[0]) : (i + 1) * (1 + sigmas.shape[0]) - 1,
            len(dataset_idcs)
            + 2 * (size_metric_plots + 1)
            + 1 : len(dataset_idcs)
            + 3 * (size_metric_plots + 1),
        ]
    )
    for i in range(num_scenarios)
]
largeaxes = [(a, b, c) for a, b, c in zip(mseaxes, psnraxes, nccaxes)]


def fill_figure(axs, imgs, sigmas, data_range, metricaxes):
    axs[0, 0].set_ylabel(
        "Original",
        fontsize=7.5 * figinches,
    )
    for ax, img in zip(axs[0], imgs):
        ax.imshow(showimg(img), rasterized=True)
        datanorm = torch.linalg.norm(img.flatten(), 2).item()
        ax.set_title(
            f"$\Vert X\Vert_2 = {num2tex(datanorm):.1e}$",
            fontsize=7.5 * figinches,
            loc="left",
        )
    reconmses, reconpsnrs, reconnccs = (
        np.zeros((sigmas.shape[0], len(imgs)), dtype=np.float64),
        np.zeros((sigmas.shape[0], len(imgs)), dtype=np.float64),
        np.zeros((sigmas.shape[0], len(imgs)), dtype=np.float64),
    )
    for i, (sigma, ax) in enumerate(zip(sigmas, axs[1:])):
        ax[0].set_ylabel(f"$\sigma={num2tex(sigma):.1e}$", fontsize=7.5 * figinches)
        for j, (img, a) in enumerate(zip(imgs, ax)):
            recon = performOptimalRecon(img, sigma, C, M)
            imgnp, reconnp = img.numpy(), recon.numpy()
            reconmses[i, j] = mse(imgnp, reconnp)
            reconpsnrs[i, j] = psnr(imgnp, reconnp, data_range=data_range)
            reconnccs[i, j] = ncc(imgnp, reconnp)
            a.imshow(showimg(recon), rasterized=True)
            a.set_title(
                f"$\mathrm{{MSE}}   ={num2tex(reconmses[i,j]):.1e}$"
                + f"\n$\mathrm{{PSNR}}={num2tex(reconpsnrs[i,j]):.1f}\mathrm{{dB}}$"
                + f"\n$\mathrm{{NCC}} ={num2tex(100.0*reconnccs[i,j]):.0f}\%$",
                fontsize=6.5 * figinches,
                loc="left",
            )
    for j in range(len(imgs)):
        metricaxes[0].plot(sigmas, reconmses[:, j], color=colors[j])
        metricaxes[1].plot(sigmas, reconpsnrs[:, j], color=colors[j])
        metricaxes[2].plot(sigmas, reconnccs[:, j], color=colors[j])
    for m in metricaxes:
        m.set_xlabel("$\sigma$", fontsize=6.5 * figinches)
        m.set_xscale("log")
        m.tick_params(labelsize=6.5 * figinches)
    metricaxes[0].set_yscale("log")
    metricaxes[0].set_ylabel("MSE", fontsize=6.5 * figinches)
    metricaxes[1].set_ylabel("PSNR", fontsize=6.5 * figinches)
    metricaxes[2].set_ylabel("NCC", fontsize=6.5 * figinches)


## Scenario 1: Normal images
dataset = torchvision.datasets.ImageNet(
    root="./data/ILSVRC2012/",
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
        ]
    ),
)
fill_figure(axs, [dataset[i][0] for i in dataset_idcs], sigmas, 1.0, largeaxes[0])
## Scenario 2: Rescaled Images
dataset = torchvision.datasets.ImageNet(
    root="./data/ILSVRC2012/",
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                224,
                interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT,
            ),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x * 255),
        ]
    ),
)
fill_figure(
    axs[1 + sigmas.shape[0] :],
    [dataset[i][0] for i in dataset_idcs],
    sigmas,
    255.0,
    largeaxes[1],
)
## Scenario 3: Resized Images
dataset = torchvision.datasets.ImageNet(
    root="./data/ILSVRC2012/",
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(64),
            torchvision.transforms.CenterCrop(64),
            torchvision.transforms.ToTensor(),
        ]
    ),
)
fill_figure(
    axs[2 * (1 + sigmas.shape[0]) :],
    [dataset[i][0] for i in dataset_idcs],
    sigmas,
    1.0,
    largeaxes[2],
)
fig.savefig("figure6.pdf", bbox_inches="tight")
# %%
