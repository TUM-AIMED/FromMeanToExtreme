# %%
import torch
import torchvision
import numpy as np
import seaborn as sn
from warnings import warn

from opacus import GradSampleModule
from opacus.optimizers import DPOptimizer
from opacus.validators import ModuleValidator


from tqdm import tqdm
from typing import Callable
from pathlib import Path


from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as mticker
from scipy import optimize
from tqdm import tqdm
from functools import partial
from fmte.prior_bounds import (
    rero_bound_without_subsampling,
    rero_bound_glrt_without_subsampling,
    inverse_rero_bound_without_subsampling,
)

from fmte.utils import torch_to_plt
from fmte.distributions import (
    mse_pdf,
    psnr_pdf,
    get_mse_dist,
    get_psnr_dist,
    mse_cdf,
    get_noise_multiplier,
    inverse_mse_cdf,
)

from skimage.metrics import (
    mean_squared_error as mse,
    peak_signal_noise_ratio as psnr,
)
from num2tex import configure as num2tex_configure

num2tex_configure(exp_format="cdot")

sn.set_theme(
    context="notebook",
    style="white",
    font="Arial",
    palette="viridis",
)
sn.despine()

colors = [
    "#ffd700",
    "#ffb14e",
    "#fa8775",
    "#ea5f94",
    "#cd34b5",
    "#9d02d7",
    "#0000ff",
]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LinearNet(torch.nn.Module):

    def __init__(
        self,
        res: tuple[int],
        bias: bool,
        M: int = 1,
        additional_layers: list[
            tuple[Callable[[torch.Tensor], torch.Tensor], torch.nn.Module]
        ] = [],
    ):
        super().__init__()
        self.linear = torch.nn.Linear(np.prod(res), M, bias=bias)
        self.additional_layers = len(additional_layers)
        self.preprocessing_layers = []
        for i, (preprocess, layer) in enumerate(additional_layers):
            self.preprocessing_layers.append(preprocess)
            self.register_module(f"add_{i}", layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        out.append(self.linear(x.reshape(1, -1)))
        for i, preprocess in zip(
            range(self.additional_layers), self.preprocessing_layers
        ):
            out.append(self.get_submodule(f"add_{i}")(preprocess(x)))
        return torch.stack(out)


def performOptimalRecon(
    input_data_batch,
    sigma,
    C,
    M=1,
    additional_layers=[],
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    assert isinstance(M, int)
    datanorm = torch.linalg.norm(input_data_batch.flatten(1), 2, axis=1)
    if torch.any(M < ((C / datanorm) ** 2)):
        warn("M < (C/|X|)²")
    net = LinearNet(
        input_data_batch.shape[1:], False, M, additional_layers=additional_layers
    ).to(device)
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
        desc="Processing batch",
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


# %%
sigmas = np.logspace(-2, 2, 1000)
# risks_sgm = rero_bound_without_subsampling(1.0 / 13.0, sigmas, 1.0)
# risks_glrt = rero_bound_glrt_without_subsampling(1.0 / 13.0, 1.0, 1.0, sigmas)
# risks_ours0 = mse_cdf(0, 1.0, sigmas, 1.0)
# risks_ours1 = mse_cdf(0.01, 1.0, sigmas, 1.0)
# risks_ours2 = mse_cdf(0.1, 1.0, sigmas, 1.0)
# risks_ours3 = mse_cdf(1.0, 1.0, sigmas, 1.0)

# %%
colors = [
    "#ffd700",
    "#ffb14e",
    "#fa8775",
    "#ea5f94",
    "#cd34b5",
    "#9d02d7",
    "#0000ff",
]
basic_alpha = 0.6

# %%
fig = plt.figure(figsize=(0.5 * 5.5, 2.0), layout="constrained")
ax = fig.add_subplot(111, projection="3d")


eta_max = 0.5
sigmas = np.logspace(-1, 1, 100)
etas = np.linspace(0, eta_max, 100)
risks_sgm = rero_bound_without_subsampling(1.0 / 13.0, sigmas, 1.0)
risks_glrt = rero_bound_glrt_without_subsampling(1.0 / 13.0, 1.0, 1.0, sigmas)

grid_sigma, grid_eta = np.meshgrid(sigmas, etas)

risks_ours = mse_cdf(grid_eta, 1.0, grid_sigma, 1.0)


eta_diffs_hayes = inverse_mse_cdf(risks_sgm, 1, sigmas, 1)
eta_diffs_kaissis = inverse_mse_cdf(risks_glrt, 1, sigmas, 1)
eta_diffs_hayes_mask = eta_diffs_hayes < eta_max
eta_diffs_kaissis_mask = eta_diffs_kaissis < eta_max
ax.plot(
    np.log10(sigmas),
    risks_sgm,
    zs=0,
    zdir="y",
    label="Hayes et al.",
    color=colors[1],
    # alpha=basic_alpha,
)
ax.plot(
    np.log10(sigmas)[eta_diffs_hayes_mask],
    risks_sgm[eta_diffs_hayes_mask],
    zs=eta_diffs_hayes[eta_diffs_hayes_mask],
    zdir="y",
    color=colors[1],
    linestyle="--",
    # alpha=basic_alpha,
)
ax.plot(
    np.log10(sigmas),
    risks_glrt,
    zs=0,
    zdir="y",
    label="Kaissis et al.",
    color=colors[3],
    # alpha=basic_alpha,
)
ax.plot(
    np.log10(sigmas)[eta_diffs_kaissis_mask],
    risks_glrt[eta_diffs_kaissis_mask],
    zs=eta_diffs_kaissis[eta_diffs_kaissis_mask],
    zdir="y",
    color=colors[3],
    linestyle="--",
    # alpha=basic_alpha,
)

ax.plot_surface(
    np.log10(grid_sigma),
    grid_eta,
    risks_ours,
    label="Ours",
    color=colors[6],
    alpha=basic_alpha,
    rstride=5,
    cstride=5,
    # cmap=cm.inferno,
    linewidth=0,
)
# ax.plot_surface(
#     np.log10(grid_sigma),
#     grid_eta,
#     np.ones_like(risks_ours) * 0.1,
#     color="lightgray",
#     alpha=0.15,
#     rstride=5,
#     cstride=5,
#     linewidth=0.8,
#     edgecolor="#4C72B0",  # thin colored border reinforces the panel-link
# )
frame_color = colors[5]
frame_lw = 1.5
sigma_min, sigma_max = np.log10(grid_sigma[0, 0]), np.log10(grid_sigma[-1, -1])
eta_min = grid_eta.min()
gamma_level = 0.1
corners_s = np.array([sigma_min, sigma_max, sigma_max, sigma_min, sigma_min])
corners_e = np.array([eta_min, eta_min, eta_max, eta_max, eta_min])
corners_g = np.ones_like(corners_s) * gamma_level

ax.plot(
    corners_s,
    corners_e,
    corners_g,
    color=frame_color,
    linewidth=frame_lw,
    zorder=5,
    alpha=0.5,
)

# ax.plot_wireframe(
#     np.log10(grid_sigma),
#     grid_eta,
#     risks_ours,
#     label="Ours",
#     color=colors[6],
#     alpha=basic_alpha,
#     rstride=50,
#     cstride=5,
# )

ax.xaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False)
ax.zaxis.set_rotate_label(False)

ax.set_xlabel("$\sigma$", fontsize=8)
ax.set_ylabel("$\eta$", fontsize=8)
ax.set_zlabel("$\gamma$", rotation=0, fontsize=8)
ax.set_ylim(etas[0], etas[-1])
ax.set_zlim(0, 1)
ax.zaxis.labelpad = -4


def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"


ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=3))
elev, azim, roll = 15, 45, 0
ax.view_init(elev, azim, roll)
ax.set_proj_type("ortho")
ax.tick_params(axis="x", labelsize=8)
ax.tick_params(axis="y", labelsize=8)
ax.tick_params(axis="z", labelsize=8)


fig.legend(fontsize=8)
fig.savefig(f"figure2a.pdf")

# %%
def animate(angle, angle_multiplier=1, start_angle=45):
    # Normalize the angle to the range [-180, 180] for display
    angle = angle * angle_multiplier + start_angle
    angle = angle % 360

    # Update the axis view and title
    ax.view_init(15, angle, 0)
    return (fig,)


anim = FuncAnimation(
    fig, partial(animate, angle_multiplier=1, start_angle=45), frames=360, repeat=True
)


class ProgressBar(tqdm):
    def update_to(self, current, total):
        self.total = total - 1
        self.update(current - self.n)


with ProgressBar(desc="Saving", unit="frames", unit_scale=True) as t:
    anim.save("figure2a_animated.mp4", fps=40, progress_callback=t.update_to)

# %%
gamma = 0.1

sigma_hayes = inverse_rero_bound_without_subsampling(gamma, 1.0 / 13, 1.0)
sigma_kaissis = optimize.bisect(
    lambda x: rero_bound_glrt_without_subsampling(1.0 / 13, 1, 1, x) - gamma,
    0.1,
    20.0,
)
risk_corridor_hayes = optimize.bisect(
    lambda x: get_noise_multiplier(gamma, x, 1.0, 1.0) - sigma_hayes, 0.05, 0.8
)
risk_corridor_kaissis = optimize.bisect(
    lambda x: get_noise_multiplier(gamma, x, 1.0, 1.0) - sigma_kaissis, 0.05, 0.8
)


etas = np.logspace(-5, 0, 100)
sigmas = get_noise_multiplier(gamma, etas, 1.0, 1.0)
# %%
fig = plt.figure(figsize=(0.5 * 5.5, 2), layout="constrained")
plt.plot(etas, sigmas, label="Ours", color=colors[6], alpha=basic_alpha)
plt.scatter(0, sigma_hayes, label="Hayes et al.", marker="x", color=colors[1])
plt.scatter(0, sigma_kaissis, label="Kaissis et al.", marker="x", color=colors[3])
plt.plot(
    [0, risk_corridor_hayes],
    [sigma_hayes, sigma_hayes],
    color="lightslategray",
    linestyle="--",
)
plt.plot(
    [0, risk_corridor_kaissis],
    [sigma_kaissis, sigma_kaissis],
    color="lightslategray",
    linestyle="--",
    label="Risk corridor",
)
# plt.xscale("log")
plt.legend(fontsize=8)
plt.xlabel("$\eta$", fontsize=8)
plt.ylabel("$\sigma$", rotation=0, fontsize=8, labelpad=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
ax2d = plt.gca()
for spine in ax2d.spines.values():
    spine.set_edgecolor(colors[5])
    spine.set_linewidth(1.5)
fig.savefig("figure2b.pdf", pad_inches=0, transparent=True)

# %%
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ]
)
# dataset = torchvision.datasets.ImageNet(
#     root="/media/alex/NVME/ILSVRC2012/",
#     transform=transform
# )

possum = torchvision.datasets.ImageFolder(root="./data/possum", transform=transform)
attempts = 0
max_attempts = 1000
while attempts < max_attempts:
    try:
        celeba = torchvision.datasets.CelebA(
            root="./data/", download=True, transform=transform
        )
        break
    except:
        attempts += 1

# %%
sigma = 5e-4
largeM = 1000

# imgs = [dataset[i][0] for i in [1, 544263, 586966, 539063]]
celeba_idcs = [5, 6]
possum_idcs = [0, 2]
imgs = [celeba[i][0] for i in celeba_idcs]
imgs += [possum[i][0] for i in possum_idcs]
datanorms = [torch.linalg.norm(img.flatten(), 2) for img in imgs]
Cs = [np.sqrt(largeM) * datanorm for datanorm in datanorms]
C = 5000
print(C)
print([(C / dn) ** 2 for dn in datanorms])

badrecons, goodrecons = [], []
for img in imgs:
    badrecons.append(performOptimalRecon(img, sigma, C, M=1))
    goodrecons.append(performOptimalRecon(img, sigma, C, M=largeM))
fig, axs = plt.subplots(1, 3, figsize=(5.5, 2), layout="constrained")
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
axs[0].imshow(convert_img_list_to_grid(imgs))
axs[1].imshow(convert_img_list_to_grid(badrecons))
axs[2].imshow(convert_img_list_to_grid(goodrecons))
axs[0].set_title("Original", fontsize=8)
axs[1].set_title("$M<\\left(\\frac{C}{\Vert X\Vert_2}\\right)^2$", fontsize=8)
axs[2].set_title("$M\\geq\\left(\\frac{C}{\Vert X\Vert_2}\\right)^2$", fontsize=8)

fig.savefig("figure3.pdf")

# %%
def find_value_ranges(sigmas, min_norm, N_dimensions, coverage=0.95):
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
    min_mse_dist = get_mse_dist(N_dimensions, min_sigma, min_norm)
    max_mse_dist = get_mse_dist(N_dimensions, max_sigma, min_norm)
    min_psnr_dist = get_psnr_dist(N_dimensions, min_sigma, 1.0, min_norm)
    max_psnr_dist = get_psnr_dist(N_dimensions, max_sigma, 1.0, min_norm)
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
def calc_recon_scores(
    data_samples, noise_multipliers, num_sigmas_eval, C, M, additional_layers
):
    torch.random.manual_seed(120496)
    np.random.seed(0)
    N_dimensions = data_samples.shape[1]
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
        noise_multipliers, min(data_norms), N_dimensions, 0.95
    )
    mse_etas_log = np.log(mse_etas)
    # ncc_bounds = ncc_bound(noise_multipliers, N_dimensions)

    # Visualization setup
    mse_eta_min, mse_eta_max = mse_etas_log
    mse_etaspace = np.logspace(mse_eta_min, mse_eta_max, 500)
    psnr_etaspace = np.linspace(psnr_etas[0], psnr_etas[1], 500)
    mseX, mseY = np.meshgrid(noise_multipliers, mse_etaspace)
    psnrX, psnrY = np.meshgrid(noise_multipliers, psnr_etaspace)
    # Compute PDFs
    mse_pdfs = np.array(
        [
            mse_pdf(mse_etaspace, N_dimensions, sigma, min(data_norms))
            for sigma in noise_multipliers
        ]
    )
    mse_pdfs /= mse_pdfs.max(axis=1, keepdims=True)
    psnr_pdfs = np.array(
        [
            psnr_pdf(psnr_etaspace, N_dimensions, sigma, 1, min(data_norms))
            for sigma in noise_multipliers
        ]
    )
    mses_per_sigma = []
    psnrs_per_sigma = []
    for j, sigma in tqdm(enumerate(noise_multipliers_eval), leave=False):
        mses, psnrs, nccs = [], [], []
        recons = performOptimalRecon(
            data_samples, sigma, C, M, additional_layers=additional_layers
        )
        for img, recon in tqdm(
            zip(data_samples, recons), leave=False, desc="metric calc"
        ):
            mses.append(mse(img.numpy(), recon.numpy()))
            psnrs.append(psnr(img.numpy(), recon.numpy(), data_range=1.0))
            # nccs.append(ncc(img.numpy(), recon.numpy()))
        mses_per_sigma.append(mses)
        psnrs_per_sigma.append(psnrs)
        # nccs = np.array(nccs)
        # nccs = nccs[~np.isnan(nccs)]
    return (
        mse_etas,
        mseX,
        mseY,
        psnrX,
        psnrY,
        mse_pdfs,
        psnr_pdfs,
        mses_per_sigma,
        psnrs_per_sigma,
        noise_multipliers_eval,
    )


def make_dist_overlap_figure_clipped(
    axs,
    mse_etas,
    mseX,
    mseY,
    psnrX,
    psnrY,
    mse_pdfs,
    psnr_pdfs,
    mses_per_sigma,
    psnrs_per_sigma,
    noise_multipliers_eval,
):

    # for i, C in enumerate(max_grad_norms):
    # Expand the range of etaspace for better coverage
    axs[0].set_ylim(*mse_etas)

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
    # axs[2].plot(np.log10(noise_multipliers), ncc_bounds, color="black", linestyle="--")
    for j, sigma in tqdm(enumerate(noise_multipliers_eval), leave=False):
        axs[0].boxplot(
            mses_per_sigma[j],
            positions=[np.log10(sigma)],
            widths=0.3,
            patch_artist=True,
            boxprops=dict(facecolor="limegreen", alpha=0.5),
            flierprops=dict(marker="x", markersize=0.2),
            meanline=True,
        )
        axs[1].boxplot(
            psnrs_per_sigma[j],
            positions=[np.log10(sigma)],
            widths=0.3,
            patch_artist=True,
            boxprops=dict(facecolor="limegreen", alpha=0.5),
            flierprops=dict(marker="x", markersize=0.2),
            meanline=True,
        )
        # axs[2].boxplot(
        #     nccs,
        #     positions=[np.log10(sigma)],
        #     widths=0.3,
        #     patch_artist=True,
        #     boxprops=dict(facecolor="limegreen", alpha=0.5),
        # )
    for i, a in enumerate(axs):
        if i == 0:
            # a.set_ylabel("MSE", fontsize=8)
            a.set_yscale("log")
        if i == 1:
            # a.set_ylabel("PSNR", fontsize=8)
            pass
        # if i == 2:
        #     a.set_ylabel("NCC")

    def update_ticks(x, pos):
        return f"$10^{{{int(x)}}}$"

    for ax in np.ravel(axs):
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.tick_params(axis="both", which="minor", labelsize=6)
        ax.tick_params(axis="x", labelrotation=45)


# %%
dataset = torchvision.datasets.CIFAR10(
    root="./data/",
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize(2),
            torchvision.transforms.ToTensor(),
        ]
    ),
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
    datasamples,
    noise_multipliers=np.logspace(-2, 2, 1000),
    num_sigmas_eval=3,
    C=1,
    M=1,
    additional_layers=[],
)
linear_net = torch.nn.Sequential(
    torch.nn.Linear(
        datasamples.shape[-1],
        int(1000000 / datasamples.shape[-1]),
        bias=False,
    ),
    torch.nn.AvgPool1d(int(1000000 / datasamples.shape[-1])),
)
print(
    f"Params Linear: {sum(p.numel() for p in linear_net.parameters() if p.requires_grad)}"
)

recon_scores["linear"] = calc_recon_scores(
    datasamples,
    noise_multipliers=np.logspace(-2, 2, 1000),
    num_sigmas_eval=3,
    C=1,
    M=1,
    additional_layers=[(torch.nn.Identity(), linear_net)],
)

transform_tensor = lambda x: x.reshape(1, 1, 2, 2).repeat(1, 3, 112, 112)
resnet = torchvision.models.resnet101(weights=None)
resnet = ModuleValidator.fix(resnet)
print(f"Param ResNet: {sum(p.numel() for p in resnet.parameters() if p.requires_grad)}")
recon_scores["resnet"] = calc_recon_scores(
    datasamples,
    noise_multipliers=np.logspace(-2, 2, 1000),
    num_sigmas_eval=3,
    C=1,
    M=1,
    additional_layers=[
        (
            transform_tensor,
            torch.nn.Sequential(resnet, torch.nn.AvgPool1d(1000)),
        )
    ],
)
# %%
fig, axs = plt.subplots(
    2, 3, figsize=(5.5, 3.09), sharex="row", sharey="row", layout="constrained"
)


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


fig.savefig("figure4.pdf")
# %%
dataset = torchvision.datasets.CIFAR10(
    root="./data/",
    download=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    ),
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
    datasamples,
    noise_multipliers=np.logspace(-2, 2, 1000),
    num_sigmas_eval=3,
    C=1,
    M=1,
    additional_layers=[],
)

linear_net = torch.nn.Sequential(
    torch.nn.Linear(
        datasamples.shape[-1],
        int(1000000 / datasamples.shape[-1]),
        bias=False,
    ),
    torch.nn.AvgPool1d(int(1000000 / datasamples.shape[-1])),
)
print(
    f"Params Linear: {sum(p.numel() for p in linear_net.parameters() if p.requires_grad)}"
)

recon_scores["linear"] = calc_recon_scores(
    datasamples,
    noise_multipliers=np.logspace(-2, 2, 1000),
    num_sigmas_eval=3,
    C=1,
    M=1,
    additional_layers=[(torch.nn.Identity(), linear_net)],
)
transform_tensor = lambda x: x.reshape(1, 3, 32, 32)
resnet = torchvision.models.resnet101(weights=None)
resnet = ModuleValidator.fix(resnet)
print(f"Param ResNet: {sum(p.numel() for p in resnet.parameters() if p.requires_grad)}")
recon_scores["resnet"] = calc_recon_scores(
    datasamples,
    noise_multipliers=np.logspace(-2, 2, 1000),
    num_sigmas_eval=3,
    C=1,
    M=1,
    additional_layers=[
        (
            transform_tensor,
            torch.nn.Sequential(resnet, torch.nn.AvgPool1d(1000)),
        )
    ],
)

# %%
fig, axs = plt.subplots(
    2, 3, figsize=(5.5, 3.09), sharex="row", sharey="row", layout="constrained"
)


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


fig.tight_layout()
fig.savefig("figure5.pdf")


# %%
