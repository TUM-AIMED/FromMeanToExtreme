"""Regenerate figure3.pdf (main text, fig:influence_M) at a shorter height.

Exact unmodified reconstruction logic from figures.py (LinearNet,
performOptimalRecon, same sigma/M/C/indices) -- only the figsize changes,
plus a pickle cache so re-running to tune height doesn't redo the reconstructions.
Main-text figure, so treated with the same restraint as Fig 4: 5.5x2.0in -> smaller.
"""
import pickle
from pathlib import Path
import torch
import torchvision
import numpy as np
from warnings import warn

from opacus import GradSampleModule
from opacus.optimizers import DPOptimizer

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from fmte.utils import torch_to_plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LinearNet(torch.nn.Module):
    def __init__(self, res, bias, M=1, additional_layers=[]):
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
    for input_data, dn in zip(input_data_batch, datanorm):
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


CACHE = Path(__file__).parent / "figure3_recons.pkl"
if CACHE.exists():
    with open(CACHE, "rb") as f:
        imgs, badrecons, goodrecons = pickle.load(f)
    print(f"Loaded cached recons from {CACHE}")
else:
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ])
    possum = torchvision.datasets.ImageFolder(root="./data/possum", transform=transform)
    attempts = 0
    while attempts < 1000:
        try:
            celeba = torchvision.datasets.CelebA(root="./data/", download=True, transform=transform)
            break
        except Exception:
            attempts += 1

    sigma = 5e-4
    largeM = 1000
    celeba_idcs = [5, 6]
    possum_idcs = [0, 2]
    imgs = [celeba[i][0] for i in celeba_idcs]
    imgs += [possum[i][0] for i in possum_idcs]
    datanorms = [torch.linalg.norm(img.flatten(), 2) for img in imgs]
    C = 5000

    badrecons, goodrecons = [], []
    for img in imgs:
        badrecons.append(performOptimalRecon(img, sigma, C, M=1))
        goodrecons.append(performOptimalRecon(img, sigma, C, M=largeM))

    with open(CACHE, "wb") as f:
        pickle.dump((imgs, badrecons, goodrecons), f)

# --- plot at a shorter height (same layout/logic as figures.py) ---
fig, axs = plt.subplots(1, 3, figsize=(5.5, 2.0), layout="constrained")
for ax in axs:
    ax.set_xticks([]); ax.set_xticklabels([])
    ax.set_yticks([]); ax.set_yticklabels([])
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
axs[1].set_title(r"$M<\left(\frac{C}{\Vert X\Vert_2}\right)^2$", fontsize=8)
axs[2].set_title(r"$M\geq\left(\frac{C}{\Vert X\Vert_2}\right)^2$", fontsize=8)

fig.savefig("figure3.pdf")
print("Saved figure3.pdf")
