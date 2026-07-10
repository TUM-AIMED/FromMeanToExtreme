"""Visual reconstruction examples for the appendix (Fig: reconexamples).

One CIFAR-10 example, 4 rows (Original, Optimal/MVUE, Geiping, DLG) x 5 noise
levels spanning the reconstruction's transition from faithful to destroyed.
Publication style: Linux Libertine, 5.5in paper width. Uses FCNet (linear first
layer) — the paper's theory architecture.
"""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import load_dataset, perform_optimal_recon, DEVICE
from exp10_prior_attacks import FCNet, compute_gradient, add_dp_noise, dlg_attack, geiping_attack, analytic_attack
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotstyle import apply_style, TEXTWIDTH

apply_style()

outdir = Path(__file__).parent.parent / "results" / "summary_figures"
outdir.mkdir(parents=True, exist_ok=True)

# Noise levels spanning the transition from faithful to destroyed reconstruction.
SIGMAS_VIS = [0.0, 0.005, 0.01, 0.03, 0.1]
N_SIGMA = len(SIGMAS_VIS)

# Fixed seeds for reproducibility: the DP noise and attack initialisations are
# stochastic (torch), so we pin the numpy, torch, and CUDA RNGs before any draw.
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

data = load_dataset("cifar10", resolution=32, num_samples=1,
                    enforced_norm=None, flatten=False, grayscale=False)
data_scale = 1.0
N_input = int(np.prod((3, 32, 32)))

x_img = data[0].unsqueeze(0)
x_flat = x_img.reshape(1, -1)
nat_img = data[0].permute(1, 2, 0).numpy()
C = float(torch.norm(data[0]).item())

N_ROWS = 5
fig, axes = plt.subplots(N_ROWS, N_SIGMA, figsize=(TEXTWIDTH, TEXTWIDTH * 1.06),
                         layout="constrained")
row_labels = ["Original", "Optimal\n(MVUE)", "Analytic\n(division)", "Geiping", "DLG"]

for si, sigma in enumerate(SIGMAS_VIS):
    # Row 0: Original (reference, repeated per column)
    axes[0, si].imshow(np.clip(nat_img, 0, 1))

    # Row 1: Optimal (MVUE) — the bound, via calibrated optimal reconstruction
    if sigma > 0:
        recon_opt = perform_optimal_recon(x_flat, sigma, C=C, M=1, device=DEVICE)
        recon_opt = np.array(recon_opt).reshape(3, 32, 32).transpose(1, 2, 0)
        axes[1, si].imshow(np.clip(recon_opt, 0, 1))
    else:
        axes[1, si].imshow(np.clip(nat_img, 0, 1))

    # Rows 2-4: Analytic (division), Geiping, DLG — fresh model per sigma,
    # all attacks read the same noised gradient.
    model = FCNet(input_dim=N_input).to(DEVICE)
    model.eval()
    y = torch.tensor([0])
    true_grads = compute_gradient(model, x_img, y)
    noisy_grads = add_dp_noise(true_grads, sigma, C)

    recon_a = analytic_attack(model, noisy_grads, (3, 32, 32), data_scale=data_scale)
    axes[2, si].imshow(np.clip(recon_a.squeeze(0).permute(1, 2, 0).numpy(), 0, 1))

    recon_g = geiping_attack(model, noisy_grads, (3, 32, 32), data_scale=data_scale)
    axes[3, si].imshow(np.clip(recon_g.squeeze(0).permute(1, 2, 0).numpy(), 0, 1))

    recon_d = dlg_attack(model, noisy_grads, (3, 32, 32), data_scale=data_scale)
    axes[4, si].imshow(np.clip(recon_d.squeeze(0).permute(1, 2, 0).numpy(), 0, 1))

    title = "σ=0" if sigma == 0 else f"σ={sigma:g}"
    axes[0, si].set_title(title, fontsize=8)
    print(f"  σ={sigma}: done")

for r in range(N_ROWS):
    # labelpad keeps the two-line labels fully in the left margin (off the images)
    axes[r, 0].set_ylabel(row_labels[r], fontsize=8, rotation=90, va="center",
                          ha="center", labelpad=14)
    for c in range(N_SIGMA):
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])

fig.suptitle(r"Reconstruction vs. DP noise (FCNet)", fontsize=9)

fig.savefig(outdir / "reconstruction_examples.pdf")
fig.savefig(outdir / "reconstruction_examples.png", dpi=200)
plt.close(fig)
print("Saved reconstruction_examples")
