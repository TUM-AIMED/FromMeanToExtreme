"""Experiment 10: Comparison with prior gradient inversion attacks under DP.

Implements three attacks on FCNet (linear first layer = paper's theory architecture):
  1. Analytic (division): x = grad_W1[i] / grad_b1[i] — the division readout used by
     Fowl et al., Boenisch et al., and Feng & Tramèr. At B=1 this recovers x exactly
     from the rank-1 linear-layer gradient; the imprint module / trap weights those
     papers add are only needed to isolate samples at B>1, so are omitted here.
  2. DLG (Zhu et al., 2019): Adam gradient matching (300 iterations).
  3. Geiping et al. (2020): Cosine similarity + TV regularization + iDLG label + restarts.

The division attack is the faithful B=1 representation of the prior analytic attacks
(Fowl/Boenisch/Feng & Tramèr), which all exploit the same rank-1 structure. DLG and
Geiping are optimization-based and do not exploit the linear-layer structure.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from common import load_dataset, DEVICE, get_mse_dist

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "exp10"

NOISE_MULTIPLIERS = np.array([0.0, 0.005, 0.02, 0.1, 1.0])
C = 1.0
NUM_SAMPLES = 100
DLG_ITERS = 300
GEIPING_ITERS = 800
GEIPING_RESTARTS = 3


class FCNet(nn.Module):
    """Fully-connected network with linear first layer.

    The first layer is nn.Linear(N_input, hidden_dim) with NO activation,
    so grad_W1 = (dL/dh) ⊗ X — a rank-1 outer product encoding X directly.
    This is the architecture the paper's optimal attack theory is derived for.
    """
    def __init__(self, input_dim=3072, hidden_dim=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SimpleConvNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def compute_gradient(model, x, y, device=DEVICE):
    model.zero_grad()
    pred = model(x.to(device))
    loss = F.cross_entropy(pred, y.to(device))
    grad = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    return [g.detach().clone() for g in grad]


def add_dp_noise(gradients, sigma, C):
    """Whole-gradient clipping + Gaussian noise (standard DP-SGD)."""
    flat = torch.cat([g.flatten() for g in gradients])
    grad_norm = torch.norm(flat)
    clip_factor = min(1.0, C / (grad_norm.item() + 1e-10))
    flat_clipped = flat * clip_factor
    noise = torch.randn_like(flat_clipped) * sigma * C
    flat_noisy = flat_clipped + noise

    noisy_grads = []
    offset = 0
    for g in gradients:
        numel = g.numel()
        noisy_grads.append(flat_noisy[offset:offset + numel].reshape(g.shape))
        offset += numel
    return noisy_grads


def analytic_attack(model, noisy_grads, input_shape, data_scale=1.0):
    """Analytic division attack (Fowl et al., Boenisch et al., Feng & Tramèr).

    For a linear first layer, grad_W1[i] = (dL/dh1)_i · x and grad_b1[i] =
    (dL/dh1)_i, so x = grad_W1[i] / grad_b1[i] for any neuron i. This is the
    "division" readout these prior analytic attacks use. At B=1 the imprint
    module / trap weights they rely on (to isolate one sample from an aggregated
    B>1 gradient) are unnecessary; we simply read off the neuron with the
    largest |grad_b1| — the highest signal-to-noise channel, i.e. the B=1
    stand-in for the neuron the imprint module would designate.
    """
    grad_W1 = noisy_grads[0].to(DEVICE)   # (hidden_dim, input_dim)
    grad_b1 = noisy_grads[1].to(DEVICE)   # (hidden_dim,)

    i = int(torch.argmax(torch.abs(grad_b1)))
    x_hat = grad_W1[i] / (grad_b1[i] + 1e-12)

    return x_hat.reshape(1, *input_shape).detach().cpu()


def dlg_attack(model, noisy_grads, input_shape, num_iters=DLG_ITERS, data_scale=1.0):
    """DLG: Adam gradient matching (adapted from Zhu et al., 2019)."""
    dummy_x = (torch.rand(1, *input_shape, device=DEVICE) * data_scale).requires_grad_(True)
    dummy_y = torch.randn(1, 10, device=DEVICE, requires_grad=True)

    optimizer = torch.optim.Adam([dummy_x, dummy_y], lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_iters // 3, gamma=0.1)

    best_loss = float('inf')
    best_x = dummy_x.data.clone()

    for _ in range(num_iters):
        optimizer.zero_grad()
        pred = model(dummy_x)
        loss = F.cross_entropy(pred, F.softmax(dummy_y, dim=1))
        dummy_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_diff = sum((dg - ng.to(DEVICE)).pow(2).sum() for dg, ng in zip(dummy_grads, noisy_grads))

        if torch.isnan(grad_diff):
            break
        grad_diff.backward()
        optimizer.step()
        scheduler.step()

        if grad_diff.item() < best_loss:
            best_loss = grad_diff.item()
            best_x = dummy_x.data.clone()

    return best_x.detach().cpu()


def geiping_attack(model, noisy_grads, input_shape, num_iters=GEIPING_ITERS,
                   num_restarts=GEIPING_RESTARTS, tv_weight=1e-4, data_scale=1.0):
    """Geiping et al. (2020): cosine similarity + TV + iDLG label + restarts."""
    last_bias_grad = noisy_grads[-1]
    recovered_label = torch.argmin(last_bias_grad).unsqueeze(0).to(DEVICE)

    target_flat = torch.cat([g.flatten() for g in noisy_grads]).to(DEVICE)

    best_x = None
    best_loss = float('inf')

    for restart in range(num_restarts):
        dummy_x = (torch.rand(1, *input_shape, device=DEVICE) * data_scale).requires_grad_(True)
        optimizer = torch.optim.Adam([dummy_x], lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters)

        for it in range(num_iters):
            optimizer.zero_grad()
            pred = model(dummy_x)
            loss = F.cross_entropy(pred, recovered_label)
            dummy_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            dummy_flat = torch.cat([dg.flatten() for dg in dummy_grads])
            cos_loss = 1 - F.cosine_similarity(dummy_flat.unsqueeze(0), target_flat.unsqueeze(0))

            tv = torch.tensor(0.0, device=DEVICE)
            if dummy_x.shape[-1] > 1:
                tv = (torch.sum(torch.abs(dummy_x[:, :, :, :-1] - dummy_x[:, :, :, 1:])) +
                      torch.sum(torch.abs(dummy_x[:, :, :-1, :] - dummy_x[:, :, 1:, :])))

            total_loss = cos_loss + tv_weight * tv
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            dummy_x.data.clamp_(0, data_scale)

        final_loss = cos_loss.item()
        if final_loss < best_loss:
            best_loss = final_loss
            best_x = dummy_x.data.clone()

    return best_x.detach().cpu()


def main():
    from common import set_seed
    set_seed(42)  # reproducibility: pin python/numpy/torch/CUDA RNGs
    print("=" * 60)
    print("Experiment 10: Prior Attack Comparison under DP (FCNet)")
    print(f"Device: {DEVICE}, Samples: {NUM_SAMPLES}")
    print(f"DLG iters: {DLG_ITERS}, Geiping iters: {GEIPING_ITERS}, Restarts: {GEIPING_RESTARTS}")
    print(f"Noise multipliers: {NOISE_MULTIPLIERS}")
    print("Architecture: FCNet (linear first layer — matches paper's theory)")
    print("=" * 60)

    data = load_dataset("cifar10", resolution=32, num_samples=NUM_SAMPLES,
                        enforced_norm=None, flatten=False, grayscale=False)
    input_shape = (3, 32, 32)
    N = int(np.prod(input_shape))

    norms = torch.tensor([torch.norm(data[i]).item() for i in range(len(data))])
    C_adaptive = float(norms.mean())
    print(f"Data shape: {data.shape}, natural norms: min={norms.min():.1f} mean={norms.mean():.1f} max={norms.max():.1f}")
    print(f"Using C={C_adaptive:.1f} (mean norm — mimics realistic DP-SGD clipping)")

    data_scale = float(data.max())
    print(f"Data value range: [{data.min():.4f}, {data.max():.4f}]")

    attacks = {"Analytic": analytic_attack, "DLG": dlg_attack, "Geiping": geiping_attack}
    all_results = {}

    for sigma in NOISE_MULTIPLIERS:
        print(f"\n=== sigma = {sigma} ===")
        all_results[float(sigma)] = {}

        for attack_name, attack_fn in attacks.items():
            print(f"\n  --- {attack_name} ---")
            mses = []
            psnrs = []
            corrs = []

            for i in tqdm(range(NUM_SAMPLES), desc=f"{attack_name} sigma={sigma:.2g}", leave=False):
                model = FCNet(input_dim=N).to(DEVICE)
                model.eval()

                x = data[i].unsqueeze(0)
                y = torch.tensor([i % 10])

                true_grads = compute_gradient(model, x, y)
                noisy_grads = add_dp_noise(true_grads, sigma, C_adaptive)

                recon = attack_fn(model, noisy_grads, input_shape, data_scale=data_scale)
                recon = recon.squeeze(0)
                orig = x.squeeze(0).cpu()

                mse = F.mse_loss(recon, orig).item()
                mses.append(mse)
                psnr = -10 * np.log10(max(mse, 1e-10))
                psnrs.append(psnr)
                corr = float(np.corrcoef(recon.numpy().flatten(), orig.numpy().flatten())[0, 1])
                corrs.append(corr if not np.isnan(corr) else 0.0)

            all_results[float(sigma)][attack_name] = {
                "mse_mean": float(np.mean(mses)),
                "mse_std": float(np.std(mses)),
                "mse_median": float(np.median(mses)),
                "psnr_mean": float(np.nanmean(psnrs)),
                "corr_mean": float(np.mean(corrs)),
            }
            r = all_results[float(sigma)][attack_name]
            print(f"    MSE={r['mse_mean']:.4f} +/- {r['mse_std']:.4f}, corr={r['corr_mean']:.4f}, PSNR={r['psnr_mean']:.1f}")

        mean_norm = float(norms.mean())
        if sigma > 0:
            theoretical_mse = sigma**2 * mean_norm**2
        else:
            theoretical_mse = 0.0
        all_results[float(sigma)]["theoretical_optimal"] = {
            "mse_mean": theoretical_mse,
            "psnr_mean": -10 * np.log10(max(theoretical_mse, 1e-10)) if theoretical_mse > 0 else float("inf"),
        }
        all_results[float(sigma)]["metadata"] = {
            "C": C_adaptive, "mean_norm": float(norms.mean()),
            "trivial_mse": float((norms.mean()**2) / N),
            "architecture": "FCNet(3072→128→128→10)",
        }
        print(f"  Theory (optimal, mean||X||={mean_norm:.1f}, C={C_adaptive:.1f}): MSE={theoretical_mse:.2f}")

    # Sigma=0 sanity check
    for atk in ["Analytic", "DLG", "Geiping"]:
        mse0 = all_results[0.0][atk]["mse_mean"]
        corr0 = all_results[0.0][atk]["corr_mean"]
        if corr0 < 0.9:
            print(f"\nWARNING: {atk} at sigma=0 has corr={corr0:.4f} — attack may not be converging properly")
        else:
            print(f"\nSANITY CHECK PASSED: {atk} at sigma=0 has corr={corr0:.4f}, MSE={mse0:.6f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "attack_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")

    # Categorical (ordinal) x-axis: identical across all three panels.
    # The sigma values span the reconstruction transition, so equal spacing
    # makes each regime equally visible and naturally includes σ=0.
    sigmas_all = list(NOISE_MULTIPLIERS)
    x_pos = np.arange(len(sigmas_all))
    sigma_labels = [("0" if s == 0 else f"{s:g}") for s in sigmas_all]
    colors = {'Analytic': '#9467bd', 'DLG': '#1f77b4', 'Geiping': '#ff7f0e', 'theoretical_optimal': '#2ca02c'}
    markers = {'Analytic': 'D', 'DLG': 'o', 'Geiping': 's', 'theoretical_optimal': '^'}

    def style_xaxis(ax):
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sigma_labels)
        ax.set_xlabel("σ (noise multiplier)")

    # Panel 1: MSE (symlog y so σ=0 / exact recovery renders cleanly)
    ax = axes[0]
    for name in ["Analytic", "DLG", "Geiping", "theoretical_optimal"]:
        vals = [all_results[float(s)].get(name, {}).get("mse_mean", float("nan")) for s in sigmas_all]
        style = "--" if name == "theoretical_optimal" else "-"
        label = {"theoretical_optimal": "Theory (MVUE)", "Analytic": "Analytic (SVD)"}.get(name, name)
        ax.plot(x_pos, vals, f"{markers[name]}{style}", label=label,
                color=colors[name], markersize=8, linewidth=2)
    trivial_mse = all_results[0.0]["metadata"]["trivial_mse"]
    ax.axhline(y=trivial_mse, color='gray', linestyle='--', alpha=0.6, label='Trivial (all-zeros)')
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_ylim(0, 2e3)
    ax.set_ylabel("MSE")
    style_xaxis(ax)
    ax.legend(fontsize=8)
    ax.set_title("MSE vs Noise")
    ax.grid(True, alpha=0.3)

    # Panel 2: Correlation
    ax = axes[1]
    for name in ["Analytic", "DLG", "Geiping"]:
        vals = [all_results[float(s)].get(name, {}).get("corr_mean", 0) for s in sigmas_all]
        label = {"Analytic": "Analytic (SVD)"}.get(name, name)
        ax.plot(x_pos, vals, f"{markers[name]}-", label=label,
                color=colors[name], markersize=8, linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4)
    ax.set_ylabel("Correlation with original")
    ax.set_ylim(-0.2, 1.05)
    style_xaxis(ax)
    ax.legend(fontsize=8)
    ax.set_title("Reconstruction Quality (Correlation)")
    ax.grid(True, alpha=0.3)

    # Panel 3: PSNR (capped for display; exact recovery → very high PSNR)
    ax = axes[2]
    PSNR_CAP = 45.0
    for name in ["Analytic", "DLG", "Geiping", "theoretical_optimal"]:
        raw = [all_results[float(s)].get(name, {}).get("psnr_mean", float("nan")) for s in sigmas_all]
        vals = [min(v, PSNR_CAP) if np.isfinite(v) else np.nan for v in raw]
        style = "--" if name == "theoretical_optimal" else "-"
        label = {"theoretical_optimal": "Theory (MVUE)", "Analytic": "Analytic (SVD)"}.get(name, name)
        ax.plot(x_pos, vals, f"{markers[name]}{style}", label=label,
                color=colors[name], markersize=8, linewidth=2)
    # Flag the capped exact-recovery point (Analytic at σ=0)
    if all_results[0.0]["Analytic"]["psnr_mean"] > PSNR_CAP:
        ax.annotate('exact\n(↑)', xy=(0, PSNR_CAP), fontsize=7, color=colors['Analytic'],
                    ha='center', va='top')
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(-25, 50)
    style_xaxis(ax)
    ax.legend(fontsize=8)
    ax.set_title("PSNR vs Noise")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Gradient Inversion Attacks vs Theoretical Bound — FCNet (linear first layer)\n"
                 "Perfect reconstruction at σ=0; partial at low noise; collapse as noise grows.",
                 fontsize=11, fontweight='bold')
    fig.savefig(OUTPUT_DIR / "attack_comparison.pdf", dpi=150)
    fig.savefig(OUTPUT_DIR / "attack_comparison.png", dpi=150)
    plt.close(fig)

    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
