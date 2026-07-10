"""Regenerate the Exp10 attack-comparison figure (main-text Fig 5) from saved JSON.

Publication style: Linux Libertine, 5.5in text width, colorblind-safe palette.
Categorical x-axis consistent across all three panels.
"""
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotstyle import apply_style, TEXTWIDTH, ATTACK_STYLE, C_TARGET

apply_style()

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "exp10"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
N = 3072  # CIFAR-10 32x32 RGB

with open(OUTPUT_DIR / "attack_comparison.json") as f:
    all_results = json.load(f)

key_by_sigma = {float(k): k for k in all_results.keys()}
sigmas_all = sorted(key_by_sigma.keys())
x_pos = np.arange(len(sigmas_all))
sigma_labels = [("0" if s == 0 else f"{s:g}") for s in sigmas_all]

LABELS = {"theoretical_optimal": "Theory (MVUE)", "Analytic": "Analytic (division)",
          "DLG": "DLG", "Geiping": "Geiping"}


def get(s, name, key):
    return all_results[key_by_sigma[s]].get(name, {}).get(key, float("nan"))


def style_xaxis(ax):
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sigma_labels)
    ax.set_xlabel(r"$\sigma$ (noise multiplier)")


fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.25), layout="constrained")

# Panel 1: MSE (symlog)
ax = axes[0]
for name in ["Analytic", "DLG", "Geiping", "theoretical_optimal"]:
    vals = [get(s, name, "mse_mean") for s in sigmas_all]
    st = ATTACK_STYLE[name]
    ls = "--" if name == "theoretical_optimal" else "-"
    ax.plot(x_pos, vals, marker=st["marker"], linestyle=ls, color=st["color"],
            label=LABELS[name], markersize=4)
trivial = all_results["0.0"]["metadata"]["trivial_mse"]
ax.axhline(y=trivial, color=C_TARGET, linestyle=(0, (1, 1)), linewidth=1.0, label="Trivial (zeros)")
ax.set_yscale("symlog", linthresh=1e-2)
ax.set_ylim(0, 2e3)
ax.set_ylabel("MSE")
style_xaxis(ax)
ax.set_title("MSE")

# Panel 2: Correlation
ax = axes[1]
for name in ["Analytic", "DLG", "Geiping"]:
    vals = [get(s, name, "corr_mean") for s in sigmas_all]
    st = ATTACK_STYLE[name]
    ax.plot(x_pos, vals, marker=st["marker"], linestyle="-", color=st["color"],
            label=LABELS[name], markersize=4)
ax.axhline(y=0, color=C_TARGET, linestyle="--", alpha=0.5, linewidth=0.8)
ax.set_ylabel("Correlation")
ax.set_ylim(-0.2, 1.05)
style_xaxis(ax)
ax.set_title("Correlation")

# Panel 3: PSNR (capped)
ax = axes[2]
PSNR_CAP = 45.0
for name in ["Analytic", "DLG", "Geiping", "theoretical_optimal"]:
    raw = [get(s, name, "psnr_mean") for s in sigmas_all]
    vals = [min(v, PSNR_CAP) if np.isfinite(v) else np.nan for v in raw]
    st = ATTACK_STYLE[name]
    ls = "--" if name == "theoretical_optimal" else "-"
    ax.plot(x_pos, vals, marker=st["marker"], linestyle=ls, color=st["color"],
            label=LABELS[name], markersize=4)
if all_results["0.0"]["Analytic"]["psnr_mean"] > PSNR_CAP:
    ax.annotate("exact", xy=(0, PSNR_CAP), fontsize=6.5, color=ATTACK_STYLE["Analytic"]["color"],
                ha="center", va="top")
ax.set_ylabel("PSNR (dB)")
ax.set_ylim(-25, 50)
style_xaxis(ax)
ax.set_title("PSNR")

# Shared legend reserved inside the canvas (constrained layout handles spacing)
handles, labels = [], []
for ax in axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h)
            labels.append(l)
fig.legend(handles, labels, loc="outside upper center", ncol=5, fontsize=7,
           columnspacing=1.2, handletextpad=0.4)

fig.savefig(OUTPUT_DIR / "attack_comparison.pdf")
fig.savefig(OUTPUT_DIR / "attack_comparison.png", dpi=200)
plt.close(fig)
print("Replotted attack comparison figure.")
