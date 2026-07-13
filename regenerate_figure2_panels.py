"""Regenerate figure2a.pdf and figure2b.pdf (the two panels of fig:sigmaetagamma)
as separate, standalone files -- exact unmodified code/sizes from figures.py.

The merged figure1_3_merged_row.pdf used in the manuscript was built by hand
from these two files (side by side, plus a connecting arrow and "A"/"B"
labels). Combining them into one matplotlib figure with subplots turned out
to change how the 3D panel renders (its legend no longer clears the surface
at the same figure size), so instead this script leaves both panels exactly
as originally coded -- same content, same size, zero risk of new rendering
bugs -- and the merge (arrow, labels, side-by-side placement) is done in
LaTeX with tikz instead of matplotlib.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from scipy import optimize

from fmte.prior_bounds import (
    rero_bound_without_subsampling,
    rero_bound_glrt_without_subsampling,
    inverse_rero_bound_without_subsampling,
)
from fmte.distributions import mse_cdf, inverse_mse_cdf, get_noise_multiplier

colors = ["#ffd700", "#ffb14e", "#fa8775", "#ea5f94", "#cd34b5", "#9d02d7", "#0000ff"]
basic_alpha = 0.6

# --- Panel A: 3D risk surface (exact code from figures.py) ---
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

ax.plot(np.log10(sigmas), risks_sgm, zs=0, zdir="y", label="Hayes et al.", color=colors[1])
ax.plot(np.log10(sigmas)[eta_diffs_hayes_mask], risks_sgm[eta_diffs_hayes_mask],
        zs=eta_diffs_hayes[eta_diffs_hayes_mask], zdir="y", color=colors[1], linestyle="--")
ax.plot(np.log10(sigmas), risks_glrt, zs=0, zdir="y", label="Kaissis et al.", color=colors[3])
ax.plot(np.log10(sigmas)[eta_diffs_kaissis_mask], risks_glrt[eta_diffs_kaissis_mask],
        zs=eta_diffs_kaissis[eta_diffs_kaissis_mask], zdir="y", color=colors[3], linestyle="--")

ax.plot_surface(np.log10(grid_sigma), grid_eta, risks_ours, label="Ours", color=colors[6],
                 alpha=basic_alpha, rstride=5, cstride=5, linewidth=0)

frame_color = colors[5]
frame_lw = 1.5
sigma_min, sigma_max = np.log10(grid_sigma[0, 0]), np.log10(grid_sigma[-1, -1])
eta_min = grid_eta.min()
gamma_level = 0.1
corners_s = np.array([sigma_min, sigma_max, sigma_max, sigma_min, sigma_min])
corners_e = np.array([eta_min, eta_min, eta_max, eta_max, eta_min])
corners_g = np.ones_like(corners_s) * gamma_level
ax.plot(corners_s, corners_e, corners_g, color=frame_color, linewidth=frame_lw, zorder=5, alpha=0.5)

ax.xaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False)
ax.zaxis.set_rotate_label(False)
ax.set_xlabel("$\\sigma$", fontsize=8)
ax.set_ylabel("$\\eta$", fontsize=8)
ax.set_zlabel("$\\gamma$", rotation=0, fontsize=8)
ax.set_ylim(etas[0], etas[-1])
ax.set_zlim(0, 1)
ax.zaxis.labelpad = -4


def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"


ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=3))
ax.view_init(15, 45, 0)
ax.set_proj_type("ortho")
ax.tick_params(axis="x", labelsize=8)
ax.tick_params(axis="y", labelsize=8)
ax.tick_params(axis="z", labelsize=8)
fig.legend(fontsize=8)
fig.savefig("figure2a.pdf")
print("Saved figure2a.pdf")

# --- Panel B: gamma=10% cutout (exact code from figures.py) ---
gamma = 0.1
sigma_hayes = inverse_rero_bound_without_subsampling(gamma, 1.0 / 13, 1.0)
sigma_kaissis = optimize.bisect(
    lambda x: rero_bound_glrt_without_subsampling(1.0 / 13, 1, 1, x) - gamma, 0.1, 20.0)
risk_corridor_hayes = optimize.bisect(
    lambda x: get_noise_multiplier(gamma, x, 1.0, 1.0) - sigma_hayes, 0.05, 0.8)
risk_corridor_kaissis = optimize.bisect(
    lambda x: get_noise_multiplier(gamma, x, 1.0, 1.0) - sigma_kaissis, 0.05, 0.8)

etas2 = np.logspace(-5, 0, 100)
sigmas2 = get_noise_multiplier(gamma, etas2, 1.0, 1.0)
fig2 = plt.figure(figsize=(0.5 * 5.5, 2.0), layout="constrained")
plt.plot(etas2, sigmas2, label="Ours", color=colors[6], alpha=basic_alpha)
plt.scatter(0, sigma_hayes, label="Hayes et al.", marker="x", color=colors[1])
plt.scatter(0, sigma_kaissis, label="Kaissis et al.", marker="x", color=colors[3])
plt.plot([0, risk_corridor_hayes], [sigma_hayes, sigma_hayes], color="lightslategray", linestyle="--")
plt.plot([0, risk_corridor_kaissis], [sigma_kaissis, sigma_kaissis], color="lightslategray",
         linestyle="--", label="Risk corridor")
plt.legend(fontsize=8)
plt.xlabel("$\\eta$", fontsize=8)
plt.ylabel("$\\sigma$", rotation=0, fontsize=8, labelpad=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
ax2d = plt.gca()
for spine in ax2d.spines.values():
    spine.set_edgecolor(colors[5])
    spine.set_linewidth(1.5)
fig2.savefig("figure2b.pdf", pad_inches=0, transparent=True)
print("Saved figure2b.pdf")
