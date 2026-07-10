"""Shared publication plotting style for ACM TOPS figures.

Matches the manuscript: Linux Libertine font, 5.5in text width, paper font sizes.
Provides a colorblind-safe palette (Wong, Nature Methods 2011) and consistent
color/marker assignments for recurring entities across figures.

Usage:
    from plotstyle import apply_style, TEXTWIDTH, PALETTE, ATTACK_STYLE, color_cycle
    apply_style()
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, TEXTWIDTH * 0.32))
"""
from pathlib import Path
import matplotlib
import matplotlib.font_manager as fm

# ACM acmsmall text width (the original figures are 396 pt = 5.5 in wide).
TEXTWIDTH = 5.5          # inches, full text width
HALFWIDTH = TEXTWIDTH / 2

_FONT_DIR = Path(__file__).parent / "fonts"

# Wong colorblind-safe palette (Bang Wong, Nature Methods 8:441, 2011).
PALETTE = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermillion": "#D55E00",
    "purple":  "#CC79A7",
    "gray":    "#999999",
}

# Ordered cycle for categorical series (datasets, architectures, etc.).
CYCLE = [PALETTE[k] for k in ["blue", "orange", "green", "vermillion", "purple", "skyblue", "black"]]

# Consistent assignments for the three attacks + theory (used in Exp10).
ATTACK_STYLE = {
    "Analytic":           {"color": PALETTE["purple"],     "marker": "D"},
    "DLG":                {"color": PALETTE["blue"],       "marker": "o"},
    "Geiping":            {"color": PALETTE["orange"],     "marker": "s"},
    "theoretical_optimal":{"color": PALETTE["green"],      "marker": "^"},
}

# Recurring semantic colors.
C_THEORY = PALETTE["green"]
C_EMPIRICAL = PALETTE["vermillion"]
C_TARGET = PALETTE["gray"]      # 95% target / trivial baseline reference lines


def _register_fonts():
    """Register the bundled Linux Libertine OTF files with matplotlib."""
    if not _FONT_DIR.exists():
        return False
    registered = False
    for otf in _FONT_DIR.glob("*.otf"):
        try:
            fm.fontManager.addfont(str(otf))
            registered = True
        except Exception:
            pass
    return registered


def apply_style():
    """Set matplotlib rcParams to the publication style."""
    have_libertine = _register_fonts()
    serif = (["Linux Libertine O", "Linux Libertine"] if have_libertine else []) + \
            ["DejaVu Serif", "Times New Roman", "serif"]
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": serif,
        "mathtext.fontset": "cm",
        # Paper-matched sizes (body is 10 pt; figure text slightly smaller).
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 9,
        # Clean, light styling.
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "legend.frameon": False,
        "axes.prop_cycle": matplotlib.cycler(color=CYCLE),
        # Save at the figure's native size (no tight crop) so the PDF width
        # exactly equals figsize; included at \linewidth this gives scale 1.0
        # and the font renders at the paper's literal pt size. Use
        # layout="constrained" + loc="outside ..." legends to keep content
        # inside the canvas.
        "savefig.bbox": None,
        "pdf.fonttype": 42,   # embed as TrueType (editable, no Type3 warnings)
        "ps.fonttype": 42,
    })
    return have_libertine


if __name__ == "__main__":
    # Quick self-test: render a figure and report whether Libertine was found.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    ok = apply_style()
    print(f"Linux Libertine registered: {ok}")
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.0))
    x = np.linspace(0, 1, 50)
    for i, name in enumerate(["blue", "orange", "green", "vermillion"]):
        ax.plot(x, np.sin(x * (i + 1) * 3), color=PALETTE[name], label=f"series {name}")
    ax.set_xlabel(r"noise multiplier $\sigma$")
    ax.set_ylabel("MSE")
    ax.set_title("Plotstyle self-test (Linux Libertine)")
    ax.legend()
    fig.savefig(Path(__file__).parent.parent / "results" / "plotstyle_test.pdf")
    print("Saved results/plotstyle_test.pdf")
