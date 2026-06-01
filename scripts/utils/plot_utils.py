'''
set_publication_style()
'''

import matplotlib.pyplot as plt


def set_publication_style(scale=1.0, font_family="sans-serif"):
    """
    Set Matplotlib style parameters for publication-quality figures.

    Parameters
    ----------
    scale : float, default=1.0
        Global scaling factor for font sizes, line widths, tick sizes,
        and marker sizes.

    font_family : str, default="sans-serif"
        Font family used by Matplotlib.
    """

    base_fontsize = 8.0 * scale

    plt.rcParams.update({
        # Font
        "font.family": font_family,
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": base_fontsize,

        # Axes
        "axes.titleweight": "bold",
        "axes.titlesize": base_fontsize * 1.15,
        "axes.labelsize": base_fontsize * 1.05,
        "axes.linewidth": 0.7 * scale,
        "axes.labelpad": 4 * scale,

        # Ticks
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "xtick.major.width": 0.7 * scale,
        "ytick.major.width": 0.7 * scale,
        "xtick.minor.width": 0.5 * scale,
        "ytick.minor.width": 0.5 * scale,
        "xtick.major.size": 3.5 * scale,
        "ytick.major.size": 3.5 * scale,
        "xtick.minor.size": 2.0 * scale,
        "ytick.minor.size": 2.0 * scale,

        # Legend
        "legend.fontsize": base_fontsize,
        "legend.title_fontsize": base_fontsize * 1.05,
        "legend.frameon": False,

        # Lines and markers
        "lines.linewidth": 0.9 * scale,
        "lines.markersize": 4.0 * scale,

        # Output
        "savefig.dpi": 600,
        "figure.dpi": 150,
        "svg.fonttype": "path",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Layout and formatting
        "figure.constrained_layout.use": True,
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
    })