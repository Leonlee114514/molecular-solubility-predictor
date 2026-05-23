"""
DisSolve - Shared matplotlib plotting utilities.
Centralises dark theme configuration and CJK font setup.
"""

import matplotlib.pyplot as plt
from ui.components import get_cjk_font

DARK_THEME = {
    "figure.facecolor": "#0a0a0f",
    "axes.facecolor": "#1e1e2e",
    "axes.edgecolor": "#2a2a3a",
    "axes.labelcolor": "#a0a0b0",
    "xtick.color": "#a0a0b0",
    "ytick.color": "#a0a0b0",
    "text.color": "#f0f0f5",
}


def setup_plt_dark():
    """Apply the DisSolve dark theme and CJK font to matplotlib globals."""
    plt.rcParams.update(DARK_THEME)
    plt.rcParams["axes.unicode_minus"] = False
    cjk = get_cjk_font()
    if cjk:
        plt.rcParams["font.family"] = cjk
