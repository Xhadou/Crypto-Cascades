"""
Publication-Quality Figure Styles

Pre-defined style dictionaries for major journals.
Call ``set_publication_style(journal)`` before creating matplotlib figures.
"""

from typing import Dict, Any

import matplotlib.pyplot as plt


JOURNAL_STYLES: Dict[str, Dict[str, Any]] = {
    'nature': {
        'font.family': 'Arial',
        'font.size': 7,
        'axes.linewidth': 0.5,
        'figure.figsize': (3.5, 2.5),   # Single column
        'figure.dpi': 300,
        'lines.linewidth': 1,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
    },
    'pnas': {
        'font.family': 'Helvetica',
        'font.size': 8,
        'axes.linewidth': 0.75,
        'figure.figsize': (3.42, 2.5),
        'figure.dpi': 300,
        'lines.linewidth': 1,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
    },
    'science': {
        'font.family': 'Helvetica',
        'font.size': 7,
        'axes.linewidth': 0.5,
        'figure.figsize': (3.5, 2.25),
        'figure.dpi': 300,
        'lines.linewidth': 0.75,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
    },
}


def set_publication_style(journal: str = 'nature') -> None:
    """
    Set matplotlib rcParams for publication-quality figures.

    Args:
        journal: One of ``'nature'``, ``'pnas'``, ``'science'``.

    Raises:
        ValueError: If *journal* is not a recognised style.
    """
    if journal not in JOURNAL_STYLES:
        raise ValueError(
            f"Unknown journal style '{journal}'. "
            f"Choose from: {list(JOURNAL_STYLES.keys())}"
        )
    plt.rcParams.update(JOURNAL_STYLES[journal])


def reset_style() -> None:
    """Reset matplotlib to default rcParams."""
    plt.rcdefaults()
