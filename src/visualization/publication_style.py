"""
Publication-Quality Figure Styles

Pre-defined style dictionaries for major journals.
Call ``set_publication_style(journal)`` before creating matplotlib figures,
or use ``apply_style()`` for automatic SciencePlots integration with
graceful fallback.
"""

from typing import Dict, Any, Optional

import matplotlib.pyplot as plt

from src.utils.logger import get_logger

logger = get_logger(__name__)


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


def apply_style(style_name: Optional[str] = None) -> None:
    """Apply publication style. Falls back gracefully if SciencePlots not installed.

    Tries SciencePlots styles first (e.g. ``['science', 'nature']``), then
    ``['science']`` alone, then falls back to the built-in ``JOURNAL_STYLES``
    rcParams dictionaries, and finally matplotlib defaults.

    Args:
        style_name: A SciencePlots / journal style name such as ``'nature'``,
            ``'ieee'``, ``'no-latex'``.  When *None*, reads
            ``visualization.publication_style`` from the project config.
    """
    if style_name is None:
        try:
            from src.utils.config_manager import ConfigManager
            cm = ConfigManager()
            config = cm.get_all()
            style_name = config.get('visualization', {}).get('publication_style')
        except Exception:
            style_name = None

    # Nothing requested -- keep current style
    if not style_name:
        logger.debug("No publication style requested; keeping matplotlib defaults")
        return

    # Attempt SciencePlots-based styles first, then built-in fallback
    styles_to_try = [
        ['science', style_name],
        ['science'],
    ]

    for styles in styles_to_try:
        try:
            plt.style.use(styles)
            logger.info("Applied SciencePlots style: %s", styles)
            return
        except OSError:
            continue

    # SciencePlots not available -- fall back to built-in JOURNAL_STYLES
    if style_name in JOURNAL_STYLES:
        plt.rcParams.update(JOURNAL_STYLES[style_name])
        logger.info(
            "SciencePlots not installed; applied built-in '%s' rcParams", style_name
        )
        return

    logger.warning(
        "Style '%s' not available (SciencePlots not installed and no built-in "
        "match); keeping matplotlib defaults",
        style_name,
    )


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
