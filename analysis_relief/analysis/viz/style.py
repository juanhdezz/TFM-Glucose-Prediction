# -*- coding: utf-8 -*-
"""
style.py  —  Tema matplotlib compartido. Importar en todos los módulos viz.
"""

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from analysis.config import (
    MPL_RC, PALETTE, DIMENSION_PALETTE, DATASET_PALETTE,
    TECHNIQUE_LABELS, DATASET_LABELS, RANGE_LABELS, DIMENSION_LABELS,
    FIGURES_DIR,
)

log = logging.getLogger(__name__)


def apply_theme():
    """Aplica el tema global. Llamar UNA VEZ al importar cada módulo viz."""
    mpl.rcParams.update(MPL_RC)


apply_theme()


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def technique_color(technique: str) -> str:
    return PALETTE.get(technique, "#888888")


def dimension_color(dim: str) -> str:
    return DIMENSION_PALETTE.get(dim, "#888888")


def dataset_color(dataset: str) -> str:
    return DATASET_PALETTE.get(dataset, "#888888")


def technique_label(technique: str) -> str:
    return TECHNIQUE_LABELS.get(technique, technique)


def dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset)


def range_label(rng: str) -> str:
    return RANGE_LABELS.get(rng, rng)


def dimension_label(dim: str) -> str:
    return DIMENSION_LABELS.get(dim, dim)


# ---------------------------------------------------------------------------
# Legend helpers
# ---------------------------------------------------------------------------

def technique_legend_handles(techniques: list[str]) -> list[mpatches.Patch]:
    """Devuelve patches para leyenda de técnicas."""
    return [
        mpatches.Patch(color=technique_color(t), label=technique_label(t))
        for t in techniques
    ]


def dataset_legend_handles(datasets: list[str]) -> list[mpatches.Patch]:
    return [
        mpatches.Patch(color=dataset_color(d), label=dataset_label(d))
        for d in datasets
    ]


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_fig(fig: plt.Figure, filename: str, subdir: str = "") -> Path:
    """
    Guarda figura en FIGURES_DIR / subdir / filename.
    filename debe incluir extensión (.pdf recomendado para vectorial, .png para rasters).
    """
    if subdir:
        out_dir = FIGURES_DIR / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = FIGURES_DIR

    path = out_dir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    log.info(f"  Figura: {path.relative_to(FIGURES_DIR.parent)}")
    return path


# ---------------------------------------------------------------------------
# Anotación de significancia estadística
# ---------------------------------------------------------------------------

def significance_stars(pval: float) -> str:
    """Convierte p-value en asteriscos de significancia."""
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    else:
        return "ns"


# ---------------------------------------------------------------------------
# Colormaps divergentes para deltas
# ---------------------------------------------------------------------------

def delta_cmap():
    """
    Colormap divergente para deltas de mejora.
    Verde = mejora, rojo = empeora, blanco = sin cambio.
    Apto para daltonismo (no usa rojo/verde puros — usa azul/naranja).
    """
    from matplotlib.colors import LinearSegmentedColormap
    # Azul (mejora) — blanco — naranja (empeora)
    colors = ["#0072B2", "#FFFFFF", "#E69F00"]
    return LinearSegmentedColormap.from_list("delta_bwr", colors, N=256)


def improvement_cmap():
    """
    Colormap unidireccional para % mejora (siempre positivo = mejor).
    De blanco a verde oscuro.
    """
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#FFFFFF", "#009E73"]
    return LinearSegmentedColormap.from_list("improvement", colors, N=256)