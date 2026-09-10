# -*- coding: utf-8 -*-
"""
distributions.py — Gráficos de distribución de métricas por fold.

Genera violin plots para cada combinación dataset × range × metric,
mostrando la dispersión de los 5 folds de cada condición.

Versiones global e intra‑familia.

CAMBIOS v2:
  - plot_distribution: modo "solo DiaTrend" (only_diatrend=True por defecto)
    para las figuras del capítulo de resultados, donde T1DiabetesGranada y
    REPLACE-BG se omiten al no alcanzar significación en Friedman.
  - Fuentes y leyenda ampliadas (fontsize mínimo 10, leyenda a ancho completo).
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from analysis.config import (
    DATASET_ORDER, DATASET_LABELS, RANGE_ORDER, RANGE_LABELS,
    ANALYSIS_METRICS,
    FAMILY_SIZE_LABELS, FAMILY_MECHANISM_LABELS,
)
from analysis.viz.style import apply_theme, save_fig, technique_color, technique_label

log = logging.getLogger(__name__)
apply_theme()

# Nombre interno de DiaTrend en DATASET_ORDER — ajusta si difiere
_DIATREND_KEY = "DIATREND"


def _cond_label(cond):
    if cond == "original": return "Original"
    parts = cond.replace("balanced_", "").split("_", 1)
    dim  = parts[0].capitalize()
    tech = technique_label(parts[1] if len(parts) > 1 else cond)
    return f"{dim} · {tech}"


def _sort_conds(conds):
    def _key(c):
        if c == "original": return (0, 0, 0)
        parts = c.replace("balanced_", "").split("_", 1)
        dim  = parts[0]
        tech = parts[1] if len(parts) > 1 else ""
        from analysis.config import TECHNIQUE_ORDER
        idx = TECHNIQUE_ORDER.index(tech) if tech in TECHNIQUE_ORDER else 99
        return (1, {"age": 0, "sex": 1}.get(dim, 2), idx)
    return sorted(conds, key=_key)


# ── Dibujo interno ────────────────────────────────────────────────────────

def _violin_plot(ax, data, x_col, y_col, hue_col, palette_dict, title):
    """
    Dibuja un violin plot agrupado por x_col, con hue=hue_col.
    Fuentes aumentadas respecto a v1.
    """
    sns.violinplot(
        ax=ax, data=data, x=x_col, y=y_col, hue=hue_col,
        palette=palette_dict, split=False, inner="quartile",
        linewidth=0.8, cut=0,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


# ──────────────────────────────────────────────────────────────────────────
# Distribuciones globales
# ──────────────────────────────────────────────────────────────────────────

def plot_distribution(fold_long, metric="RMSE", rng="ENTIRE",
                      only_diatrend=True):
    """
    Violin plot por dataset.

    Parámetros
    ----------
    only_diatrend : bool (default True)
        Si es True, genera una figura con un único panel (DiaTrend) y añade
        una nota al pie explicando la omisión de los otros dos datasets por
        no significación del test de Friedman.
        Si es False, comportamiento original con los tres datasets.
    """
    sub = fold_long[
        (fold_long["metric"] == metric) &
        (fold_long["range"]  == rng)
    ].copy()
    if sub.empty:
        return

    all_datasets = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]

    if only_diatrend:
        datasets = [d for d in all_datasets if d == _DIATREND_KEY]
        if not datasets:
            log.warning(f"plot_distribution: DiaTrend no encontrado en datos ({metric}/{rng})")
            return
    else:
        datasets = all_datasets

    n_ds = len(datasets)
    # Figura más alta cuando hay un solo panel para dar espacio a la nota al pie
    fig_h = 7 if only_diatrend else 6
    fig, axes = plt.subplots(1, n_ds, figsize=(n_ds * 7, fig_h), sharey=True)
    if n_ds == 1:
        axes = [axes]

    conds = _sort_conds(sub["condition"].unique().tolist())
    sub["condition"] = pd.Categorical(sub["condition"], categories=conds, ordered=True)

    for ax, ds in zip(axes, datasets):
        ds_data = sub[sub["dataset"] == ds]
        if ds_data.empty:
            continue
        unique_conds = ds_data["condition"].unique()
        palette = {cond: technique_color(
            cond.replace("balanced_age_", "").replace("balanced_sex_", "")
            if cond != "original" else "original"
        ) for cond in unique_conds}
        _violin_plot(ax, ds_data, x_col="condition", y_col="value",
                     hue_col="condition", palette_dict=palette,
                     title=f"{DATASET_LABELS.get(ds, ds)} — {metric} · {RANGE_LABELS.get(rng, rng)}")
        ax.set_xticklabels(
            [_cond_label(c.get_text()) for c in ax.get_xticklabels()],
            rotation=45, ha="right", fontsize=12,   # ← aumentado de 8
        )
        ax.set_xlabel("", fontsize=10)
        ax.set_ylabel(metric, fontsize=11)
        ax.tick_params(axis="y", labelsize=10)


    fig.suptitle(f"Distribución de folds — {metric} · {RANGE_LABELS.get(rng, rng)}",
                 fontsize=14, fontweight="bold", y=1.02)

    # Leyenda a ancho completo debajo de la figura
    handles_labels = axes[0].get_legend_handles_labels()
    all_conds_in_plot = _sort_conds(
        sub[sub["dataset"].isin(datasets)]["condition"].unique().tolist()
    )
    legend_handles = []
    legend_labels  = []
    seen = set()
    for cond in all_conds_in_plot:
        label = _cond_label(cond)
        if label in seen:
            continue
        seen.add(label)
        color = palette.get(cond, "#888")
        import matplotlib.patches as mpatches
        legend_handles.append(mpatches.Patch(facecolor=color, label=label))
        legend_labels.append(label)

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="lower center",
            ncol=min(len(legend_handles), 5),   # hasta 5 por fila
            bbox_to_anchor=(0.5, -0.12),
            fontsize=10,
            frameon=True,
            framealpha=0.95,
            edgecolor="0.8",
            handlelength=1.5,
            handleheight=1.2,
        )
        plt.subplots_adjust(bottom=0.22)

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    save_fig(fig, f"dist_{metric}_{rng}.pdf", subdir="distributions")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Distribuciones intra‑familia
# ──────────────────────────────────────────────────────────────────────────

def plot_distribution_family(fold_long, family_col="family_size",
                             metric="RMSE", rng="ENTIRE"):
    """
    Violin plots por familia de técnicas.
    Mantiene los tres datasets (las figuras de familia no se usan en
    el capítulo principal de resultados, por lo que no se aplica el
    filtrado de DiaTrend).
    Fuentes y leyenda aumentadas respecto a v1.
    """
    label_map = FAMILY_SIZE_LABELS if family_col == "family_size" else FAMILY_MECHANISM_LABELS
    for fam in fold_long[family_col].dropna().unique():
        sub = fold_long[
            (fold_long[family_col] == fam) &
            (fold_long["metric"] == metric) &
            (fold_long["range"]  == rng)
        ].copy()
        if sub.empty:
            continue
        datasets = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
        n_ds = len(datasets)
        fig, axes = plt.subplots(1, n_ds, figsize=(n_ds * 7, 6), sharey=True)
        if n_ds == 1:
            axes = [axes]

        conds = _sort_conds(sub["condition"].unique().tolist())
        sub["condition"] = pd.Categorical(sub["condition"], categories=conds, ordered=True)

        for ax, ds in zip(axes, datasets):
            ds_data = sub[sub["dataset"] == ds]
            if ds_data.empty:
                continue
            unique_conds = ds_data["condition"].unique()
            palette = {cond: technique_color(
                cond.replace("balanced_age_", "").replace("balanced_sex_", "")
                if cond != "original" else "original"
            ) for cond in unique_conds}
            _violin_plot(ax, ds_data, x_col="condition", y_col="value",
                         hue_col="condition", palette_dict=palette,
                         title=f"{DATASET_LABELS.get(ds, ds)} — {metric} · {RANGE_LABELS.get(rng, rng)}")
            ax.set_xticklabels(
                [_cond_label(c.get_text()) for c in ax.get_xticklabels()],
                rotation=45, ha="right", fontsize=10,
            )
            ax.set_xlabel("", fontsize=10)
            ax.set_ylabel(metric, fontsize=11)
            ax.tick_params(axis="y", labelsize=10)

        fam_label = label_map.get(fam, fam)
        fig.suptitle(f"Distribución — {metric} · {RANGE_LABELS.get(rng, rng)}  |  {fam_label}",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        subdir = f"distributions/{family_col}/{fam}"
        save_fig(fig, f"dist_{metric}_{rng}_{fam}.pdf", subdir=subdir)
        plt.close(fig)


def plot_all_distributions(fold_long):
    log.info("  Generando distribuciones...")
    # Globales — solo DiaTrend (Friedman no significativo en T1D y REPLACE-BG)
    for metric in ["RMSE", "MAE"]:
        for rng in ["ENTIRE", "TBR_2", "TBR_1", "TIR"]:
            plot_distribution(fold_long, metric=metric, rng=rng, only_diatrend=True)

    # Intra‑familia — tres datasets (figuras de análisis secundario)
    for family_col in ["family_size", "family_mechanism"]:
        for metric in ["RMSE", "MAE"]:
            for rng in ["ENTIRE", "TBR_2", "TBR_1", "TIR"]:
                plot_distribution_family(fold_long, family_col=family_col,
                                         metric=metric, rng=rng)
    n_global = 2 * 4
    n_family = 2 * 2 * 4
    log.info(f"  Distribuciones: {n_global} globales + {n_family} intra‑familia = {n_global+n_family} figuras.")