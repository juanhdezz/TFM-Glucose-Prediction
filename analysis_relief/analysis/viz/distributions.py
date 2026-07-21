# -*- coding: utf-8 -*-
"""
distributions.py — Gráficos de distribución de métricas por fold.

Genera violin plots para cada combinación dataset × range × metric,
mostrando la dispersión de los 5 folds de cada condición.

Versiones global e intra‑familia.
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
    """
    sns.violinplot(
        ax=ax, data=data, x=x_col, y=y_col, hue=hue_col,
        palette=palette_dict, split=False, inner="quartile",
        linewidth=0.8, cut=0,
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    # Eliminar leyenda solo si existe
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


# ──────────────────────────────────────────────────────────────────────────
# Distribuciones globales
# ──────────────────────────────────────────────────────────────────────────

def plot_distribution(fold_long, metric="RMSE", rng="ENTIRE"):
    """
    Un violin plot por dataset, mostrando la distribución de los folds
    para cada condición.
    """
    sub = fold_long[
        (fold_long["metric"] == metric) &
        (fold_long["range"]  == rng)
    ].copy()
    if sub.empty:
        return

    datasets = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
    n_ds     = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(n_ds * 6, 6), sharey=True)
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
        ax.set_xticklabels([_cond_label(c.get_text()) for c in ax.get_xticklabels()],
                           rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel(metric)

    fig.suptitle(f"Distribución de folds — {metric} · {RANGE_LABELS.get(rng, rng)}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, f"dist_{metric}_{rng}.pdf", subdir="distributions")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Distribuciones intra‑familia
# ──────────────────────────────────────────────────────────────────────────

def plot_distribution_family(fold_long, family_col="family_size",
                             metric="RMSE", rng="ENTIRE"):
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
        fig, axes = plt.subplots(1, n_ds, figsize=(n_ds * 6, 6), sharey=True)
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
            ax.set_xticklabels([_cond_label(c.get_text()) for c in ax.get_xticklabels()],
                               rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("")
            ax.set_ylabel(metric)

        fam_label = label_map.get(fam, fam)
        fig.suptitle(f"Distribución — {metric} · {RANGE_LABELS.get(rng, rng)}  |  {fam_label}",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        subdir = f"distributions/{family_col}/{fam}"
        save_fig(fig, f"dist_{metric}_{rng}_{fam}.pdf", subdir=subdir)
        plt.close(fig)


def plot_all_distributions(fold_long):
    log.info("  Generando distribuciones...")
    # Globales
    for metric in ["RMSE", "MAE"]:
        for rng in ["ENTIRE", "TBR_2", "TBR_1", "TIR"]:
            plot_distribution(fold_long, metric=metric, rng=rng)

    # Intra‑familia
    for family_col in ["family_size", "family_mechanism"]:
        for metric in ["RMSE", "MAE"]:
            for rng in ["ENTIRE", "TBR_2", "TBR_1", "TIR"]:
                plot_distribution_family(fold_long, family_col=family_col,
                                         metric=metric, rng=rng)
    n_global = 2 * 4  # 8
    n_family = 2 * 2 * 4  # 16
    log.info(f"  Distribuciones: {n_global} globales + {n_family} intra‑familia = {n_global+n_family} figuras.")