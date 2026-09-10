# -*- coding: utf-8 -*-
"""
heatmaps.py — Heatmaps de calidad de publicación.

F1  heatmap_delta_{metric}_{rng}.pdf   mejora% por (técnica×dim) × dataset
F2  heatmap_ranking_{metric}.pdf       posición de ranking por técnica × (dataset×range)

CAMBIOS v2:
  - plot_delta_heatmap: modo only_diatrend=True — muestra solo la columna
    DiaTrend y añade nota al pie para T1DiabetesGranada y REPLACE-BG.
  - Fuentes de títulos, ejes y colorbar aumentadas.
  - plot_ranking_heatmap sin cambios de filtrado (contiene los tres datasets).
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap

from analysis.config import (
    DATASET_ORDER, DATASET_LABELS, TECHNIQUE_ORDER, TECHNIQUE_LABELS,
    RANGE_ORDER, RANGE_LABELS, ANALYSIS_METRICS, LOWER_IS_BETTER,
)
from analysis.viz.style import apply_theme, save_fig

log = logging.getLogger(__name__)
apply_theme()

_DIATREND_KEY = "DIATREND"  # ajusta si difiere del valor en DATASET_ORDER


def _delta_cmap():
    return LinearSegmentedColormap.from_list(
        "delta", ["#0072B2", "#FFFFFF", "#E69F00"], N=256)

def _rank_cmap():
    return LinearSegmentedColormap.from_list(
        "rank", ["#009E73", "#FFFFFF", "#D55E00"], N=256)

def _sort_cond(conds):
    def _key(c):
        if c == "original":
            return (0, 0, 0)
        parts = c.replace("balanced_", "").split("_", 1)
        dim  = parts[0]
        tech = parts[1] if len(parts) > 1 else ""
        return (1, {"age": 0, "sex": 1}.get(dim, 2),
                TECHNIQUE_ORDER.index(tech) if tech in TECHNIQUE_ORDER else 99)
    return sorted(conds, key=_key)

def _row_label(cond):
    if cond == "original":
        return "Original (baseline)"
    parts = cond.replace("balanced_", "").split("_", 1)
    dim  = parts[0].capitalize()
    tech = TECHNIQUE_LABELS.get(parts[1] if len(parts) > 1 else cond, cond)
    return f"{dim} · {tech}"


# ── F1: Delta heatmap ────────────────────────────────────────────────────────

def plot_delta_heatmap(master, metric="RMSE", rng="ENTIRE",
                       only_diatrend=True):
    """
    Heatmap de mejora relativa respecto al baseline.

    Parámetros
    ----------
    only_diatrend : bool (default True)
        Si True, muestra solo la columna de DiaTrend y añade nota al pie
        explicando la omisión de T1DiabetesGranada y REPLACE-BG.
    """
    sub = master[
        (~master["is_original"]) &
        (master["metric"] == metric) &
        (master["range"]  == rng)
    ].copy()
    if sub.empty:
        log.warning(f"delta_heatmap: sin datos metric={metric} range={rng}")
        return

    pivot = sub.pivot_table(
        index="condition", columns="dataset",
        values="improvement_pct", aggfunc="mean"
    )
    pivot = pivot.loc[_sort_cond([c for c in pivot.index])]

    # Selección de columnas
    if only_diatrend:
        col_order = [d for d in DATASET_ORDER
                     if d in pivot.columns and d == _DIATREND_KEY]
    else:
        col_order = [d for d in DATASET_ORDER if d in pivot.columns]

    if not col_order:
        log.warning(f"delta_heatmap: DiaTrend no encontrado en pivot ({metric}/{rng})")
        return

    pivot = pivot[col_order]
    row_labels = [_row_label(c) for c in pivot.index]
    col_labels = [DATASET_LABELS.get(d, d) for d in pivot.columns]
    data = pivot.values.astype(float)

    n_rows, n_cols = data.shape
    # Figura más ancha y alta para dar margen a fuentes mayores
    fig, ax = plt.subplots(figsize=(max(8, n_cols * 4.0), max(4, n_rows * 0.40 + 1.5)))

    vmax = max(np.nanpercentile(np.abs(data[~np.isnan(data)]), 95), 2.0)
    im = ax.imshow(data, cmap=_delta_cmap(), vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="#888")
                continue
            txt   = f"{v:+.1f}%"
            norm  = abs(v) / vmax
            color = "white" if norm > 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=13, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xlabel("Dataset", fontsize=12, labelpad=10)

    # Separador Age / Sex
    age_idx = [i for i, c in enumerate(pivot.index) if "balanced_age" in c]
    sex_idx = [i for i, c in enumerate(pivot.index) if "balanced_sex" in c]
    if age_idx and sex_idx:
        ax.axhline(max(age_idx) + 0.5, color="0.4", lw=1.5, ls="--")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(f"Improvement vs. Original (%)  ▲ better", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    range_lbl = RANGE_LABELS.get(rng, rng)
    ax.set_title(f"Relative improvement vs. baseline  |  {metric} · {range_lbl}",
                 fontsize=13, pad=22, fontweight="bold")

    

    plt.tight_layout()
    save_fig(fig, f"heatmap_delta_{metric}_{rng}.pdf", subdir="heatmaps")
    plt.close(fig)


# ── F2: Ranking heatmap ───────────────────────────────────────────────────────

def plot_ranking_heatmap(master, metric="RMSE"):
    """
    Heatmap de posición de ranking por técnica × (dataset × rango).
    Mantiene los tres datasets (figura de análisis global, no de resultados
    principales).
    Fuentes aumentadas respecto a v1.
    """
    sub = master[master["metric"] == metric].copy()
    if sub.empty:
        return

    records = []
    for (ds, rng), grp in sub.groupby(["dataset", "range"]):
        ranked = grp.sort_values("mean", ascending=(metric in LOWER_IS_BETTER)).reset_index(drop=True)
        for pos, row in ranked.iterrows():
            records.append({
                "condition": row["condition"],
                "col_key": f"{DATASET_LABELS.get(ds,ds)}\n{RANGE_LABELS.get(rng,rng)}",
                "rank": pos + 1,
                "n":    len(ranked),
            })
    df_rank = pd.DataFrame(records)

    col_order = []
    for ds in DATASET_ORDER:
        ds_lbl = DATASET_LABELS.get(ds, ds)
        for rng in RANGE_ORDER:
            key = f"{ds_lbl}\n{RANGE_LABELS.get(rng, rng)}"
            if key in df_rank["col_key"].unique():
                col_order.append(key)

    pivot = df_rank.pivot_table(index="condition", columns="col_key",
                                values="rank", aggfunc="mean")
    pivot = pivot.loc[_sort_cond(list(pivot.index))]
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    row_labels = [_row_label(c) for c in pivot.index]
    data = pivot.values.astype(float)
    n_rows, n_cols = data.shape
    n_max = int(np.nanmax(data)) if not np.all(np.isnan(data)) else 13

    fig, ax = plt.subplots(figsize=(max(14, n_cols * 0.90), max(5, n_rows * 0.56 + 1.5)))
    im = ax.imshow(data, cmap=_rank_cmap(), vmin=1, vmax=n_max, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if np.isnan(v):
                continue
            norm  = (v - 1) / max(n_max - 1, 1)
            color = "white" if norm > 0.62 else "black"
            ax.text(j, i, str(int(round(v))), ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pivot.columns, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)

    age_idx = [i for i, c in enumerate(pivot.index) if "balanced_age" in c]
    sex_idx = [i for i, c in enumerate(pivot.index) if "balanced_sex" in c]
    if age_idx and sex_idx:
        ax.axhline(max(age_idx) + 0.5, color="0.4", lw=1.5, ls="--")

    ds_sizes = {}
    for ds in DATASET_ORDER:
        ds_lbl = DATASET_LABELS.get(ds, ds)
        count  = sum(1 for c in pivot.columns if c.startswith(ds_lbl))
        ds_sizes[ds_lbl] = count
    x = -0.5
    for ds_lbl, cnt in ds_sizes.items():
        x += cnt
        if x < n_cols - 0.5:
            ax.axvline(x, color="black", lw=2)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1.5%", pad=0.1)
    cb = fig.colorbar(im, cax=cax, ticks=[1, n_max])
    cb.set_ticklabels(["1 (best)", f"{n_max} (worst)"], fontsize=9)

    ax.set_title(f"Ranking position by technique × dataset × glucose range  |  {metric}",
                 fontsize=13, pad=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, f"heatmap_ranking_{metric}.pdf", subdir="heatmaps")
    plt.close(fig)


def plot_all_heatmaps(master):
    log.info("  Generando heatmaps...")
    for metric in ["RMSE", "MAE"]:
        for rng in ["ENTIRE", "TBR_2", "TBR_1", "TIR"]:
            # Solo DiaTrend en los heatmaps de resultados principales
            plot_delta_heatmap(master, metric=metric, rng=rng, only_diatrend=True)
    for metric in ["RMSE", "MAE"]:
        plot_ranking_heatmap(master, metric=metric)
    log.info(f"  Heatmaps: {2*4 + 2} figuras generadas.")