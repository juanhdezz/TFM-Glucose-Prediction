# -*- coding: utf-8 -*-
"""
dumbbell.py — Dumbbell plots y slope charts.

F3  dumbbell_{metric}_{rng}.pdf    baseline vs mejor técnica, por dataset
F4  slope_{metric}_by_dim.pdf      pendiente original→balanceado por dimensión
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from analysis.config import (
    DATASET_ORDER, DATASET_LABELS, TECHNIQUE_ORDER, TECHNIQUE_LABELS,
    RANGE_ORDER, RANGE_LABELS, LOWER_IS_BETTER, PALETTE, DIMENSION_PALETTE,
    DIMENSION_LABELS,
)
from analysis.viz.style import apply_theme, save_fig, technique_color, technique_label

log = logging.getLogger(__name__)
apply_theme()


# ── F3: Dumbbell plot ─────────────────────────────────────────────────────────

def plot_dumbbell(master, metric="RMSE", rng="ENTIRE"):
    """
    Un panel por dataset (3 columnas).
    Cada fila = condición balanceada.
    Punto izquierdo = original (baseline), punto derecho = balanceado.
    Línea conecta ambos. Color del punto derecho = técnica.
    """
    sub = master[(master["metric"] == metric) & (master["range"] == rng)].copy()
    if sub.empty:
        return

    ascending = metric in LOWER_IS_BETTER
    datasets  = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
    n_ds      = len(datasets)

    fig, axes = plt.subplots(1, n_ds, figsize=(n_ds * 5.5, 7), sharey=False)
    if n_ds == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        ds_data    = sub[sub["dataset"] == ds]
        orig_val   = ds_data[ds_data["is_original"]]["mean"].values
        if len(orig_val) == 0:
            continue
        orig_val   = orig_val[0]

        bal = ds_data[~ds_data["is_original"]].copy()
        bal = bal.sort_values("mean", ascending=not ascending)  # worst at top, best at bottom

        y_pos = np.arange(len(bal))

        for y, (_, row) in zip(y_pos, bal.iterrows()):
            color = technique_color(row["technique"])
            # Connecting line
            ax.plot([orig_val, row["mean"]], [y, y],
                    color="0.75", lw=1.5, zorder=1)
            # Baseline dot
            ax.scatter(orig_val, y, color="#333333", s=55, zorder=3,
                       marker="D", linewidths=0)
            # Technique dot
            ax.scatter(row["mean"], y, color=color, s=90, zorder=4,
                       edgecolors="white", linewidths=0.8)
            # Std error bar
            if not np.isnan(row.get("std", np.nan)):
                ax.errorbar(row["mean"], y, xerr=row["std"],
                            fmt="none", ecolor=color, elinewidth=1.2,
                            capsize=3, zorder=2, alpha=0.7)

        # Baseline vertical line
        ax.axvline(orig_val, color="#333333", lw=1.2, ls="--", alpha=0.6, zorder=0)

        # y-axis labels
        cond_labels = []
        for _, row in bal.iterrows():
            dim  = DIMENSION_LABELS.get(row["dimension"], "")
            tech = TECHNIQUE_LABELS.get(row["technique"], row["technique"])
            cond_labels.append(f"{dim} · {tech}" if dim else tech)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(cond_labels, fontsize=8)
        ax.set_xlabel(f"{metric} (mg/dL)", fontsize=10)
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)

        # Shade improvement direction
        xlim = ax.get_xlim()
        if ascending:  # lower is better → shade left of baseline
            ax.axvspan(xlim[0], orig_val, alpha=0.04, color="#009E73")
        else:
            ax.axvspan(orig_val, xlim[1], alpha=0.04, color="#009E73")
        ax.set_xlim(xlim)

    # Shared legend for techniques
    handles = [mlines.Line2D([], [], color="#333333", marker="D", markersize=6,
                              linestyle="None", label="Original (baseline)")]
    seen = set()
    for _, row in master[~master["is_original"]].iterrows():
        t = row["technique"]
        if t not in seen:
            handles.append(mlines.Line2D([], [], color=technique_color(t),
                                          marker="o", markersize=7,
                                          linestyle="None", label=technique_label(t)))
            seen.add(t)

    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
               bbox_to_anchor=(0.5, -0.04), frameon=True, fontsize=9)

    range_lbl = RANGE_LABELS.get(rng, rng)
    fig.suptitle(f"Dumbbell plot — {metric} · {range_lbl}  |  ◆ = Original baseline",
                 fontsize=13, fontweight="bold", y=1.01)

    save_fig(fig, f"dumbbell_{metric}_{rng}.pdf", subdir="dumbbell")


# ── F4: Slope chart by dimension ──────────────────────────────────────────────

def plot_slope_by_dimension(master, metric="RMSE", rng="ENTIRE"):
    """
    Slope chart: eje izquierdo = original, eje derecho = media de cada técnica.
    Líneas coloreadas por técnica, separadas en dos paneles (Age / Sex).
    Permite ver claramente qué técnicas mejoran y cuáles empeoran.
    """
    sub = master[(master["metric"] == metric) & (master["range"] == rng)].copy()
    if sub.empty:
        return

    dims = ["age", "sex"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)

    for ax, dim in zip(axes, dims):
        dim_data = sub[sub["dimension"].isin([dim, None])].copy()
        orig     = dim_data[dim_data["is_original"]].groupby("dataset")["mean"].mean()
        bal      = dim_data[~dim_data["is_original"]]

        x_orig, x_bal = 0, 1
        techniques = [t for t in TECHNIQUE_ORDER if t != "original" and
                      t in bal["technique"].unique()]

        for tech in techniques:
            tech_data = bal[bal["technique"] == tech]
            for ds in DATASET_ORDER:
                ds_orig = orig.get(ds, np.nan)
                ds_bal  = tech_data[tech_data["dataset"] == ds]["mean"].values
                if np.isnan(ds_orig) or len(ds_bal) == 0:
                    continue
                color = technique_color(tech)
                ax.plot([x_orig, x_bal], [ds_orig, ds_bal[0]],
                        color=color, lw=1.8, alpha=0.85, zorder=2)
                ax.scatter([x_orig, x_bal], [ds_orig, ds_bal[0]],
                           color=color, s=50, zorder=3, edgecolors="white", lw=0.7)

        ax.set_xticks([x_orig, x_bal])
        ax.set_xticklabels(["Original", "Balanced"], fontsize=11, fontweight="bold")
        ax.set_ylabel(f"{metric} (mg/dL)", fontsize=10) if dim == "age" else None
        ax.set_title(f"Dimension: {DIMENSION_LABELS.get(dim, dim)}",
                     fontsize=12, fontweight="bold")
        ax.set_xlim(-0.3, 1.3)
        ax.grid(axis="y", alpha=0.3)

        # Annotate direction
        arrow_dir = "↓ better" if metric in LOWER_IS_BETTER else "↑ better"
        ax.text(1.25, ax.get_ylim()[0], arrow_dir,
                fontsize=8, color="0.5", va="bottom", ha="right")

    # Legend
    handles = [mlines.Line2D([], [], color=technique_color(t), lw=2,
                              label=technique_label(t))
               for t in techniques]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
               bbox_to_anchor=(0.5, -0.05), frameon=True, fontsize=9)

    range_lbl = RANGE_LABELS.get(rng, rng)
    fig.suptitle(f"Slope chart — {metric} · {range_lbl}",
                 fontsize=13, fontweight="bold", y=1.01)
    save_fig(fig, f"slope_{metric}_{rng}_by_dim.pdf", subdir="dumbbell")


def plot_all_dumbbells(master):
    log.info("  Generando dumbbell / slope plots...")
    for metric in ["RMSE", "MAE"]:
        for rng in ["ENTIRE", "TBR_2", "TBR_1", "TIR"]:
            plot_dumbbell(master, metric=metric, rng=rng)
            plot_slope_by_dimension(master, metric=metric, rng=rng)
    log.info(f"  Dumbbell plots: {2*4*2} figuras generadas.")