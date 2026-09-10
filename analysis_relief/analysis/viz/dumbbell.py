# -*- coding: utf-8 -*-
"""
dumbbell.py — Dumbbell plots y slope charts.

F3  dumbbell_{metric}_{rng}.pdf    baseline vs mejor técnica, por dataset
F4  slope_{metric}_by_dim.pdf      pendiente original→balanceado por dimensión

CAMBIOS v3:
  - Aumentada altura de figura para evitar solapamiento de etiquetas Y
  - Etiquetas del eje Y con ha='right' y mayor separación
  - Abreviación automática de técnicas largas (>25 caracteres)
  - Ajuste de márgenes izquierdo para dar espacio a etiquetas
  - Leyenda a ancho completo con fuente más grande
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


def _abbreviate_label(label, max_len=28):
    """Abrevia una etiqueta si supera max_len caracteres."""
    if len(label) <= max_len:
        return label
    # Intentar abreviar de forma inteligente
    parts = label.split(" · ")
    if len(parts) == 2:
        dim, tech = parts
        # Si la técnica es muy larga, abreviar
        if len(tech) > 20:
            tech = tech[:18] + "…"
        return f"{dim} · {tech}"
    return label[:max_len-1] + "…"


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

    # Calcular altura dinámica: más técnicas = más altura
    max_techniques = 0
    for ds in datasets:
        ds_data = sub[sub["dataset"] == ds]
        n_tech = len(ds_data[~ds_data["is_original"]])
        if n_tech > max_techniques:
            max_techniques = n_tech
    
    # Altura base: 1.2 unidades por técnica + margen
    fig_height = max(8, max_techniques * 0.55 + 2.5)
    fig_width = n_ds * 6.0

    fig, axes = plt.subplots(1, n_ds, figsize=(fig_width, fig_height), sharey=False)
    if n_ds == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        ds_data    = sub[sub["dataset"] == ds]
        orig_val   = ds_data[ds_data["is_original"]]["mean"].values
        if len(orig_val) == 0:
            continue
        orig_val   = orig_val[0]

        bal = ds_data[~ds_data["is_original"]].copy()
        bal = bal[bal["mean"].notna()]
        if bal.empty:
            continue
            
        bal = bal.sort_values("mean", ascending=not ascending)

        y_pos = np.arange(len(bal))

        for y, (_, row) in zip(y_pos, bal.iterrows()):
            color = technique_color(row["technique"])
            # Connecting line
            ax.plot([orig_val, row["mean"]], [y, y],
                    color="0.75", lw=1.5, zorder=1)
            # Baseline dot
            ax.scatter(orig_val, y, color="#333333", s=50, zorder=3,
                       marker="D", linewidths=0)
            # Technique dot
            ax.scatter(row["mean"], y, color=color, s=80, zorder=4,
                       edgecolors="white", linewidths=0.8)
            # Std error bar
            if not np.isnan(row.get("std", np.nan)):
                ax.errorbar(row["mean"], y, xerr=row["std"],
                            fmt="none", ecolor=color, elinewidth=1.2,
                            capsize=3, zorder=2, alpha=0.7)

        # Baseline vertical line
        ax.axvline(orig_val, color="#333333", lw=1.2, ls="--", alpha=0.6, zorder=0)

        # --- ETIQUETAS DEL EJE Y MEJORADAS ---
        cond_labels = []
        for _, row in bal.iterrows():
            dim  = DIMENSION_LABELS.get(row["dimension"], "")
            tech = TECHNIQUE_LABELS.get(row["technique"], row["technique"])
            label = f"{dim} · {tech}" if dim else tech
            label = _abbreviate_label(label, max_len=30)
            cond_labels.append(label)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(cond_labels, fontsize=9, ha='right', va='center')
        ax.tick_params(axis='y', pad=8)  # más separación entre etiqueta y eje
        
        # Ajustar margen izquierdo para que las etiquetas no se corten
        ax.set_xlabel(f"{metric} (mg/dL)", fontsize=11)
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)

        # Shade improvement direction
        xlim = ax.get_xlim()
        if ascending:  # lower is better → shade left of baseline
            ax.axvspan(xlim[0], orig_val, alpha=0.04, color="#009E73")
        else:
            ax.axvspan(orig_val, xlim[1], alpha=0.04, color="#009E73")
        ax.set_xlim(xlim)
        
        # Margen superior e inferior para que las etiquetas no se corten
        ax.set_ylim(-0.8, len(bal) - 0.2)

    # --- LEYENDA A ANCHO COMPLETO ---
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

    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5),
               bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=10,
               handlelength=2.0, handleheight=1.5)

    range_lbl = RANGE_LABELS.get(rng, rng)
    fig.suptitle(f"Dumbbell plot — {metric} · {range_lbl}  |  ◆ = Original baseline",
                 fontsize=14, fontweight="bold", y=1.01)

    # Ajuste final con más espacio a la izquierda para etiquetas Y
    plt.tight_layout(rect=[0.03, 0.04, 1, 0.98])
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

        annotated_orig = set()
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

                if ds not in annotated_orig:
                    ax.text(
                        x_orig - 0.04, ds_orig,
                        DATASET_LABELS.get(ds, ds),
                        fontsize=8, color="0.35",
                        ha="right", va="center",
                        fontstyle="italic",
                    )
                    annotated_orig.add(ds)

        ax.set_xticks([x_orig, x_bal])
        ax.set_xticklabels(["Original", "Balanced"], fontsize=11, fontweight="bold")
        ax.set_ylabel(f"{metric} (mg/dL)", fontsize=11) if dim == "age" else None
        ax.set_title(f"Dimension: {DIMENSION_LABELS.get(dim, dim)}",
                     fontsize=12, fontweight="bold")
        ax.set_xlim(-0.55, 1.3)
        ax.grid(axis="y", alpha=0.3)

        arrow_dir = "↓ better" if metric in LOWER_IS_BETTER else "↑ better"
        ax.text(1.25, ax.get_ylim()[0], arrow_dir,
                fontsize=9, color="0.5", va="bottom", ha="right")

    # Leyenda
    handles = [mlines.Line2D([], [], color=technique_color(t), lw=2,
                              label=technique_label(t))
               for t in techniques]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
               bbox_to_anchor=(0.5, -0.05), frameon=True, fontsize=10)

    range_lbl = RANGE_LABELS.get(rng, rng)
    fig.suptitle(f"Slope chart — {metric} · {range_lbl}",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(fig, f"slope_{metric}_{rng}_by_dim.pdf", subdir="dumbbell")


def plot_all_dumbbells(master):
    log.info("  Generando dumbbell / slope plots...")
    for metric in ["RMSE", "MAE"]:
        for rng in ["ENTIRE", "TBR_2", "TBR_1", "TIR"]:
            plot_dumbbell(master, metric=metric, rng=rng)
            plot_slope_by_dimension(master, metric=metric, rng=rng)
    log.info(f"  Dumbbell plots: {2*4*2} figuras generadas.")