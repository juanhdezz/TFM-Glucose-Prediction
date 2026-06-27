# -*- coding: utf-8 -*-
"""
distributions.py — Distribuciones inter-fold y consistencia.

F5  strip_fold_{metric}_{ds}.pdf     strip/dot plot de RMSE por fold × condición
F6  violin_fold_{metric}.pdf         violin plots por dataset (3-panel)
F7  consistency_matrix.pdf           correlación de Spearman entre rankings por fold
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from analysis.config import (
    DATASET_ORDER, DATASET_LABELS, TECHNIQUE_ORDER, TECHNIQUE_LABELS,
    RANGE_ORDER, RANGE_LABELS, LOWER_IS_BETTER, PALETTE,
    DIMENSION_LABELS, N_FOLDS,
)
from analysis.viz.style import apply_theme, save_fig, technique_color, technique_label

log = logging.getLogger(__name__)
apply_theme()

FOLD_MARKERS = {1: "o", 2: "s", 3: "^", 4: "D", 5: "v"}
FOLD_COLORS  = {1:"#1a1a1a", 2:"#444", 3:"#777", 4:"#aaa", 5:"#ccc"}


def _cond_label(cond):
    if cond == "original":
        return "Original"
    parts = cond.replace("balanced_", "").split("_", 1)
    dim  = DIMENSION_LABELS.get(parts[0], parts[0])
    tech = TECHNIQUE_LABELS.get(parts[1] if len(parts)>1 else cond, cond)
    return f"{dim} · {tech}"

def _sort_conds(conds):
    def _k(c):
        if c == "original": return (0,0,0)
        parts = c.replace("balanced_","").split("_",1)
        dim = parts[0]; tech = parts[1] if len(parts)>1 else ""
        return (1, {"age":0,"sex":1}.get(dim,2),
                TECHNIQUE_ORDER.index(tech) if tech in TECHNIQUE_ORDER else 99)
    return sorted(conds, key=_k)


# ── F5: Strip plot per dataset ────────────────────────────────────────────────

def plot_strip_fold(fold_long, metric="RMSE", rng="ENTIRE", dataset=None):
    """
    Dot/strip plot: una fila por condición, puntos = folds individuales.
    La media se marca con una línea horizontal.
    Permite detectar si una mejora es consistente o artefacto de un fold.
    """
    sub = fold_long[
        (fold_long["metric"] == metric) &
        (fold_long["range"]  == rng)
    ].copy()
    if dataset:
        sub = sub[sub["dataset"] == dataset]
    if sub.empty:
        return

    datasets = [dataset] if dataset else [d for d in DATASET_ORDER if d in sub["dataset"].unique()]

    for ds in datasets:
        ds_data = sub[sub["dataset"] == ds]
        conds   = _sort_conds(ds_data["condition"].unique().tolist())
        n_conds = len(conds)

        fig, ax = plt.subplots(figsize=(8, max(4, n_conds * 0.55 + 1.5)))
        y_pos   = {c: i for i, c in enumerate(conds)}

        for cond in conds:
            cdata = ds_data[ds_data["condition"] == cond]
            y     = y_pos[cond]
            color = technique_color(cond.replace("balanced_age_","").replace("balanced_sex_","")
                                    if cond != "original" else "original")

            # Mean line
            mean_val = cdata["value"].mean()
            ax.plot([mean_val - 0.5, mean_val + 0.5] if False else
                    [mean_val], [y], marker="|", color=color,
                    markersize=18, markeredgewidth=2.5, zorder=3)

            # Individual fold dots with jitter
            jitter = np.linspace(-0.18, 0.18, len(cdata))
            for (_, row), jit in zip(cdata.iterrows(), jitter):
                fold_n = int(row["fold"]) if not np.isnan(row.get("fold", np.nan)) else 1
                ax.scatter(row["value"], y + jit,
                           marker=FOLD_MARKERS.get(fold_n, "o"),
                           color=color, s=42, zorder=4,
                           edgecolors="white", linewidths=0.6, alpha=0.9)

            # Std range shading
            std_val = cdata["value"].std()
            ax.barh(y, std_val * 2, left=mean_val - std_val,
                    height=0.35, color=color, alpha=0.12, zorder=1)

        ax.set_yticks(range(n_conds))
        ax.set_yticklabels([_cond_label(c) for c in conds], fontsize=8.5)
        ax.set_xlabel(f"{metric} (mg/dL)", fontsize=10)
        ax.set_title(
            f"Fold consistency — {DATASET_LABELS.get(ds,ds)} | {metric} · {RANGE_LABELS.get(rng,rng)}",
            fontsize=11, fontweight="bold"
        )

        # Fold legend
        fold_handles = [
            Line2D([0],[0], marker=FOLD_MARKERS[f], color="0.4", linestyle="None",
                   markersize=6, label=f"Fold {f}")
            for f in range(1, N_FOLDS+1)
        ]
        fold_handles.append(
            Line2D([0],[0], marker="|", color="0.4", markersize=10,
                   markeredgewidth=2, linestyle="None", label="Mean")
        )
        ax.legend(handles=fold_handles, loc="lower right", fontsize=8,
                  ncol=3, framealpha=0.9)

        ax.grid(axis="x", alpha=0.3)
        save_fig(fig, f"strip_fold_{metric}_{rng}_{ds}.pdf", subdir="distributions")


# ── F6: Violin plots ──────────────────────────────────────────────────────────

def plot_violin_fold(fold_long, metric="RMSE", rng="ENTIRE"):
    """
    Violin plots: distribución de la métrica a través de los 5 folds.
    3 paneles (un panel por dataset). Violines coloreados por técnica.
    """
    sub = fold_long[
        (fold_long["metric"] == metric) &
        (fold_long["range"]  == rng)
    ].copy()
    if sub.empty:
        return

    datasets = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
    n_ds     = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(n_ds * 5.5, 6), sharey=False)
    if n_ds == 1: axes = [axes]

    for ax, ds in zip(axes, datasets):
        ds_data = sub[sub["dataset"] == ds]
        conds   = _sort_conds(ds_data["condition"].unique().tolist())
        n_c     = len(conds)

        for i, cond in enumerate(conds):
            cdata  = ds_data[ds_data["condition"] == cond]["value"].dropna().values
            if len(cdata) < 2:
                continue
            color  = technique_color(
                cond.replace("balanced_age_","").replace("balanced_sex_","")
                if cond != "original" else "original"
            )
            vp = ax.violinplot([cdata], positions=[i], widths=0.7,
                               showmeans=False, showmedians=True, showextrema=True)
            for pc in vp["bodies"]:
                pc.set_facecolor(color); pc.set_alpha(0.6)
            for part in ("cmedians","cmins","cmaxes","cbars"):
                if part in vp:
                    vp[part].set_edgecolor(color); vp[part].set_linewidth(1.5)
            # mean dot
            ax.scatter([i], [cdata.mean()], color=color, s=50, zorder=5,
                       edgecolors="white", lw=1)

        ax.set_xticks(range(n_c))
        ax.set_xticklabels([_cond_label(c) for c in conds],
                           rotation=40, ha="right", fontsize=7.5)
        ax.set_ylabel(f"{metric} (mg/dL)", fontsize=10) if ds == datasets[0] else None
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    range_lbl = RANGE_LABELS.get(rng, rng)
    fig.suptitle(f"Distribution across folds — {metric} · {range_lbl}  |  ● = mean, — = median",
                 fontsize=12, fontweight="bold", y=1.01)
    save_fig(fig, f"violin_fold_{metric}_{rng}.pdf", subdir="distributions")


# ── F7: Consistency matrix (Spearman rank correlation between folds) ──────────

def plot_fold_consistency(fold_long, metric="RMSE", rng="ENTIRE"):
    """
    Matriz de correlación de Spearman entre los rankings de folds.
    Responde: ¿la técnica que gana en Fold1 también gana en Fold5?
    Alta correlación = el ranking es estable.
    """
    from scipy.stats import spearmanr
    sub = fold_long[
        (fold_long["metric"] == metric) &
        (fold_long["range"]  == rng)
    ].copy()
    if sub.empty:
        return

    datasets = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
    n_ds     = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(n_ds * 4.5, 4.5))
    if n_ds == 1: axes = [axes]

    for ax, ds in zip(axes, datasets):
        ds_data = sub[sub["dataset"] == ds]
        folds   = sorted(ds_data["fold"].dropna().unique())
        n_f     = len(folds)

        # Rankings per fold: rank conditions by metric value
        corr_matrix = np.full((n_f, n_f), np.nan)
        fold_ranks  = {}
        for f in folds:
            fd = ds_data[ds_data["fold"] == f]
            ranked = fd.sort_values("value", ascending=metric in LOWER_IS_BETTER)
            fold_ranks[f] = {c: r+1 for r, c in enumerate(ranked["condition"])}

        conds_all = list(fold_ranks[folds[0]].keys()) if folds else []
        for i, f1 in enumerate(folds):
            for j, f2 in enumerate(folds):
                v1 = [fold_ranks[f1].get(c, np.nan) for c in conds_all]
                v2 = [fold_ranks[f2].get(c, np.nan) for c in conds_all]
                if any(np.isnan(v1)) or any(np.isnan(v2)):
                    continue
                corr_matrix[i, j], _ = spearmanr(v1, v2)

        im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        for i in range(n_f):
            for j in range(n_f):
                v = corr_matrix[i,j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color="black" if abs(v) < 0.7 else "white")

        fold_labels = [f"Fold {int(f)}" for f in folds]
        ax.set_xticks(range(n_f)); ax.set_xticklabels(fold_labels, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(n_f)); ax.set_yticklabels(fold_labels, fontsize=9)
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=11, fontweight="bold")

        divider = __import__("mpl_toolkits.axes_grid1", fromlist=["make_axes_locatable"]).make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        fig.colorbar(im, cax=cax).set_label("Spearman ρ", fontsize=8)

    range_lbl = RANGE_LABELS.get(rng, rng)
    fig.suptitle(f"Ranking consistency across folds (Spearman ρ) — {metric} · {range_lbl}\n"
                 f"ρ = 1: identical ranking, ρ ≈ 0: no agreement",
                 fontsize=11, fontweight="bold", y=1.03)
    save_fig(fig, f"consistency_folds_{metric}_{rng}.pdf", subdir="distributions")


def plot_all_distributions(fold_long):
    log.info("  Generando distribuciones...")
    for metric in ["RMSE", "MAE"]:
        for rng in ["ENTIRE", "TBR_2", "TIR"]:
            plot_strip_fold(fold_long, metric=metric, rng=rng)
            plot_violin_fold(fold_long, metric=metric, rng=rng)
    plot_fold_consistency(fold_long, metric="RMSE", rng="ENTIRE")
    log.info("  Distribuciones completadas.")