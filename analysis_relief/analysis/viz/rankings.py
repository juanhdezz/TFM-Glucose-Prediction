# -*- coding: utf-8 -*-
"""
rankings.py — Critical Difference diagram, bump chart, Borda ranking plot.

F8   cd_diagram_{metric}_{rng}_{ds}.pdf   CD diagram por dataset
F9   bump_chart_{metric}.pdf              Cómo varía el ranking según la métrica
F10  borda_lollipop.pdf                   Borda score global por técnica × dimensión
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import scipy.stats as ss

from analysis.config import (
    DATASET_ORDER, DATASET_LABELS, TECHNIQUE_ORDER, TECHNIQUE_LABELS,
    RANGE_ORDER, RANGE_LABELS, ANALYSIS_METRICS, LOWER_IS_BETTER,
    ALPHA, PALETTE, DIMENSION_PALETTE, DIMENSION_LABELS,
)
from analysis.viz.style import apply_theme, save_fig, technique_color, technique_label

log = logging.getLogger(__name__)
apply_theme()


def _cond_label(cond):
    if cond == "original": return "Original"
    parts = cond.replace("balanced_","").split("_",1)
    dim  = DIMENSION_LABELS.get(parts[0], parts[0])
    tech = TECHNIQUE_LABELS.get(parts[1] if len(parts)>1 else cond, cond)
    return f"{dim}·{tech}"


# ── F8: Critical Difference Diagram (Demšar 2006) ────────────────────────────

def _average_ranks(pivot: pd.DataFrame, ascending=True) -> pd.Series:
    """Calcula ranking promedio por condición (Friedman average rank)."""
    ranks = pivot.rank(axis=1, ascending=ascending, method="average")
    return ranks.mean(axis=0).sort_values()

def _critical_difference(n_classifiers, n_datasets, alpha=0.05):
    """Valor crítico para el CD diagram (Nemenyi test)."""
    # Tabla de valores q_alpha (Student range / sqrt(2)) para alpha=0.05
    q_table = {
        2:1.960, 3:2.344, 4:2.569, 5:2.728, 6:2.850,
        7:2.949, 8:3.031, 9:3.102, 10:3.164, 12:3.268, 14:3.376
    }
    k = min(n_classifiers, 14)
    for key in sorted(q_table.keys(), reverse=True):
        if k >= key:
            q = q_table[key]; break
    else:
        q = 1.960
    return q * np.sqrt(n_classifiers * (n_classifiers + 1) / (6 * n_datasets))

def plot_cd_diagram(fold_long, metric="RMSE", rng="ENTIRE", dataset=None):
    """
    CD diagram estándar (Demšar 2006).
    Condiciones con diferencia < CD se conectan con una barra gruesa (no significativas).
    Eje X: ranking promedio. Mejor = izquierda (rank 1).
    """
    sub = fold_long[
        (fold_long["metric"] == metric) &
        (fold_long["range"]  == rng)
    ].copy()
    if dataset:
        sub  = sub[sub["dataset"] == dataset]
        label = DATASET_LABELS.get(dataset, dataset)
    else:
        label = "All Datasets"
    if sub.empty:
        return

    pivot = sub.pivot_table(index="fold", columns="condition",
                            values="value", aggfunc="mean")
    pivot.dropna(axis=1, inplace=True)
    if pivot.shape[1] < 3 or pivot.shape[0] < 2:
        return

    asc    = metric in LOWER_IS_BETTER
    avg_r  = _average_ranks(pivot, ascending=asc)
    cd     = _critical_difference(len(avg_r), len(pivot))
    conds  = avg_r.index.tolist()
    n      = len(conds)

    # ---- layout ----
    fig_h = max(3.5, n * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(-1, n * 0.5 + 1.5)

    # horizontal axis at top
    ax.axhline(n * 0.5 + 0.8, color="black", lw=1.5, xmin=0, xmax=1)
    ax.set_xticks(np.arange(1, n+1))
    ax.set_xticklabels([f"{r:.1f}" for r in np.arange(1, n+1)], fontsize=9)
    ax.xaxis.tick_top()
    ax.set_xlabel("Average rank  (1 = best)", fontsize=10, labelpad=8)
    ax.xaxis.set_label_position("top")
    ax.yaxis.set_visible(False)
    ax.spines[["left","right","bottom"]].set_visible(False)

    # CD bracket at top-left
    cd_y = n * 0.5 + 0.3
    ax.annotate("", xy=(1 + cd, cd_y), xytext=(1, cd_y),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(1 + cd/2, cd_y + 0.18, f"CD = {cd:.2f}", ha="center", fontsize=8)

    # Tick marks on axis
    for r in avg_r.values:
        ax.plot([r, r], [n*0.5+0.8, n*0.5+0.7], color="black", lw=1)

    # Conditions on alternating sides
    left_conds  = conds[:n//2]
    right_conds = conds[n//2:]

    def _y(i, side): return (n//2 - i) * 0.45 if side == "left" else (i - n//2) * 0.45

    positions = {}
    for i, c in enumerate(left_conds):
        rank = avg_r[c]
        y    = _y(i, "left")
        color = technique_color(c.replace("balanced_age_","").replace("balanced_sex_","")
                                if c != "original" else "original")
        ax.plot([rank, rank], [n*0.5+0.8, y+0.05], color=color, lw=1.2, ls=":")
        ax.plot([0.5, rank], [y, y], color=color, lw=1.5)
        ax.scatter([rank], [y], color=color, s=50, zorder=5)
        ax.text(0.45, y, _cond_label(c), ha="right", va="center", fontsize=8)
        positions[c] = (rank, y)

    for i, c in enumerate(right_conds):
        rank = avg_r[c]
        y    = _y(i, "right")
        color = technique_color(c.replace("balanced_age_","").replace("balanced_sex_","")
                                if c != "original" else "original")
        ax.plot([rank, rank], [n*0.5+0.8, y+0.05], color=color, lw=1.2, ls=":")
        ax.plot([rank, n+0.5], [y, y], color=color, lw=1.5)
        ax.scatter([rank], [y], color=color, s=50, zorder=5)
        ax.text(n+0.55, y, _cond_label(c), ha="left", va="center", fontsize=8)
        positions[c] = (rank, y)

    # Non-significant cliques (rank difference < CD)
    ax_bottom = -0.8
    clique_y  = ax_bottom
    drawn     = set()
    for i, c1 in enumerate(conds):
        clique = [c1]
        for c2 in conds[i+1:]:
            if abs(avg_r[c1] - avg_r[c2]) < cd:
                clique.append(c2)
        if len(clique) > 1:
            key = tuple(sorted(clique))
            if key not in drawn:
                x_start = avg_r[clique[0]]
                x_end   = avg_r[clique[-1]]
                ax.plot([x_start, x_end], [clique_y, clique_y],
                        color="black", lw=4, solid_capstyle="round", alpha=0.7)
                clique_y -= 0.25
                drawn.add(key)

    ds_sfx = f"_{dataset}" if dataset else "_global"
    ax.set_title(
        f"Critical Difference Diagram — {metric} · {RANGE_LABELS.get(rng,rng)}  |  {label}\n"
        f"Connected groups: no significant difference (Nemenyi, α={ALPHA})",
        fontsize=10, fontweight="bold", pad=20
    )
    save_fig(fig, f"cd_diagram_{metric}_{rng}{ds_sfx}.pdf", subdir="rankings")


# ── F9: Bump chart ────────────────────────────────────────────────────────────

def plot_bump_chart(master, metric_list=None, rng="ENTIRE"):
    """
    Bump chart: cómo cambia el ranking de cada condición según la métrica usada.
    Eje X: métrica. Eje Y: posición. Líneas coloreadas por técnica.
    """
    if metric_list is None:
        metric_list = ["RMSE", "MAE"]  # expandir cuando haya más métricas

    sub = master[
        (master["range"] == rng) &
        (master["metric"].isin(metric_list))
    ].copy()
    if sub.empty:
        return

    datasets = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
    n_ds     = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(n_ds * 5, 7), sharey=True)
    if n_ds == 1: axes = [axes]

    for ax, ds in zip(axes, datasets):
        ds_data = sub[sub["dataset"] == ds]
        # Rank per metric
        rank_dfs = []
        for m in metric_list:
            mdata = ds_data[ds_data["metric"] == m].copy()
            mdata = mdata.sort_values("mean", ascending=m in LOWER_IS_BETTER)
            mdata["rank"]   = range(1, len(mdata)+1)
            mdata["metric_label"] = m
            rank_dfs.append(mdata[["condition","technique","dimension","rank","metric_label"]])

        if not rank_dfs:
            continue
        df_r = pd.concat(rank_dfs)
        n_conds = df_r["rank"].max()

        for cond in df_r["condition"].unique():
            cdata  = df_r[df_r["condition"] == cond].set_index("metric_label")
            tech   = cdata["technique"].iloc[0]
            dim    = cdata["dimension"].iloc[0] if pd.notna(cdata["dimension"].iloc[0]) else None
            color  = technique_color(tech)
            ls     = "-" if dim == "age" else "--" if dim == "sex" else "-."
            ranks  = [cdata.loc[m,"rank"] if m in cdata.index else np.nan for m in metric_list]

            ax.plot(range(len(metric_list)), ranks, color=color,
                    lw=2, ls=ls, alpha=0.85, zorder=2)
            ax.scatter(range(len(metric_list)), ranks, color=color,
                       s=60, zorder=3, edgecolors="white", lw=0.8)

        ax.set_xticks(range(len(metric_list)))
        ax.set_xticklabels(metric_list, fontsize=11, fontweight="bold")
        ax.set_yticks(range(1, int(n_conds)+1))
        ax.set_yticklabels([f"#{i}" for i in range(1, int(n_conds)+1)], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=12, fontweight="bold")
        ax.set_ylabel("Ranking position (1 = best)", fontsize=10) if ds == datasets[0] else None

    # Legend
    seen = set()
    handles = []
    for _, row in master[~master["is_original"]].iterrows():
        t = row["technique"]
        if t not in seen:
            handles.append(Line2D([0],[0], color=technique_color(t), lw=2, label=technique_label(t)))
            seen.add(t)
    handles += [
        Line2D([0],[0], color="0.5", lw=2, ls="-",  label="Age"),
        Line2D([0],[0], color="0.5", lw=2, ls="--", label="Sex"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5),
               bbox_to_anchor=(0.5, -0.05), fontsize=9, frameon=True)

    range_lbl = RANGE_LABELS.get(rng, rng)
    fig.suptitle(f"Bump chart — Ranking stability across metrics  |  {range_lbl}",
                 fontsize=12, fontweight="bold", y=1.01)
    save_fig(fig, f"bump_chart_{rng}.pdf", subdir="rankings")


# ── F10: Borda lollipop ───────────────────────────────────────────────────────

def plot_borda_lollipop(borda_df: pd.DataFrame):
    """
    Lollipop chart del score de Borda por (técnica × dimensión).
    Menor score = mejor posición acumulada. Un panel por dataset.
    """
    if borda_df is None or borda_df.empty:
        log.warning("borda_df vacío — skip lollipop")
        return

    datasets = [d for d in DATASET_ORDER if d in borda_df["dataset"].unique()]
    n_ds     = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(n_ds * 5, 6), sharey=False)
    if n_ds == 1: axes = [axes]

    for ax, ds in zip(axes, datasets):
        ds_data = borda_df[borda_df["dataset"] == ds].copy()
        ds_data = ds_data.sort_values("borda_score", ascending=True)
        n = len(ds_data)

        for i, (_, row) in enumerate(ds_data.iterrows()):
            tech  = row["technique"]
            dim   = row["dimension"] if pd.notna(row.get("dimension")) else None
            score = row["borda_score"]
            color = technique_color(tech)

            # Lollipop stem
            ax.plot([0, score], [i, i], color=color, lw=2, alpha=0.7, zorder=1)
            # Head
            marker = "o" if dim == "age" else "s" if dim == "sex" else "D"
            ax.scatter([score], [i], color=color, s=90, marker=marker,
                       zorder=3, edgecolors="white", lw=0.8)
            # Label
            cond = row.get("condition","")
            ax.text(score + 0.5, i, _cond_label(cond), va="center", fontsize=8)

        ax.set_yticks([])
        ax.set_xlabel("Borda score (lower = better)", fontsize=10)
        ax.set_title(f"{DATASET_LABELS.get(ds,ds)}", fontsize=12, fontweight="bold")
        ax.axvline(ds_data["borda_score"].min(), color="0.7", lw=1, ls="--")

    # legend
    handles = [
        Line2D([0],[0], marker="o", color="0.4", linestyle="None", markersize=7, label="Age"),
        Line2D([0],[0], marker="s", color="0.4", linestyle="None", markersize=7, label="Sex"),
        Line2D([0],[0], marker="D", color="0.2", linestyle="None", markersize=7, label="Original"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), fontsize=9, frameon=True)

    fig.suptitle("Borda ranking — Aggregate performance across metrics and glucose ranges",
                 fontsize=12, fontweight="bold", y=1.01)
    save_fig(fig, "borda_lollipop.pdf", subdir="rankings")


def plot_all_rankings(master, fold_long, borda_df=None):
    log.info("  Generando rankings...")
    for ds in [d for d in DATASET_ORDER if d in fold_long["dataset"].unique()]:
        plot_cd_diagram(fold_long, metric="RMSE", rng="ENTIRE", dataset=ds)
        plot_cd_diagram(fold_long, metric="RMSE", rng="TBR_2",  dataset=ds)
    plot_bump_chart(master, metric_list=["RMSE","MAE"], rng="ENTIRE")
    plot_bump_chart(master, metric_list=["RMSE","MAE"], rng="TBR_1")
    if borda_df is not None and not borda_df.empty:
        plot_borda_lollipop(borda_df)
    log.info("  Rankings completados.")