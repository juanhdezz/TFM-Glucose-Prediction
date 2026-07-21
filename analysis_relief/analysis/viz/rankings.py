# -*- coding: utf-8 -*-
"""
rankings.py — Critical Difference diagram, bump chart, Borda ranking plot.

Versiones global e intra‑familia completamente implementadas.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.stats as ss

from analysis.config import (
    DATASET_ORDER, DATASET_LABELS, TECHNIQUE_ORDER, TECHNIQUE_LABELS,
    RANGE_ORDER, RANGE_LABELS, ANALYSIS_METRICS, LOWER_IS_BETTER,
    ALPHA, DIMENSION_LABELS,
    FAMILY_SIZE_LABELS, FAMILY_MECHANISM_LABELS,
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


# ── Helpers para CD ─────────────────────────────────────────────────────────

def _average_ranks(pivot, ascending=True):
    ranks = pivot.rank(axis=1, ascending=ascending, method="average")
    return ranks.mean(axis=0).sort_values()

def _critical_difference(n_classifiers, n_datasets, alpha=0.05):
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


# ── F8: CD Diagram ─────────────────────────────────────────────────────────

def plot_cd_diagram(fold_long, metric="RMSE", rng="ENTIRE", dataset=None):
    sub = fold_long[(fold_long["metric"] == metric) & (fold_long["range"] == rng)].copy()
    if dataset:
        sub = sub[sub["dataset"] == dataset]
        label = DATASET_LABELS.get(dataset, dataset)
    else:
        label = "All Datasets"
    if sub.empty:
        return

    pivot = sub.pivot_table(index="fold", columns="condition", values="value", aggfunc="mean")
    pivot.dropna(axis=1, inplace=True)
    if pivot.shape[1] < 3 or pivot.shape[0] < 2:
        return

    asc    = metric in LOWER_IS_BETTER
    avg_r  = _average_ranks(pivot, ascending=asc)
    cd     = _critical_difference(len(avg_r), len(pivot))
    conds  = avg_r.index.tolist()
    n      = len(conds)

    fig_h = max(3.5, n * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(-1, n * 0.5 + 1.5)

    ax.axhline(n * 0.5 + 0.8, color="black", lw=1.5, xmin=0, xmax=1)
    ax.set_xticks(np.arange(1, n+1))
    ax.set_xticklabels([f"{r:.1f}" for r in np.arange(1, n+1)], fontsize=9)
    ax.xaxis.tick_top()
    ax.set_xlabel("Average rank  (1 = best)", fontsize=10, labelpad=8)
    ax.xaxis.set_label_position("top")
    ax.yaxis.set_visible(False)
    ax.spines[["left","right","bottom"]].set_visible(False)

    cd_y = n * 0.5 + 0.3
    ax.annotate("", xy=(1 + cd, cd_y), xytext=(1, cd_y),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(1 + cd/2, cd_y + 0.18, f"CD = {cd:.2f}", ha="center", fontsize=8)

    for r in avg_r.values:
        ax.plot([r, r], [n*0.5+0.8, n*0.5+0.7], color="black", lw=1)

    left_conds  = conds[:n//2]
    right_conds = conds[n//2:]

    def _y(i, side): return (n//2 - i) * 0.45 if side == "left" else (i - n//2) * 0.45

    for i, c in enumerate(left_conds):
        rank = avg_r[c]
        y    = _y(i, "left")
        color = technique_color(c.replace("balanced_age_","").replace("balanced_sex_","") if c != "original" else "original")
        ax.plot([rank, rank], [n*0.5+0.8, y+0.05], color=color, lw=1.2, ls=":")
        ax.plot([0.5, rank], [y, y], color=color, lw=1.5)
        ax.scatter([rank], [y], color=color, s=50, zorder=5)
        ax.text(0.45, y, _cond_label(c), ha="right", va="center", fontsize=8)

    for i, c in enumerate(right_conds):
        rank = avg_r[c]
        y    = _y(i, "right")
        color = technique_color(c.replace("balanced_age_","").replace("balanced_sex_","") if c != "original" else "original")
        ax.plot([rank, rank], [n*0.5+0.8, y+0.05], color=color, lw=1.2, ls=":")
        ax.plot([rank, n+0.5], [y, y], color=color, lw=1.5)
        ax.scatter([rank], [y], color=color, s=50, zorder=5)
        ax.text(n+0.55, y, _cond_label(c), ha="left", va="center", fontsize=8)

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
    plt.close(fig)


# ── F9: Bump chart ─────────────────────────────────────────────────────────

def _draw_bump_chart(ax, master, metric_list, rng):
    sub = master[(master["range"] == rng) & (master["metric"].isin(metric_list))].copy()
    datasets = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
    for ds in datasets:
        ds_data = sub[sub["dataset"] == ds]
        rank_dfs = []
        for m in metric_list:
            mdata = ds_data[ds_data["metric"] == m].copy()
            mdata = mdata.sort_values("mean", ascending=m in LOWER_IS_BETTER)
            mdata["rank"] = range(1, len(mdata)+1)
            mdata["metric_label"] = m
            rank_dfs.append(mdata[["condition","technique","dimension","rank","metric_label"]])
        if not rank_dfs:
            continue
        df_r = pd.concat(rank_dfs)
        n_conds = df_r["rank"].max()
        for cond in df_r["condition"].unique():
            cdata = df_r[df_r["condition"] == cond].set_index("metric_label")
            tech  = cdata["technique"].iloc[0]
            dim   = cdata["dimension"].iloc[0] if pd.notna(cdata["dimension"].iloc[0]) else None
            color = technique_color(tech)
            ls    = "-" if dim == "age" else "--" if dim == "sex" else "-."
            ranks = [cdata.loc[m,"rank"] if m in cdata.index else np.nan for m in metric_list]
            ax.plot(range(len(metric_list)), ranks, color=color, lw=2, ls=ls, alpha=0.85)
            ax.scatter(range(len(metric_list)), ranks, color=color, s=60, edgecolors="white", lw=0.8)
        ax.set_xticks(range(len(metric_list)))
        ax.set_xticklabels(metric_list, fontsize=11, fontweight="bold")
        ax.set_yticks(range(1, int(n_conds)+1))
        ax.set_yticklabels([f"#{i}" for i in range(1, int(n_conds)+1)], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=12, fontweight="bold")
        ax.set_ylabel("Ranking position (1 = best)")


def plot_bump_chart(master, metric_list=None, rng="ENTIRE"):
    if metric_list is None:
        metric_list = ["RMSE", "MAE"]
    datasets = [d for d in DATASET_ORDER if d in master["dataset"].unique()]
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(n_ds * 5, 7), sharey=True)
    if n_ds == 1: axes = [axes]
    for ax, ds in zip(axes, datasets):
        _draw_bump_chart(ax, master[master["dataset"] == ds], metric_list, rng)
    # leyenda
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
    fig.suptitle(f"Bump chart — Ranking stability across metrics  |  {RANGE_LABELS.get(rng, rng)}",
                 fontsize=12, fontweight="bold", y=1.01)
    save_fig(fig, f"bump_chart_{rng}.pdf", subdir="rankings")
    plt.close(fig)


# ── F10: Borda lollipop ────────────────────────────────────────────────────

def _draw_borda_lollipop(ax, borda_df, ds):
    ds_data = borda_df[borda_df["dataset"] == ds].copy()
    ds_data = ds_data.sort_values("borda_score", ascending=True)
    for i, (_, row) in enumerate(ds_data.iterrows()):
        tech  = row["technique"]
        dim   = row["dimension"] if pd.notna(row.get("dimension")) else None
        score = row["borda_score"]
        color = technique_color(tech)
        ax.plot([0, score], [i, i], color=color, lw=2, alpha=0.7)
        marker = "o" if dim == "age" else "s" if dim == "sex" else "D"
        ax.scatter([score], [i], color=color, s=90, marker=marker, edgecolors="white", lw=0.8)
        ax.text(score + 0.5, i, _cond_label(row["condition"]), va="center", fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Borda score (lower = better)", fontsize=10)
    ax.set_title(f"{DATASET_LABELS.get(ds,ds)}", fontsize=12, fontweight="bold")
    ax.axvline(ds_data["borda_score"].min(), color="0.7", lw=1, ls="--")


def plot_borda_lollipop(borda_df: pd.DataFrame):
    if borda_df is None or borda_df.empty:
        log.warning("borda_df vacío — skip lollipop")
        return
    datasets = [d for d in DATASET_ORDER if d in borda_df["dataset"].unique()]
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(n_ds * 5, 6), sharey=False)
    if n_ds == 1: axes = [axes]
    for ax, ds in zip(axes, datasets):
        _draw_borda_lollipop(ax, borda_df, ds)
    handles = [
        Line2D([0],[0], marker="o", color="0.4", linestyle="None", markersize=7, label="Age"),
        Line2D([0],[0], marker="s", color="0.4", linestyle="None", markersize=7, label="Sex"),
        Line2D([0],[0], marker="D", color="0.2", linestyle="None", markersize=7, label="Original"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), fontsize=9, frameon=True)
    fig.suptitle("Borda ranking — Aggregate performance", fontsize=12, fontweight="bold", y=1.01)
    save_fig(fig, "borda_lollipop.pdf", subdir="rankings")
    plt.close(fig)


# ── Intra‑family functions ─────────────────────────────────────────────────

def plot_cd_diagram_family(fold_long, family_col="family_size", metric="RMSE", rng="ENTIRE", dataset=None):
    label_map = FAMILY_SIZE_LABELS if family_col == "family_size" else FAMILY_MECHANISM_LABELS
    for fam in fold_long[family_col].dropna().unique():
        sub = fold_long[(fold_long[family_col] == fam) & (fold_long["metric"] == metric) & (fold_long["range"] == rng)]
        if dataset:
            sub = sub[sub["dataset"] == dataset]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="fold", columns="condition", values="value", aggfunc="mean")
        pivot.dropna(axis=1, inplace=True)
        if pivot.shape[1] < 3 or pivot.shape[0] < 2:
            continue
        asc   = metric in LOWER_IS_BETTER
        avg_r = _average_ranks(pivot, ascending=asc)
        cd    = _critical_difference(len(avg_r), len(pivot))
        conds = avg_r.index.tolist()
        n = len(conds)
        fig_h = max(3.5, n * 0.45 + 2)
        fig, ax = plt.subplots(figsize=(8, fig_h))
        ax.set_xlim(0.5, n + 0.5)
        ax.set_ylim(-1, n * 0.5 + 1.5)
        ax.axhline(n * 0.5 + 0.8, color="black", lw=1.5, xmin=0, xmax=1)
        ax.set_xticks(np.arange(1, n+1))
        ax.set_xticklabels([f"{r:.1f}" for r in np.arange(1, n+1)], fontsize=9)
        ax.xaxis.tick_top()
        ax.set_xlabel("Average rank  (1 = best)", fontsize=10, labelpad=8)
        ax.xaxis.set_label_position("top")
        ax.yaxis.set_visible(False)
        ax.spines[["left","right","bottom"]].set_visible(False)
        cd_y = n * 0.5 + 0.3
        ax.annotate("", xy=(1 + cd, cd_y), xytext=(1, cd_y), arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
        ax.text(1 + cd/2, cd_y + 0.18, f"CD = {cd:.2f}", ha="center", fontsize=8)
        for r in avg_r.values:
            ax.plot([r, r], [n*0.5+0.8, n*0.5+0.7], color="black", lw=1)
        left_conds  = conds[:n//2]
        right_conds = conds[n//2:]
        def _y(i, side): return (n//2 - i) * 0.45 if side == "left" else (i - n//2) * 0.45
        for i, c in enumerate(left_conds):
            rank = avg_r[c]
            y = _y(i, "left")
            color = technique_color(c.replace("balanced_age_","").replace("balanced_sex_","") if c != "original" else "original")
            ax.plot([rank, rank], [n*0.5+0.8, y+0.05], color=color, lw=1.2, ls=":")
            ax.plot([0.5, rank], [y, y], color=color, lw=1.5)
            ax.scatter([rank], [y], color=color, s=50, zorder=5)
            ax.text(0.45, y, _cond_label(c), ha="right", va="center", fontsize=8)
        for i, c in enumerate(right_conds):
            rank = avg_r[c]
            y = _y(i, "right")
            color = technique_color(c.replace("balanced_age_","").replace("balanced_sex_","") if c != "original" else "original")
            ax.plot([rank, rank], [n*0.5+0.8, y+0.05], color=color, lw=1.2, ls=":")
            ax.plot([rank, n+0.5], [y, y], color=color, lw=1.5)
            ax.scatter([rank], [y], color=color, s=50, zorder=5)
            ax.text(n+0.55, y, _cond_label(c), ha="left", va="center", fontsize=8)
        ax_bottom = -0.8
        clique_y = ax_bottom
        drawn = set()
        for i, c1 in enumerate(conds):
            clique = [c1]
            for c2 in conds[i+1:]:
                if abs(avg_r[c1] - avg_r[c2]) < cd:
                    clique.append(c2)
            if len(clique) > 1:
                key = tuple(sorted(clique))
                if key not in drawn:
                    x_start = avg_r[clique[0]]
                    x_end = avg_r[clique[-1]]
                    ax.plot([x_start, x_end], [clique_y, clique_y],
                            color="black", lw=4, solid_capstyle="round", alpha=0.7)
                    clique_y -= 0.25
                    drawn.add(key)
        fam_label = label_map.get(fam, fam)
        ds_sfx = f"_{dataset}" if dataset else "_global"
        ax.set_title(f"CD Diagram — {fam_label} | {metric} · {RANGE_LABELS.get(rng,rng)}\n"
                     f"Connected groups: no significant difference (Nemenyi, α={ALPHA})",
                     fontsize=10, fontweight="bold", pad=20)
        save_fig(fig, f"cd_diagram_{metric}_{rng}{ds_sfx}_{fam}.pdf",
                 subdir=f"rankings/{family_col}/{fam}")
        plt.close(fig)


def plot_bump_chart_family(master, family_col="family_size", metric_list=None, rng="ENTIRE"):
    label_map = FAMILY_SIZE_LABELS if family_col == "family_size" else FAMILY_MECHANISM_LABELS
    if metric_list is None:
        metric_list = ["RMSE", "MAE"]
    for fam in master[family_col].dropna().unique():
        sub = master[(master[family_col] == fam) & (master["range"] == rng) & (master["metric"].isin(metric_list))]
        if sub.empty:
            continue
        datasets = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
        n_ds = len(datasets)
        fig, axes = plt.subplots(1, n_ds, figsize=(n_ds * 5, 7), sharey=True)
        if n_ds == 1: axes = [axes]
        for ax, ds in zip(axes, datasets):
            _draw_bump_chart(ax, sub[sub["dataset"] == ds], metric_list, rng)

        # ---- Añadir leyenda ----
        seen = set()
        handles = []
        # Solo técnicas presentes en esta familia
        for t in sub[~sub["is_original"]]["technique"].unique():
            if t not in seen:
                handles.append(Line2D([0],[0], color=technique_color(t), lw=2, label=technique_label(t)))
                seen.add(t)
        handles += [
            Line2D([0],[0], color="0.5", lw=2, ls="-",  label="Age"),
            Line2D([0],[0], color="0.5", lw=2, ls="--", label="Sex"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
                   bbox_to_anchor=(0.5, -0.05), fontsize=9, frameon=True)

        fam_label = label_map.get(fam, fam)
        fig.suptitle(f"Bump chart — {fam_label} | {RANGE_LABELS.get(rng,rng)}",
                     fontsize=12, fontweight="bold", y=1.01)
        save_fig(fig, f"bump_chart_{rng}_{fam}.pdf", subdir=f"rankings/{family_col}/{fam}")
        plt.close(fig)


def plot_borda_lollipop_family(borda_df: pd.DataFrame, family_col="family_size"):
    label_map = FAMILY_SIZE_LABELS if family_col == "family_size" else FAMILY_MECHANISM_LABELS
    for fam in borda_df[family_col].dropna().unique():
        fam_borda = borda_df[borda_df[family_col] == fam]
        if fam_borda.empty:
            continue
        datasets = [d for d in DATASET_ORDER if d in fam_borda["dataset"].unique()]
        n_ds = len(datasets)
        fig, axes = plt.subplots(1, n_ds, figsize=(n_ds * 5, 6), sharey=False)
        if n_ds == 1: axes = [axes]
        for ax, ds in zip(axes, datasets):
            _draw_borda_lollipop(ax, fam_borda, ds)
        fam_label = label_map.get(fam, fam)
        fig.suptitle(f"Borda ranking — {fam_label}", fontsize=12, fontweight="bold", y=1.01)
        save_fig(fig, f"borda_lollipop_{fam}.pdf", subdir=f"rankings/{family_col}/{fam}")
        plt.close(fig)


def plot_all_rankings(master, fold_long, borda_df=None):
    log.info("  Generando rankings...")
    # Globales
    for ds in [d for d in DATASET_ORDER if d in fold_long["dataset"].unique()]:
        plot_cd_diagram(fold_long, metric="RMSE", rng="ENTIRE", dataset=ds)
        plot_cd_diagram(fold_long, metric="RMSE", rng="TBR_2",  dataset=ds)
    plot_bump_chart(master, metric_list=["RMSE","MAE"], rng="ENTIRE")
    plot_bump_chart(master, metric_list=["RMSE","MAE"], rng="TBR_1")
    if borda_df is not None and not borda_df.empty:
        plot_borda_lollipop(borda_df)

    # Intra‑familia
    for family_col in ["family_size", "family_mechanism"]:
        for ds in [d for d in DATASET_ORDER if d in fold_long["dataset"].unique()]:
            plot_cd_diagram_family(fold_long, family_col, metric="RMSE", rng="ENTIRE", dataset=ds)
            plot_cd_diagram_family(fold_long, family_col, metric="RMSE", rng="TBR_2", dataset=ds)
        plot_bump_chart_family(master, family_col, metric_list=["RMSE","MAE"], rng="ENTIRE")
        plot_bump_chart_family(master, family_col, metric_list=["RMSE","MAE"], rng="TBR_1")
        if borda_df is not None and not borda_df.empty:
            # Los borda_df globales no tienen la columna family; se puede filtrar con el master
            # Para usar los borda por familia se necesitan los resultados de stats
            pass  # Se omite, o se podría pasar los borda específicos si se cargan desde stats
    log.info("  Rankings completados.")