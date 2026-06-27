# -*- coding: utf-8 -*-
"""
dashboards.py — Figuras compuestas multi-panel para el TFM.

D1  dashboard_global.pdf      heatmap delta + Borda ranking + CD diagram (1 página)
D2  dashboard_per_dataset.pdf dumbbell + distribución por fold (3 columnas × dataset)
D3  dashboard_clinical.pdf    perfiles glucémicos + Clarke zones (visión clínica)
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from analysis.config import (
    DASHBOARDS_DIR, DATASET_ORDER, DATASET_LABELS,
    TECHNIQUE_ORDER, TECHNIQUE_LABELS, RANGE_ORDER, RANGE_LABELS,
    LOWER_IS_BETTER, PALETTE, DIMENSION_LABELS,
)
from analysis.viz.style import apply_theme, technique_color, technique_label

log = logging.getLogger(__name__)
apply_theme()


def _cond_label(cond):
    if cond == "original": return "Original"
    parts = cond.replace("balanced_","").split("_",1)
    dim  = DIMENSION_LABELS.get(parts[0], parts[0])
    tech = TECHNIQUE_LABELS.get(parts[1] if len(parts)>1 else cond, cond)
    return f"{dim}·{tech}"

def _sort_conds(conds):
    def _k(c):
        if c == "original": return (0,0,0)
        parts = c.replace("balanced_","").split("_",1)
        dim = parts[0]; tech = parts[1] if len(parts)>1 else ""
        return (1,{"age":0,"sex":1}.get(dim,2),
                TECHNIQUE_ORDER.index(tech) if tech in TECHNIQUE_ORDER else 99)
    return sorted(conds, key=_k)

def _save_dashboard(fig, fname):
    path = DASHBOARDS_DIR / fname
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    log.info(f"  Dashboard: {path.name}")


# ── D1: Global overview dashboard ─────────────────────────────────────────────

def dashboard_global(master: pd.DataFrame, borda_df: pd.DataFrame = None) -> None:
    """
    Layout 2×2:
    [Top-left]  Delta heatmap (RMSE · ENTIRE)
    [Top-right] Borda score lollipop global
    [Bottom-left]  Mean RMSE by technique (bar, grouped by dimension)
    [Bottom-right] Delta heatmap (RMSE · TBR_1 — clinical focus)
    """
    metric = "RMSE"
    fig = plt.figure(figsize=(18, 13))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)

    ax_heat_all = fig.add_subplot(gs[0, 0])
    ax_borda    = fig.add_subplot(gs[0, 1])
    ax_bar      = fig.add_subplot(gs[1, 0])
    ax_heat_hyp = fig.add_subplot(gs[1, 1])

    # ── Panel A: delta heatmap ENTIRE ──
    _mini_delta_heatmap(ax_heat_all, master, metric=metric, rng="ENTIRE",
                        title="A  |  RMSE improvement vs. baseline  (All ranges)")

    # ── Panel B: Borda lollipop ──
    _mini_borda(ax_borda, master, borda_df, metric=metric,
                title="B  |  Aggregate Borda ranking (lower = better)")

    # ── Panel C: Mean RMSE bar grouped by dim ──
    _mini_bar_by_dim(ax_bar, master, metric=metric, rng="ENTIRE",
                     title="C  |  Mean RMSE by technique and dimension")

    # ── Panel D: delta heatmap Hypo L1 ──
    _mini_delta_heatmap(ax_heat_hyp, master, metric=metric, rng="TBR_1",
                        title="D  |  RMSE improvement vs. baseline  (Hypo L1 · 54–69 mg/dL)")

    fig.suptitle(
        "Global experimental overview — LSTM glucose prediction with balancing techniques",
        fontsize=15, fontweight="bold", y=0.99
    )
    _save_dashboard(fig, "dashboard_global.pdf")


def _mini_delta_heatmap(ax, master, metric, rng, title):
    sub = master[
        (~master["is_original"]) &
        (master["metric"] == metric) &
        (master["range"]  == rng)
    ].copy()
    if sub.empty:
        ax.set_visible(False); return

    pivot = sub.pivot_table(index="condition", columns="dataset",
                            values="improvement_pct", aggfunc="mean")
    pivot = pivot.loc[_sort_conds(list(pivot.index))]
    pivot = pivot[[d for d in DATASET_ORDER if d in pivot.columns]]

    data = pivot.values.astype(float)
    cmap = LinearSegmentedColormap.from_list("d",["#0072B2","#FFFFFF","#E69F00"],N=256)
    vmax = max(np.nanpercentile(np.abs(data[~np.isnan(data)]), 95) if not np.all(np.isnan(data)) else 5, 2)

    ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    row_labels = [_cond_label(c) for c in pivot.index]
    col_labels = [DATASET_LABELS.get(d,d) for d in pivot.columns]
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=7)
    ax.xaxis.tick_top()

    n_rows, n_cols = data.shape
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i,j]
            if np.isnan(v): continue
            norm  = abs(v) / vmax
            color = "white" if norm > 0.55 else "black"
            ax.text(j, i, f"{v:+.1f}%", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    # separator age/sex
    age_idx = [i for i, c in enumerate(pivot.index) if "balanced_age" in c]
    sex_idx = [i for i, c in enumerate(pivot.index) if "balanced_sex" in c]
    if age_idx and sex_idx:
        ax.axhline(max(age_idx)+0.5, color="0.4", lw=1.2, ls="--")

    ax.set_title(title, fontsize=10, fontweight="bold", pad=14)


def _mini_borda(ax, master, borda_df, metric, title):
    """If borda_df available use it, else compute simple aggregate rank."""
    sub = master[master["metric"] == metric].copy()
    if sub.empty:
        ax.set_visible(False); return

    if borda_df is not None and not borda_df.empty:
        # aggregate across datasets
        agg = (borda_df.groupby(["condition","technique","dimension"])
               ["borda_score"].sum().reset_index()
               .sort_values("borda_score"))
    else:
        # fallback: average rank per condition
        records = []
        for (ds, rng), grp in sub.groupby(["dataset","range"]):
            ranked = grp.sort_values("mean", ascending=metric in LOWER_IS_BETTER).reset_index(drop=True)
            for pos, (_, row) in enumerate(ranked.iterrows(), 1):
                records.append({"condition":row["condition"],
                                 "technique":row["technique"],
                                 "dimension":row["dimension"],
                                 "borda_score":pos})
        agg = (pd.DataFrame(records)
               .groupby(["condition","technique","dimension"])
               ["borda_score"].sum().reset_index()
               .sort_values("borda_score"))

    conds  = agg["condition"].tolist()
    scores = agg["borda_score"].tolist()
    colors = [technique_color(t) for t in agg["technique"]]
    markers= ["o" if str(d)=="age" else ("s" if str(d)=="sex" else "D")
              for d in agg["dimension"]]

    y_pos = np.arange(len(conds))
    for y, score, color, marker, cond in zip(y_pos, scores, colors, markers, conds):
        ax.plot([0, score], [y, y], color=color, lw=1.8, alpha=0.7)
        ax.scatter([score], [y], color=color, s=55, marker=marker,
                   zorder=3, edgecolors="white", lw=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([_cond_label(c) for c in conds], fontsize=7)
    ax.set_xlabel("Borda score (lower = better)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)


def _mini_bar_by_dim(ax, master, metric, rng, title):
    sub = master[
        (~master["is_original"]) &
        (master["metric"] == metric) &
        (master["range"]  == rng)
    ].copy()
    if sub.empty:
        ax.set_visible(False); return

    techs    = [t for t in TECHNIQUE_ORDER if t!="original" and t in sub["technique"].unique()]
    dims     = ["age","sex"]
    x        = np.arange(len(techs))
    bar_w    = 0.3
    dim_off  = {"age":-bar_w/2, "sex":bar_w/2}
    dim_color= {"age":"#0072B2", "sex":"#D55E00"}

    for dim in dims:
        dsub = sub[sub["dimension"]==dim]
        means = [dsub[dsub["technique"]==t]["mean"].mean() for t in techs]
        stds  = [dsub[dsub["technique"]==t]["std"].mean() for t in techs]
        ax.bar(x + dim_off[dim], means, bar_w, label=dim.capitalize(),
               color=dim_color[dim], alpha=0.82, edgecolor="white", lw=0.5)
        ax.errorbar(x + dim_off[dim], means, yerr=stds,
                    fmt="none", ecolor="black", elinewidth=1, capsize=3, alpha=0.7)

    # Baseline line per dataset (average)
    orig_mean = master[master["is_original"] & (master["metric"]==metric) &
                       (master["range"]==rng)]["mean"].mean()
    ax.axhline(orig_mean, color="black", lw=1.5, ls="--", alpha=0.6, label="Baseline mean")

    ax.set_xticks(x)
    ax.set_xticklabels([TECHNIQUE_LABELS.get(t,t) for t in techs],
                       rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(f"{metric} (mg/dL)", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


# ── D2: Per-dataset dashboard ─────────────────────────────────────────────────

def dashboard_per_dataset(master: pd.DataFrame, fold_long: pd.DataFrame) -> None:
    """
    3 columns (one per dataset), 2 rows:
    Row 1: dumbbell RMSE·ENTIRE
    Row 2: strip plot fold consistency
    """
    metric   = "RMSE"
    rng      = "ENTIRE"
    datasets = [d for d in DATASET_ORDER if d in master["dataset"].unique()]
    n_ds     = len(datasets)

    fig = plt.figure(figsize=(n_ds * 5.5, 12))
    gs  = gridspec.GridSpec(2, n_ds, figure=fig, hspace=0.45, wspace=0.35)

    for col, ds in enumerate(datasets):
        ax_db  = fig.add_subplot(gs[0, col])
        ax_str = fig.add_subplot(gs[1, col])

        # ── Row 1: dumbbell ──
        sub = master[
            (master["dataset"] == ds) &
            (master["metric"]  == metric) &
            (master["range"]   == rng)
        ].copy()
        orig_val = sub[sub["is_original"]]["mean"].values
        orig_val = orig_val[0] if len(orig_val) else np.nan
        bal      = sub[~sub["is_original"]].sort_values("mean", ascending=False)
        y_pos    = np.arange(len(bal))

        for y, (_, row) in zip(y_pos, bal.iterrows()):
            color = technique_color(row["technique"])
            ax_db.plot([orig_val, row["mean"]], [y, y], color="0.78", lw=1.5, zorder=1)
            ax_db.scatter([orig_val], [y], color="#333", s=45, marker="D", zorder=3)
            ax_db.scatter([row["mean"]], [y], color=color, s=75, zorder=4,
                          edgecolors="white", lw=0.7)
            if not np.isnan(row.get("std",np.nan)):
                ax_db.errorbar([row["mean"]], [y], xerr=[[row["std"]]],
                               fmt="none", ecolor=color, elinewidth=1, capsize=2, alpha=0.6)

        ax_db.axvline(orig_val, color="#333", lw=1.2, ls="--", alpha=0.5)
        ax_db.set_yticks(y_pos)
        ax_db.set_yticklabels([_cond_label(c) for c in bal["condition"]], fontsize=7.5)
        ax_db.set_xlabel(f"{metric} (mg/dL)", fontsize=9)
        ax_db.set_title(f"{DATASET_LABELS.get(ds,ds)}\nDumbbell: {metric}·All ranges",
                        fontsize=10, fontweight="bold")

        # ── Row 2: strip fold ──
        sub_f = fold_long[
            (fold_long["dataset"] == ds) &
            (fold_long["metric"]  == metric) &
            (fold_long["range"]   == rng)
        ].copy()
        conds = _sort_conds(sub_f["condition"].unique().tolist())
        y_map = {c: i for i, c in enumerate(conds)}
        FOLD_M = {1:"o",2:"s",3:"^",4:"D",5:"v"}

        for cond in conds:
            cdata = sub_f[sub_f["condition"]==cond]
            y     = y_map[cond]
            color = technique_color(
                cond.replace("balanced_age_","").replace("balanced_sex_","")
                if cond!="original" else "original"
            )
            mean_v = cdata["value"].mean()
            ax_str.plot([mean_v],[y],marker="|",color=color,
                        markersize=16,markeredgewidth=2.5,zorder=3)
            jitters = np.linspace(-0.15,0.15,len(cdata))
            for (_, row), jit in zip(cdata.iterrows(), jitters):
                fn = int(row["fold"]) if not np.isnan(row.get("fold",np.nan)) else 1
                ax_str.scatter([row["value"]], [y+jit],
                               marker=FOLD_M.get(fn,"o"), color=color,
                               s=35, zorder=4, edgecolors="white", lw=0.5, alpha=0.9)

        ax_str.set_yticks(range(len(conds)))
        ax_str.set_yticklabels([_cond_label(c) for c in conds], fontsize=7.5)
        ax_str.set_xlabel(f"{metric} (mg/dL)", fontsize=9)
        ax_str.set_title(f"Fold consistency — | = mean", fontsize=10, fontweight="bold")
        ax_str.grid(axis="x", alpha=0.3)

    # Shared fold legend
    fold_handles = [Line2D([0],[0],marker=FOLD_M[f],color="0.4",linestyle="None",
                            markersize=6,label=f"Fold {f}") for f in range(1,6)]
    fig.legend(handles=fold_handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5,-0.02), fontsize=8, frameon=True)

    fig.suptitle(f"Per-dataset overview — {metric} · All ranges  |  ◆ = baseline",
                 fontsize=14, fontweight="bold", y=1.01)
    _save_dashboard(fig, "dashboard_per_dataset.pdf")


# ── D3: Clinical dashboard ─────────────────────────────────────────────────────

def dashboard_clinical(master: pd.DataFrame) -> None:
    """
    Focus clínico:
    Row 1: delta heatmap TBR_2 (hipoglucemia severa) + delta heatmap TBR_1
    Row 2: bar chart Clarke zone A por condición (% puntos clínicamente correctos)
    Row 3: delta heatmap TIR + delta heatmap TAR_1
    """
    fig = plt.figure(figsize=(18, 16))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.35)

    _mini_delta_heatmap(fig.add_subplot(gs[0,0]), master, "RMSE", "TBR_2",
                        "A  |  RMSE improvement · Hypo L2 (<54 mg/dL) — CRITICAL")
    _mini_delta_heatmap(fig.add_subplot(gs[0,1]), master, "RMSE", "TBR_1",
                        "B  |  RMSE improvement · Hypo L1 (54–69 mg/dL)")

    # Clarke zone A bar
    ax_ceg = fig.add_subplot(gs[1, :])
    _mini_clarke_bar(ax_ceg, master, zone="A",
                     title="C  |  Clarke Zone A (%) — clinically safe predictions  ↑ better")

    _mini_delta_heatmap(fig.add_subplot(gs[2,0]), master, "RMSE", "TIR",
                        "D  |  RMSE improvement · In Range (70–180 mg/dL)")
    _mini_delta_heatmap(fig.add_subplot(gs[2,1]), master, "RMSE", "TAR_1",
                        "E  |  RMSE improvement · Hyper L1 (181–250 mg/dL)")

    fig.suptitle("Clinical safety dashboard — Impact of balancing on glucose prediction accuracy",
                 fontsize=14, fontweight="bold", y=1.01)
    _save_dashboard(fig, "dashboard_clinical.pdf")


def _mini_clarke_bar(ax, master, zone="A", title=""):
    sub = master[
        (master["metric"] == zone) &
        (master["range"]  == "ENTIRE")
    ].copy()
    if sub.empty:
        ax.set_visible(False); return

    datasets = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
    conds    = _sort_conds(sub["condition"].unique().tolist())
    n_conds  = len(conds)
    n_ds     = len(datasets)
    bar_w    = 0.8 / n_ds
    x        = np.arange(n_conds)
    from analysis.config import DATASET_PALETTE

    for k, ds in enumerate(datasets):
        ds_sub = sub[sub["dataset"]==ds]
        means  = [ds_sub[ds_sub["condition"]==c]["mean"].values[0]
                  if not ds_sub[ds_sub["condition"]==c].empty else np.nan
                  for c in conds]
        stds   = [ds_sub[ds_sub["condition"]==c]["std"].values[0]
                  if not ds_sub[ds_sub["condition"]==c].empty else np.nan
                  for c in conds]
        offset = (k - n_ds/2 + 0.5) * bar_w
        ax.bar(x + offset, means, bar_w, label=DATASET_LABELS.get(ds,ds),
               color=DATASET_PALETTE.get(ds,"#888"), alpha=0.82, edgecolor="white", lw=0.5)
        ax.errorbar(x + offset, means, yerr=stds,
                    fmt="none", ecolor="black", elinewidth=1, capsize=2, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([_cond_label(c) for c in conds], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"Clarke Zone {zone} (%)", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0]-2))


def generate_all_dashboards(master: pd.DataFrame, fold_long: pd.DataFrame = None,
                            borda_df: pd.DataFrame = None) -> None:
    log.info("  Generando dashboards...")
    dashboard_global(master, borda_df=borda_df)
    if fold_long is not None and not fold_long.empty:
        dashboard_per_dataset(master, fold_long)
    dashboard_clinical(master)
    log.info("  Dashboards completados (3 figuras).")