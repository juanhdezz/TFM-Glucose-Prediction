# -*- coding: utf-8 -*-
"""
profiles.py — Perfiles de rendimiento por rango glucémico.

F11  parallel_coords_{metric}.pdf   coordenadas paralelas: RMSE en los 6 rangos
F12  radar_{ds}_{dim}.pdf           radar chart por dataset y dimensión
F13  range_profile_facet.pdf        small multiples: una fila por técnica, columna=rango

Versiones intra‑familia.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from analysis.config import (
    DATASET_ORDER, DATASET_LABELS, TECHNIQUE_ORDER, TECHNIQUE_LABELS,
    RANGE_ORDER, RANGE_LABELS, LOWER_IS_BETTER,
    DIMENSION_LABELS, PALETTE, DATASET_PALETTE,
    FAMILY_SIZE_LABELS, FAMILY_MECHANISM_LABELS,
)
from analysis.viz.style import apply_theme, save_fig, technique_color, technique_label

log = logging.getLogger(__name__)
apply_theme()

RANGES_FOR_PROFILE = ["TBR_2", "TBR_1", "TIR", "TAR_1", "TAR_2"]  # excluye ENTIRE


def _cond_label(cond):
    if cond == "original": return "Original"
    parts = cond.replace("balanced_","").split("_",1)
    dim  = DIMENSION_LABELS.get(parts[0], parts[0])
    tech = TECHNIQUE_LABELS.get(parts[1] if len(parts)>1 else cond, cond)
    return f"{dim}·{tech}"


# ── F11: Parallel coordinates ─────────────────────────────────────────────────

def plot_parallel_coords(master, metric="RMSE"):
    """
    Coordenadas paralelas globales.
    """
    sub = master[
        (master["metric"] == metric) &
        (master["range"].isin(RANGES_FOR_PROFILE))
    ].copy()
    if sub.empty:
        return

    datasets = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
    n_ds     = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(n_ds * 6, 5.5), sharey=False)
    if n_ds == 1: axes = [axes]

    for ax, ds in zip(axes, datasets):
        ds_data = sub[sub["dataset"] == ds]
        n_axes  = len(RANGES_FOR_PROFILE)

        range_stats = {}
        for rng in RANGES_FOR_PROFILE:
            vals = ds_data[ds_data["range"] == rng]["mean"].dropna()
            if vals.empty:
                range_stats[rng] = (0, 1)
            else:
                range_stats[rng] = (vals.min(), vals.max())

        def normalize(val, rng):
            lo, hi = range_stats[rng]
            if hi == lo: return 0.5
            n = (val - lo) / (hi - lo)
            return 1 - n if metric in LOWER_IS_BETTER else n

        x_positions = np.linspace(0, 1, n_axes)

        for x in x_positions:
            ax.axvline(x, color="0.8", lw=1, zorder=0)

        conds = ds_data["condition"].unique()
        for cond in conds:
            cdata = ds_data[ds_data["condition"] == cond].set_index("range")
            tech  = cdata["technique"].iloc[0] if "technique" in cdata.columns else "original"
            color = technique_color(tech)
            lw    = 2.5 if cond == "original" else 1.5
            ls    = "-" if cond == "original" else (
                "-" if cdata["dimension"].iloc[0] == "age" else "--")
            alpha = 0.95 if cond == "original" else 0.65

            ys = []
            for rng in RANGES_FOR_PROFILE:
                if rng in cdata.index:
                    ys.append(normalize(cdata.loc[rng, "mean"], rng))
                else:
                    ys.append(np.nan)

            valid = [(x, y) for x, y in zip(x_positions, ys) if not np.isnan(y)]
            if len(valid) < 2:
                continue
            xs_, ys_ = zip(*valid)
            ax.plot(xs_, ys_, color=color, lw=lw, ls=ls, alpha=alpha, zorder=2)
            ax.scatter(xs_, ys_, color=color, s=30, zorder=3,
                       edgecolors="white", lw=0.6, alpha=alpha)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([RANGE_LABELS.get(r, r) for r in RANGES_FOR_PROFILE],
                           fontsize=9, rotation=20, ha="right")
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["Worst","Mid","Best"], fontsize=8)
        ax.set_ylim(-0.08, 1.08)
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=12, fontweight="bold")
        ax.set_ylabel("Relative performance (↑ better)", fontsize=9) if ds == datasets[0] else None

    seen = set()
    handles = [Line2D([0],[0], color="#000", lw=2.5, label="Original (baseline)")]
    for _, row in master[~master["is_original"]].iterrows():
        t = row["technique"]
        if t not in seen:
            handles.append(Line2D([0],[0], color=technique_color(t),
                                   lw=1.5, label=technique_label(t)))
            seen.add(t)
    handles += [
        Line2D([0],[0], color="0.5", lw=1.5, ls="-",  label="Age dim."),
        Line2D([0],[0], color="0.5", lw=1.5, ls="--", label="Sex dim."),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5),
               bbox_to_anchor=(0.5, -0.07), fontsize=9, frameon=True)

    fig.suptitle(f"Performance profile across glucose ranges — {metric}  |  ↑ = better on each axis",
                 fontsize=12, fontweight="bold", y=1.01)
    save_fig(fig, f"parallel_coords_{metric}.pdf", subdir="profiles")
    plt.close(fig)


# ── F12: Radar chart ─────────────────────────────────────────────────────────

def _radar_factory(ax, n_vars, frame="polygon"):
    angles = np.linspace(0, 2*np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return angles


def plot_radar(master, metric="RMSE", dataset=None, dimension="age"):
    sub = master[
        (master["metric"] == metric) &
        (master["range"].isin(RANGES_FOR_PROFILE)) &
        (master["dataset"] == dataset if dataset else True) &
        (master["dimension"].isin([dimension, None]))
    ].copy()
    if sub.empty or dataset is None:
        return

    n_vars  = len(RANGES_FOR_PROFILE)
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    angles  = _radar_factory(ax, n_vars)

    range_stats = {}
    for rng in RANGES_FOR_PROFILE:
        vals = sub[sub["range"] == rng]["mean"].dropna()
        range_stats[rng] = (vals.min(), vals.max()) if not vals.empty else (0, 1)

    def norm(val, rng):
        lo, hi = range_stats[rng]
        if hi == lo: return 0.5
        n = (val - lo) / (hi - lo)
        return 1 - n if metric in LOWER_IS_BETTER else n

    for level in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles, [level]*(n_vars+1), color="0.85", lw=0.8, ls="--", zorder=0)

    conds = sub["condition"].unique()
    for cond in conds:
        cdata = sub[sub["condition"] == cond].set_index("range")
        tech  = cdata["technique"].iloc[0] if "technique" in cdata.columns else "original"
        color = technique_color(tech)
        lw    = 2.5 if cond == "original" else 1.8
        alpha_fill = 0.08 if cond != "original" else 0.12

        vals = [norm(cdata.loc[r,"mean"], r) if r in cdata.index else 0
                for r in RANGES_FOR_PROFILE]
        vals += vals[:1]

        ax.plot(angles, vals, color=color, lw=lw, zorder=2)
        ax.fill(angles, vals, color=color, alpha=alpha_fill)
        ax.scatter(angles[:-1], vals[:-1], color=color, s=30, zorder=3)

    for angle, rng in zip(angles[:-1], RANGES_FOR_PROFILE):
        ax.text(angle, 1.18, RANGE_LABELS.get(rng, rng),
                ha="center", va="center", fontsize=9)

    dim_lbl = DIMENSION_LABELS.get(dimension, dimension)
    ax.set_title(
        f"Glucose range profile — {DATASET_LABELS.get(dataset,dataset)}\n"
        f"{metric} · Dimension: {dim_lbl}  |  outer = better",
        fontsize=10, fontweight="bold", pad=20
    )

    seen = set()
    handles = []
    for cond in conds:
        cdata = sub[sub["condition"] == cond]
        tech  = cdata["technique"].iloc[0]
        if tech not in seen:
            handles.append(mpatches.Patch(color=technique_color(tech),
                                          label=_cond_label(cond), alpha=0.7))
            seen.add(tech)
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=8, frameon=True)

    fname = f"radar_{dataset}_{metric}_{dimension}.pdf".replace("-","")
    save_fig(fig, fname, subdir="profiles")
    plt.close(fig)


# ── F13: Small multiples facet ───────────────────────────────────────────────

def plot_range_profile_facet(master, metric="RMSE"):
    sub = master[
        (master["metric"] == metric) &
        (master["range"].isin(RANGES_FOR_PROFILE))
    ].copy()
    if sub.empty:
        return

    techniques  = [t for t in TECHNIQUE_ORDER if t != "original" and
                   t in sub["technique"].unique()]
    n_tech      = len(techniques)
    n_ranges    = len(RANGES_FOR_PROFILE)
    datasets    = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
    ds_colors   = [DATASET_PALETTE.get(d, "#888") for d in datasets]

    fig, axes = plt.subplots(n_tech, n_ranges,
                             figsize=(n_ranges * 2.8, n_tech * 2.0 + 1.2),
                             sharey="row", sharex=True)
    if n_tech == 1:  axes = [axes]
    if n_ranges == 1: axes = [[ax] for ax in axes]

    for row_i, tech in enumerate(techniques):
        for col_j, rng in enumerate(RANGES_FOR_PROFILE):
            ax = axes[row_i][col_j]
            for k, (ds, color) in enumerate(zip(datasets, ds_colors)):
                orig_val = sub[
                    (sub["dataset"] == ds) & (sub["range"] == rng) &
                    (sub["is_original"])
                ]["mean"].values
                bal_val  = sub[
                    (sub["dataset"] == ds) & (sub["range"] == rng) &
                    (sub["technique"] == tech)
                ]["mean"].values

                if len(orig_val) == 0 or len(bal_val) == 0:
                    continue

                ax.bar(k - 0.125, orig_val[0], 0.25,
                       color=color, alpha=0.3, hatch="//", edgecolor=color, lw=0.8)
                ax.bar(k + 0.125, bal_val[0], 0.25,
                       color=color, alpha=0.85, edgecolor="white", lw=0.5)

            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", alpha=0.25)

            if col_j == 0:
                ax.set_ylabel(TECHNIQUE_LABELS.get(tech, tech), fontsize=8,
                              rotation=0, ha="right", va="center", labelpad=60)
            if row_i == 0:
                ax.set_title(RANGE_LABELS.get(rng, rng), fontsize=9, fontweight="bold")

    ds_handles = [
        mpatches.Patch(facecolor=DATASET_PALETTE.get(d,"#888"),
                       label=DATASET_LABELS.get(d,d), alpha=0.85)
        for d in datasets
    ]
    ds_handles += [
        mpatches.Patch(facecolor="0.7", hatch="//", edgecolor="0.5",
                       label="Baseline (hatched)"),
        mpatches.Patch(facecolor="0.4", label="Balanced (solid)"),
    ]
    fig.legend(handles=ds_handles, loc="lower center", ncol=len(ds_handles),
               bbox_to_anchor=(0.5, -0.02), fontsize=8, frameon=True)

    fig.suptitle(
        f"Small multiples — {metric} by technique × glucose range\n"
        f"Row = technique  |  Column = glucose range  |  Hatched = baseline",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    save_fig(fig, f"range_profile_facet_{metric}.pdf", subdir="profiles")
    plt.close(fig)


# ── Intra‑family versions ────────────────────────────────────────────────────

# Añadir al final de profiles.py las funciones intra‑familia completas

def plot_parallel_coords_family(master, family_col="family_size", metric="RMSE"):
    label_map = FAMILY_SIZE_LABELS if family_col == "family_size" else FAMILY_MECHANISM_LABELS
    for fam in master[family_col].dropna().unique():
        sub = master[(master[family_col] == fam) & (master["metric"] == metric) &
                     (master["range"].isin(RANGES_FOR_PROFILE))]
        if sub.empty: continue
        datasets = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
        n_ds = len(datasets)
        fig, axes = plt.subplots(1, n_ds, figsize=(n_ds*6, 5.5), sharey=False)
        if n_ds == 1: axes = [axes]

        for ax, ds in zip(axes, datasets):
            ds_data = sub[sub["dataset"] == ds]
            n_axes = len(RANGES_FOR_PROFILE)
            range_stats = {rng: (ds_data[ds_data["range"]==rng]["mean"].min(),
                                ds_data[ds_data["range"]==rng]["mean"].max())
                          for rng in RANGES_FOR_PROFILE}
            def norm(val, rng):
                lo, hi = range_stats[rng]
                if hi == lo: return 0.5
                n = (val-lo)/(hi-lo)
                return 1-n if metric in LOWER_IS_BETTER else n
            x_positions = np.linspace(0, 1, n_axes)
            for x in x_positions: ax.axvline(x, color="0.8", lw=1)
            for cond in ds_data["condition"].unique():
                cdata = ds_data[ds_data["condition"]==cond].set_index("range")
                tech = cdata["technique"].iloc[0]
                color = technique_color(tech)
                lw = 2.5 if cond=="original" else 1.5
                ls = "-" if cond=="original" else ("-" if cdata["dimension"].iloc[0]=="age" else "--")
                alpha = 0.95 if cond=="original" else 0.65
                ys = [norm(cdata.loc[r,"mean"], r) if r in cdata.index else np.nan for r in RANGES_FOR_PROFILE]
                valid = [(x,y) for x,y in zip(x_positions, ys) if not np.isnan(y)]
                if len(valid)<2: continue
                xs_, ys_ = zip(*valid)
                ax.plot(xs_, ys_, color=color, lw=lw, ls=ls, alpha=alpha)
                ax.scatter(xs_, ys_, color=color, s=30, edgecolors="white", lw=0.6, alpha=alpha)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([RANGE_LABELS.get(r,r) for r in RANGES_FOR_PROFILE],
                               fontsize=9, rotation=20, ha="right")
            ax.set_yticks([0,0.5,1]); ax.set_yticklabels(["Worst","Mid","Best"], fontsize=8)
            ax.set_ylim(-0.08, 1.08)
            ax.set_title(DATASET_LABELS.get(ds,ds), fontsize=12, fontweight="bold")

        # ---- Añadir leyenda ----
        seen = set()
        handles = [Line2D([0],[0], color="#000", lw=2.5, label="Original (baseline)")]
        for t in sub[~sub["is_original"]]["technique"].unique():
            if t not in seen:
                handles.append(Line2D([0],[0], color=technique_color(t), lw=1.5, label=technique_label(t)))
                seen.add(t)
        handles += [
            Line2D([0],[0], color="0.5", lw=1.5, ls="-",  label="Age dim."),
            Line2D([0],[0], color="0.5", lw=1.5, ls="--", label="Sex dim."),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
                   bbox_to_anchor=(0.5, -0.07), fontsize=9, frameon=True)

        fam_label = label_map.get(fam, fam)
        fig.suptitle(f"Parallel coordinates — {metric} | {fam_label}",
                     fontsize=12, fontweight="bold", y=1.01)
        save_fig(fig, f"parallel_coords_{metric}_{fam}.pdf", subdir=f"profiles/{family_col}/{fam}")
        plt.close(fig)


def plot_radar_family(master, family_col="family_size", metric="RMSE"):
    label_map = FAMILY_SIZE_LABELS if family_col == "family_size" else FAMILY_MECHANISM_LABELS
    for fam in master[family_col].dropna().unique():
        for ds in [d for d in DATASET_ORDER if d in master["dataset"].unique()]:
            for dim in ["age", "sex"]:
                sub = master[(master[family_col]==fam)&(master["metric"]==metric)&
                             (master["range"].isin(RANGES_FOR_PROFILE))&
                             (master["dataset"]==ds)&(master["dimension"].isin([dim, None]))]
                if sub.empty: continue
                n_vars = len(RANGES_FOR_PROFILE)
                fig, ax = plt.subplots(figsize=(6,6), subplot_kw={"projection":"polar"})
                angles = _radar_factory(ax, n_vars)
                range_stats = {rng: (sub[sub["range"]==rng]["mean"].min(),
                                    sub[sub["range"]==rng]["mean"].max())
                              for rng in RANGES_FOR_PROFILE}
                def norm(val, rng):
                    lo, hi = range_stats[rng]
                    if hi == lo: return 0.5
                    n = (val-lo)/(hi-lo)
                    return 1-n if metric in LOWER_IS_BETTER else n
                for level in [0.25,0.5,0.75,1.0]:
                    ax.plot(angles, [level]*(n_vars+1), color="0.85", lw=0.8, ls="--")
                conds = sub["condition"].unique()
                for cond in conds:
                    cdata = sub[sub["condition"]==cond].set_index("range")
                    tech = cdata["technique"].iloc[0]
                    color = technique_color(tech)
                    lw = 2.5 if cond=="original" else 1.8
                    alpha_fill = 0.08 if cond!="original" else 0.12
                    vals = [norm(cdata.loc[r,"mean"],r) if r in cdata.index else 0
                            for r in RANGES_FOR_PROFILE]
                    vals += vals[:1]
                    ax.plot(angles, vals, color=color, lw=lw)
                    ax.fill(angles, vals, color=color, alpha=alpha_fill)
                    ax.scatter(angles[:-1], vals[:-1], color=color, s=30)
                for angle, rng in zip(angles[:-1], RANGES_FOR_PROFILE):
                    ax.text(angle, 1.18, RANGE_LABELS.get(rng,rng),
                            ha="center", va="center", fontsize=9)
                ax.set_title(f"{DATASET_LABELS.get(ds,ds)} | {DIMENSION_LABELS.get(dim,dim)} | {label_map.get(fam,fam)}",
                             fontsize=10, fontweight="bold", pad=20)

                # ---- Añadir leyenda ----
                seen = set()
                handles = []
                for cond in conds:
                    cdata = sub[sub["condition"]==cond]
                    tech = cdata["technique"].iloc[0]
                    if tech not in seen:
                        handles.append(mpatches.Patch(color=technique_color(tech),
                                                      label=_cond_label(cond), alpha=0.7))
                        seen.add(tech)
                ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.35, 1.15),
                          fontsize=8, frameon=True)

                fname = f"radar_{ds}_{metric}_{dim}_{fam}.pdf".replace("-","")
                save_fig(fig, fname, subdir=f"profiles/{family_col}/{fam}")
                plt.close(fig)


def plot_range_profile_facet_family(master, family_col="family_size", metric="RMSE"):
    label_map = FAMILY_SIZE_LABELS if family_col == "family_size" else FAMILY_MECHANISM_LABELS
    for fam in master[family_col].dropna().unique():
        sub = master[(master[family_col]==fam)&(master["metric"]==metric)&(master["range"].isin(RANGES_FOR_PROFILE))]
        if sub.empty: continue
        techniques = [t for t in TECHNIQUE_ORDER if t!="original" and t in sub["technique"].unique()]
        if not techniques: continue
        n_tech = len(techniques)
        n_ranges = len(RANGES_FOR_PROFILE)
        datasets = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
        ds_colors = [DATASET_PALETTE.get(d,"#888") for d in datasets]
        fig, axes = plt.subplots(n_tech, n_ranges, figsize=(n_ranges*2.8, n_tech*2.0+1.2), sharey="row", sharex=True)
        if n_tech==1: axes = [axes]
        if n_ranges==1: axes = [[ax] for ax in axes]
        for row_i, tech in enumerate(techniques):
            for col_j, rng in enumerate(RANGES_FOR_PROFILE):
                ax = axes[row_i][col_j]
                for k, (ds, color) in enumerate(zip(datasets, ds_colors)):
                    orig_val = sub[(sub["dataset"]==ds)&(sub["range"]==rng)&(sub["is_original"])]["mean"].values
                    bal_val  = sub[(sub["dataset"]==ds)&(sub["range"]==rng)&(sub["technique"]==tech)]["mean"].values
                    if len(orig_val)==0 or len(bal_val)==0: continue
                    ax.bar(k-0.125, orig_val[0], 0.25, color=color, alpha=0.3, hatch="//")
                    ax.bar(k+0.125, bal_val[0], 0.25, color=color, alpha=0.85, edgecolor="white")
                ax.set_xticks([])
                ax.grid(axis="y", alpha=0.25)
                if col_j==0: ax.set_ylabel(TECHNIQUE_LABELS.get(tech,tech), fontsize=8, rotation=0, ha="right", labelpad=60)
                if row_i==0: ax.set_title(RANGE_LABELS.get(rng,rng), fontsize=9, fontweight="bold")
        fig.suptitle(f"Small multiples — {metric} | {label_map.get(fam,fam)}", fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0,0.05,1,0.97])
        save_fig(fig, f"range_profile_facet_{metric}_{fam}.pdf", subdir=f"profiles/{family_col}/{fam}")
        plt.close(fig)


def plot_all_profiles(master):
    log.info("  Generando perfiles glucémicos...")
    # Globales
    for metric in ["RMSE", "MAE"]:
        plot_parallel_coords(master, metric=metric)
        plot_range_profile_facet(master, metric=metric)
    for ds in [d for d in DATASET_ORDER if d in master["dataset"].unique()]:
        for dim in ["age", "sex"]:
            plot_radar(master, metric="RMSE", dataset=ds, dimension=dim)

    # Intra‑familia
    for family_col in ["family_size", "family_mechanism"]:
        for metric in ["RMSE", "MAE"]:
            plot_parallel_coords_family(master, family_col, metric=metric)
            plot_range_profile_facet_family(master, family_col, metric=metric)
        for ds in [d for d in DATASET_ORDER if d in master["dataset"].unique()]:
            for dim in ["age", "sex"]:
                plot_radar_family(master, family_col, metric="RMSE")
    log.info("  Perfiles completados.")