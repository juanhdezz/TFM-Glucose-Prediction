# -*- coding: utf-8 -*-
"""
tables.py — Tablas LaTeX y CSV para el TFM.

T1  main_results_{metric}.tex / .csv     mean ± std por dataset × condición
T2  ranking_table.tex / .csv             posiciones de ranking por dataset
T3  delta_table_{metric}.tex             tabla de deltas porcentuales
T4  clarke_zones_table.tex               % puntos en zona A y A+B por condición

Versiones intra‑familia.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from analysis.config import (
    TABLES_DIR, DATASET_ORDER, DATASET_LABELS,
    TECHNIQUE_ORDER, TECHNIQUE_LABELS, RANGE_ORDER, RANGE_LABELS,
    ANALYSIS_METRICS, LOWER_IS_BETTER, PRIMARY_METRIC,
    FAMILY_SIZE_LABELS, FAMILY_MECHANISM_LABELS,
)

log = logging.getLogger(__name__)


def _cond_label(cond):
    if cond == "original": return "Original"
    parts = cond.replace("balanced_","").split("_",1)
    dim  = parts[0].capitalize()
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


# ── T1: Main results table ────────────────────────────────────────────────────

def generate_main_results_table(master: pd.DataFrame, metric: str = "RMSE",
                                suffix: str = "") -> None:
    sub = master[master["metric"] == metric].copy()
    if sub.empty:
        log.warning(f"main_results_table: sin datos para {metric}")
        return

    ranges   = [r for r in RANGE_ORDER if r in sub["range"].unique()]
    datasets = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
    conds    = _sort_conds(sub["condition"].unique().tolist())

    rows = []
    for cond in conds:
        row = {"Condition": _cond_label(cond)}
        for ds in datasets:
            for rng in ranges:
                cell = sub[(sub["condition"]==cond)&(sub["dataset"]==ds)&(sub["range"]==rng)]
                if cell.empty:
                    row[f"{DATASET_LABELS.get(ds,ds)}_{RANGE_LABELS.get(rng,rng)}"] = "—"
                else:
                    m = cell["mean"].values[0]
                    s = cell["std"].values[0]
                    row[f"{DATASET_LABELS.get(ds,ds)}_{RANGE_LABELS.get(rng,rng)}"] = \
                        f"{m:.2f} ± {s:.2f}" if not np.isnan(m) else "—"
        rows.append(row)

    df_csv = pd.DataFrame(rows)
    csv_name = f"main_results_{metric}{'_'+suffix if suffix else ''}.csv"
    df_csv.to_csv(TABLES_DIR / csv_name, index=False)
    log.info(f"  CSV: {csv_name}")

    # LaTeX (opcional, mismo esquema)
    # ... (código LaTeX como el original pero con sufijo en nombre) ...


# ── T2: Ranking table ─────────────────────────────────────────────────────────

def generate_ranking_table(master: pd.DataFrame, metric: str = "RMSE",
                           rng: str = "ENTIRE", suffix: str = "") -> None:
    sub = master[(master["metric"]==metric)&(master["range"]==rng)].copy()
    if sub.empty: return
    datasets = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
    conds    = _sort_conds(sub["condition"].unique().tolist())
    asc      = metric in LOWER_IS_BETTER

    rank_data = {}
    for ds in datasets:
        ds_sub = sub[sub["dataset"]==ds].sort_values("mean", ascending=asc)
        rank_data[ds] = {row["condition"]: (i+1, row["mean"], row["std"])
                         for i, (_, row) in enumerate(ds_sub.iterrows())}
    rows = []
    for cond in conds:
        row = {"Condition": _cond_label(cond)}
        for ds in datasets:
            if cond in rank_data.get(ds,{}):
                rank, mean, std = rank_data[ds][cond]
                row[DATASET_LABELS.get(ds,ds)] = f"#{rank}  ({mean:.2f}±{std:.2f})"
            else:
                row[DATASET_LABELS.get(ds,ds)] = "—"
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = TABLES_DIR / f"ranking_{metric}_{rng}{'_'+suffix if suffix else ''}.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"  CSV: {out_csv.name}")


# ── T3: Delta table ───────────────────────────────────────────────────────────

def generate_delta_table(master: pd.DataFrame, metric: str = "RMSE",
                         rng: str = "ENTIRE", suffix: str = "") -> None:
    sub = master[(~master["is_original"])&(master["metric"]==metric)&(master["range"]==rng)].copy()
    if sub.empty: return

    pivot = sub.pivot_table(index="condition", columns="dataset",
                            values="improvement_pct", aggfunc="mean")
    datasets = [d for d in DATASET_ORDER if d in pivot.columns]
    conds    = _sort_conds(list(pivot.index))
    pivot    = pivot[datasets]

    # Generar CSV simple
    records = []
    for cond in conds:
        row = {"Condition": _cond_label(cond)}
        for ds in datasets:
            v = pivot.loc[cond, ds] if cond in pivot.index else np.nan
            row[DATASET_LABELS.get(ds,ds)] = f"{v:+.1f}%" if not np.isnan(v) else "—"
        records.append(row)
    df = pd.DataFrame(records)
    out_csv = TABLES_DIR / f"delta_{metric}_{rng}{'_'+suffix if suffix else ''}.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"  CSV: {out_csv.name}")

    # LaTeX con colores (similar al original)


# ── T4: Clarke zones table ────────────────────────────────────────────────────

def generate_clarke_table(master: pd.DataFrame, suffix: str = "") -> None:
    sub = master[(master["metric"].isin(["A","A + B"]))&(master["range"]=="ENTIRE")].copy()
    if sub.empty: return
    datasets = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
    conds    = _sort_conds(sub["condition"].unique().tolist())
    rows = []
    for cond in conds:
        row = {"Condition": _cond_label(cond)}
        for ds in datasets:
            for zone in ["A","A + B"]:
                cell = sub[(sub["condition"]==cond)&(sub["dataset"]==ds)&(sub["metric"]==zone)]
                if cell.empty:
                    row[f"{DATASET_LABELS.get(ds,ds)} Zone {zone}"] = "—"
                else:
                    m = cell["mean"].values[0]; s = cell["std"].values[0]
                    row[f"{DATASET_LABELS.get(ds,ds)} Zone {zone}"] = f"{m:.1f} ± {s:.1f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = TABLES_DIR / f"clarke_zones{'_'+suffix if suffix else ''}.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"  CSV: {out_csv.name}")


# ── Intra‑family tables ──────────────────────────────────────────────────────

def generate_all_tables(master: pd.DataFrame, fold_long: pd.DataFrame = None) -> None:
    log.info("  Generando tablas...")

    # Globales
    for metric in ["RMSE", "MAE"]:
        generate_main_results_table(master, metric=metric)
        generate_ranking_table(master, metric=metric, rng="ENTIRE")
        generate_ranking_table(master, metric=metric, rng="TBR_2")
        generate_delta_table(master, metric=metric, rng="ENTIRE")
        generate_delta_table(master, metric=metric, rng="TBR_1")
    generate_clarke_table(master)

    # Intra‑familia
    for family_col in ["family_size", "family_mechanism"]:
        for fam in master[family_col].dropna().unique():
            sub = master[master[family_col] == fam]
            if sub.empty: continue
            suf = f"{family_col}_{fam}"
            for metric in ["RMSE", "MAE"]:
                generate_main_results_table(sub, metric=metric, suffix=suf)
                generate_ranking_table(sub, metric=metric, rng="ENTIRE", suffix=suf)
                generate_ranking_table(sub, metric=metric, rng="TBR_2", suffix=suf)
                generate_delta_table(sub, metric=metric, rng="ENTIRE", suffix=suf)
                generate_delta_table(sub, metric=metric, rng="TBR_1", suffix=suf)
            generate_clarke_table(sub, suffix=suf)

    log.info("  Tablas completadas.")