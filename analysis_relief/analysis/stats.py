# -*- coding: utf-8 -*-
"""
stats.py  —  Tests estadísticos y rankings.

Genera en outputs/stats/:
  - friedman_results.csv       Test de Friedman por dataset × métrica
  - nemenyi_*.csv              Matrices de p-valores post-hoc (cuando p<α)
  - cohens_d_vs_original.csv   Tamaño del efecto de cada técnica vs original
  - rankings_*.csv             Rankings por dataset y global (Borda)
"""

import logging
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats as ss

from analysis.config import (
    STATS_DIR, DATASETS, ANALYSIS_METRICS, N_FOLDS, ALPHA,
    PRIMARY_METRIC, RANGE_ORDER, LOWER_IS_BETTER, TECHNIQUE_LABELS,
)
from analysis.preprocessing import make_friedman_pivot

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Friedman + Nemenyi
# ---------------------------------------------------------------------------

def run_friedman(fold_long: pd.DataFrame) -> pd.DataFrame:
    """
    Test de Friedman para cada combinación de (dataset, range, metric).
    Devuelve DataFrame con resultados y guarda CSV.
    """
    results = []

    datasets  = fold_long["dataset"].unique()
    ranges    = [r for r in RANGE_ORDER if r in fold_long["range"].unique()]
    metrics   = [m for m in ANALYSIS_METRICS if m in fold_long["metric"].unique()]

    for dataset in datasets:
        for rng in ranges:
            for metric in metrics:
                pivot = make_friedman_pivot(fold_long, metric=metric, rng=rng, dataset=dataset)

                if pivot.empty or pivot.shape[1] < 3:
                    continue  # Friedman requiere ≥3 grupos

                try:
                    stat, pval = ss.friedmanchisquare(
                        *[pivot[c].values for c in pivot.columns]
                    )
                except Exception as e:
                    log.warning(f"Friedman error ({dataset}/{rng}/{metric}): {e}")
                    continue

                results.append({
                    "dataset":         dataset,
                    "range":           rng,
                    "metric":          metric,
                    "n_conditions":    pivot.shape[1],
                    "n_folds":         pivot.shape[0],
                    "friedman_stat":   round(stat, 4),
                    "friedman_pval":   round(pval, 6),
                    "significant":     pval < ALPHA,
                })

                # Post-hoc Nemenyi si significativo
                if pval < ALPHA:
                    _run_nemenyi(pivot, dataset, rng, metric)

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        out = STATS_DIR / "friedman_results.csv"
        df_results.to_csv(out, index=False)
        n_sig = df_results["significant"].sum()
        log.info(f"Friedman: {len(df_results)} tests, {n_sig} significativos → {out.name}")

    return df_results


def _run_nemenyi(pivot: pd.DataFrame, dataset: str, rng: str, metric: str) -> None:
    """Calcula post-hoc Nemenyi y guarda CSV."""
    try:
        import scikit_posthocs as sp
        nemenyi = sp.posthoc_nemenyi_friedman(pivot.values)
        nemenyi.index   = pivot.columns
        nemenyi.columns = pivot.columns
        fname = f"nemenyi_{dataset}_{rng}_{metric}.csv".replace(" ", "_").replace("+", "plus")
        nemenyi.round(4).to_csv(STATS_DIR / fname)
        log.info(f"  Nemenyi guardado: {fname}")
    except ImportError:
        log.warning("scikit-posthocs no instalado. pip install scikit-posthocs")
    except Exception as e:
        log.warning(f"Nemenyi error: {e}")


# ---------------------------------------------------------------------------
# Cohen's d vs original
# ---------------------------------------------------------------------------

def compute_cohens_d(fold_long: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula Cohen's d de cada condición balanceada vs el original
    del mismo dataset, para cada (range, metric).

    d = (mean_original - mean_balanced) / pooled_std
    d > 0: el balanceo reduce el error (mejora)
    d < 0: el balanceo aumenta el error (empeora)
    """
    results = []

    datasets  = fold_long["dataset"].unique()
    ranges    = [r for r in RANGE_ORDER if r in fold_long["range"].unique()]
    metrics   = [m for m in ANALYSIS_METRICS if m in fold_long["metric"].unique()]

    for dataset in datasets:
        ds = fold_long[fold_long["dataset"] == dataset]
        original_cond = ds[ds["is_original"]]

        for rng in ranges:
            for metric in metrics:
                orig_vals = original_cond[
                    (original_cond["range"]  == rng) &
                    (original_cond["metric"] == metric)
                ]["value"].dropna().values

                if len(orig_vals) < 2:
                    continue

                conditions = ds[~ds["is_original"]]["condition"].unique()

                for cond in conditions:
                    bal_vals = ds[
                        (ds["condition"] == cond) &
                        (ds["range"]     == rng) &
                        (ds["metric"]    == metric)
                    ]["value"].dropna().values

                    if len(bal_vals) < 2:
                        continue

                    # Pooled std
                    n1, n2 = len(orig_vals), len(bal_vals)
                    pooled = np.sqrt(
                        ((n1 - 1) * np.std(orig_vals, ddof=1)**2 +
                         (n2 - 1) * np.std(bal_vals, ddof=1)**2)
                        / (n1 + n2 - 2)
                    )
                    d = (np.mean(orig_vals) - np.mean(bal_vals)) / pooled if pooled > 0 else np.nan

                    # Metadatos de la condición
                    cond_meta = ds[ds["condition"] == cond].iloc[0]

                    results.append({
                        "dataset":     dataset,
                        "condition":   cond,
                        "dimension":   cond_meta["dimension"],
                        "technique":   cond_meta["technique"],
                        "range":       rng,
                        "metric":      metric,
                        "mean_original":  round(np.mean(orig_vals), 4),
                        "mean_balanced":  round(np.mean(bal_vals),  4),
                        "cohens_d":       round(d, 4) if not np.isnan(d) else np.nan,
                        "effect_size":    _interpret_d(d),
                    })

    df = pd.DataFrame(results)
    if not df.empty:
        out = STATS_DIR / "cohens_d_vs_original.csv"
        df.to_csv(out, index=False)
        log.info(f"Cohen's d: {len(df)} filas → {out.name}")

    return df


def _interpret_d(d: float) -> str:
    if np.isnan(d):
        return "—"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------

def compute_rankings(master: pd.DataFrame) -> pd.DataFrame:
    """
    Ranking por (dataset, metric, range) basado en la media del mta_report.
    Para métricas LOWER_IS_BETTER: menor valor = mejor posición.

    Devuelve DataFrame con columnas:
        dataset, condition, dimension, technique, range, metric, mean, rank
    """
    results = []

    for (dataset, rng, metric), grp in master.groupby(["dataset", "range", "metric"]):
        ascending = metric in LOWER_IS_BETTER
        ranked = grp.sort_values("mean", ascending=ascending).copy()
        ranked["rank"] = range(1, len(ranked) + 1)
        results.append(ranked)

    if not results:
        return pd.DataFrame()

    df = pd.concat(results, ignore_index=True)
    out = STATS_DIR / "rankings_by_dataset_range_metric.csv"
    df[["dataset", "condition", "dimension", "technique",
        "range", "metric", "mean", "std", "rank"]].to_csv(out, index=False)
    log.info(f"Rankings guardados: {out.name}")
    return df


def compute_borda_ranking(master: pd.DataFrame,
                          metrics: list[str] | None = None,
                          ranges: list[str] | None = None) -> pd.DataFrame:
    """
    Ranking de Borda multicriterio.

    Para cada combinación (dataset, metric, range) se asigna posición a cada condición.
    El score de Borda es la suma de posiciones (menor = mejor).

    Parámetros:
        metrics: lista de métricas a incluir (default: ANALYSIS_METRICS)
        ranges:  lista de rangos a incluir (default: todos excepto ENTIRE duplicado)
    """
    from analysis.config import ANALYSIS_METRICS as DEFAULT_METRICS

    if metrics is None:
        metrics = DEFAULT_METRICS
    if ranges is None:
        ranges = RANGE_ORDER

    sub = master[
        master["metric"].isin(metrics) &
        master["range"].isin(ranges)
    ].copy()

    borda_rows = []

    for (dataset, rng, metric), grp in sub.groupby(["dataset", "range", "metric"]):
        ascending = metric in LOWER_IS_BETTER
        ranked = grp.sort_values("mean", ascending=ascending).copy()
        ranked["borda_score"] = range(1, len(ranked) + 1)

        for _, row in ranked.iterrows():
            borda_rows.append({
                "dataset":      dataset,
                "condition":    row["condition"],
                "dimension":    row["dimension"],
                "technique":    row["technique"],
                "range":        rng,
                "metric":       metric,
                "borda_score":  row["borda_score"],
            })

    df_borda = pd.DataFrame(borda_rows)
    if df_borda.empty:
        return df_borda

    # Suma de scores por (dataset, condition) — menor = mejor
    borda_summary = (
        df_borda.groupby(["dataset", "condition", "dimension", "technique"])
        ["borda_score"]
        .sum()
        .reset_index()
        .sort_values(["dataset", "borda_score"])
    )
    borda_summary["global_rank"] = borda_summary.groupby("dataset")["borda_score"].rank(
        method="min", ascending=True
    ).astype(int)

    out = STATS_DIR / "borda_ranking.csv"
    borda_summary.to_csv(out, index=False)
    log.info(f"Borda ranking: {out.name}")
    return borda_summary


# ---------------------------------------------------------------------------
# Entrada pública
# ---------------------------------------------------------------------------

def run_all_stats(master: pd.DataFrame, fold_long: pd.DataFrame) -> dict:
    """
    Ejecuta todo el pipeline estadístico.
    Devuelve dict con todos los DataFrames de resultados.
    """
    log.info("--- Estadísticos ---")
    out = {}

    out["friedman"]  = run_friedman(fold_long)
    out["cohens_d"]  = compute_cohens_d(fold_long)
    out["rankings"]  = compute_rankings(master)
    out["borda"]     = compute_borda_ranking(master)

    # Guardar tabla master
    master.to_csv(STATS_DIR / "master_table.csv", index=False)
    fold_long.to_csv(STATS_DIR / "fold_table.csv", index=False)
    log.info("Tablas master y fold guardadas en stats/")

    return out