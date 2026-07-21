# -*- coding: utf-8 -*-
"""
preprocessing.py  —  Construye el DataFrame maestro y calcula deltas.

Genera dos DataFrames públicos:

1. master  — Una fila por (dataset × condition × range × metric).
   Columnas: dataset, condition, dimension, technique, is_original,
             range, metric, mean, std, delta_abs, delta_pct

2. fold_long — Una fila por (dataset × condition × fold × range × metric).
   Columnas: dataset, condition, dimension, technique, is_original,
             fold, range, metric, value
   Usado para tests estadísticos (Friedman, Nemenyi) y plots de distribución.
"""

import logging
import numpy as np
import pandas as pd

from analysis.config import (
    ANALYSIS_METRICS, ERROR_METRICS, CEG_METRICS, RANGE_ORDER, PRIMARY_METRIC,FAMILY_SIZE,FAMILY_MECHANISM
)

log = logging.getLogger(__name__)

# Todas las columnas métricas presentes en los CSVs (en orden original)
ALL_METRIC_COLS = ["A", "B", "C", "D", "E", "A + B", "RMSE", "MSE", "MAE", "MAPE"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _meta_fields(exp: dict) -> dict:
    m = exp["metadata"]
    tech = m["technique"]
    return {
        "dataset":     m["dataset"],
        "condition":   m["condition"],
        "dimension":   m["dimension"],
        "technique":   tech,
        "is_original": m["is_original"],
        "family_size":       FAMILY_SIZE.get(tech, "unknown"),
        "family_mechanism":  FAMILY_MECHANISM.get(tech, "unknown"),
    }


def _melt_perf(df_mean: pd.DataFrame, df_std: pd.DataFrame, meta: dict) -> list[dict]:
    """
    Transforma df_mean y df_std (columnas=métricas, filas=rangos)
    en lista de dicts con una entrada por (range, metric).
    """
    rows = []
    metric_cols = [c for c in df_mean.columns if c != "Range"]

    for _, row_mean in df_mean.iterrows():
        rng = row_mean["Range"]
        # Buscar la std correspondiente (mismo Range)
        std_row = df_std[df_std["Range"] == rng]

        for col in metric_cols:
            mean_val = row_mean[col]
            std_val  = std_row[col].values[0] if not std_row.empty else np.nan

            rows.append({
                **meta,
                "range":  rng,
                "metric": col,
                "mean":   float(mean_val) if pd.notna(mean_val) else np.nan,
                "std":    float(std_val)  if pd.notna(std_val)  else np.nan,
            })
    return rows


def _melt_folds(fold_dfs: list[pd.DataFrame], meta: dict) -> list[dict]:
    """
    Transforma la lista de DataFrames de folds individuales en lista de dicts.
    Una entrada por (fold, range, metric).
    """
    rows = []
    for df_fold in fold_dfs:
        fold_n = int(df_fold["Fold"].iloc[0]) if "Fold" in df_fold.columns else -1
        metric_cols = [c for c in df_fold.columns if c not in ("Range", "Fold")]

        for _, row in df_fold.iterrows():
            rng = row["Range"]
            for col in metric_cols:
                rows.append({
                    **meta,
                    "fold":   fold_n,
                    "range":  rng,
                    "metric": col,
                    "value":  float(row[col]) if pd.notna(row[col]) else np.nan,
                })
    return rows


# ---------------------------------------------------------------------------
# Construcción de tablas maestras
# ---------------------------------------------------------------------------

def build_master_table(experiments: list[dict]) -> pd.DataFrame:
    """
    DataFrame con una fila por (dataset × condition × range × metric).
    Fuente: mta_report (mean ± std).
    """
    rows = []
    for exp in experiments:
        meta = _meta_fields(exp)

        if exp["perf_mean"] is None:
            log.warning(f"Sin perf_mean: {meta['dataset']}/{meta['condition']}")
            continue

        rows.extend(_melt_perf(exp["perf_mean"], exp["perf_std"], meta))

    master = pd.DataFrame(rows)

    if master.empty:
        log.error("Master table vacía — verificar OUTPUT_ROOT y estructura de directorios.")
        return master

    # Ordenar rangos y métricas canónicamente
    master["range"]  = pd.Categorical(master["range"],  categories=RANGE_ORDER,      ordered=True)
    master["metric"] = pd.Categorical(master["metric"], categories=ALL_METRIC_COLS,  ordered=True)
    master = master.sort_values(["dataset", "condition", "range", "metric"]).reset_index(drop=True)

    log.info(f"Master table: {master.shape[0]} filas, "
             f"{master['condition'].nunique()} condiciones, "
             f"{master['dataset'].nunique()} datasets")
    return master


def build_fold_table(experiments: list[dict]) -> pd.DataFrame:
    """
    DataFrame con una fila por (dataset × condition × fold × range × metric).
    Fuente: Fold1..N CSVs (floats directos).
    Usado para tests estadísticos y distribuciones.
    """
    rows = []
    for exp in experiments:
        meta = _meta_fields(exp)
        fold_dfs = exp.get("fold_metrics", [])
        if not fold_dfs:
            log.warning(f"Sin folds: {meta['dataset']}/{meta['condition']}")
            continue
        rows.extend(_melt_folds(fold_dfs, meta))

    fold_long = pd.DataFrame(rows)

    if not fold_long.empty:
        fold_long["range"]  = pd.Categorical(fold_long["range"],  categories=RANGE_ORDER,     ordered=True)
        fold_long["metric"] = pd.Categorical(fold_long["metric"], categories=ALL_METRIC_COLS, ordered=True)
        fold_long = fold_long.sort_values(
            ["dataset", "condition", "fold", "range", "metric"]
        ).reset_index(drop=True)

    log.info(f"Fold table: {fold_long.shape[0]} filas")
    return fold_long


# ---------------------------------------------------------------------------
# Cálculo de deltas
# ---------------------------------------------------------------------------

def compute_deltas(master: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas delta_abs y delta_pct al master DataFrame.

    delta_abs = mean_balanceado - mean_original   (para el mismo dataset/range/metric)
    delta_pct = delta_abs / mean_original * 100

    Convención de signo para métricas donde MENOR = MEJOR (RMSE, MAE, MAPE, MSE):
        delta_pct < 0  →  MEJORA  (el balanceo reduce el error)
        delta_pct > 0  →  EMPEORA

    Para CEG zonas (MAYOR = MEJOR, e.g. % puntos en zona A):
        delta_pct > 0  →  MEJORA
    """
    if master.empty:
        return master

    # Extraer baseline (is_original == True)
    baseline = (
        master[master["is_original"]]
        [["dataset", "range", "metric", "mean"]]
        .rename(columns={"mean": "baseline_mean"})
    )

    master = master.merge(
        baseline,
        on=["dataset", "range", "metric"],
        how="left",
    )

    master["delta_abs"] = master["mean"] - master["baseline_mean"]

    master["delta_pct"] = np.where(
        master["baseline_mean"].notna() & (master["baseline_mean"] != 0),
        master["delta_abs"] / master["baseline_mean"] * 100,
        np.nan,
    )

    # improvement_pct: siempre positivo = mejor (normaliza signo)
    # Para LOWER_IS_BETTER: improvement = -delta_pct
    # Para HIGHER_IS_BETTER: improvement = +delta_pct
    from analysis.config import LOWER_IS_BETTER, HIGHER_IS_BETTER_CEG
    master["improvement_pct"] = np.where(
        master["metric"].isin(LOWER_IS_BETTER),
        -master["delta_pct"],
        np.where(
            master["metric"].isin(HIGHER_IS_BETTER_CEG),
            master["delta_pct"],
            np.nan,
        )
    )

    n_with_delta = master["delta_abs"].notna().sum()
    log.info(f"Deltas calculados: {n_with_delta}/{len(master)} filas con baseline.")
    return master


# ---------------------------------------------------------------------------
# Tabla pivot para estadísticos
# ---------------------------------------------------------------------------

def make_friedman_pivot(
    fold_long: pd.DataFrame,
    metric: str = "RMSE",
    rng: str = "ENTIRE",
    dataset: str | None = None,
) -> pd.DataFrame:
    """
    Construye tabla pivot para test de Friedman.

    Filas: folds (1..N)
    Columnas: condiciones (técnicas)
    Valores: métrica en ese fold/rango

    Si dataset es None, usa todos los datasets (suma/promedio entre datasets).
    """
    sub = fold_long[
        (fold_long["metric"] == metric) &
        (fold_long["range"]  == rng)
    ].copy()

    if dataset is not None:
        sub = sub[sub["dataset"] == dataset]

    if sub.empty:
        log.warning(f"Sin datos para Friedman: metric={metric}, range={rng}, dataset={dataset}")
        return pd.DataFrame()

    pivot = sub.pivot_table(
        index="fold",
        columns="condition",
        values="value",
        aggfunc="mean",   # por si hay múltiples datasets
    )

    # Eliminar columnas con NaN (condiciones incompletas)
    pivot = pivot.dropna(axis=1)

    return pivot