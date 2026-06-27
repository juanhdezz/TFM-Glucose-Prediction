# -*- coding: utf-8 -*-
"""
loader.py  —  Descubrimiento y carga de todos los experimentos.

Estructura real de directorios:
  results/
    {DATASET}/
      {balancing_condition}/          e.g. "balanced_age_jittering", "original"
        EXP-{date}-...-v{N}/
          results/evaluation/metrics/
            metrics_performance_by_range_RMSE_LSTM_H4_ALL_mta_report.csv
            metrics_performance_by_range_RMSE_LSTM_H4_Fold1.csv  ... Fold5.csv
            number_of_points_by_zones_*
            predictions_out_of_limits_*

Estructura real de mta_report.csv:
    Columnas: Range, A, B, C, D, E, A + B, RMSE, MSE, MAE, MAPE
    Filas:    ENTIRE, TBR_2, TBR_1, TIR, TAR_1, TAR_2
    Valores:  "mean ± std"  (strings)

Estructura real de FoldN.csv:
    Mismas columnas pero valores float, más columna Fold.
"""

import re
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from analysis.config import OUTPUT_ROOT, N_FOLDS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parser de balancing_condition
# ---------------------------------------------------------------------------

_BALANCED_RE = re.compile(
    r"^balanced_(?P<dimension>age|sex)_(?P<technique>.+)$"
)


def parse_condition(condition: str) -> dict:
    """
    'original'                            -> {dimension: None,  technique: 'original'}
    'balanced_age_jittering'              -> {dimension: 'age', technique: 'jittering'}
    'balanced_sex_patient_aware_...'      -> {dimension: 'sex', technique: 'patient_aware_...'}
    """
    if condition == "original":
        return {"dimension": None, "technique": "original"}
    m = _BALANCED_RE.match(condition)
    if not m:
        log.warning(f"Condición no reconocida: '{condition}' — tratada como técnica desconocida")
        return {"dimension": None, "technique": condition}
    return {
        "dimension": m.group("dimension"),
        "technique": m.group("technique"),
    }


# ---------------------------------------------------------------------------
# Parseo de celdas "mean ± std"
# ---------------------------------------------------------------------------

_PLUSMINUS_RE = re.compile(r"^\s*(?P<mean>[\d.]+)\s*±\s*(?P<std>[\d.]+)\s*$")


def parse_mean_std(cell: str) -> tuple[float, float]:
    """
    '45.98 ± 2.11'  ->  (45.98, 2.11)
    Retorna (NaN, NaN) si el formato no coincide.
    """
    if pd.isna(cell):
        return np.nan, np.nan
    m = _PLUSMINUS_RE.match(str(cell).strip())
    if not m:
        log.debug(f"Celda sin formato mean±std: '{cell}'")
        return np.nan, np.nan
    return float(m.group("mean")), float(m.group("std"))


def split_mta_report(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recibe el mta_report con celdas 'mean ± std'.
    Devuelve dos DataFrames con los mismos índices/columnas:
        df_mean: solo medias (float)
        df_std:  solo desviaciones (float)
    """
    metric_cols = [c for c in df.columns if c != "Range"]
    df_mean = df[["Range"]].copy()
    df_std  = df[["Range"]].copy()

    for col in metric_cols:
        means, stds = zip(*df[col].map(parse_mean_std))
        df_mean[col] = list(means)
        df_std[col]  = list(stds)

    return df_mean, df_std


# ---------------------------------------------------------------------------
# Descubrimiento de directorios
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"-v(\d+)$")


def _latest_exp_dir(condition_dir: Path) -> Path | None:
    """Devuelve el subdirectorio EXP-...-vN con mayor N."""
    candidates = [
        d for d in condition_dir.iterdir()
        if d.is_dir() and d.name.startswith("EXP-")
    ]
    if not candidates:
        return None
    def version_num(p: Path) -> int:
        m = _VERSION_RE.search(p.name)
        return int(m.group(1)) if m else 0
    return max(candidates, key=version_num)


def discover_experiments() -> list[dict]:
    """
    Recorre OUTPUT_ROOT y devuelve lista de dicts con metadatos de cada experimento.

    Cada dict:
        dataset       str    "DIATREND"
        condition     str    "balanced_age_jittering"
        dimension     str|None   "age" | "sex" | None
        technique     str    "jittering" | "original"
        exp_dir       Path
        metrics_dir   Path
    """
    if not OUTPUT_ROOT.exists():
        log.error(f"OUTPUT_ROOT no existe: {OUTPUT_ROOT}")
        return []

    experiments = []

    for dataset_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name

        for condition_dir in sorted(dataset_dir.iterdir()):
            if not condition_dir.is_dir():
                continue
            condition = condition_dir.name
            parsed    = parse_condition(condition)

            exp_dir = _latest_exp_dir(condition_dir)
            if exp_dir is None:
                log.warning(f"Sin dir EXP en: {condition_dir}")
                continue

            metrics_dir = exp_dir / "results" / "evaluation" / "metrics"
            if not metrics_dir.exists():
                log.warning(f"Sin metrics/ en: {exp_dir.name}")
                continue

            experiments.append({
                "dataset":     dataset,
                "condition":   condition,
                "dimension":   parsed["dimension"],
                "technique":   parsed["technique"],
                "is_original": parsed["technique"] == "original",
                "exp_dir":     exp_dir,
                "metrics_dir": metrics_dir,
            })

    log.info(f"Experimentos descubiertos: {len(experiments)}")
    return experiments


# ---------------------------------------------------------------------------
# Carga de CSVs individuales
# ---------------------------------------------------------------------------

def _find_csv(metrics_dir: Path, glob: str) -> Path | None:
    hits = sorted(metrics_dir.glob(glob))
    if not hits:
        log.debug(f"No encontrado '{glob}' en {metrics_dir.parent.parent.parent.name}")
        return None
    if len(hits) > 1:
        log.warning(f"Múltiples hits para '{glob}': {[h.name for h in hits]}")
    return hits[0]


def _load_mta_report(metrics_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    """
    Carga el mta_report de métricas de rendimiento.
    Devuelve (df_mean, df_std) con floats separados.
    """
    path = _find_csv(metrics_dir, "metrics_performance_by_range*_mta_report.csv")
    if path is None:
        return None, None
    try:
        df = pd.read_csv(path)
        return split_mta_report(df)
    except Exception as e:
        log.error(f"Error leyendo mta_report {path.name}: {e}")
        return None, None


def _load_fold_metrics(metrics_dir: Path) -> list[pd.DataFrame]:
    """
    Carga los N CSVs de folds individuales de métricas de rendimiento.
    Cada uno tiene columna Fold con valor 1..N y valores float directos.
    """
    dfs = []
    for fold_n in range(1, N_FOLDS + 1):
        path = _find_csv(metrics_dir, f"metrics_performance_by_range*_Fold{fold_n}.csv")
        if path is None:
            log.warning(f"Fold {fold_n} no encontrado en {metrics_dir.parent.parent.parent.name}")
            continue
        try:
            df = pd.read_csv(path)
            # Asegurar que la columna Fold existe y tiene el valor correcto
            df["Fold"] = fold_n
            dfs.append(df)
        except Exception as e:
            log.error(f"Error leyendo Fold{fold_n}: {e}")
    return dfs


def _load_npoints_mta(metrics_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    """Carga número de puntos por zona (Clarke EG) — mta_report."""
    path = _find_csv(metrics_dir, "number_of_points_by_zones*_mta_report.csv")
    if path is None:
        return None, None
    try:
        df = pd.read_csv(path)
        return split_mta_report(df)
    except Exception as e:
        log.error(f"Error leyendo npoints mta_report: {e}")
        return None, None


def _load_out_of_limits_mta(metrics_dir: Path) -> pd.DataFrame | None:
    """Carga predicciones fuera de límites del sensor — mta_report."""
    path = _find_csv(metrics_dir, "predictions_out_of_limits*_mta_report.csv")
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        log.error(f"Error leyendo out_of_limits: {e}")
        return None


# ---------------------------------------------------------------------------
# Entrada pública
# ---------------------------------------------------------------------------

def load_all_experiments() -> list[dict]:
    """
    Descubre y carga todos los experimentos.

    Devuelve lista de dicts:
    {
        metadata:         dict  (dataset, condition, dimension, technique, is_original, paths)
        perf_mean:        pd.DataFrame | None   (mta_report, medias)
        perf_std:         pd.DataFrame | None   (mta_report, std)
        fold_metrics:     list[pd.DataFrame]    (Fold1..N, valores float)
        npoints_mean:     pd.DataFrame | None
        npoints_std:      pd.DataFrame | None
        out_of_limits:    pd.DataFrame | None
    }
    """
    experiments = discover_experiments()
    loaded = []

    for meta in experiments:
        md = meta["metrics_dir"]

        perf_mean, perf_std   = _load_mta_report(md)
        npts_mean, npts_std   = _load_npoints_mta(md)
        fold_dfs              = _load_fold_metrics(md)
        out                   = _load_out_of_limits_mta(md)

        entry = {
            "metadata":      meta,
            "perf_mean":     perf_mean,
            "perf_std":      perf_std,
            "fold_metrics":  fold_dfs,
            "npoints_mean":  npts_mean,
            "npoints_std":   npts_std,
            "out_of_limits": out,
        }

        n_folds_loaded = len(fold_dfs)
        if perf_mean is None:
            log.warning(f"[{meta['dataset']}/{meta['condition']}] Sin mta_report!")
        if n_folds_loaded < N_FOLDS:
            log.warning(f"[{meta['dataset']}/{meta['condition']}] Solo {n_folds_loaded}/{N_FOLDS} folds cargados")

        loaded.append(entry)
        log.debug(f"OK: {meta['dataset']}/{meta['condition']}  ({n_folds_loaded} folds)")

    log.info(f"Total cargados: {len(loaded)} experimentos")
    return loaded