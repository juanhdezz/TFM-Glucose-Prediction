#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_analysis_project.py
=========================
Genera la estructura de directorios y ficheros esqueleto del sistema
de análisis experimental RELIEF-T1D.

Uso:
    python setup_analysis_project.py [--root /ruta/a/tu/proyecto]

Si no se especifica --root, crea la carpeta 'analysis_relief/' en el
directorio actual.
"""

import argparse
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Definición de la estructura
# ---------------------------------------------------------------------------

DIRS = [
    "analysis",
    "analysis/viz",
    "analysis/tests",
    "outputs",
    "outputs/figures",
    "outputs/tables",
    "outputs/dashboards",
    "outputs/stats",
]

# (path_relativo, contenido)
FILES = [

    # -----------------------------------------------------------------------
    # Raíz del proyecto
    # -----------------------------------------------------------------------
    ("README.md", """\
# RELIEF-T1D — Sistema de Análisis Experimental

Análisis completo de los resultados de 39 experimentos LSTM
sobre 3 datasets de Diabetes Tipo 1, con 6 técnicas de balanceo
aplicadas sobre 2 dimensiones demográficas (Age / Sex).

## Estructura

```
analysis/           Código fuente del sistema de análisis
outputs/            Figuras, tablas y estadísticos generados
main.py             Punto de entrada principal
setup_analysis_project.py   Este script de inicialización
```

## Uso rápido

```bash
# 1. Configurar rutas en analysis/config.py
# 2. Ejecutar pipeline completo
python main.py

# Ejecutar solo carga y preprocesado
python main.py --only load

# Ejecutar solo visualizaciones
python main.py --only viz

# Ejecutar solo estadísticos
python main.py --only stats

# Ejecutar solo tablas
python main.py --only tables
```

## Requisitos

```bash
pip install pandas numpy scipy scikit-posthocs matplotlib seaborn
pip install ptitprince  # raincloud plots
```
"""),

    ("requirements.txt", """\
pandas>=2.0
numpy>=1.24
scipy>=1.11
scikit-posthocs>=0.9
matplotlib>=3.8
seaborn>=0.13
ptitprince>=0.2
"""),

    # -----------------------------------------------------------------------
    # main.py
    # -----------------------------------------------------------------------
    ("main.py", '''\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py  —  Orquestador del pipeline de análisis RELIEF-T1D.

Fases:
  1. load    Descubrimiento y carga de todos los CSVs en un DataFrame maestro.
  2. stats   Tests estadísticos (Friedman, Nemenyi, Cohen d, rankings).
  3. viz     Generación de todas las figuras.
  4. tables  Tablas LaTeX / CSV para el TFM.
"""

import argparse
import logging
from pathlib import Path

from analysis.loader import load_all_experiments
from analysis.preprocessing import build_master_table, compute_deltas
from analysis.stats import run_all_stats
from analysis.tables import generate_all_tables
from analysis.dashboards import generate_all_dashboards
from analysis.viz.heatmaps import plot_all_heatmaps
from analysis.viz.dumbbell import plot_all_dumbbells
from analysis.viz.distributions import plot_all_distributions
from analysis.viz.rankings import plot_all_rankings
from analysis.viz.profiles import plot_all_profiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Pipeline de análisis RELIEF-T1D")
    p.add_argument(
        "--only",
        choices=["load", "stats", "viz", "tables", "dashboards"],
        default=None,
        help="Ejecutar solo una fase del pipeline.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Mostrar logs de depuración.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    log.info("=== RELIEF-T1D  Analysis Pipeline ===")

    # --- Fase 1: Carga ---
    log.info("[1/4] Cargando experimentos...")
    raw = load_all_experiments()
    master = build_master_table(raw)
    master = compute_deltas(master)
    log.info(f"      {len(master)} filas cargadas ({master['experiment_id'].nunique()} experimentos).")

    if args.only == "load":
        log.info("Modo --only load. Saliendo.")
        return

    # --- Fase 2: Estadísticos ---
    if args.only in (None, "stats"):
        log.info("[2/4] Ejecutando análisis estadístico...")
        run_all_stats(master)

    if args.only == "stats":
        return

    # --- Fase 3: Visualizaciones ---
    if args.only in (None, "viz"):
        log.info("[3/4] Generando figuras...")
        plot_all_heatmaps(master)
        plot_all_dumbbells(master)
        plot_all_distributions(master)
        plot_all_rankings(master)
        plot_all_profiles(master)

    if args.only == "viz":
        return

    # --- Fase 4: Tablas ---
    if args.only in (None, "tables"):
        log.info("[4/4] Generando tablas...")
        generate_all_tables(master)

    # --- Dashboards (siempre al final) ---
    if args.only in (None, "dashboards"):
        log.info("[+]   Generando dashboards compuestos...")
        generate_all_dashboards(master)

    log.info("=== Pipeline completado. Outputs en outputs/ ===")


if __name__ == "__main__":
    main()
'''),

    # -----------------------------------------------------------------------
    # analysis/__init__.py
    # -----------------------------------------------------------------------
    ("analysis/__init__.py", ""),

    # -----------------------------------------------------------------------
    # analysis/config.py
    # -----------------------------------------------------------------------
    ("analysis/config.py", '''\
# -*- coding: utf-8 -*-
"""
config.py  —  Configuración central del sistema de análisis.

AJUSTA LAS RUTAS antes de ejecutar el pipeline.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# RUTAS  —  AJUSTA ESTO
# ---------------------------------------------------------------------------

#: Raíz del directorio de salida de experimentos
#: Debe contener subcarpetas: T1DiabetesGranada/, DIATREND/, REPLACE-BG/
OUTPUT_ROOT = Path("/home/juanhdez/data/output")

#: Carpeta donde se guardarán las figuras, tablas y estadísticos
ANALYSIS_OUT = Path(__file__).parent.parent / "outputs"

FIGURES_DIR   = ANALYSIS_OUT / "figures"
TABLES_DIR    = ANALYSIS_OUT / "tables"
STATS_DIR     = ANALYSIS_OUT / "stats"
DASHBOARDS_DIR = ANALYSIS_OUT / "dashboards"

for _d in [FIGURES_DIR, TABLES_DIR, STATS_DIR, DASHBOARDS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# NOMBRES CANÓNICOS
# ---------------------------------------------------------------------------

DATASETS = ["T1DiabetesGranada", "DIATREND", "REPLACE-BG"]

DIMENSIONS = ["age", "sex"]

#: Las 6 técnicas de balanceo —  ajusta a los nombres reales de tus ficheros
TECHNIQUES = [
    "oversampling",
    "undersampling",
    "smote",
    "smote_tomek",
    "adasyn",
    "borderline_smote",
]

#: Etiquetas de visualización para cada técnica
TECHNIQUE_LABELS = {
    "oversampling":      "Oversampling",
    "undersampling":     "Undersampling",
    "smote":             "SMOTE",
    "smote_tomek":       "SMOTE+Tomek",
    "adasyn":            "ADASYN",
    "borderline_smote":  "Borderline SMOTE",
    "original":          "Original",
}

DATASET_LABELS = {
    "T1DiabetesGranada": "T1DGranada",
    "DIATREND":          "DiaTrend",
    "REPLACE-BG":        "ReplaceBG",
}

DIMENSION_LABELS = {
    "age": "Age",
    "sex": "Sex",
}


# ---------------------------------------------------------------------------
# MÉTRICAS
# ---------------------------------------------------------------------------

#: Métrica primaria (función de pérdida de entrenamiento)
PRIMARY_METRIC = "RMSE_ALL"

#: Métricas para el análisis multicriterio
ANALYSIS_METRICS = [
    "RMSE_ALL",
    "RMSE_Hypoglycemia_L1",
    "RMSE_Hypoglycemia_L2",
    "RMSE_In_Range",
    "MAE_ALL",
]

#: Rangos glucémicos disponibles
GLUCOSE_RANGES = [
    "Hypoglycemia_L2",
    "Hypoglycemia_L1",
    "In_Range",
    "Hyperglycemia_L1",
    "Hyperglycemia_L2",
    "ALL",
]

#: Número de folds
N_FOLDS = 5

#: Nivel de significancia estadística
ALPHA = 0.05


# ---------------------------------------------------------------------------
# ESTILO VISUAL
# ---------------------------------------------------------------------------

#: Paleta Wong (2011) — colorblind-safe, 7 colores
#: Asignación: original + 6 técnicas
PALETTE = {
    "original":          "#000000",   # negro
    "oversampling":      "#E69F00",   # naranja
    "undersampling":     "#56B4E9",   # azul cielo
    "smote":             "#009E73",   # verde
    "smote_tomek":       "#F0E442",   # amarillo
    "adasyn":            "#0072B2",   # azul oscuro
    "borderline_smote":  "#D55E00",   # rojo-naranja
}

DIMENSION_PALETTE = {
    "age": "#CC79A7",   # rosa/morado  (Wong)
    "sex": "#0072B2",   # azul oscuro  (Wong)
}

#: Configuración matplotlib base
MPL_RC = {
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.1,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
}
'''),

    # -----------------------------------------------------------------------
    # analysis/loader.py
    # -----------------------------------------------------------------------
    ("analysis/loader.py", '''\
# -*- coding: utf-8 -*-
"""
loader.py  —  Descubrimiento y carga de todos los CSVs de experimentos.

Estrategia:
  - Recorre OUTPUT_ROOT/{dataset}/{balancing_condition}/EXP-*/
  - Para cada experimento localiza el fichero
    metrics_performance_by_range_*_mta_report.csv  (fuente primaria)
    y los Fold1..Fold5 individuales (para tests estadísticos).
  - Devuelve un dict con DataFrames listos para preprocessing.py.

El parser de nombres infiere: dataset, dimension, technique
a partir del balancing_condition del directorio.
"""

import re
import logging
from pathlib import Path

import pandas as pd

from analysis.config import (
    OUTPUT_ROOT, DATASETS, N_FOLDS, TECHNIQUES, DIMENSIONS
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parser de balancing_condition
# ---------------------------------------------------------------------------

_COND_RE = re.compile(
    r"^balanced_(?P<dimension>age|sex)_(?P<technique>.+)$"
)


def parse_condition(condition: str) -> dict:
    """
    'original'                       -> {dimension: None,  technique: 'original'}
    'balanced_age_smote'             -> {dimension: 'age', technique: 'smote'}
    'balanced_sex_borderline_smote'  -> {dimension: 'sex', technique: 'borderline_smote'}
    """
    if condition == "original":
        return {"dimension": None, "technique": "original"}
    m = _COND_RE.match(condition)
    if not m:
        log.warning(f"Condición de balanceo no reconocida: '{condition}'")
        return {"dimension": None, "technique": condition}
    return {"dimension": m.group("dimension"), "technique": m.group("technique")}


# ---------------------------------------------------------------------------
# Descubrimiento de directorios de experimento
# ---------------------------------------------------------------------------

def _latest_exp_dir(condition_dir: Path) -> Path | None:
    """
    Dentro de condition_dir/ puede haber varias versiones EXP-...-vN.
    Devuelve la de mayor N (más reciente).
    """
    candidates = sorted(
        [d for d in condition_dir.iterdir() if d.is_dir() and d.name.startswith("EXP-")],
        key=lambda d: int(re.search(r"-v(\d+)$", d.name).group(1))
                      if re.search(r"-v(\d+)$", d.name) else 0,
        reverse=True,
    )
    return candidates[0] if candidates else None


def discover_experiments() -> list[dict]:
    """
    Devuelve lista de dicts con metadatos de cada experimento encontrado:
        dataset, condition, dimension, technique, exp_dir, metrics_dir
    """
    experiments = []

    if not OUTPUT_ROOT.exists():
        log.error(f"OUTPUT_ROOT no existe: {OUTPUT_ROOT}")
        return experiments

    for dataset_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name

        for condition_dir in sorted(dataset_dir.iterdir()):
            if not condition_dir.is_dir():
                continue
            condition = condition_dir.name
            parsed = parse_condition(condition)

            exp_dir = _latest_exp_dir(condition_dir)
            if exp_dir is None:
                log.warning(f"Sin directorio EXP en: {condition_dir}")
                continue

            metrics_dir = exp_dir / "results" / "evaluation" / "metrics"
            if not metrics_dir.exists():
                log.warning(f"Sin metrics/ en: {exp_dir}")
                continue

            experiments.append({
                "dataset":    dataset,
                "condition":  condition,
                "dimension":  parsed["dimension"],
                "technique":  parsed["technique"],
                "exp_dir":    exp_dir,
                "metrics_dir": metrics_dir,
            })

    log.info(f"Experimentos descubiertos: {len(experiments)}")
    return experiments


# ---------------------------------------------------------------------------
# Carga de CSVs
# ---------------------------------------------------------------------------

def _find_csv(metrics_dir: Path, pattern: str) -> Path | None:
    hits = list(metrics_dir.glob(pattern))
    if not hits:
        log.debug(f"No encontrado '{pattern}' en {metrics_dir}")
        return None
    if len(hits) > 1:
        log.warning(f"Múltiples hits para '{pattern}' en {metrics_dir}: {hits}")
    return hits[0]


def _load_mta_report(metrics_dir: Path) -> pd.DataFrame | None:
    """Carga el fichero mta_report (estadísticos agregados de los 5 folds)."""
    path = _find_csv(metrics_dir, "*_mta_report.csv")
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
        df["_source_file"] = path.name
        return df
    except Exception as e:
        log.error(f"Error leyendo {path}: {e}")
        return None


def _load_fold_metrics(metrics_dir: Path) -> list[pd.DataFrame]:
    """Carga los 5 CSVs de fold individuales (Metric-Then-Aggregate por fold)."""
    dfs = []
    for fold_n in range(1, N_FOLDS + 1):
        path = _find_csv(metrics_dir, f"*_Fold{fold_n}.csv")
        if path is None:
            continue
        try:
            df = pd.read_csv(path)
            df["fold"] = fold_n
            df["_source_file"] = path.name
            dfs.append(df)
        except Exception as e:
            log.error(f"Error leyendo {path}: {e}")
    return dfs


def _load_npoints(metrics_dir: Path) -> pd.DataFrame | None:
    """Carga number_of_points_by_zones mta_report."""
    path = _find_csv(metrics_dir, "number_of_points_by_zones*_mta_report.csv")
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        log.error(f"Error leyendo {path}: {e}")
        return None


def _load_out_of_limits(metrics_dir: Path) -> pd.DataFrame | None:
    """Carga predictions_out_of_limits mta_report."""
    path = _find_csv(metrics_dir, "predictions_out_of_limits*_mta_report.csv")
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        log.error(f"Error leyendo {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Entrada pública
# ---------------------------------------------------------------------------

def load_all_experiments() -> list[dict]:
    """
    Descubre y carga todos los experimentos.

    Devuelve lista de dicts, uno por experimento:
        {
          metadata:       dict con dataset/condition/dimension/technique/paths,
          mta_report:     pd.DataFrame | None,
          fold_metrics:   list[pd.DataFrame],
          npoints:        pd.DataFrame | None,
          out_of_limits:  pd.DataFrame | None,
        }
    """
    experiments = discover_experiments()
    results = []

    for meta in experiments:
        md = meta["metrics_dir"]
        entry = {
            "metadata":      meta,
            "mta_report":    _load_mta_report(md),
            "fold_metrics":  _load_fold_metrics(md),
            "npoints":       _load_npoints(md),
            "out_of_limits": _load_out_of_limits(md),
        }

        missing = [k for k, v in entry.items()
                   if k != "metadata" and v is None and k != "out_of_limits"]
        if missing:
            log.warning(f"[{meta['dataset']} / {meta['condition']}] Falta: {missing}")

        results.append(entry)
        log.debug(f"Cargado: {meta['dataset']} / {meta['condition']}")

    log.info(f"Total experimentos cargados: {len(results)}")
    return results
'''),

    # -----------------------------------------------------------------------
    # analysis/preprocessing.py
    # -----------------------------------------------------------------------
    ("analysis/preprocessing.py", '''\
# -*- coding: utf-8 -*-
"""
preprocessing.py  —  Construye el DataFrame maestro y calcula deltas.

Salida principal: DataFrame `master` con una fila por
(dataset × condition × rango × métrica × estadístico).

También genera `master_folds` con una fila por
(dataset × condition × fold × rango × métrica).

Los deltas se calculan siempre respecto al baseline 'original'
del mismo dataset.
"""

import logging
import numpy as np
import pandas as pd

from analysis.config import ANALYSIS_METRICS, PRIMARY_METRIC

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Columnas esperadas en mta_report
# (ajustar si los nombres reales difieren — lo haremos tras ver los CSVs)
# ---------------------------------------------------------------------------
STAT_COLS = ["mean", "std", "min", "max", "median"]


def _meta_row(exp: dict) -> dict:
    """Extrae campos de metadatos de un experimento."""
    m = exp["metadata"]
    return {
        "dataset":   m["dataset"],
        "condition": m["condition"],
        "dimension": m["dimension"],    # None para 'original'
        "technique": m["technique"],
        "is_original": m["technique"] == "original",
    }


def build_master_table(experiments: list[dict]) -> pd.DataFrame:
    """
    Consolida mta_report de todos los experimentos en un único DataFrame.

    Columnas resultantes:
        dataset, condition, dimension, technique, is_original,
        range, metric, mean, std, min, max, median
    """
    rows = []

    for exp in experiments:
        meta = _meta_row(exp)
        df = exp.get("mta_report")

        if df is None:
            log.warning(f"Sin mta_report: {meta['dataset']} / {meta['condition']}")
            continue

        # El mta_report tiene una fila por estadístico ('mean', 'std', …)
        # y columnas del tipo RMSE_ALL, RMSE_Hypoglycemia_L1, MAE_ALL …
        # Pivotamos a formato largo: una fila por (range, metric, stat)
        # NOTA: ajustaremos esto tras ver la estructura real de los CSVs.
        try:
            df_long = _pivot_mta_report(df)
        except Exception as e:
            log.error(f"Error pivotando {meta['dataset']}/{meta['condition']}: {e}")
            continue

        for _, row in df_long.iterrows():
            rows.append({**meta, **row.to_dict()})

    master = pd.DataFrame(rows)
    log.info(f"Master table: {master.shape}  columnas={list(master.columns)}")
    return master


def _pivot_mta_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma el mta_report en formato largo.

    Estructura esperada del mta_report (a confirmar con CSVs reales):
        Columna 'stat' con valores: mean, std, min, max, ...
        Columnas de métricas: RMSE_ALL, RMSE_Hypoglycemia_L1, ...

    TODO: actualizar este método una vez confirmada la estructura real.
    """
    # Placeholder — se completará con estructura real de los CSVs
    # Por ahora asumimos que la primera columna identifica el estadístico
    if df.empty:
        return pd.DataFrame()

    # Detectar columna de estadístico (puede llamarse 'stat', index, o ser el índice)
    stat_col = None
    for candidate in ["stat", "Stat", "statistic", "index"]:
        if candidate in df.columns:
            stat_col = candidate
            break

    if stat_col is None:
        # Puede que el estadístico esté en el índice
        df = df.reset_index()
        if "index" in df.columns:
            stat_col = "index"

    metric_cols = [c for c in df.columns if c != stat_col]

    records = []
    for _, row in df.iterrows():
        stat = row.get(stat_col, "unknown")
        for col in metric_cols:
            # Intentar parsear nombre de columna: RMSE_Hypoglycemia_L1 -> metric=RMSE, range=Hypoglycemia_L1
            parts = col.split("_", 1)
            metric = parts[0] if len(parts) >= 1 else col
            rng    = parts[1] if len(parts) == 2 else "ALL"
            records.append({
                "range":  rng,
                "metric": metric,
                "stat":   stat,
                "value":  row[col],
            })

    return pd.DataFrame(records)


def build_fold_table(experiments: list[dict]) -> pd.DataFrame:
    """
    Construye tabla larga con métricas por fold individual.
    Usada para tests estadísticos (Friedman, Nemenyi).
    """
    rows = []
    for exp in experiments:
        meta = _meta_row(exp)
        for df_fold in exp.get("fold_metrics", []):
            fold_n = df_fold["fold"].iloc[0] if "fold" in df_fold.columns else -1
            # Mismo pivotado que mta_report pero sin estadísticos
            for col in df_fold.columns:
                if col in ["fold", "_source_file", "Fold"]:
                    continue
                parts = col.split("_", 1)
                metric = parts[0] if len(parts) >= 1 else col
                rng    = parts[1] if len(parts) == 2 else "ALL"
                for _, row in df_fold.iterrows():
                    rows.append({
                        **meta,
                        "fold":   fold_n,
                        "range":  rng,
                        "metric": metric,
                        "value":  row[col],
                    })
    return pd.DataFrame(rows)


def compute_deltas(master: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas delta_abs y delta_pct respecto al baseline original
    del mismo dataset.

    delta_abs = value_balanceado - value_original
    delta_pct = (value_balanceado - value_original) / value_original * 100

    Para métricas donde menor = mejor (RMSE, MAE):
        delta_pct negativo = MEJORA
    """
    if master.empty:
        return master

    # Obtener baseline por (dataset, range, metric, stat)
    baseline = (
        master[master["is_original"]]
        [["dataset", "range", "metric", "stat", "value"]]
        .rename(columns={"value": "baseline_value"})
    )

    master = master.merge(
        baseline,
        on=["dataset", "range", "metric", "stat"],
        how="left",
    )

    master["delta_abs"] = master["value"] - master["baseline_value"]
    master["delta_pct"] = np.where(
        master["baseline_value"] != 0,
        (master["value"] - master["baseline_value"]) / master["baseline_value"] * 100,
        np.nan,
    )

    log.info("Deltas calculados correctamente.")
    return master
'''),

    # -----------------------------------------------------------------------
    # analysis/stats.py
    # -----------------------------------------------------------------------
    ("analysis/stats.py", '''\
# -*- coding: utf-8 -*-
"""
stats.py  —  Análisis estadístico: Friedman, Nemenyi, Cohen d, rankings.

Genera CSVs en outputs/stats/.
"""

import logging
import numpy as np
import pandas as pd
import scipy.stats as ss

from analysis.config import (
    STATS_DIR, DATASETS, ANALYSIS_METRICS, N_FOLDS, ALPHA, PRIMARY_METRIC
)

log = logging.getLogger(__name__)


def _friedman_and_nemenyi(pivot: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    pivot: filas=folds, columnas=condiciones, valores=métrica.
    Devuelve tabla de p-values pairwise (Nemenyi).
    """
    try:
        import scikit_posthocs as sp
    except ImportError:
        log.error("scikit-posthocs no instalado. pip install scikit-posthocs")
        return pd.DataFrame()

    stat, pval = ss.friedmanchisquare(*[pivot[c].values for c in pivot.columns])
    log.info(f"Friedman [{label}]: χ²={stat:.3f}  p={pval:.4f}")

    result = {"label": label, "friedman_stat": stat, "friedman_pval": pval}

    nemenyi = pd.DataFrame()
    if pval < ALPHA:
        nemenyi = sp.posthoc_nemenyi_friedman(pivot.values)
        nemenyi.index   = pivot.columns
        nemenyi.columns = pivot.columns
        log.info(f"  → Diferencias significativas. Nemenyi calculado.")
    else:
        log.info(f"  → Sin diferencias significativas (p≥{ALPHA}).")

    return nemenyi


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d entre dos arrays (pooled std)."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled_std = np.sqrt(((n1 - 1) * np.std(a, ddof=1)**2 +
                          (n2 - 1) * np.std(b, ddof=1)**2) / (n1 + n2 - 2))
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else np.nan


def compute_rankings(fold_table: pd.DataFrame,
                     metric: str = "RMSE",
                     rng: str = "ALL") -> pd.DataFrame:
    """
    Ranking de Borda por dataset.
    Devuelve DataFrame con columnas: dataset, condition, mean, rank.
    """
    sub = fold_table[
        (fold_table["metric"] == metric) &
        (fold_table["range"] == rng)
    ].copy()

    rankings = []
    for dataset, grp in sub.groupby("dataset"):
        means = grp.groupby("condition")["value"].mean().sort_values()
        for rank, (cond, val) in enumerate(means.items(), 1):
            rankings.append({
                "dataset":   dataset,
                "condition": cond,
                "mean":      val,
                "rank":      rank,
            })

    return pd.DataFrame(rankings)


def run_all_stats(master: pd.DataFrame) -> None:
    """
    Ejecuta todos los análisis estadísticos y guarda resultados en STATS_DIR.
    """
    log.info("Iniciando análisis estadístico...")

    # TODO: llamar a build_fold_table si necesitamos datos por fold
    # Por ahora operamos sobre el master (mta_report)

    # Guardar tabla maestra
    out = STATS_DIR / "master_table.csv"
    master.to_csv(out, index=False)
    log.info(f"  Tabla maestra guardada: {out}")

    # Rankings por métrica primaria
    # (se completará con fold_table real)
    log.info("  Rankings pendientes de fold_table — completar tras ver CSVs reales.")
    log.info("Análisis estadístico completado.")
'''),

    # -----------------------------------------------------------------------
    # analysis/viz/style.py
    # -----------------------------------------------------------------------
    ("analysis/viz/style.py", '''\
# -*- coding: utf-8 -*-
"""
style.py  —  Tema matplotlib compartido para todas las figuras.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

from analysis.config import MPL_RC, PALETTE, DIMENSION_PALETTE


def apply_theme():
    """Aplica el tema global. Llamar una vez al inicio de cada módulo viz."""
    mpl.rcParams.update(MPL_RC)


def get_technique_color(technique: str) -> str:
    return PALETTE.get(technique, "#888888")


def get_dimension_color(dimension: str) -> str:
    return DIMENSION_PALETTE.get(dimension, "#888888")


def save_fig(fig, path, tight=True):
    """Guarda figura con configuración estándar (300 dpi, bbox_inches=tight)."""
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    import logging
    logging.getLogger(__name__).info(f"Figura guardada: {path}")
'''),

    ("analysis/viz/__init__.py", ""),

    # -----------------------------------------------------------------------
    # Stubs de los módulos viz restantes
    # -----------------------------------------------------------------------
    ("analysis/viz/heatmaps.py", '''\
# -*- coding: utf-8 -*-
"""heatmaps.py — Heatmaps de delta y rankings."""

import logging
import pandas as pd

from analysis.config import FIGURES_DIR
from analysis.viz.style import apply_theme

log = logging.getLogger(__name__)
apply_theme()


def plot_delta_heatmap(master: pd.DataFrame) -> None:
    """Heatmap: filas=condición, columnas=dataset×métrica, color=delta_pct."""
    # TODO: implementar
    log.info("plot_delta_heatmap: pendiente de implementar.")


def plot_ranking_heatmap(master: pd.DataFrame) -> None:
    """Heatmap de posiciones de ranking por dataset."""
    # TODO: implementar
    log.info("plot_ranking_heatmap: pendiente de implementar.")


def plot_all_heatmaps(master: pd.DataFrame) -> None:
    plot_delta_heatmap(master)
    plot_ranking_heatmap(master)
'''),

    ("analysis/viz/dumbbell.py", '''\
# -*- coding: utf-8 -*-
"""dumbbell.py — Dumbbell plots: baseline vs. balanceado."""

import logging
import pandas as pd

from analysis.config import FIGURES_DIR
from analysis.viz.style import apply_theme

log = logging.getLogger(__name__)
apply_theme()


def plot_dumbbell_by_dataset(master: pd.DataFrame) -> None:
    """Un panel por dataset. Eje X: RMSE_ALL. Puntos: original vs técnica."""
    # TODO: implementar
    log.info("plot_dumbbell_by_dataset: pendiente de implementar.")


def plot_all_dumbbells(master: pd.DataFrame) -> None:
    plot_dumbbell_by_dataset(master)
'''),

    ("analysis/viz/distributions.py", '''\
# -*- coding: utf-8 -*-
"""distributions.py — Raincloud / violin / strip plots por fold."""

import logging
import pandas as pd

from analysis.config import FIGURES_DIR
from analysis.viz.style import apply_theme

log = logging.getLogger(__name__)
apply_theme()


def plot_fold_distributions(master: pd.DataFrame) -> None:
    """Distribución de RMSE_ALL a través de los 5 folds por condición."""
    # TODO: implementar
    log.info("plot_fold_distributions: pendiente de implementar.")


def plot_all_distributions(master: pd.DataFrame) -> None:
    plot_fold_distributions(master)
'''),

    ("analysis/viz/rankings.py", '''\
# -*- coding: utf-8 -*-
"""rankings.py — CD diagrams, bump charts, Borda ranking."""

import logging
import pandas as pd

from analysis.config import FIGURES_DIR
from analysis.viz.style import apply_theme

log = logging.getLogger(__name__)
apply_theme()


def plot_cd_diagram(master: pd.DataFrame) -> None:
    """Critical Difference diagram (Demšar 2006) por dataset."""
    # TODO: implementar
    log.info("plot_cd_diagram: pendiente de implementar.")


def plot_bump_chart(master: pd.DataFrame) -> None:
    """Bump chart: cómo varía el ranking según la métrica usada."""
    # TODO: implementar
    log.info("plot_bump_chart: pendiente de implementar.")


def plot_all_rankings(master: pd.DataFrame) -> None:
    plot_cd_diagram(master)
    plot_bump_chart(master)
'''),

    ("analysis/viz/profiles.py", '''\
# -*- coding: utf-8 -*-
"""profiles.py — Parallel coordinates y radar por rango glucémico."""

import logging
import pandas as pd

from analysis.config import FIGURES_DIR
from analysis.viz.style import apply_theme

log = logging.getLogger(__name__)
apply_theme()


def plot_glucose_range_profiles(master: pd.DataFrame) -> None:
    """Parallel coordinates: RMSE en los 5 rangos glucémicos por condición."""
    # TODO: implementar
    log.info("plot_glucose_range_profiles: pendiente de implementar.")


def plot_all_profiles(master: pd.DataFrame) -> None:
    plot_glucose_range_profiles(master)
'''),

    # -----------------------------------------------------------------------
    # analysis/tables.py
    # -----------------------------------------------------------------------
    ("analysis/tables.py", '''\
# -*- coding: utf-8 -*-
"""tables.py — Tablas LaTeX y CSV para el TFM."""

import logging
import pandas as pd

from analysis.config import TABLES_DIR, ANALYSIS_METRICS

log = logging.getLogger(__name__)


def generate_main_results_table(master: pd.DataFrame) -> None:
    """
    Tabla principal: dataset × condición con mean ± std de métricas clave.
    Guarda .csv y .tex.
    """
    # TODO: implementar tras confirmar estructura de master
    log.info("generate_main_results_table: pendiente de implementar.")


def generate_ranking_table(master: pd.DataFrame) -> None:
    """Tabla de rankings por dataset y global."""
    # TODO: implementar
    log.info("generate_ranking_table: pendiente de implementar.")


def generate_all_tables(master: pd.DataFrame) -> None:
    generate_main_results_table(master)
    generate_ranking_table(master)
'''),

    # -----------------------------------------------------------------------
    # analysis/dashboards.py
    # -----------------------------------------------------------------------
    ("analysis/dashboards.py", '''\
# -*- coding: utf-8 -*-
"""dashboards.py — Figuras compuestas multi-panel."""

import logging
import pandas as pd

from analysis.config import DASHBOARDS_DIR
from analysis.viz.style import apply_theme

log = logging.getLogger(__name__)
apply_theme()


def dashboard_global_overview(master: pd.DataFrame) -> None:
    """
    Dashboard 1: heatmap de deltas + ranking de Borda.
    Una figura compacta para la sección de resultados del TFM.
    """
    # TODO: implementar
    log.info("dashboard_global_overview: pendiente de implementar.")


def dashboard_per_dataset(master: pd.DataFrame) -> None:
    """
    Dashboard 2: 3 paneles (uno por dataset).
    Dumbbell plot + distribución fold por dataset.
    """
    # TODO: implementar
    log.info("dashboard_per_dataset: pendiente de implementar.")


def generate_all_dashboards(master: pd.DataFrame) -> None:
    dashboard_global_overview(master)
    dashboard_per_dataset(master)
'''),

    # -----------------------------------------------------------------------
    # analysis/tests/__init__.py
    # -----------------------------------------------------------------------
    ("analysis/tests/__init__.py", ""),
    ("analysis/tests/test_loader.py", '''\
# -*- coding: utf-8 -*-
"""Tests básicos del loader (ejecutar con pytest)."""

from analysis.loader import parse_condition


def test_parse_original():
    r = parse_condition("original")
    assert r["technique"] == "original"
    assert r["dimension"] is None


def test_parse_balanced_age():
    r = parse_condition("balanced_age_smote")
    assert r["dimension"] == "age"
    assert r["technique"] == "smote"


def test_parse_balanced_sex_compound():
    r = parse_condition("balanced_sex_borderline_smote")
    assert r["dimension"] == "sex"
    assert r["technique"] == "borderline_smote"
'''),
]


# ---------------------------------------------------------------------------
# Creación de estructura
# ---------------------------------------------------------------------------

def create_structure(root: Path):
    print(f"\n📁  Creando estructura en: {root.resolve()}\n")

    # Directorios
    for d in DIRS:
        target = root / d
        target.mkdir(parents=True, exist_ok=True)
        print(f"  [DIR]  {d}/")

    # Ficheros
    print()
    for rel_path, content in FILES:
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            print(f"  [SKIP] {rel_path}  (ya existe)")
        else:
            target.write_text(textwrap.dedent(content), encoding="utf-8")
            print(f"  [FILE] {rel_path}")

    print(f"""
✅  Estructura creada correctamente.

Próximos pasos:
  1. Edita  {root}/analysis/config.py
     → Ajusta OUTPUT_ROOT a la ruta de tus experimentos.
     → Verifica los nombres de TECHNIQUES (deben coincidir con tus ficheros).

  2. Ejecuta el pipeline de carga para validar el descubrimiento:
       cd {root}
       python main.py --only load --debug

  3. Comparte la salida del debug con Claude para calibrar
     el parser y el pivotado del mta_report.
""")


def main():
    parser = argparse.ArgumentParser(
        description="Genera la estructura del proyecto de análisis RELIEF-T1D."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("analysis_relief"),
        help="Directorio raíz del proyecto (default: ./analysis_relief/)",
    )
    args = parser.parse_args()
    create_structure(args.root)


if __name__ == "__main__":
    import argparse
    main()