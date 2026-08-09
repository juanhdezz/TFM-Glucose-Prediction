# -*- coding: utf-8 -*-
"""
balancing_viz.py — Análisis 1: Efecto del balanceo sobre el conjunto de entrenamiento.

Figuras generadas (todas en outputs/figures/balancing/):

  FIG-B1  heatmap_balancing_{dataset}_{dimension}.pdf          (6 fig)
          Heatmap de variación Δn y Δ% por técnica × clase demográfica.
          Filas = técnicas, columnas = grupos de edad o sexo.
          Escala de color UNIFICADA entre datasets: una escala común para
          los 3 heatmaps de "sex" y otra (distinta) para los 3 de "age",
          cada una con su propia paleta de color para no confundirlas.

  FIG-B2  stacked_bars_{dataset}.pdf                           (3 fig)
          Barras horizontales apiladas por rango glucémico.
          Una barra por (clase × técnica) × dimensión.
          Eje X idéntico entre el panel de "age" y el de "sex" del mismo
          dataset (misma escala para poder comparar ambos paneles).
          Valores = MEDIA por fold (no la suma de los 5 folds), para que
          la barra sea representativa del tamaño real de un train set.

  FIG-B3  trainsize_bars_{dimension}.pdf                       (2 fig)
          Barras verticales del tamaño de train por técnica, con media
          y desviación estándar (error bars) sobre los 5 folds.
          La línea discontinua "original" usa la media de los 5 folds
          originales.

  FIG-B4  delta_glycemic_{dataset}.pdf                         (3 fig)
          Barras divergentes: Δ% (no puntos porcentuales) por rango
          glucémico para cada técnica vs. original. Lectura clínica directa.

  FIG-B5  tradeoff_{rango}_vs_tir.pdf                          (4 fig)
          Scatter: Δ% de un rango glucémico (eje X) vs. Δ% TIR (eje Y)
          por técnica × dimensión. Se genera una figura para cada uno de
          los 4 rangos fuera de TIR: TBR_2, TBR_1, TAR_1, TAR_2.
          Visualiza el trade-off clínico más importante del balanceo.

Total: 18 figuras.

Asunciones verificadas:
  - Columna `y`: glucosa en mg/dL directamente.
  - Parquets balanceados: incluyen train + val + test; se filtra por split == 'train'.
  - Columna patient_id: presente en todos los parquets.
  - Las tablas agregadas (orig_glyc, tech_glyc, orig_demo, tech_demo) representan
    la MEDIA sobre los 5 folds de CV, no la suma — la suma multiplicaba por ~5
    el tamaño real del dataset y distorsionaba las cifras (ver FIG-B2/B3).
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "input"
OUT_DIR  = ROOT_DIR / "outputs" / "figures" / "balancing"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DATASETS = {
    "DIATREND":          DATA_DIR / "DIATREND",
    "REPLACE-BG":        DATA_DIR / "REPLACE-BG",
    "T1DiabetesGranada": DATA_DIR / "T1DiabetesGranada",
}
DATASET_LABELS = {
    "DIATREND":          "DiaTrend",
    "REPLACE-BG":        "ReplaceBG",
    "T1DiabetesGranada": "T1DiabetesGranada",
}
DATASET_ORDER = ["T1DiabetesGranada", "DIATREND", "REPLACE-BG"]

DIMENSIONS = ["age", "sex"]
N_FOLDS    = 5
GLUCOSE_COL = "y"

AGE_BINS   = [-np.inf, 30, 45, 65, np.inf]
AGE_LABELS = ["<31", "31-45", "46-65", ">=66"]

TECHNIQUES = [
    "undersampling",
    "oversampling",
    "smote",
    "tomek_links",
    "patient_aware_undersampling",
    "jittering",
    "undersampling_oversampling",
    "undersampling_smote",
    "smote_tomek",
    "oversampling_tomek",
]
TECHNIQUE_LABELS = {
    "undersampling":               "RUS",
    "oversampling":                "ROS",
    "smote":                       "SMOTE",
    "tomek_links":                 "Tomek Links",
    "patient_aware_undersampling": "PA-Undersampling",
    "jittering":                   "Jittering",
    "undersampling_oversampling":  "RUS + ROS",
    "undersampling_smote":         "RUS + SMOTE",
    "smote_tomek":                 "SMOTE + Tomek",
    "oversampling_tomek":          "ROS + Tomek",
}
TECHNIQUE_ORDER = list(TECHNIQUES)  # orden canónico

# Familias de técnicas — para colorear FIG-B5
FAMILY_COLOR = {
    "undersampling":               "#56B4E9",
    "oversampling":                "#E69F00",
    "smote":                       "#D55E00",
    "tomek_links":                 "#F0E442",
    "patient_aware_undersampling": "#0072B2",
    "jittering":                   "#009E73",
    "undersampling_oversampling":  "#BC8F8F",
    "undersampling_smote":         "#984EA3",
    "smote_tomek":                 "#FF7F00",
    "oversampling_tomek":          "#999999",
}
FAMILY_MARKER = {"age": "o", "sex": "s"}   # círculo=age, cuadrado=sex

# Rangos glucémicos
GLYCEMIC_RANGES = ["TBR_2", "TBR_1", "TIR", "TAR_1", "TAR_2"]
GLYCEMIC_LABELS = {
    "TBR_2": "Hypo L2  (<54 mg/dL)",
    "TBR_1": "Hypo L1  (54–69 mg/dL)",
    "TIR":   "In Range (70–180 mg/dL)",
    "TAR_1": "Hyper L1 (181–250 mg/dL)",
    "TAR_2": "Hyper L2 (>250 mg/dL)",
}
GLYCEMIC_COLORS = {
    "TBR_2": "#FA8072",  # rosa salmón
    "TBR_1": "#E63946",  # rojo
    "TIR":   "#2CA02C",  # verde
    "TAR_1": "#F1C40F",  # amarillo
    "TAR_2": "#F28C28",  # naranja
}

# Rangos evaluados en FIG-B5 (trade-off vs. TIR) — todos menos TIR
TRADEOFF_RANGES = ["TBR_2", "TBR_1", "TAR_1", "TAR_2"]
# Para la lectura clínica de cada rango: ¿es hipoglucemia o hiperglucemia?
RANGE_KIND = {"TBR_2": "hypo", "TBR_1": "hypo", "TAR_1": "hyper", "TAR_2": "hyper"}

MPL_RC = {
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.framealpha":  0.9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.grid":          False,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
}
mpl.rcParams.update(MPL_RC)


# ===========================================================================
# CARGA DE DATOS
# ===========================================================================

def _find_col(df: pd.DataFrame, candidates: list) -> str:
    norm = {c.strip().lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower().replace(" ", "_")
        if key in norm:
            return norm[key]
    raise KeyError(f"Columna no encontrada entre {candidates}. Disponibles: {list(df.columns)}")


def load_original(dataset_dir: Path) -> pd.DataFrame:
    candidates = sorted(
        p for p in dataset_dir.glob("windows_with_5folds_*.parquet")
        if "balanced_outputs" not in str(p)
    )
    if not candidates:
        raise FileNotFoundError(f"No se encontró parquet original en {dataset_dir}")
    return pd.read_parquet(candidates[0])


def get_patient_info(dataset_dir: Path) -> pd.DataFrame:
    for name in ["Patient_info.parquet", "patient_info.parquet",
                 "Patient_info.csv", "patient_info.csv"]:
        p = dataset_dir / name
        if p.exists():
            return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"No se encontró patient_info en {dataset_dir}")


def find_balanced_file(dataset_dir: Path, fold_idx: int,
                       group: str, technique: str) -> Path | None:
    bal_dir = dataset_dir / "balanced_outputs"
    if not bal_dir.exists():
        return None
    matches = sorted(bal_dir.glob(f"*_fold{fold_idx}_{group}_{technique}.parquet"))
    return matches[0] if matches else None


def get_train_split(df: pd.DataFrame, fold_idx: int) -> pd.DataFrame:
    """Extrae el split de train para un fold dado.
    Soporta tanto la columna fold_{n} con valores 'train'/'val'/'test'
    como la columna 'split' directa (parquets balanceados).
    """
    # Parquet original: columna fold_{n}
    col = f"fold_{fold_idx}"
    if col in df.columns:
        return df[df[col].str.lower() == "train"].reset_index(drop=True)
    # Parquet balanceado: columna split
    if "split" in df.columns:
        return df[df["split"].str.lower() == "train"].reset_index(drop=True)
    # Fallback: devolver todo (no debería ocurrir)
    log.warning("No se encontró columna de split — usando todo el DataFrame")
    return df.reset_index(drop=True)


# ===========================================================================
# LOOKUPS DEMOGRÁFICOS
# ===========================================================================

def build_lookup(pat_info: pd.DataFrame, group: str) -> dict:
    pi  = pat_info.copy()
    pid = _find_col(pi, ["patient_id", "Patient_ID", "patientid"])
    pi["_pid"] = pi[pid].astype(str).str.strip()

    if group == "age":
        try:
            age_c = _find_col(pi, ["Age", "age"])
            pi["_age"] = pd.to_numeric(pi[age_c], errors="coerce")
        except KeyError:
            birth_c = _find_col(pi, ["Birth_year", "birth_year"])
            pi["_age"] = 2026 - pd.to_numeric(pi[birth_c], errors="coerce")
        pi["_grp"] = pd.cut(pi["_age"], bins=AGE_BINS, labels=AGE_LABELS,
                            include_lowest=True, right=True).astype(str)
    else:  # sex
        sex_c = _find_col(pi, ["Sex", "sex", "Gender", "gender"])
        pi["_grp"] = pi[sex_c].astype(str).str.strip().str.upper().str[0]  # M/F

    return pi.set_index("_pid")["_grp"].to_dict()


def add_group_col(df: pd.DataFrame, lookup: dict, gcol: str) -> pd.DataFrame:
    out = df.copy()
    out[gcol] = out["patient_id"].astype(str).str.strip().map(lookup).fillna("Unknown")
    return out[out[gcol] != "Unknown"]   # descartar ventanas sin info demográfica


# ===========================================================================
# RANGO GLUCÉMICO
# ===========================================================================

def assign_glycemic_range(df: pd.DataFrame) -> pd.DataFrame:
    g = df[GLUCOSE_COL]
    out = df.copy()
    out["glycemic_range"] = np.select(
        [g < 54,
         (g >= 54)  & (g <= 69),
         (g >= 70)  & (g <= 180),
         (g >= 181) & (g <= 250),
         g > 250],
        GLYCEMIC_RANGES,
        default="TIR"
    )
    return out


# ===========================================================================
# AGREGACIÓN CENTRAL — produce todas las tablas necesarias en un solo paso
# ===========================================================================

class DatasetAggregator:
    """
    Para un dataset dado, precalcula sobre los 5 folds:

      self.orig_demo[group]         → pd.Series  index=clases,  values=n medio POR FOLD
      self.tech_demo[group][tech]   → pd.Series  index=clases,  values=n medio balanceado POR FOLD

      self.orig_glyc[group]         → pd.DataFrame  index=clases, cols=rangos (n medio POR FOLD)
      self.tech_glyc[group][tech]   → pd.DataFrame  index=clases, cols=rangos (n medio bal POR FOLD)

      self.fold_sizes[group][tech]  → list[float]   tamaño total train POR fold (crudo, sin agregar)
      self.fold_sizes_orig[group]   → list[float]   tamaño total train original POR fold (crudo)

    IMPORTANTE: todos los valores agregados (orig_demo/tech_demo/orig_glyc/tech_glyc)
    son la MEDIA sobre los folds disponibles, no la suma. Sumar los 5 folds de una
    validación cruzada da un total ~5x mayor que el dataset real (cada fold es un
    train set casi completo), lo que generaba cifras engañosas (p. ej. un dataset de
    22.6M de muestras mostrando 26-36M en las barras). fold_sizes / fold_sizes_orig sí
    quedan sin agregar para poder calcular media ± desviación estándar en FIG-B3.
    """

    def __init__(self, ds_name: str, dataset_dir: Path):
        self.ds_name     = ds_name
        self.dataset_dir = dataset_dir
        self.df_orig     = load_original(dataset_dir)
        self.pat_info    = get_patient_info(dataset_dir)

        self.orig_demo:       dict = {}
        self.tech_demo:       dict = {}
        self.orig_glyc:       dict = {}
        self.tech_glyc:       dict = {}
        self.fold_sizes:      dict = {}
        self.fold_sizes_orig: dict = {}

        for group in DIMENSIONS:
            self._aggregate_group(group)

    def _aggregate_group(self, group: str):
        lookup = build_lookup(self.pat_info, group)
        gcol   = f"{group}_group"

        class_order = AGE_LABELS if group == "age" else ["M", "F"]

        orig_demo_folds  = []
        orig_glyc_folds  = []
        orig_sizes       = []

        tech_demo_folds  = {t: [] for t in TECHNIQUES}
        tech_glyc_folds  = {t: [] for t in TECHNIQUES}
        tech_sizes_folds = {t: [] for t in TECHNIQUES}

        for fold_idx in range(N_FOLDS):
            orig_train = get_train_split(self.df_orig, fold_idx)
            orig_g     = assign_glycemic_range(add_group_col(orig_train, lookup, gcol))

            orig_demo_folds.append(orig_g[gcol].value_counts())
            orig_glyc_folds.append(
                pd.crosstab(orig_g[gcol], orig_g["glycemic_range"])
                  .reindex(columns=GLYCEMIC_RANGES, fill_value=0)
            )
            orig_sizes.append(len(orig_g))

            for tech in TECHNIQUES:
                bal_file = find_balanced_file(self.dataset_dir, fold_idx, group, tech)
                if bal_file is None:
                    continue
                bal_df    = pd.read_parquet(bal_file)
                bal_train = get_train_split(bal_df, fold_idx)
                bal_g     = assign_glycemic_range(add_group_col(bal_train, lookup, gcol))

                tech_demo_folds[tech].append(bal_g[gcol].value_counts())
                tech_glyc_folds[tech].append(
                    pd.crosstab(bal_g[gcol], bal_g["glycemic_range"])
                      .reindex(columns=GLYCEMIC_RANGES, fill_value=0)
                )
                tech_sizes_folds[tech].append(len(bal_g))

        # -- MEDIA sobre folds: representa el tamaño real de UN train set --
        # Cada fold se reindexa primero a las clases/rangos canónicos (rellenando
        # con 0 lo ausente) y luego se promedia — así un fold sin una clase no
        # falsea la media al excluirla en vez de contarla como 0.
        def mean_series(folds, idx):
            if not folds:
                return pd.Series(0.0, index=idx)
            aligned = [f.reindex(idx, fill_value=0) for f in folds]
            return pd.concat(aligned, axis=1).mean(axis=1)

        def mean_df(folds, idx):
            if not folds:
                return pd.DataFrame(0.0, index=idx, columns=GLYCEMIC_RANGES)
            aligned = [f.reindex(index=idx, columns=GLYCEMIC_RANGES, fill_value=0)
                       for f in folds]
            stacked = np.stack([a.values for a in aligned])
            return pd.DataFrame(stacked.mean(axis=0), index=idx, columns=GLYCEMIC_RANGES)

        self.orig_demo[group]       = mean_series(orig_demo_folds, class_order)
        self.orig_glyc[group]       = mean_df(orig_glyc_folds, class_order)
        self.fold_sizes_orig[group] = [float(s) for s in orig_sizes]  # crudo, sin agregar

        self.tech_demo[group]  = {}
        self.tech_glyc[group]  = {}
        self.fold_sizes[group] = {}

        for tech in TECHNIQUES:
            if not tech_demo_folds[tech]:
                continue
            self.tech_demo[group][tech]  = mean_series(tech_demo_folds[tech], class_order)
            self.tech_glyc[group][tech]  = mean_df(tech_glyc_folds[tech], class_order)
            self.fold_sizes[group][tech] = [float(s) for s in tech_sizes_folds[tech]]  # crudo


# ===========================================================================
# HELPERS GRÁFICOS
# ===========================================================================

def _save(fig, filename: str):
    path = OUT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    log.info(f"  → {path.name}")


def _delta_cmap(group: str):
    """Paleta divergente para el heatmap FIG-B1. Distinta por dimensión
    para que 'age' y 'sex' no se confundan visualmente aunque compartan
    la misma figura de referencia (Blue/Green = sex, Purple/Orange = age)."""
    if group == "sex":
        colors = ["#0072B2", "#FFFFFF", "#009E73"]   # azul – blanco – verde
    else:
        colors = ["#7B3294", "#FFFFFF", "#E66101"]   # morado – blanco – naranja
    return LinearSegmentedColormap.from_list(f"bal_delta_{group}", colors, N=256)


def _class_order(group: str) -> list:
    return AGE_LABELS if group == "age" else ["M", "F"]


def _dim_label(group: str) -> str:
    return "Age group" if group == "age" else "Sex"


# ===========================================================================
# FIG-B1: Heatmap de variación demográfica  (6 figuras)
# ===========================================================================

def _heatmap_deltas(agg: DatasetAggregator, group: str):
    """Calcula las matrices Δabs y Δ% (técnica × clase) para un (dataset, dimensión)."""
    class_order = _class_order(group)
    n_techs     = len(TECHNIQUES)
    n_classes   = len(class_order)

    delta_pct = np.full((n_techs, n_classes), np.nan)
    delta_abs = np.full((n_techs, n_classes), np.nan)

    orig = agg.orig_demo[group]
    for ti, tech in enumerate(TECHNIQUES):
        if tech not in agg.tech_demo[group]:
            continue
        bal = agg.tech_demo[group][tech]
        for ci, cls in enumerate(class_order):
            o = orig.get(cls, 0)
            b = bal.get(cls, 0)
            delta_abs[ti, ci] = b - o
            delta_pct[ti, ci] = (b - o) / o * 100 if o > 0 else np.nan

    return delta_pct, delta_abs, class_order


def _global_heatmap_vmax(aggregators: list, group: str) -> float:
    """
    Vmax de la escala de color de FIG-B1, calculado sobre TODOS los datasets
    para una misma dimensión (age o sex). Así los 3 heatmaps de "sex" son
    directamente comparables entre sí (y lo mismo para "age"), en vez de que
    cada dataset tenga su propia escala distorsionando la percepción visual.
    """
    all_valid = []
    for agg in aggregators:
        delta_pct, _, _ = _heatmap_deltas(agg, group)
        all_valid.append(delta_pct[~np.isnan(delta_pct)])
    valid = np.concatenate(all_valid) if all_valid else np.array([])
    return max(np.percentile(np.abs(valid), 95), 5.0) if len(valid) else 20.0


def plot_heatmap_balancing(agg: DatasetAggregator, group: str, vmax: float):
    """
    Heatmap Δn / Δ% por técnica × clase demográfica.
    Una figura por (dataset × dimensión). `vmax` se calcula una única vez
    para todos los datasets de la misma dimensión (ver _global_heatmap_vmax),
    de forma que la escala de color sea homogénea y comparable entre datasets.
    """
    delta_pct, delta_abs, class_order = _heatmap_deltas(agg, group)
    n_techs   = len(TECHNIQUES)
    n_classes = len(class_order)

    fig_h = max(5.5, n_techs * 0.70 + 2.2)
    fig_w = max(5.0, n_classes * 2.4 + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(delta_pct, cmap=_delta_cmap(group), vmin=-vmax, vmax=vmax,
                   aspect="auto", interpolation="nearest")

    for ti in range(n_techs):
        for ci in range(n_classes):
            dp = delta_pct[ti, ci]
            da = delta_abs[ti, ci]
            if np.isnan(dp):
                ax.text(ci, ti, "n/a", ha="center", va="center",
                        fontsize=8, color="#aaa")
                continue
            fg = "white" if abs(dp) / vmax > 0.55 else "#1a1a1a"
            ax.text(ci, ti - 0.14, f"{int(round(da)):+,}",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color=fg)
            ax.text(ci, ti + 0.20, f"({dp:+.1f}%)",
                    ha="center", va="center",
                    fontsize=7.5, color=fg, alpha=0.88)

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_order, fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_techs))
    ax.set_yticklabels([TECHNIQUE_LABELS.get(t, t) for t in TECHNIQUES], fontsize=9)
    # Labels de clase demográfica ABAJO (posición por defecto del eje x)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    ax.set_xlabel(_dim_label(group), fontsize=11, labelpad=10)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.12)
    cb  = fig.colorbar(im, cax=cax)
    cb.set_label("Relative change vs. original (%)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)

    ds_lbl = DATASET_LABELS.get(agg.ds_name, agg.ds_name)
    dim_lbl = "Age" if group == "age" else "Sex"
    scale_note = ("Blue = loss  ·  Green = gain" if group == "sex"
                  else "Purple = loss  ·  Orange = gain")
    ax.set_title(
        f"{ds_lbl}  ·  Balancing effect on train composition  ({dim_lbl} dimension)\n"
        f"Cell: mean Δsamples (bold) and Δ% across {N_FOLDS} folds  "
        f"|  {scale_note}",
        fontsize=10, pad=14, loc="left"
    )
    fig.text(0.01, -0.01,
             "Values averaged across 5 cross-validation folds. "
             "Positive = samples added (oversampling); negative = removed (undersampling). "
             f"Color scale shared across all {len(DATASET_ORDER)} datasets for this dimension.",
             fontsize=7, color="0.5")

    _save(fig, f"heatmap_balancing_{agg.ds_name}_{group}.pdf")


# ===========================================================================
# FIG-B2: Barras horizontales apiladas por rango glucémico  (3 figuras)
# ===========================================================================

def plot_stacked_bars(agg: DatasetAggregator):
    """
    Una figura por dataset con dos paneles (age | sex).
    Etiquetas de técnica al final (derecha) de cada barra, no en el eje Y.
    Eje Y: solo la clase demográfica (centrada en el bloque).
    xmax: ÚNICO para ambos paneles (age y sex) de este dataset, para que las
    dos escalas sean directamente comparables entre sí.
    Valores = MEDIA por fold (no la suma de los 5 folds), representando el
    tamaño real de un train set.
    """
    ds_lbl = DATASET_LABELS.get(agg.ds_name, agg.ds_name)

    # xmax = max ancho de barra individual (suma de rangos de UNA clase),
    # calculado sobre AMBAS dimensiones (age y sex) para que los dos paneles
    # compartan la misma escala, más un margen para que las etiquetas al
    # final de la barra no queden cortadas.
    def _panel_xmax(orig_glyc, tech_glyc):
        candidates = list(orig_glyc.sum(axis=1).values)   # total por clase, original
        for df in tech_glyc.values():
            candidates.extend(df.sum(axis=1).values)       # total por clase, cada técnica
        return float(max(candidates)) if candidates else 1.0

    xmax_shared = max(
        _panel_xmax(agg.orig_glyc[g], agg.tech_glyc[g]) for g in DIMENSIONS
    ) * 1.28  # margen derecho para etiquetas

    fig, axes = plt.subplots(1, 2, figsize=(24, 20),
                             gridspec_kw={"wspace": 0.10})

    for ax, group in zip(axes, DIMENSIONS):
        class_order = _class_order(group)
        orig_glyc   = agg.orig_glyc[group]
        tech_glyc   = agg.tech_glyc[group]
        available   = [t for t in TECHNIQUES if t in tech_glyc]

        xmax = xmax_shared  # misma escala X para age y sex

        BAR_H     = 0.30
        ORIG_H    = 0.42
        GAP_INNER = 0.05
        GAP_CLS   = 0.80   # espacio extra entre clases para separación visual

        y_pos     = 0.0
        cls_ticks = []
        all_ys    = []
        sep_lines = []   # posiciones y de separadores entre clases

        def draw_bar(row_data, yc, height, alpha=1.0, lw=0.5, ec="white"):
            left  = 0.0
            total = 0.0
            for rng in GLYCEMIC_RANGES:
                val = float(row_data.get(rng, 0.0))
                if val > 0:
                    ax.barh(yc, val, height=height, left=left,
                            color=GLYCEMIC_COLORS[rng], alpha=alpha,
                            edgecolor=ec, linewidth=lw)
                    left  += val
                    total += val
            return total   # devuelve el ancho total de la barra

        for i_cls, cls in enumerate(reversed(class_order)):  # clase más baja arriba
            block_top = y_pos

            # — Original —
            orig_row  = orig_glyc.loc[cls] if cls in orig_glyc.index else pd.Series(dtype=float)
            bar_total = draw_bar(orig_row, y_pos, ORIG_H, alpha=1.0, lw=0.9, ec="#333")
            # Etiqueta "Original" al final de la barra
            ax.text(bar_total + xmax * 0.012, y_pos,
                    "Original", ha="left", va="center",
                    fontsize=8, fontweight="bold", color="#222")
            all_ys.append(y_pos)
            y_pos -= (ORIG_H + GAP_INNER * 1.6)

            # — Técnicas —
            for tech in available:
                df_t    = tech_glyc[tech]
                t_row   = df_t.loc[cls] if cls in df_t.index else pd.Series(dtype=float)
                bar_tot = draw_bar(t_row, y_pos, BAR_H, alpha=0.84, lw=0.3, ec="white")
                ax.text(bar_tot + xmax * 0.012, y_pos,
                        TECHNIQUE_LABELS.get(tech, tech),
                        ha="left", va="center", fontsize=7.2, color="#444")
                all_ys.append(y_pos)
                y_pos -= (BAR_H + GAP_INNER)

            # Centro del bloque para el tick de clase
            cls_center = (block_top + y_pos + BAR_H) / 2
            cls_ticks.append((cls_center, cls))

            # Guardar posición del separador (excepto después del último bloque)
            if i_cls < len(class_order) - 1:
                sep_lines.append(y_pos - GAP_CLS * 0.45)

            y_pos -= GAP_CLS

        # — Ejes y decoración —
        ax.set_xlim(0, xmax)
        if all_ys:
            ax.set_ylim(min(all_ys) - ORIG_H - 0.1,
                        max(all_ys) + ORIG_H + 0.4)

        # Eje Y: solo etiquetas de clase demográfica, centradas en cada bloque
        ax.set_yticks([y for y, _ in cls_ticks])
        ax.set_yticklabels([lbl for _, lbl in cls_ticks],
                           fontsize=13, fontweight="bold")
        ax.tick_params(axis="y", length=0, pad=6)

        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_xlabel("Mean training samples per fold", fontsize=10)
        ax.grid(axis="x", alpha=0.20, linestyle="--", color="0.6")
        ax.set_axisbelow(True)

        # Línea vertical: tamaño de la clase original más grande
        orig_max_class = float(orig_glyc.sum(axis=1).max()) if not orig_glyc.empty else 0
        if orig_max_class > 0:
            ax.axvline(orig_max_class, color="#222", lw=1.2, ls=":", alpha=0.5, zorder=0)
            ax.text(orig_max_class + xmax * 0.005,
                    max(all_ys) + ORIG_H + 0.25,
                    f"Orig. group samples\n{int(orig_max_class):,}",
                    fontsize=7.5, color="#555", va="bottom")

        # Separadores horizontales entre grupos de clase
        for sy in sep_lines:
            ax.axhline(sy, color="0.65", lw=1.0, ls="--", alpha=0.7)

        dim_lbl = "Age group" if group == "age" else "Sex"
        ax.set_title(dim_lbl, fontsize=13, fontweight="bold", pad=10)

        # Eliminar spine izquierdo (las etiquetas de clase ya son suficiente separador)
        ax.spines["left"].set_visible(False)

    # Leyenda compartida de rangos glucémicos en la parte inferior
    patches = [mpatches.Patch(color=GLYCEMIC_COLORS[r], label=GLYCEMIC_LABELS[r])
               for r in GLYCEMIC_RANGES]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.03), fontsize=10,
               framealpha=0.95, edgecolor="0.8",
               title="Glycemic range", title_fontsize=10)

    fig.suptitle(
        f"{ds_lbl}  ·  Train set composition by glycemic range — Original vs. Balanced\n"
        f"Each bar = mean training samples per fold (averaged across {N_FOLDS} folds)  "
        f"|  Bold-border bar = original baseline  "
        f"|  X-axis shared between both panels",
        fontsize=11, y=1.01
    )

    _save(fig, f"stacked_bars_{agg.ds_name}.pdf")


# ===========================================================================
# FIG-B3: Boxplot de tamaño de train  (2 figuras — una por dimensión)
# ===========================================================================

def plot_trainsize_bars(aggregators: list, group: str):
    """
    FIG-B3: Gráfico de barras verticales — tamaño medio del train por técnica,
    con barra de error mostrando la desviación estándar sobre los 5 folds.
    La línea discontinua "Original" usa la media de los 5 folds originales.
    Una figura con un panel por dataset, compacta y sin solapamientos.
    """
    dim_lbl  = "Age" if group == "age" else "Sex"
    datasets = [d for d in DATASET_ORDER
                if any(agg.ds_name == d for agg in aggregators)]
    n_ds     = len(datasets)

    fig, axes = plt.subplots(1, n_ds, figsize=(n_ds * 6.0, 5.5),
                             gridspec_kw={"wspace": 0.35})
    if n_ds == 1:
        axes = [axes]

    tech_labels = [TECHNIQUE_LABELS.get(t, t) for t in TECHNIQUES]
    x_pos = np.arange(len(TECHNIQUES))

    for ax, ds in zip(axes, datasets):
        agg = next((a for a in aggregators if a.ds_name == ds), None)
        if agg is None:
            continue

        # Tamaño original: media ± std sobre los 5 folds
        orig_sizes = agg.fold_sizes_orig[group]
        orig_mean  = float(np.mean(orig_sizes)) if orig_sizes else 0.0

        # Tamaño por técnica: media ± std sobre los folds disponibles
        heights, errs = [], []
        for tech in TECHNIQUES:
            sizes = agg.fold_sizes[group].get(tech, [])
            if sizes:
                heights.append(float(np.mean(sizes)))
                errs.append(float(np.std(sizes, ddof=1)) if len(sizes) > 1 else 0.0)
            else:
                heights.append(0.0)
                errs.append(0.0)

        bars = ax.bar(x_pos, heights, yerr=errs, width=0.65,
                      color=[FAMILY_COLOR.get(t, "#888") for t in TECHNIQUES],
                      edgecolor="white", linewidth=0.5, alpha=0.85, zorder=2,
                      capsize=3.5,
                      error_kw={"ecolor": "#333", "elinewidth": 1.0, "zorder": 4})

        # Línea horizontal del original (media de los folds originales)
        ax.axhline(orig_mean, color="#111", lw=1.4, ls="--",
                   alpha=0.75, zorder=3)
        ax.text(len(TECHNIQUES) - 0.5, orig_mean * 1.012,
                f"Original (mean): {int(orig_mean):,}",
                ha="right", va="bottom", fontsize=8,
                fontweight="bold", color="#111")

        # Valor sobre cada barra (compacto, rotado para no solapar)
        ymax = max([h + e for h, e in zip(heights, errs)] + [orig_mean]) if heights else orig_mean
        for xi, (h, e) in enumerate(zip(heights, errs)):
            if h > 0:
                ax.text(xi, h + e + ymax * 0.015, f"{int(h):,}",
                        ha="center", va="bottom",
                        fontsize=6.5, color="#333", rotation=90)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(tech_labels, rotation=38, ha="right", fontsize=8.5)
        ax.set_ylabel("Mean training samples ± SD (5 folds)", fontsize=9.5)
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=12, fontweight="bold")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_ylim(0, ymax * 1.25)   # margen para etiquetas rotadas
        ax.grid(axis="y", alpha=0.22, linestyle="--", color="0.6")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Mean train set size per balancing technique  ({dim_lbl} dimension)\n"
        f"Bars = mean across {N_FOLDS} folds  ·  error bars = ± 1 SD across folds  "
        f"|  Dashed line = mean original dataset size",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    _save(fig, f"trainsize_bars_{group}.pdf")


# ===========================================================================
# FIG-B4: Barras divergentes Δ% por rango glucémico  (3 fig)
# ===========================================================================

def plot_delta_glycemic(agg: DatasetAggregator):
    """
    Para cada técnica: barra horizontal que muestra cuántos puntos de %
    gana (+) o pierde (-) cada rango glucémico respecto al original.
    Panel izquierdo = age, panel derecho = sex.
    Subpaneles por rango glucémico (5 rangos, 5 filas de barras).
    """
    ds_lbl = DATASET_LABELS.get(agg.ds_name, agg.ds_name)
    n_techs = len(TECHNIQUES)

    fig, axes = plt.subplots(
        len(GLYCEMIC_RANGES), 2,
        figsize=(16, n_techs * 0.72 + 2.5),
        gridspec_kw={"hspace": 0.55, "wspace": 0.38}
    )

    for col_idx, group in enumerate(DIMENSIONS):
        orig_glyc = agg.orig_glyc[group]
        tech_glyc = agg.tech_glyc[group]

        # Calcular totales originales por clase → distribución porcentual
        orig_total = orig_glyc.sum(axis=1)  # total por clase

        for row_idx, rng in enumerate(GLYCEMIC_RANGES):
            ax = axes[row_idx, col_idx]

            deltas = []
            labels = []
            colors_list = []

            for tech in TECHNIQUES:
                if tech not in tech_glyc:
                    deltas.append(np.nan)
                    labels.append(TECHNIQUE_LABELS.get(tech, tech))
                    colors_list.append("#ccc")
                    continue

                df_bal = tech_glyc[tech]
                # Δ puntos porcentuales: media sobre clases (ponderada por tamaño original)
                dp_list = []
                for cls in _class_order(group):
                    o_total = orig_total.get(cls, 0)
                    if o_total == 0:
                        continue
                    o_rng = orig_glyc.loc[cls, rng] if (cls in orig_glyc.index and rng in orig_glyc.columns) else 0
                    b_rng = df_bal.loc[cls, rng]    if (cls in df_bal.index   and rng in df_bal.columns)   else 0
                    b_total = df_bal.loc[cls].sum() if cls in df_bal.index else 0
                    # pp = % en rango tras balanceo - % en rango antes
                    o_pct = o_rng / o_total * 100
                    b_pct = b_rng / b_total * 100 if b_total > 0 else 0
                    dp_list.append(b_pct - o_pct)

                delta = float(np.mean(dp_list)) if dp_list else np.nan
                deltas.append(delta)
                labels.append(TECHNIQUE_LABELS.get(tech, tech))
                colors_list.append(GLYCEMIC_COLORS[rng])

            y_pos = np.arange(n_techs)
            valid  = [d for d in deltas if not np.isnan(d)]
            xmax   = max(max(np.abs(valid)), 0.5) * 1.15 if valid else 1.0

            for yi, (delta, lab, col) in enumerate(zip(deltas, labels, colors_list)):
                if np.isnan(delta):
                    continue
                ax.barh(yi, delta, height=0.6,
                        color=col, alpha=0.82,
                        edgecolor="white", linewidth=0.4)
                txt_x   = delta + xmax * 0.02 * np.sign(delta)
                txt_ha  = "left" if delta >= 0 else "right"
                ax.text(txt_x, yi, f"{delta:+.2f}%",
                        ha=txt_ha, va="center", fontsize=7.5, color="#222")

            ax.axvline(0, color="#333", lw=1.0, zorder=3)
            ax.set_xlim(-xmax, xmax)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("Δ%  (share of range, vs. original)", fontsize=8.5)
            ax.grid(axis="x", alpha=0.2, linestyle="--")
            ax.set_axisbelow(True)

            rng_lbl = GLYCEMIC_LABELS.get(rng, rng).split("(")[0].strip()
            dim_lbl = "Age" if group == "age" else "Sex"
            ax.set_title(f"{rng_lbl}  [{dim_lbl}]", fontsize=9,
                         color=GLYCEMIC_COLORS[rng], fontweight="bold")

    fig.suptitle(
        f"{ds_lbl}  ·  Change in glycemic range composition (Δ% vs. original)\n"
        f"Positive = technique increases share of this range  "
        f"|  Negative = technique reduces it  "
        f"|  Averaged across folds and demographic classes",
        fontsize=10, y=1.01
    )
    _save(fig, f"delta_glycemic_{agg.ds_name}.pdf")


# ===========================================================================
# FIG-B5: Scatter trade-off {rango} vs. TIR  (4 figuras: TBR_2, TBR_1, TAR_1, TAR_2)
# ===========================================================================

def plot_tradeoff_scatter(aggregators: list, vs_range: str):
    """
    Scatter: Δ% de `vs_range` (eje X) vs. Δ% TIR (eje Y, en rango).
    Se llama una vez por cada rango en TRADEOFF_RANGES (TBR_2, TBR_1, TAR_1, TAR_2),
    generando una figura independiente por rango — así se visualiza el trade-off
    clínico tanto en el extremo de hipoglucemia como en el de hiperglucemia.
    Un punto por (técnica × dimensión × dataset).
    Marcador = dimensión (○ age, □ sex), color = técnica.
    """
    kind      = RANGE_KIND.get(vs_range, "hypo")            # "hypo" o "hyper"
    rng_lbl   = GLYCEMIC_LABELS.get(vs_range, vs_range).split("(")[0].strip()
    rng_units = GLYCEMIC_LABELS.get(vs_range, vs_range).split("(")[-1].rstrip(")")

    records = []
    for agg in aggregators:
        for group in DIMENSIONS:
            orig_glyc = agg.orig_glyc[group]
            orig_pct  = orig_glyc.div(orig_glyc.sum(axis=1), axis=0) * 100
            orig_pct  = orig_pct.fillna(0)

            for tech in TECHNIQUES:
                if tech not in agg.tech_glyc[group]:
                    continue
                bal_glyc = agg.tech_glyc[group][tech]
                bal_pct  = bal_glyc.div(bal_glyc.sum(axis=1), axis=0) * 100
                bal_pct  = bal_pct.fillna(0)

                # Δ% ponderado por clase (media simple entre clases)
                classes  = [c for c in _class_order(group)
                            if c in orig_pct.index and c in bal_pct.index]
                if not classes:
                    continue
                d_rng = float(np.mean([
                    bal_pct.loc[c, vs_range] - orig_pct.loc[c, vs_range]
                    for c in classes
                ]))
                d_tir  = float(np.mean([
                    bal_pct.loc[c, "TIR"]   - orig_pct.loc[c, "TIR"]
                    for c in classes
                ]))
                records.append({
                    "technique": tech,
                    "dimension": group,
                    "dataset":   agg.ds_name,
                    "d_rng":     d_rng,
                    "d_tir":     d_tir,
                })

    if not records:
        log.warning(f"FIG-B5 ({vs_range}): sin datos")
        return

    df = pd.DataFrame(records)
    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    n_ds = len(datasets)

    fig, axes = plt.subplots(1, n_ds, figsize=(n_ds * 6.5, 6), sharey=True)
    if n_ds == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        for _, row in sub.iterrows():
            marker = FAMILY_MARKER.get(row["dimension"], "o")
            color  = FAMILY_COLOR.get(row["technique"], "#888")
            ax.scatter(row["d_rng"], row["d_tir"],
                       color=color, marker=marker, s=90,
                       edgecolors="white", linewidths=0.7, zorder=3)
            # Etiquetar solo la dimensión age para no saturar
            if row["dimension"] == "age":
                ax.annotate(
                    TECHNIQUE_LABELS.get(row["technique"], row["technique"]),
                    (row["d_rng"], row["d_tir"]),
                    fontsize=6.5, color="#333",
                    xytext=(4, 3), textcoords="offset points"
                )

        ax.axhline(0, color="#333", lw=1.0, ls="--", alpha=0.5)
        ax.axvline(0, color="#333", lw=1.0, ls="--", alpha=0.5)
        ax.set_xlabel(f"Δ% {rng_lbl} ({rng_units})", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Δ% TIR (In Range, 70–180 mg/dL)", fontsize=10)
        ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=12, fontweight="bold")
        ax.grid(alpha=0.2, linestyle="--")

        # Cuadrantes clínicos — más Δrango a la derecha siempre es peor
        # (más tiempo fuera de rango), tanto si el rango es hipo como hiper.
        risk_word = "hypo" if kind == "hypo" else "hyper"
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.text(xlim[1] * 0.95, ylim[1] * 0.95,
                f"↑ TIR\n→ more {risk_word}\n(worsens safety)",
                ha="right", va="top", fontsize=7.5, color="#c0392b", alpha=0.7)
        ax.text(xlim[1] * 0.95, ylim[0] * 0.95,
                f"↓ TIR\n→ more {risk_word}\n(both worsen)",
                ha="right", va="bottom", fontsize=7.5, color="#888", alpha=0.7)
        ax.text(xlim[0] * 0.95, ylim[1] * 0.95,
                f"↑ TIR\n← less {risk_word}\n(ideal)",
                ha="left", va="top", fontsize=7.5, color="#27ae60", alpha=0.8)

    # Leyenda técnicas
    tech_handles = [
        mpatches.Patch(color=FAMILY_COLOR.get(t, "#888"),
                       label=TECHNIQUE_LABELS.get(t, t))
        for t in TECHNIQUES
    ]
    # Leyenda dimensión
    dim_handles = [
        plt.scatter([], [], marker="o", color="0.4", s=60, label="Age"),
        plt.scatter([], [], marker="s", color="0.4", s=60, label="Sex"),
    ]

    fig.legend(handles=tech_handles + dim_handles,
               loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.08),
               fontsize=8.5, framealpha=0.95,
               edgecolor="0.8", title="Technique  (○=Age, □=Sex)",
               title_fontsize=8.5)

    fig.suptitle(
        f"Clinical trade-off: {rng_lbl} representation vs. In-Range after balancing\n"
        "Each point: one technique × dimension, averaged across folds and demographic classes  "
        f"|  Ideal quadrant: top-left  (↑ TIR, no increase in {rng_lbl})",
        fontsize=10, y=1.02
    )
    plt.tight_layout()
    fname = f"tradeoff_{vs_range.lower().replace('_', '')}_vs_tir.pdf"
    _save(fig, fname)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Salida: {OUT_DIR}\n")

    # -- Carga y agregación --
    log.info("Cargando y agregando datos...")
    aggregators = []
    for ds_name in DATASET_ORDER:
        dataset_dir = DATASETS.get(ds_name)
        if dataset_dir is None:
            continue
        try:
            agg = DatasetAggregator(ds_name, dataset_dir)
            aggregators.append(agg)
            log.info(f"  ✓ {ds_name}")
        except FileNotFoundError as e:
            log.warning(f"  ✗ {ds_name}: {e}")

    if not aggregators:
        log.error("Sin datasets cargados. Revisa DATA_DIR y estructura de directorios.")
        return

    # -- FIG-B1: Heatmaps demográficos (6) --
    # vmax se calcula UNA vez por dimensión, sobre todos los datasets juntos,
    # para que la escala de color sea homogénea y comparable entre datasets.
    log.info("\nFIG-B1: Heatmaps de variación demográfica...")
    vmax_by_group = {group: _global_heatmap_vmax(aggregators, group) for group in DIMENSIONS}
    for agg in aggregators:
        for group in DIMENSIONS:
            plot_heatmap_balancing(agg, group, vmax=vmax_by_group[group])

    # -- FIG-B2: Barras apiladas glucémicas (3) --
    # xmax compartido entre los paneles age/sex de un mismo dataset;
    # valores = media por fold (no la suma de los 5 folds).
    log.info("\nFIG-B2: Barras apiladas por rango glucémico...")
    for agg in aggregators:
        plot_stacked_bars(agg)

    # -- FIG-B3: Barras de tamaño de train, media ± std (2) --
    log.info("\nFIG-B3: Barras de tamaño de train (media ± SD)...")
    for group in DIMENSIONS:
        plot_trainsize_bars(aggregators, group)

    # -- FIG-B4: Δ% por rango glucémico (3) --
    log.info("\nFIG-B4: Barras divergentes por rango glucémico...")
    for agg in aggregators:
        plot_delta_glycemic(agg)

    # -- FIG-B5: Scatter trade-off vs. TIR, uno por rango (4) --
    log.info("\nFIG-B5: Scatter trade-off clínico (TBR_2, TBR_1, TAR_1, TAR_2 vs. TIR)...")
    for rng in TRADEOFF_RANGES:
        plot_tradeoff_scatter(aggregators, rng)

    n = len(aggregators)
    n_b5 = len(TRADEOFF_RANGES)
    total = n * 2 + n + 2 + n + n_b5
    log.info(f"""
Resumen de figuras generadas:
  FIG-B1  heatmap_balancing_*       {n * 2:>3} figuras  (1 por dataset × dimensión, escala unificada por dimensión)
  FIG-B2  stacked_bars_*            {n:>3} figuras  (1 por dataset, eje X compartido age/sex, medias por fold)
  FIG-B3  trainsize_bars_*            2 figuras  (1 por dimensión, media ± SD)
  FIG-B4  delta_glycemic_*          {n:>3} figuras  (1 por dataset, en %)
  FIG-B5  tradeoff_*_vs_tir         {n_b5:>3} figuras  (TBR_2, TBR_1, TAR_1, TAR_2 vs. TIR)
  ────────────────────────────────────────────
  Total                             {total:>3} figuras
""")


if __name__ == "__main__":
    main()