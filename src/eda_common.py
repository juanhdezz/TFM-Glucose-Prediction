# -*- coding: utf-8 -*-
"""
eda_common.py — Funciones compartidas para EDA de datasets originales.
RELIEF-T1D · TFM DATCOM UGR

Estilo visual alineado con config.py (paleta Wong 2011, colorblind-safe).
Leyendas siempre fuera del área de datos o gestionadas con bbox_to_anchor
para garantizar que NUNCA se solapen con las visualizaciones.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# IMPORTAR CONFIGURACIÓN CENTRAL
# ---------------------------------------------------------------------------
try:
    from analysis_relief.analysis.config import (
        OUTPUT_ROOT, DATASET_LABELS, DATASET_PALETTE,
        FIGURES_DIR, TABLES_DIR, STATS_DIR, DASHBOARDS_DIR,
        PALETTE, DIMENSION_PALETTE, MPL_RC,
    )
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    OUTPUT_ROOT  = PROJECT_ROOT / "results"
    FIGURES_DIR  = OUTPUT_ROOT / "figures"
    TABLES_DIR   = OUTPUT_ROOT / "tables"
    STATS_DIR    = OUTPUT_ROOT / "stats"
    DASHBOARDS_DIR = OUTPUT_ROOT / "dashboards"
    DATASET_LABELS  = {"DIATREND": "DiaTrend", "REPLACE-BG": "ReplaceBG", "T1DiabetesGranada": "T1DiabetesGranada"}
    DATASET_PALETTE = {"DIATREND": "#009E73", "REPLACE-BG": "#E69F00", "T1DiabetesGranada": "#CC79A7"}
    DIMENSION_PALETTE = {"age": "#0072B2", "sex": "#D55E00", "original": "#000000"}
    MPL_RC = {
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9, "legend.framealpha": 0.92, "legend.edgecolor": "0.75",
        "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    }

# ---------------------------------------------------------------------------
# CONSTANTES CLÍNICAS / DEMOGRÁFICAS
# ---------------------------------------------------------------------------
SENSOR_LIMITS = {
    "DIATREND":          (39.0, 401.0),
    "REPLACE-BG":        (39.0, 401.0),
    "T1DiabetesGranada": (40.0, 500.0),
}

AGE_BINS   = [-np.inf, 30, 45, 65, np.inf]
AGE_LABELS = ["<31", "31–45", "46–65", "≥66"]

# Paletas alineadas con config.py (DIMENSION_PALETTE)
SEX_PALETTE = {"F": "#D55E00", "M": "#0072B2", "Unknown": "#999999"}
AGE_PALETTE = {"<31": "#56B4E9", "31–45": "#009E73", "46–65": "#E69F00", "≥66": "#D55E00"}

# Rangos glucémicos — 5 bandas clínicas (aligned con RANGE_INFO de config.py)
GLUCOSE_RANGE_PALETTE = {
    "TBR_2": "#4575B4", "TBR_1": "#91BFDB",
    "TIR":   "#7DCB8A",
    "TAR_1": "#FC8D59", "TAR_2": "#D73027",
}
GLUCOSE_RANGE_LABELS = {
    "TBR_2": "Hypo L2 (<54 mg/dL)",
    "TBR_1": "Hypo L1 (54–69 mg/dL)",
    "TIR":   "In Range (70–180 mg/dL)",
    "TAR_1": "Hyper L1 (181–250 mg/dL)",
    "TAR_2": "Hyper L2 (>250 mg/dL)",
}
RANGE_ORDER_FINE = ["TBR_2", "TBR_1", "TIR", "TAR_1", "TAR_2"]

SEX_MAP = {
    "F": "F", "FEMALE": "F", "FEM": "F",
    "M": "M", "MALE": "M",  "MASC": "M",
}

# ---------------------------------------------------------------------------
# ESTILO GLOBAL
# ---------------------------------------------------------------------------

def setup_plot_style():
    """Aplica estilo matplotlib homogéneo con config.py."""
    matplotlib.rcParams.update(MPL_RC)
    # Añadimos refinamientos de publicación sobre el rc base
    matplotlib.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
        "axes.linewidth":    0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.borderpad":  0.5,
        "legend.handlelength": 1.5,
        "legend.handleheight": 0.8,
    })


def _legend_outside(ax, loc="upper left", title=None, ncol=1, **kw):
    """
    Coloca la leyenda fuera del área de datos (a la derecha o abajo)
    para garantizar que NUNCA se solape con las visualizaciones.

    Uso:
        _legend_outside(ax)             → derecha
        _legend_outside(ax, "bottom")   → debajo, centrado
    """
    if loc == "bottom":
        leg = ax.legend(
            title=title, ncol=ncol,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            frameon=True, framealpha=0.92,
            edgecolor="0.75", **kw
        )
    else:
        leg = ax.legend(
            title=title, ncol=ncol,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            frameon=True, framealpha=0.92,
            edgecolor="0.75", **kw
        )
    return leg


def _fig_legend_outside(fig, handles, labels, ncol=4, title=None, y_offset=-0.04):
    """Leyenda de figura centrada debajo de todos los subplots."""
    leg = fig.legend(
        handles, labels,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, y_offset),
        frameon=True, framealpha=0.92,
        edgecolor="0.75",
        title=title,
    )
    return leg


def _savefig(fig, path: Path, tight_layout: bool = True, extra_bottom: float = 0.0):
    """
    Guarda en PNG y PDF.
    extra_bottom (0–0.3): espacio adicional abajo para leyendas de figura.
    """
    if tight_layout:
        fig.tight_layout()
    if extra_bottom > 0:
        fig.subplots_adjust(bottom=extra_bottom)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FUNCIONES DE CARGA / PREPARACIÓN
# ---------------------------------------------------------------------------

def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    norm_cols = {c.strip().lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower().replace(" ", "_")
        if key in norm_cols:
            return norm_cols[key]
    raise KeyError(f"Ninguna de {candidates} encontrada en columnas.")


def infer_age_series(patient_info: pd.DataFrame) -> pd.Series:
    for col in ["Age", "age"]:
        try:
            age_col = find_column(patient_info, [col])
            return pd.to_numeric(patient_info[age_col], errors="coerce")
        except KeyError:
            continue
    for col in ["Birth_year", "birth_year", "Birth Year"]:
        try:
            birth_col = find_column(patient_info, [col])
            birth_year = pd.to_numeric(patient_info[birth_col], errors="coerce")
            return datetime.now().year - birth_year
        except KeyError:
            continue
    raise KeyError("No se pudo inferir la edad: falta 'Age' o 'Birth_year'.")


def build_demographic_lookup(patient_info: pd.DataFrame) -> pd.DataFrame:
    meta    = patient_info.copy()
    pid_col = find_column(meta, ["patient_id", "Patient_ID", "patientid"])
    meta["_patient_key"] = meta[pid_col].astype(str).str.strip()

    sex_col = find_column(meta, ["Sex", "sex"])
    meta["sex_group"] = (
        meta[sex_col].astype(str).str.strip().str.upper()
        .map(SEX_MAP).fillna("Unknown")
    )

    meta["age"] = infer_age_series(meta)
    meta["age_group"] = pd.cut(
        meta["age"], bins=AGE_BINS, labels=AGE_LABELS,
        include_lowest=True, right=True
    ).astype("object").fillna("Unknown")

    known_ages = [g for g in AGE_LABELS if g in meta["age_group"].unique()]
    if known_ages:
        meta["age_group"] = pd.Categorical(
            meta["age_group"],
            categories=known_ages + ["Unknown"],
            ordered=True,
        )
    return (
        meta[["_patient_key", "sex_group", "age", "age_group"]]
        .drop_duplicates(subset=["_patient_key"])
        .set_index("_patient_key")
    )


def load_original_data(dataset_name: str, data_root: Path) -> pd.DataFrame:
    dataset_dir = data_root / dataset_name
    patterns = [
        f"Glucose_measurements_*{dataset_name}*FILTERED*.parquet",
        f"Glucose_measurements_*{dataset_name}*FILTERED*.csv",
        f"*{dataset_name}*Glucose_measurements*.parquet",
        f"*{dataset_name}*Glucose_measurements*.csv",
        "Glucose_measurements_*.parquet",
        "Glucose_measurements_*.csv",
    ]
    for pat in patterns:
        candidates = list(dataset_dir.glob(pat))
        if candidates:
            fp = candidates[0]
            return pd.read_parquet(fp) if fp.suffix.lower() == ".parquet" else pd.read_csv(fp)
    raise FileNotFoundError(f"No se encontró archivo original para {dataset_name} en {dataset_dir}")


def load_patient_info(dataset_name: str, data_root: Path) -> pd.DataFrame:
    dataset_dir = data_root / dataset_name
    for ext in [".parquet", ".csv"]:
        candidates = (
            list(dataset_dir.glob(f"*patient_info*{ext}")) +
            list(dataset_dir.glob(f"*Patient_info*{ext}"))
        )
        if candidates:
            fp = candidates[0]
            return pd.read_parquet(fp) if ext == ".parquet" else pd.read_csv(fp)
    raise FileNotFoundError(f"No se encontró patient_info para {dataset_name} en {dataset_dir}")


def clean_measurements(df: pd.DataFrame, min_val: float, max_val: float) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean["measurement"] = pd.to_numeric(df_clean["measurement"], errors="coerce")
    df_clean = df_clean[df_clean["measurement"].notna()]
    return df_clean[(df_clean["measurement"] >= min_val) & (df_clean["measurement"] <= max_val)]


def prepare_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "measurement_date" in df.columns:
        df["measurement_date"] = df["measurement_date"].astype(str)
    if "measurement_time" in df.columns:
        df["measurement_time"] = df["measurement_time"].astype(str)
    df["timestamp"] = pd.to_datetime(
        df["measurement_date"] + " " + df["measurement_time"], errors="coerce"
    )
    for col in ["15min", "5min"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def summarize_patients(df: pd.DataFrame, demo_lookup: pd.DataFrame) -> pd.DataFrame:
    pid_col = find_column(df, ["patient_id", "Patient_ID", "patientid"])
    df["_patient_key"] = df[pid_col].astype(str).str.strip()
    df_with_demo = df.join(demo_lookup, on="_patient_key", how="left")

    summary = df_with_demo.groupby("_patient_key").agg(
        n_measurements=("measurement", "count"),
        first_date=("timestamp", "min"),
        last_date=("timestamp", "max"),
        mean_glucose=("measurement", "mean"),
        std_glucose=("measurement", "std"),
        min_glucose=("measurement", "min"),
        max_glucose=("measurement", "max"),
        sex_group=("sex_group", "first"),
        age_group=("age_group", "first"),
        age=("age", "first"),
    ).reset_index()

    summary["days_followup"] = (summary["last_date"] - summary["first_date"]).dt.days
    summary["frequency_per_day"] = summary["n_measurements"] / summary["days_followup"].clip(lower=1)
    return summary


def _assign_fine_range(series: pd.Series) -> pd.Series:
    """Clasifica glucosa en 5 rangos clínicos (aligned con config.py RANGE_INFO)."""
    out = pd.Series("unknown", index=series.index)
    out[series < 54]                        = "TBR_2"
    out[(series >= 54) & (series < 70)]     = "TBR_1"
    out[(series >= 70) & (series <= 180)]   = "TIR"
    out[(series > 180) & (series <= 250)]   = "TAR_1"
    out[series > 250]                       = "TAR_2"
    return out


def glucose_range_category(value: float) -> str:
    """Clasificación simple TBR / TIR / TAR para compatibilidad legada."""
    if value < 70:   return "TBR (<70)"
    if value <= 180: return "TIR (70–180)"
    return "TAR (>180)"


def compute_range_composition(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    df["range_cat"] = df["measurement"].apply(glucose_range_category)
    pivot = df.groupby([group_col, "range_cat"]).size().unstack(fill_value=0)
    pct   = pivot.div(pivot.sum(axis=1), axis=0) * 100
    if isinstance(df[group_col].dtype, pd.CategoricalDtype):
        ordered = [c for c in df[group_col].cat.categories if c in pct.index]
        pct = pct.reindex(ordered)
    return pct


# ---------------------------------------------------------------------------
# FIGURAS — calidad de publicación
# ---------------------------------------------------------------------------

def _plot_glucose_distribution(df_clean, dataset_name, fig_dir, min_val, max_val):
    """Fig 1: Histograma + KDE de la distribución global de glucosa."""
    ds_color = DATASET_PALETTE.get(dataset_name, "#56B4E9")
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        df_clean["measurement"], bins=60, kde=True, ax=ax,
        color=ds_color, alpha=0.65, linewidth=0,
        line_kws={"linewidth": 2, "color": ds_color},
    )
    # Bandas clínicas de fondo
    ax.axvspan(ax.get_xlim()[0], 70,  alpha=0.06, color="#4575B4", zorder=0)
    ax.axvspan(70, 180,              alpha=0.06, color="#7DCB8A", zorder=0)
    ax.axvspan(180, ax.get_xlim()[1] if ax.get_xlim()[1] > 180 else max_val,
               alpha=0.06, color="#D73027", zorder=0)

    ax.axvline(min_val, color="#555555", linestyle=":", linewidth=1.2, label=f"Sensor min ({min_val:.0f})")
    ax.axvline(max_val, color="#555555", linestyle=":",  linewidth=1.2, label=f"Sensor max ({max_val:.0f})")
    ax.axvline(70,  color="#4575B4", linestyle="--", linewidth=1.2, label="Hypo threshold (70)")
    ax.axvline(180, color="#D73027", linestyle="--", linewidth=1.2, label="Hyper threshold (180)")

    ax.set_xlabel("Glucose (mg/dL)")
    ax.set_ylabel("Count")
    ax.set_title(f"Glucose Distribution — {DATASET_LABELS.get(dataset_name, dataset_name)}")
    _legend_outside(ax, loc="upper left")

    _savefig(fig, fig_dir / "glucose_distribution_global")


def _plot_measurements_per_patient(patient_summary, dataset_name, stats_patients, fig_dir):
    """Fig 2: Distribución de mediciones por paciente."""
    ds_color = DATASET_PALETTE.get(dataset_name, "#56B4E9")
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(patient_summary["n_measurements"], bins=30, kde=True, ax=ax,
                 color=ds_color, alpha=0.65, linewidth=0,
                 line_kws={"linewidth": 2, "color": ds_color})
    ax.axvline(stats_patients["mean"], color="#D55E00", linestyle="--",
               linewidth=1.5, label=f"Mean: {stats_patients['mean']:.0f}")
    ax.axvline(stats_patients["50%"],  color="#009E73", linestyle="--",
               linewidth=1.5, label=f"Median: {stats_patients['50%']:.0f}")

    ax.set_xlabel("N measurements per patient")
    ax.set_ylabel("Count")
    ax.set_title(f"Measurements per Patient — {DATASET_LABELS.get(dataset_name, dataset_name)}")
    _legend_outside(ax, loc="upper left")

    _savefig(fig, fig_dir / "measurements_per_patient_dist")


def _plot_glucose_by_sex(df_sex, dataset_name, fig_dir):
    """Fig 3a: Violinplot glucosa por sexo."""
    if df_sex.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))

    vp = ax.violinplot(
        [df_sex[df_sex["sex_group"] == g]["measurement"].dropna().values
         for g in ["F", "M"] if g in df_sex["sex_group"].unique()],
        positions=range(len([g for g in ["F", "M"] if g in df_sex["sex_group"].unique()])),
        showmedians=True, showextrema=False,
    )
    sex_groups = [g for g in ["F", "M"] if g in df_sex["sex_group"].unique()]
    for body, grp in zip(vp["bodies"], sex_groups):
        body.set_facecolor(SEX_PALETTE[grp])
        body.set_alpha(0.7)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(2)

    ax.axhline(70,  color="#4575B4", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(180, color="#D73027", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(range(len(sex_groups)))
    ax.set_xticklabels(["Female" if g == "F" else "Male" for g in sex_groups])
    ax.set_xlabel("Sex")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title(f"Glucose by Sex — {DATASET_LABELS.get(dataset_name, dataset_name)}")

    # Leyenda de umbrales fuera del área
    handles = [
        Line2D([0], [0], color="#4575B4", linestyle="--", linewidth=1, label="Hypo (70 mg/dL)"),
        Line2D([0], [0], color="#D73027", linestyle="--", linewidth=1, label="Hyper (180 mg/dL)"),
    ]
    _legend_outside(ax, loc="upper left")
    ax.add_artist(ax.legend(handles=handles, loc="upper left",
                            bbox_to_anchor=(1.02, 1.0), frameon=True,
                            framealpha=0.92, edgecolor="0.75"))
    _savefig(fig, fig_dir / "glucose_by_sex")


def _plot_range_stacked(range_df, group_label, title, out_path):
    """
    Gráfico de barras apiladas con rangos glucémicos (5 bandas).
    Leyenda siempre fuera — debajo del gráfico.
    """
    # Columnas presentes en orden canónico
    cols = [c for c in RANGE_ORDER_FINE if c in range_df.columns]
    if not cols:
        # Fallback: 3 rangos legados
        cols = [c for c in ["TBR (<70)", "TIR (70–180)", "TAR (>180)"] if c in range_df.columns]
        colors = ["#4575B4", "#7DCB8A", "#D73027"]
        labels_map = {"TBR (<70)": "TBR (<70)", "TIR (70–180)": "TIR (70–180)", "TAR (>180)": "TAR (>180)"}
    else:
        colors = [GLUCOSE_RANGE_PALETTE[c] for c in cols]
        labels_map = GLUCOSE_RANGE_LABELS

    n_groups = len(range_df)
    fig, ax = plt.subplots(figsize=(max(6, n_groups * 1.4 + 2.5), 4.5))

    bottom = np.zeros(n_groups)
    bars_handles = []
    for col, color in zip(cols, colors):
        if col not in range_df.columns:
            continue
        vals = range_df[col].values
        bars = ax.bar(range(n_groups), vals, bottom=bottom,
                      color=color, label=labels_map.get(col, col),
                      edgecolor="white", linewidth=0.5, alpha=0.92)
        bars_handles.append(bars[0])
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 4:
                ax.text(xi, b + v / 2, f"{v:.1f}%",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if color in ("#4575B4", "#D73027") else "black",
                        fontweight="bold")
        bottom += vals

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(range_df.index, rotation=0)
    ax.set_xlabel(group_label)
    ax.set_ylabel("% of measurements")
    ax.set_ylim(0, 107)
    ax.set_title(title)

    # Leyenda centrada debajo — NUNCA sobre las barras
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=len(cols),
        frameon=True, framealpha=0.92, edgecolor="0.75",
        fontsize=8,
    )
    _savefig(fig, out_path, tight_layout=True, extra_bottom=0.18)


def _plot_glucose_by_age(df_age, age_order, dataset_name, fig_dir):
    """Fig 4a: Violinplot glucosa por grupo de edad."""
    if df_age.empty:
        return
    groups = [g for g in age_order if g in df_age["age_group"].unique()]
    fig, ax = plt.subplots(figsize=(max(7, len(groups) * 1.6), 5))

    vp = ax.violinplot(
        [df_age[df_age["age_group"] == g]["measurement"].dropna().values for g in groups],
        positions=range(len(groups)),
        showmedians=True, showextrema=False,
    )
    for body, grp in zip(vp["bodies"], groups):
        body.set_facecolor(AGE_PALETTE.get(grp, "#999999"))
        body.set_alpha(0.7)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(2)

    ax.axhline(70,  color="#4575B4", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(180, color="#D73027", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_xlabel("Age group")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title(f"Glucose by Age Group — {DATASET_LABELS.get(dataset_name, dataset_name)}")

    handles = [
        mpatches.Patch(facecolor=AGE_PALETTE.get(g, "#999"), label=g) for g in groups
    ] + [
        Line2D([0], [0], color="#4575B4", linestyle="--", linewidth=1, label="Hypo (70 mg/dL)"),
        Line2D([0], [0], color="#D73027", linestyle="--", linewidth=1, label="Hyper (180 mg/dL)"),
    ]
    _legend_outside(ax, loc="upper left")
    ax.add_artist(ax.legend(handles=handles, loc="upper left",
                            bbox_to_anchor=(1.02, 1.0), frameon=True,
                            framealpha=0.92, edgecolor="0.75", fontsize=8))
    _savefig(fig, fig_dir / "glucose_by_age")


def _plot_trend(df_monthly, hue_col, hue_order, palette, title, xlabel, ylabel, out_path):
    """
    Evolución temporal mensual.
    Leyenda fuera a la derecha para no solaparse con las líneas.
    """
    fig, ax = plt.subplots(figsize=(12, 4.5))

    for grp in hue_order:
        sub = df_monthly[df_monthly[hue_col] == grp]
        if sub.empty:
            continue
        color = palette.get(grp, "#999999")
        ax.plot(sub["timestamp"], sub["measurement"], linewidth=1.8,
                color=color, label=grp, alpha=0.9)
        # Banda de confianza suave (±1 SD si disponible)

    ax.axhline(70,  color="#4575B4", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.axhline(180, color="#D73027", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30)
    _legend_outside(ax, loc="upper left")
    _savefig(fig, out_path)


def _plot_monthly_counts(monthly_counts, dataset_name, fig_dir):
    """Fig 7: Mediciones por mes — barras con color del dataset."""
    ds_color = DATASET_PALETTE.get(dataset_name, "#56B4E9")
    fig, ax = plt.subplots(figsize=(13, 4.5))

    x_pos = range(len(monthly_counts))
    ax.bar(x_pos, monthly_counts.values, color=ds_color, alpha=0.75, linewidth=0)
    ax.set_xticks(list(x_pos)[::max(1, len(monthly_counts)//12)])
    ax.set_xticklabels(
        [str(monthly_counts.index[i]) for i in range(0, len(monthly_counts), max(1, len(monthly_counts)//12))],
        rotation=40, ha="right", fontsize=8,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("N measurements")
    ax.set_title(f"Measurements per Month — {DATASET_LABELS.get(dataset_name, dataset_name)}")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    _savefig(fig, fig_dir / "monthly_counts")


# ---------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL DE REPORTE
# ---------------------------------------------------------------------------

def generate_report_and_figures(
    df: pd.DataFrame,
    demo_lookup: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    min_val: float,
    max_val: float,
):
    """Genera todas las tablas, figuras y el reporte Markdown."""
    setup_plot_style()

    fig_dir   = output_dir / "figures"
    table_dir = output_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    label = DATASET_LABELS.get(dataset_name, dataset_name)

    # ── Preparar datos ────────────────────────────────────────────────────────
    df_clean = clean_measurements(df, min_val, max_val)
    df_clean = prepare_datetime_columns(df_clean)
    df_clean["_patient_key"] = df_clean["patient_id"].astype(str).str.strip()
    df_demo  = df_clean.join(demo_lookup, on="_patient_key", how="left")

    patient_summary = summarize_patients(df_clean, demo_lookup)

    total_patients    = len(patient_summary)
    total_measurements= len(df_clean)
    sex_counts        = patient_summary["sex_group"].value_counts().to_dict()
    age_counts        = patient_summary["age_group"].value_counts().to_dict()
    date_min          = df_clean["timestamp"].min()
    date_max          = df_clean["timestamp"].max()
    days_span         = (date_max - date_min).days if pd.notna(date_min) and pd.notna(date_max) else None

    n_meas = patient_summary["n_measurements"]
    stats_patients = {
        "count": total_patients, "mean": n_meas.mean(), "std": n_meas.std(),
        "min": n_meas.min(), "25%": n_meas.quantile(0.25),
        "50%": n_meas.quantile(0.50), "75%": n_meas.quantile(0.75), "max": n_meas.max(),
    }

    coverage = {}
    for col in ["15min", "5min"]:
        if col in df_clean.columns:
            coverage[col] = df_clean[col].notna().sum() / len(df_clean) * 100
        else:
            coverage[col] = 0.0

    stats = {
        "dataset": dataset_name, "total_patients": total_patients,
        "total_measurements": total_measurements,
        "sex_counts": sex_counts, "age_counts": age_counts,
        "date_range": {"start": str(date_min), "end": str(date_max), "days": days_span},
        "measurement_min":  float(df_clean["measurement"].min()),
        "measurement_max":  float(df_clean["measurement"].max()),
        "measurement_mean": float(df_clean["measurement"].mean()),
        "measurement_std":  float(df_clean["measurement"].std()),
        "null_percentage": {
            col: df[col].isna().sum() / len(df) * 100
            for col in df.columns
            if col in ["measurement_date", "measurement_time", "measurement", "15min", "5min"]
        },
        "coverage_15min_5min": coverage,
        "measurements_per_patient": stats_patients,
    }

    with open(output_dir / "summary_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    patient_summary.to_csv(table_dir / "patient_summary.csv", index=False)

    # ── Enriquecer con rango glucémico fino ───────────────────────────────────
    df_demo["glucose_range"] = _assign_fine_range(df_demo["measurement"])

    # ── Figuras ───────────────────────────────────────────────────────────────
    # 1. Distribución global
    _plot_glucose_distribution(df_clean, dataset_name, fig_dir, min_val, max_val)

    # 2. Mediciones por paciente
    _plot_measurements_per_patient(patient_summary, dataset_name, stats_patients, fig_dir)

    # 3. Glucosa y rangos por sexo
    df_sex = df_demo[df_demo["sex_group"].isin(["F", "M"])].copy()
    if not df_sex.empty:
        _plot_glucose_by_sex(df_sex, dataset_name, fig_dir)

        # Rango glucémico 5 bandas por sexo
        df_sex["glucose_range"] = _assign_fine_range(df_sex["measurement"])
        pivot_sex = (
            df_sex.groupby(["sex_group", "glucose_range"])["measurement"]
            .count().unstack(fill_value=0)
            .reindex(columns=RANGE_ORDER_FINE, fill_value=0)
        )
        pivot_sex_pct = pivot_sex.div(pivot_sex.sum(axis=1), axis=0) * 100
        pivot_sex_pct.to_csv(table_dir / "glucose_range_by_sex.csv")
        _plot_range_stacked(
            pivot_sex_pct, "Sex",
            f"Glycemic Range Distribution by Sex — {label}",
            fig_dir / "glucose_range_by_sex",
        )

    # 4. Glucosa y rangos por edad
    df_age = df_demo[df_demo["age_group"] != "Unknown"].copy()
    if not df_age.empty:
        age_order = [c for c in AGE_LABELS if c in df_age["age_group"].unique()]
        _plot_glucose_by_age(df_age, age_order, dataset_name, fig_dir)

        df_age["glucose_range"] = _assign_fine_range(df_age["measurement"])
        pivot_age = (
            df_age.groupby(["age_group", "glucose_range"])["measurement"]
            .count().unstack(fill_value=0)
            .reindex(index=age_order, columns=RANGE_ORDER_FINE, fill_value=0)
        )
        pivot_age_pct = pivot_age.div(pivot_age.sum(axis=1), axis=0) * 100
        pivot_age_pct.to_csv(table_dir / "glucose_range_by_age.csv")
        _plot_range_stacked(
            pivot_age_pct, "Age group",
            f"Glycemic Range Distribution by Age — {label}",
            fig_dir / "glucose_range_by_age",
        )

    # 5. Tendencia temporal por sexo
    if not df_sex.empty:
        df_sex_monthly = (
            df_sex.groupby([pd.Grouper(key="timestamp", freq="ME"), "sex_group"])["measurement"]
            .mean().reset_index()
        )
        sex_in_data = [g for g in ["F", "M"] if g in df_sex_monthly["sex_group"].unique()]
        _plot_trend(
            df_sex_monthly, "sex_group", sex_in_data, SEX_PALETTE,
            f"Monthly Mean Glucose by Sex — {label}",
            "Date", "Mean glucose (mg/dL)",
            fig_dir / "glucose_trend_by_sex",
        )

    # 6. Tendencia temporal por edad
    if not df_age.empty:
        if not isinstance(df_age["age_group"].dtype, pd.CategoricalDtype):
            df_age["age_group"] = pd.Categorical(df_age["age_group"], categories=age_order, ordered=True)
        df_age_monthly = (
            df_age.groupby([pd.Grouper(key="timestamp", freq="ME"), "age_group"])["measurement"]
            .mean().reset_index()
        )
        _plot_trend(
            df_age_monthly, "age_group", age_order, AGE_PALETTE,
            f"Monthly Mean Glucose by Age Group — {label}",
            "Date", "Mean glucose (mg/dL)",
            fig_dir / "glucose_trend_by_age",
        )

    # 7. Mediciones por mes
    df_clean["year_month"] = df_clean["timestamp"].dt.to_period("M")
    monthly_counts = df_clean.groupby("year_month").size()
    _plot_monthly_counts(monthly_counts, dataset_name, fig_dir)

    # ── Reporte Markdown ──────────────────────────────────────────────────────
    _write_report(
        output_dir, dataset_name, label, total_patients, total_measurements,
        date_min, date_max, days_span, stats, stats_patients,
        sex_counts, age_counts, coverage, df_demo, df_sex, df_age,
    )

    print(f"✅ EDA completado para {dataset_name}. Resultados en {output_dir}")


def _write_report(
    output_dir, dataset_name, label, total_patients, total_measurements,
    date_min, date_max, days_span, stats, stats_patients,
    sex_counts, age_counts, coverage, df_demo, df_sex, df_age,
):
    """Escribe el informe Markdown."""
    lines = [
        f"# EDA Report — {label}",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## General Summary",
        f"- **Patients:** {total_patients}",
        f"- **Measurements:** {total_measurements:,}",
        f"- **Date range:** {date_min} to {date_max} ({days_span} days)",
        f"- **Glucose:** mean {stats['measurement_mean']:.1f} mg/dL, SD {stats['measurement_std']:.1f} mg/dL",
        "",
        "## Demographic Distribution",
        "",
        "### Sex",
        "| Sex | N patients | % patients | N measurements | % measurements |",
        "|-----|-----------|------------|----------------|----------------|",
    ]
    for sex, count in sex_counts.items():
        n_meas_sex = len(df_demo[df_demo["sex_group"] == sex])
        lines.append(
            f"| {sex} | {count} | {count/total_patients*100:.1f}% "
            f"| {n_meas_sex:,} | {n_meas_sex/len(df_demo)*100:.1f}% |"
        )
    lines += [
        "", "### Age (ascending order)",
        "| Age group | N patients | % patients | N measurements | % measurements |",
        "|-----------|-----------|------------|----------------|----------------|",
    ]
    for age in AGE_LABELS:
        count = age_counts.get(age, 0)
        if count > 0:
            n_meas_age = len(df_demo[df_demo["age_group"] == age])
            lines.append(
                f"| {age} | {count} | {count/total_patients*100:.1f}% "
                f"| {n_meas_age:,} | {n_meas_age/len(df_demo)*100:.1f}% |"
            )
    lines += [
        "", "## Measurements per Patient",
        "| Metric | Value |", "|--------|-------|",
    ]
    for k, v in stats_patients.items():
        lines.append(f"| {k} | {v:.1f} |" if k != "count" else f"| Total patients | {v} |")

    lines += [
        "", "## Data Quality", "| Column | % Null |", "|--------|--------|",
    ]
    for col, pct in stats["null_percentage"].items():
        lines.append(f"| {col} | {pct:.2f}% |")
    lines += ["", "### Timestamp coverage (15min / 5min)"]
    for col, pct in coverage.items():
        lines.append(f"- **{col}**: {pct:.2f}% non-null")

    lines += ["", "---", "*Auto-generated for exploratory purposes.*"]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# PUNTO DE ENTRADA
# ---------------------------------------------------------------------------

def run_eda(dataset_name: str, data_root: Path, output_root: Path):
    """Función principal que orquesta el EDA para un dataset."""
    print(f"=== EDA iniciado: {dataset_name} ===")
    df_orig      = load_original_data(dataset_name, data_root)
    patient_info = load_patient_info(dataset_name, data_root)
    demo_lookup  = build_demographic_lookup(patient_info)

    eda_dir = output_root / "EDA" / f"eda_{dataset_name}"
    eda_dir.mkdir(parents=True, exist_ok=True)

    min_val, max_val = SENSOR_LIMITS.get(dataset_name, (39.0, 401.0))
    generate_report_and_figures(df_orig, demo_lookup, dataset_name, eda_dir, min_val, max_val)