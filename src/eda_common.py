# -*- coding: utf-8 -*-
"""eda_common.py — Funciones compartidas para EDA de datasets originales."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Importar configuración central (ajustar si es necesario)
try:
    from config import (
        OUTPUT_ROOT, DATASET_LABELS, DATASET_PALETTE,
        FIGURES_DIR, TABLES_DIR, STATS_DIR, DASHBOARDS_DIR,
        PALETTE, DIMENSION_PALETTE, MPL_RC,
    )
except ImportError:
    # Definir valores por defecto si no existe config.py
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    OUTPUT_ROOT = PROJECT_ROOT / "results"
    FIGURES_DIR = OUTPUT_ROOT / "figures"
    TABLES_DIR = OUTPUT_ROOT / "tables"
    STATS_DIR = OUTPUT_ROOT / "stats"
    DASHBOARDS_DIR = OUTPUT_ROOT / "dashboards"
    DATASET_LABELS = {"DIATREND": "DiaTrend", "REPLACE-BG": "ReplaceBG", "T1DiabetesGranada": "T1DGranada"}
    DATASET_PALETTE = {"DIATREND": "#009E73", "REPLACE-BG": "#E69F00", "T1DiabetesGranada": "#CC79A7"}
    DIMENSION_PALETTE = {"age": "#0072B2", "sex": "#D55E00", "original": "#000000"}
    MPL_RC = {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
    }

# Límites de sensor por dataset
SENSOR_LIMITS = {
    "DIATREND": (39.0, 401.0),
    "REPLACE-BG": (39.0, 401.0),
    "T1DiabetesGranada": (40.0, 500.0),
}

# Rangos de edad (según especificación) - ORDEN ASCENDENTE
AGE_BINS = [-np.inf, 30, 45, 65, np.inf]
AGE_LABELS = ["<31", "31-45", "46-65", ">=66"]   # Este orden se mantendrá

# Mapeo de sexo (ampliado)
SEX_MAP = {
    "F": "F", "FEMALE": "F", "FEM": "F",
    "M": "M", "MALE": "M", "MASC": "M",
}


def setup_plot_style():
    """Aplica estilo matplotlib definido en config."""
    plt.rcParams.update(MPL_RC)
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")


def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """Encuentra la primera columna que coincida (normalizando nombres)."""
    norm_cols = {c.strip().lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower().replace(" ", "_")
        if key in norm_cols:
            return norm_cols[key]
    raise KeyError(f"None of {candidates} found in columns.")


def infer_age_series(patient_info: pd.DataFrame) -> pd.Series:
    """Infiere la edad numérica a partir de Age o Birth_year."""
    for col in ["Age", "age"]:
        try:
            age_col = find_column(patient_info, [col])
            return pd.to_numeric(patient_info[age_col], errors="coerce")
        except KeyError:
            continue

    for col in ["Birth_year", "birth_year", "Birth Year", "birth year"]:
        try:
            birth_col = find_column(patient_info, [col])
            birth_year = pd.to_numeric(patient_info[birth_col], errors="coerce")
            ref_year = datetime.now().year
            return ref_year - birth_year
        except KeyError:
            continue

    raise KeyError("No se pudo inferir la edad: falta 'Age' o 'Birth_year'.")


def build_demographic_lookup(patient_info: pd.DataFrame) -> pd.DataFrame:
    """Construye DataFrame con sex_group y age_group por patient_id.
       age_group se convierte en categórico con orden ascendente.
    """
    meta = patient_info.copy()
    pid_col = find_column(meta, ["patient_id", "Patient_ID", "patientid"])
    meta["_patient_key"] = meta[pid_col].astype(str).str.strip()

    sex_col = find_column(meta, ["Sex", "sex"])
    meta["sex_group"] = (
        meta[sex_col].astype(str).str.strip().str.upper().map(SEX_MAP).fillna("Unknown")
    )

    meta["age"] = infer_age_series(meta)
    meta["age_group"] = pd.cut(
        meta["age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        include_lowest=True,
        right=True
    ).astype("object").fillna("Unknown")

    # Convertir age_group a categórico con el orden deseado (ascendente)
    # Los valores "Unknown" se mantienen al final (no en el orden)
    known_ages = [g for g in AGE_LABELS if g in meta["age_group"].unique()]
    if known_ages:
        meta["age_group"] = pd.Categorical(
            meta["age_group"],
            categories=known_ages + ["Unknown"],
            ordered=True
        )

    return (
        meta[["_patient_key", "sex_group", "age", "age_group"]]
        .drop_duplicates(subset=["_patient_key"])
        .set_index("_patient_key")
    )


def load_original_data(dataset_name: str, data_root: Path) -> pd.DataFrame:
    """Carga el archivo original del dataset (parquet o csv) que contenga 'Glucose_measurements' y el nombre del dataset."""
    dataset_dir = data_root / dataset_name
    patterns = [
        f"Glucose_measurements_*{dataset_name}*FILTERED*.parquet",
        f"Glucose_measurements_*{dataset_name}*FILTERED*.csv",
        f"*{dataset_name}*Glucose_measurements*.parquet",
        f"*{dataset_name}*Glucose_measurements*.csv",
        f"Glucose_measurements_*.parquet",
        f"Glucose_measurements_*.csv",
    ]
    for pat in patterns:
        candidates = list(dataset_dir.glob(pat))
        if candidates:
            file_path = candidates[0]
            if file_path.suffix.lower() == ".parquet":
                return pd.read_parquet(file_path)
            else:
                return pd.read_csv(file_path)
    raise FileNotFoundError(f"No se encontró archivo original para {dataset_name} en {dataset_dir}")


def load_patient_info(dataset_name: str, data_root: Path) -> pd.DataFrame:
    """Carga patient_info.parquet o .csv desde el mismo directorio."""
    dataset_dir = data_root / dataset_name
    for ext in [".parquet", ".csv"]:
        candidates = list(dataset_dir.glob(f"*patient_info*{ext}")) + \
                     list(dataset_dir.glob(f"*Patient_info*{ext}"))
        if candidates:
            file_path = candidates[0]
            if ext == ".parquet":
                return pd.read_parquet(file_path)
            else:
                return pd.read_csv(file_path)
    raise FileNotFoundError(f"No se encontró patient_info para {dataset_name} en {dataset_dir}")


def clean_measurements(df: pd.DataFrame, min_val: float, max_val: float) -> pd.DataFrame:
    """Filtra valores de measurement fuera de rango y nulos."""
    df_clean = df.copy()
    df_clean["measurement"] = pd.to_numeric(df_clean["measurement"], errors="coerce")
    df_clean = df_clean[df_clean["measurement"].notna()]
    df_clean = df_clean[(df_clean["measurement"] >= min_val) & (df_clean["measurement"] <= max_val)]
    return df_clean


def prepare_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte measurement_date y measurement_time a datetime, y crea timestamp."""
    df = df.copy()
    if "measurement_date" in df.columns:
        df["measurement_date"] = df["measurement_date"].astype(str)
    if "measurement_time" in df.columns:
        df["measurement_time"] = df["measurement_time"].astype(str)

    df["timestamp"] = pd.to_datetime(
        df["measurement_date"] + " " + df["measurement_time"],
        errors="coerce"
    )
    for col in ["15min", "5min"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def summarize_patients(df: pd.DataFrame, demo_lookup: pd.DataFrame) -> pd.DataFrame:
    """Agrupa por patient_id y calcula estadísticas básicas."""
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


def glucose_range_category(value: float) -> str:
    """Clasifica un valor de glucosa en TBR, TIR o TAR según estándares."""
    if value < 70:
        return "TBR (<70)"
    elif value <= 180:
        return "TIR (70-180)"
    else:
        return "TAR (>180)"


def compute_range_composition(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Calcula el porcentaje de mediciones en TBR, TIR, TAR para cada grupo.
       Mantiene el orden de las categorías si son categóricas.
    """
    df["range_cat"] = df["measurement"].apply(glucose_range_category)
    pivot = df.groupby([group_col, "range_cat"]).size().unstack(fill_value=0)
    percentages = pivot.div(pivot.sum(axis=1), axis=0) * 100
    # Asegurar que el orden de las filas sea el orden de las categorías (si es categórico)
    if isinstance(df[group_col].dtype, pd.CategoricalDtype):
        # Reordenar según el orden de las categorías
        ordered_cats = [c for c in df[group_col].cat.categories if c in percentages.index]
        percentages = percentages.reindex(ordered_cats)
    return percentages


def generate_report_and_figures(
    df: pd.DataFrame,
    demo_lookup: pd.DataFrame,
    dataset_name: str,
    output_dir: Path,
    min_val: float,
    max_val: float
):
    """Genera todas las tablas, figuras y el reporte Markdown."""
    setup_plot_style()

    fig_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    # Limpiar y preparar
    df_clean = clean_measurements(df, min_val, max_val)
    df_clean = prepare_datetime_columns(df_clean)
    df_clean["_patient_key"] = df_clean["patient_id"].astype(str).str.strip()
    df_demo = df_clean.join(demo_lookup, on="_patient_key", how="left")

    patient_summary = summarize_patients(df_clean, demo_lookup)

    # ---- Estadísticas generales ----
    total_patients = len(patient_summary)
    total_measurements = len(df_clean)
    sex_counts = patient_summary["sex_group"].value_counts().to_dict()
    age_counts = patient_summary["age_group"].value_counts().to_dict()
    date_min = df_clean["timestamp"].min()
    date_max = df_clean["timestamp"].max()
    days_span = (date_max - date_min).days if pd.notna(date_min) and pd.notna(date_max) else None

    # Estadísticas de mediciones por paciente
    n_meas = patient_summary["n_measurements"]
    stats_patients = {
        "count": total_patients,
        "mean": n_meas.mean(),
        "std": n_meas.std(),
        "min": n_meas.min(),
        "25%": n_meas.quantile(0.25),
        "50%": n_meas.quantile(0.50),
        "75%": n_meas.quantile(0.75),
        "max": n_meas.max(),
    }

    # Cobertura de 15min y 5min
    coverage = {}
    for col in ["15min", "5min"]:
        if col in df_clean.columns:
            non_null = df_clean[col].notna().sum()
            coverage[col] = (non_null / len(df_clean)) * 100
        else:
            coverage[col] = 0.0

    stats = {
        "dataset": dataset_name,
        "total_patients": total_patients,
        "total_measurements": total_measurements,
        "sex_counts": sex_counts,
        "age_counts": age_counts,
        "date_range": {"start": str(date_min), "end": str(date_max), "days": days_span},
        "measurement_min": float(df_clean["measurement"].min()),
        "measurement_max": float(df_clean["measurement"].max()),
        "measurement_mean": float(df_clean["measurement"].mean()),
        "measurement_std": float(df_clean["measurement"].std()),
        "null_percentage": {
            col: (df[col].isna().sum() / len(df)) * 100
            for col in df.columns if col in ["measurement_date", "measurement_time", "measurement", "15min", "5min"]
        },
        "coverage_15min_5min": coverage,
        "measurements_per_patient": stats_patients,
    }

    with open(output_dir / "summary_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    # ---- Tablas CSV ----
    patient_summary.to_csv(table_dir / "patient_summary.csv", index=False)

    # ---- Figuras ----
    # 1. Distribución global
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_clean["measurement"], bins=50, kde=True, ax=ax)
    ax.set_title(f"Distribución de glucosa - {dataset_name}")
    ax.set_xlabel("Glucosa (mg/dL)")
    ax.axvline(min_val, color="red", linestyle="--", label=f"Límite inferior ({min_val})")
    ax.axvline(max_val, color="red", linestyle="--", label=f"Límite superior ({max_val})")
    ax.legend()
    fig.savefig(fig_dir / "glucose_distribution_global.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Número de mediciones por paciente (distribución)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(patient_summary["n_measurements"], bins=30, kde=True, ax=ax)
    ax.set_title(f"Distribución del número de mediciones por paciente - {dataset_name}")
    ax.set_xlabel("Nº de mediciones por paciente")
    ax.axvline(stats_patients["mean"], color="red", linestyle="--", label=f"Media: {stats_patients['mean']:.0f}")
    ax.axvline(stats_patients["50%"], color="green", linestyle="--", label=f"Mediana: {stats_patients['50%']:.0f}")
    ax.legend()
    fig.savefig(fig_dir / "measurements_per_patient_dist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Glucosa por sexo
    sex_palette = {"F": "#D55E00", "M": "#0072B2", "Unknown": "#999999"}
    df_sex = df_demo[df_demo["sex_group"] != "Unknown"]
    if not df_sex.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=df_sex, x="sex_group", y="measurement", hue="sex_group", palette=sex_palette, legend=False, ax=ax)
        ax.set_title(f"Distribución de glucosa por sexo - {dataset_name}")
        ax.set_xlabel("Sexo")
        ax.set_ylabel("Glucosa (mg/dL)")
        fig.savefig(fig_dir / "glucose_by_sex.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Rango glucémico por sexo
        range_sex = compute_range_composition(df_sex, "sex_group")
        range_sex.to_csv(table_dir / "glucose_range_by_sex.csv")
        fig, ax = plt.subplots(figsize=(8, 5))
        range_sex.plot(kind="bar", stacked=True, ax=ax, color=["#d73027", "#fee08b", "#1a9850"])
        ax.set_title(f"Composición por rango glucémico y sexo - {dataset_name}")
        ax.set_xlabel("Sexo")
        ax.set_ylabel("Porcentaje de mediciones")
        ax.legend(title="Rango")
        fig.savefig(fig_dir / "glucose_range_by_sex.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 4. Glucosa por edad (con orden ascendente)
    df_age = df_demo[df_demo["age_group"] != "Unknown"]
    if not df_age.empty:
        # Obtener el orden de las categorías de age_group (ya debería ser categórico)
        age_order = [c for c in AGE_LABELS if c in df_age["age_group"].unique()]
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.violinplot(data=df_age, x="age_group", y="measurement", 
                       hue="age_group", palette="Blues_d", legend=False,
                       order=age_order, ax=ax)
        ax.set_title(f"Distribución de glucosa por grupo de edad - {dataset_name}")
        ax.set_xlabel("Grupo de edad")
        ax.set_ylabel("Glucosa (mg/dL)")
        fig.savefig(fig_dir / "glucose_by_age.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Rango glucémico por edad (con orden)
        range_age = compute_range_composition(df_age, "age_group")
        range_age.to_csv(table_dir / "glucose_range_by_age.csv")
        fig, ax = plt.subplots(figsize=(10, 5))
        # Reordenar el índice si es necesario
        range_age = range_age.reindex(age_order, fill_value=0)
        range_age.plot(kind="bar", stacked=True, ax=ax, color=["#d73027", "#fee08b", "#1a9850"])
        ax.set_title(f"Composición por rango glucémico y edad - {dataset_name}")
        ax.set_xlabel("Grupo de edad")
        ax.set_ylabel("Porcentaje de mediciones")
        ax.legend(title="Rango")
        fig.savefig(fig_dir / "glucose_range_by_age.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 5. Evolución temporal de la media por sexo (resample mensual)
    if not df_sex.empty:
        df_sex_monthly = df_sex.groupby([pd.Grouper(key="timestamp", freq="M"), "sex_group"])["measurement"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(14, 5))
        sns.lineplot(data=df_sex_monthly, x="timestamp", y="measurement", hue="sex_group", ax=ax, palette=sex_palette)
        ax.set_title(f"Evolución de la glucosa media mensual por sexo - {dataset_name}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Glucosa media (mg/dL)")
        fig.savefig(fig_dir / "glucose_trend_by_sex.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 6. Evolución temporal de la media por edad (con orden en leyenda)
    if not df_age.empty:
        # Asegurar que age_group sea categórico para el orden en la leyenda
        if not isinstance(df_age["age_group"].dtype, pd.CategoricalDtype):
            df_age["age_group"] = pd.Categorical(df_age["age_group"], categories=age_order, ordered=True)
        df_age_monthly = df_age.groupby([pd.Grouper(key="timestamp", freq="M"), "age_group"])["measurement"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(14, 5))
        sns.lineplot(data=df_age_monthly, x="timestamp", y="measurement", hue="age_group", 
                     ax=ax, palette="Blues_d", hue_order=age_order)
        ax.set_title(f"Evolución de la glucosa media mensual por edad - {dataset_name}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Glucosa media (mg/dL)")
        fig.savefig(fig_dir / "glucose_trend_by_age.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 7. Mediciones por mes (global)
    df_clean["year_month"] = df_clean["timestamp"].dt.to_period("M")
    monthly_counts = df_clean.groupby("year_month").size()
    fig, ax = plt.subplots(figsize=(14, 5))
    monthly_counts.plot(kind="bar", ax=ax, color=DATASET_PALETTE.get(dataset_name, "blue"))
    ax.set_title(f"Número de mediciones por mes - {dataset_name}")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Nº mediciones")
    fig.savefig(fig_dir / "monthly_counts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- Reporte Markdown ----
    report_lines = [
        f"# Informe EDA - {DATASET_LABELS.get(dataset_name, dataset_name)}",
        "",
        f"**Fecha de generación:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Resumen general",
        f"- **Pacientes:** {total_patients}",
        f"- **Mediciones totales:** {total_measurements:,}",
        f"- **Rango de fechas:** {date_min} a {date_max} ({days_span} días)",
        f"- **Glucosa:** media {stats['measurement_mean']:.1f} mg/dL, desv. {stats['measurement_std']:.1f} mg/dL",
        f"- **Límites del sensor:** {min_val} - {max_val} mg/dL",
        "",
        "## Distribución demográfica",
        "",
        "### Sexo",
        "| Sexo | Pacientes | % Pacientes | Mediciones | % Mediciones |",
        "|------|-----------|-------------|------------|--------------|",
    ]
    for sex, count in sex_counts.items():
        n_meas_sex = len(df_demo[df_demo["sex_group"] == sex])
        pct_pat = count / total_patients * 100
        pct_meas = n_meas_sex / total_measurements * 100
        report_lines.append(f"| {sex} | {count} | {pct_pat:.1f}% | {n_meas_sex:,} | {pct_meas:.1f}% |")

    report_lines.append("")
    report_lines.append("### Edad (orden ascendente)")
    report_lines.append("| Grupo de edad | Pacientes | % Pacientes | Mediciones | % Mediciones |")
    report_lines.append("|---------------|-----------|-------------|------------|--------------|")
    # Asegurar orden ascendente según AGE_LABELS
    for age in AGE_LABELS:
        count = age_counts.get(age, 0)
        if count > 0:
            n_meas_age = len(df_demo[df_demo["age_group"] == age])
            pct_pat = count / total_patients * 100
            pct_meas = n_meas_age / total_measurements * 100
            report_lines.append(f"| {age} | {count} | {pct_pat:.1f}% | {n_meas_age:,} | {pct_meas:.1f}% |")
    # Añadir "Unknown" al final si existe
    if "Unknown" in age_counts:
        count = age_counts["Unknown"]
        n_meas_age = len(df_demo[df_demo["age_group"] == "Unknown"])
        pct_pat = count / total_patients * 100
        pct_meas = n_meas_age / total_measurements * 100
        report_lines.append(f"| Unknown | {count} | {pct_pat:.1f}% | {n_meas_age:,} | {pct_meas:.1f}% |")

    report_lines.append("")
    report_lines.append("## Estadísticas por paciente")
    report_lines.append("| Métrica | Valor |")
    report_lines.append("|---------|-------|")
    for k, v in stats_patients.items():
        if k == "count":
            report_lines.append(f"| Total pacientes | {v} |")
        else:
            report_lines.append(f"| {k} | {v:.1f} |")

    report_lines.append("")
    report_lines.append("## Figuras generadas")
    report_lines.append("")
    report_lines.append("### Distribución global de glucosa")
    report_lines.append(f"![Distribución global](figures/glucose_distribution_global.png)")
    report_lines.append("")
    report_lines.append("### Distribución de mediciones por paciente")
    report_lines.append(f"![Mediciones por paciente](figures/measurements_per_patient_dist.png)")
    report_lines.append("")
    report_lines.append("### Glucosa por sexo")
    report_lines.append(f"![Glucosa por sexo](figures/glucose_by_sex.png)")
    report_lines.append("")
    report_lines.append("### Composición por rango glucémico y sexo")
    report_lines.append(f"![Rango glucémico por sexo](figures/glucose_range_by_sex.png)")
    report_lines.append("")
    report_lines.append("### Glucosa por edad")
    report_lines.append(f"![Glucosa por edad](figures/glucose_by_age.png)")
    report_lines.append("")
    report_lines.append("### Composición por rango glucémico y edad")
    report_lines.append(f"![Rango glucémico por edad](figures/glucose_range_by_age.png)")
    report_lines.append("")
    report_lines.append("### Evolución temporal de la glucosa media por sexo")
    report_lines.append(f"![Tendencia por sexo](figures/glucose_trend_by_sex.png)")
    report_lines.append("")
    report_lines.append("### Evolución temporal de la glucosa media por edad")
    report_lines.append(f"![Tendencia por edad](figures/glucose_trend_by_age.png)")
    report_lines.append("")
    report_lines.append("### Mediciones por mes")
    report_lines.append(f"![Mediciones mensuales](figures/monthly_counts.png)")
    report_lines.append("")
    report_lines.append("## Tablas")
    report_lines.append(f"- [Resumen por paciente](tables/patient_summary.csv)")
    if not df_sex.empty:
        report_lines.append(f"- [Rango glucémico por sexo](tables/glucose_range_by_sex.csv)")
    if not df_age.empty:
        report_lines.append(f"- [Rango glucémico por edad](tables/glucose_range_by_age.csv)")
    report_lines.append("")
    report_lines.append("## Calidad de datos")
    report_lines.append("| Columna | % Nulos |")
    report_lines.append("|---------|---------|")
    for col, pct in stats["null_percentage"].items():
        report_lines.append(f"| {col} | {pct:.2f}% |")
    report_lines.append("")
    report_lines.append("### Cobertura de timestamps agregados (15min / 5min)")
    for col, pct in coverage.items():
        report_lines.append(f"- **{col}**: {pct:.2f}% de filas con valor no nulo")
    report_lines.append("")
    report_lines.append("## Observaciones")
    # Comentarios automáticos
    # Desbalanceo de sexo
    if len(sex_counts) == 2:
        ratio = max(sex_counts.values()) / min(sex_counts.values())
        if ratio > 2:
            report_lines.append(f"- **Desbalanceo de sexo severo:** ratio {ratio:.1f}:1 entre grupos (el grupo mayoritario tiene {max(sex_counts.values())} pacientes, el minoritario {min(sex_counts.values())}).")
        else:
            report_lines.append("- El sexo está razonablemente balanceado.")
    else:
        report_lines.append("- Solo hay un grupo de sexo (o muchos desconocidos) → no se puede analizar diferencia por sexo.")

    # Desbalanceo de edad (considerando solo grupos conocidos)
    known_age_counts = {k: v for k, v in age_counts.items() if k != "Unknown"}
    if len(known_age_counts) > 1:
        max_age = max(known_age_counts.values())
        min_age = min(known_age_counts.values())
        if max_age / min_age > 3:
            report_lines.append(f"- **Desbalanceo de edad severo:** el grupo más grande tiene {max_age} pacientes, el más pequeño {min_age} (ratio {max_age/min_age:.1f}:1).")
        else:
            report_lines.append("- Los grupos de edad están razonablemente balanceados.")
    else:
        report_lines.append("- Solo hay un grupo de edad (o muchos desconocidos) → no se puede analizar diferencias por edad.")

    # Pacientes con muy pocas mediciones
    low_meas = patient_summary[patient_summary["n_measurements"] < 10]
    if len(low_meas) > 0:
        report_lines.append(f"- **{len(low_meas)} pacientes** tienen menos de 10 mediciones. Esto puede afectar la fiabilidad de los modelos para esos pacientes.")

    # Cobertura de timestamps
    if coverage.get("15min", 0) < 50:
        report_lines.append("- **Baja cobertura de `15min`**: puede dificultar el etiquetado en ventanas de 15 minutos.")
    if coverage.get("5min", 0) < 50:
        report_lines.append("- **Baja cobertura de `5min`**: puede dificultar el etiquetado en ventanas de 5 minutos.")

    # Rango glucémico
    global_range = compute_range_composition(df_clean, "range_cat").sum(axis=0)
    tbr_pct = global_range.get("TBR (<70)", 0)
    tir_pct = global_range.get("TIR (70-180)", 0)
    tar_pct = global_range.get("TAR (>180)", 0)
    if tbr_pct > 5:
        report_lines.append(f"- **Alta prevalencia de hipoglucemia ({tbr_pct:.1f}% de las mediciones)** → importante para modelos de predicción de hipos.")
    if tar_pct > 20:
        report_lines.append(f"- **Alta prevalencia de hiperglucemia ({tar_pct:.1f}% de las mediciones)** → puede indicar control glucémico deficiente.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("*Este informe se ha generado automáticamente con fines exploratorios.*")

    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"✅ EDA completado para {dataset_name}. Resultados en {output_dir}")


def run_eda(dataset_name: str, data_root: Path, output_root: Path):
    """Función principal que orquesta el EDA para un dataset."""
    print(f"=== Iniciando EDA para {dataset_name} ===")
    df_orig = load_original_data(dataset_name, data_root)
    patient_info = load_patient_info(dataset_name, data_root)
    demo_lookup = build_demographic_lookup(patient_info)

    eda_dir = output_root / "EDA" / f"eda_{dataset_name}"
    eda_dir.mkdir(parents=True, exist_ok=True)

    min_val, max_val = SENSOR_LIMITS.get(dataset_name, (39.0, 401.0))
    generate_report_and_figures(df_orig, demo_lookup, dataset_name, eda_dir, min_val, max_val)