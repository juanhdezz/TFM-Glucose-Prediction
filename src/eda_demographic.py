# -*- coding: utf-8 -*-
"""
eda_demographic.py — Análisis Exploratorio de Datos (EDA) demográfico
Proyecto RELIEF-T1D — TFM DATCOM UGR

Objetivo: caracterización de los tres datasets (T1DiabetesGranada, DIATREND,
REPLACE-BG) con foco en las dimensiones demográficas de interés (edad y sexo)
que articulan el diseño experimental del pipeline de balanceo.

Salidas: figuras (.pdf + .png), tablas LaTeX y estadísticos CSV en ROOT/EDA/
  ├── cross/          — comparativas entre los tres datasets
  ├── T1DiabetesGranada/
  ├── DIATREND/
  └── REPLACE-BG/
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — importar desde config.py si está disponible
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(root: Path):
    """Intenta cargar config.py del proyecto; si no, usa valores inline."""
    cfg_path = root / "config.py"
    if cfg_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", cfg_path)
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        return cfg
    return None


# Paleta Wong (2011) — colorblind-safe, consistente con config.py
PALETTE_TECHNIQUE = {
    "original":                    "#000000",
    "jittering":                   "#009E73",
    "oversampling":                "#E69F00",
    "patient_aware_undersampling": "#0072B2",
    "smote":                       "#D55E00",
    "undersampling":               "#56B4E9",
    "tomek_links":                 "#F0E442",
    "undersampling_oversampling":  "#BC8F8F",
    "undersampling_smote":         "#984EA3",
    "smote_tomek":                 "#FF7F00",
    "oversampling_tomek":          "#999999",
}

DATASET_PALETTE = {
    "DIATREND":          "#009E73",
    "REPLACE-BG":        "#E69F00",
    "T1DiabetesGranada": "#CC79A7",
}

DATASET_LABELS = {
    "DIATREND":          "DiaTrend",
    "REPLACE-BG":        "ReplaceBG",
    "T1DiabetesGranada": "T1DGranada",
}

DIMENSION_PALETTE = {
    "age":      "#0072B2",
    "sex":      "#D55E00",
    "original": "#000000",
}

# Paleta específica para grupos de edad
AGE_PALETTE = {
    "<31":   "#56B4E9",
    "31-45": "#009E73",
    "46-65": "#E69F00",
    ">=66":  "#D55E00",
}

SEX_PALETTE = {
    "F": "#CC79A7",
    "M": "#0072B2",
}

GLUCOSE_RANGE_PALETTE = {
    "TBR_2": "#7EA6FF",
    "TBR_1": "#A8C4FF",
    "TIR":   "#7DCB8A",
    "TAR_1": "#F4A261",
    "TAR_2": "#D95F02",
}

GLUCOSE_RANGE_LABELS = {
    "TBR_2": "Hypo L2 (<54)",
    "TBR_1": "Hypo L1 (54–69)",
    "TIR":   "In Range (70–180)",
    "TAR_1": "Hyper L1 (181–250)",
    "TAR_2": "Hyper L2 (>250)",
}

# Grupos demográficos fijos del TFM
AGE_BINS   = [-np.inf, 30, 45, 65, np.inf]
AGE_LABELS = ["<31", "31-45", "46-65", ">=66"]

SEX_MAP = {"F": "F", "M": "M"}

MPL_RC = {
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":    9,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "0.8",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches":0.15,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.axisbelow":    True,
}

# ─────────────────────────────────────────────────────────────────────────────
# DETECCIÓN DEL PROYECTO
# ─────────────────────────────────────────────────────────────────────────────

def find_project_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        for ds in ["T1DiabetesGranada", "DIATREND", "REPLACE-BG"]:
            if (candidate / "data" / ds).exists():
                return candidate
    raise FileNotFoundError(
        "No se encontró ningún dataset bajo data/<DATASET> desde el directorio actual."
    )


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def _norm_patient_id(df: pd.DataFrame) -> pd.DataFrame:
    for alias in ["Patient_Id", "patient_id", "PatientID", "PATIENT_ID"]:
        if alias in df.columns and "Patient_ID" not in df.columns:
            df = df.rename(columns={alias: "Patient_ID"})
    df["Patient_ID"] = df["Patient_ID"].astype(str)
    return df


def _parse_glucose_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Detecta y parsea columnas temporales en mediciones de glucosa."""
    if {"Measurement_date", "Measurement_time"}.issubset(df.columns):
        df["timestamp"] = pd.to_datetime(
            df["Measurement_date"].astype(str) + " " + df["Measurement_time"].astype(str),
            errors="coerce",
        )
    elif "Measurement_date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["Measurement_date"], errors="coerce")
    elif "15min" in df.columns:
        df["timestamp"] = pd.to_datetime(df["15min"], errors="coerce")
    elif "5min" in df.columns:
        df["timestamp"] = pd.to_datetime(df["5min"], errors="coerce")
    elif "datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        raise ValueError(f"No se encontraron columnas de fecha/hora. Columnas: {list(df.columns)}")
    return df


def _norm_measurement(df: pd.DataFrame) -> pd.DataFrame:
    for alias in ["measurement", "Glucose", "glucose", "value", "Value"]:
        if alias in df.columns and "Measurement" not in df.columns:
            df = df.rename(columns={alias: "Measurement"})
    df["Measurement"] = pd.to_numeric(df["Measurement"], errors="coerce").astype(float)
    return df


def _assign_glucose_ranges(series: pd.Series) -> pd.Series:
    out = pd.Series("unknown", index=series.index)
    out[series < 54]                       = "TBR_2"
    out[(series >= 54) & (series < 70)]    = "TBR_1"
    out[(series >= 70) & (series <= 180)]  = "TIR"
    out[(series > 180) & (series <= 250)]  = "TAR_1"
    out[series > 250]                      = "TAR_2"
    return out


_PARQUET_MAGIC = b"PAR1"          # primeros 4 bytes de todo fichero Parquet


def _is_parquet(path: Path) -> bool:
    """Detecta formato Parquet por magic bytes, ignorando la extensión."""
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == _PARQUET_MAGIC
    except OSError:
        return False


def _read_tabular(path: Path) -> pd.DataFrame:
    """
    Lee un fichero tabular eligiendo el lector correcto por contenido real,
    no por extensión. Prueba encodings habituales para CSV.
    """
    if _is_parquet(path):
        return pd.read_parquet(path)

    # Es un fichero de texto — probar encodings en orden de probabilidad
    for enc in ("utf-8", "latin-1", "cp1252", "utf-16"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError(f"No se pudo leer {path} con ningún encoding conocido.")


def load_dataset(data_dir: Path, dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga patient_info y glucose para un dataset dado.
    Retorna (patient_info_df, glucose_df) normalizados.
    """
    ds_dir = data_dir / dataset_name

    # ── patient_info ──────────────────────────────────────────────────────────
    pi_candidates = (
        list(ds_dir.glob("Patient_info*")) +
        list(ds_dir.glob("patient_info*")) +
        list(ds_dir.glob("PatientInfo*")) +
        list(ds_dir.glob("patients*"))
    )
    if not pi_candidates:
        raise FileNotFoundError(f"No se encontró patient_info en {ds_dir}")
    patient_info = _read_tabular(pi_candidates[0])
    patient_info = _norm_patient_id(patient_info)

    # ── glucose ───────────────────────────────────────────────────────────────
    # Prioridad: parquet > csv; nombres explícitos primero, wildcard al final
    gl_candidates = (
        list(ds_dir.glob("Glucose_measurements*.parquet")) +
        list(ds_dir.glob("glucose*.parquet")) +
        list(ds_dir.glob("*.parquet")) +
        list(ds_dir.glob("Glucose_measurements*.csv")) +
        list(ds_dir.glob("glucose*.csv")) +
        list(ds_dir.glob("*measurements*.csv"))
    )
    gl_candidates = [p for p in gl_candidates if "patient" not in p.name.lower()]
    if not gl_candidates:
        raise FileNotFoundError(f"No se encontró fichero de glucosa en {ds_dir}")

    # Elegir el primer candidato que realmente sea Parquet (si existe),
    # o el primero de la lista si no hay ninguno Parquet.
    parquet_first = next((p for p in gl_candidates if _is_parquet(p)), None)
    gl_path = parquet_first if parquet_first else gl_candidates[0]
    print(f"  [{dataset_name}] glucose ← {gl_path.name} (parquet={_is_parquet(gl_path)})")

    glucose = _read_tabular(gl_path)
    glucose = _norm_patient_id(glucose)
    glucose = _parse_glucose_timestamp(glucose)
    glucose = _norm_measurement(glucose)
    glucose = glucose.sort_values(["Patient_ID", "timestamp"]).reset_index(drop=True)

    print(f"  [{dataset_name}] patient_info={patient_info.shape} | glucose={glucose.shape}")
    return patient_info, glucose


# ─────────────────────────────────────────────────────────────────────────────
# ENRIQUECIMIENTO DEMOGRÁFICO
# ─────────────────────────────────────────────────────────────────────────────

def enrich_demographics(patient_info: pd.DataFrame, glucose: pd.DataFrame,
                        ref_year: int = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Añade age_group y sex_label a patient_info y hace merge en glucose.
    """
    pi = patient_info.copy()

    # Año de referencia
    if ref_year is None:
        if "timestamp" in glucose.columns:
            ref_year = int(glucose["timestamp"].dt.year.max())
        else:
            ref_year = 2024

    # Edad
    birth_col = next((c for c in ["Birth_year", "birth_year", "BirthYear", "Year_of_birth"]
                      if c in pi.columns), None)
    age_col   = next((c for c in ["Age", "age", "AGE"] if c in pi.columns), None)

    if birth_col:
        pi["_age_num"] = ref_year - pd.to_numeric(pi[birth_col], errors="coerce")
    elif age_col:
        pi["_age_num"] = pd.to_numeric(pi[age_col], errors="coerce")
    else:
        pi["_age_num"] = np.nan

    pi["age_group"] = pd.cut(
        pi["_age_num"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True
    ).astype(str)
    pi.loc[pi["_age_num"].isna(), "age_group"] = "Unknown"

    # Sexo
    sex_col = next((c for c in ["Sex", "sex", "Gender", "gender", "SEX"]
                    if c in pi.columns), None)
    if sex_col:
        pi["sex_label"] = (
            pi[sex_col].astype(str).str.strip().str.upper()
            .map({"F": "F", "FEMALE": "F", "WOMAN": "F", "0": "F",
                  "M": "M", "MALE":   "M", "MAN":   "M", "1": "M"})
            .fillna("Unknown")
        )
    else:
        pi["sex_label"] = "Unknown"

    # Merge en glucose
    demo_cols = ["Patient_ID", "age_group", "sex_label", "_age_num"]
    gl = glucose.merge(pi[demo_cols], on="Patient_ID", how="left")
    gl["glucose_range"] = _assign_glucose_ranges(gl["Measurement"])

    return pi, gl


# ─────────────────────────────────────────────────────────────────────────────
# ESTRUCTURA DE DIRECTORIOS DE SALIDA
# ─────────────────────────────────────────────────────────────────────────────

from typing import NamedTuple


class OutDirs(NamedTuple):
    """
    Agrupa los directorios de salida:
      cross   — figuras y tablas comparativas entre los tres datasets
      per_ds  — dict {ds_name: Path} con una subcarpeta por dataset
    """
    cross:  Path
    per_ds: dict   # {ds_name: Path}


def build_out_dirs(eda_root: Path, ds_names: list) -> OutDirs:
    cross = eda_root / "cross"
    cross.mkdir(parents=True, exist_ok=True)
    per_ds = {}
    for name in ds_names:
        d = eda_root / name
        d.mkdir(parents=True, exist_ok=True)
        per_ds[name] = d
    return OutDirs(cross=cross, per_ds=per_ds)


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE ESCRITURA
# ─────────────────────────────────────────────────────────────────────────────

def _savefig(fig, out_dir: Path, stem: str):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)


def _save_latex(df: pd.DataFrame, out_dir: Path, stem: str, caption: str = "",
                label: str = "", float_fmt: str = "%.2f"):
    tex = df.to_latex(
        index=True,
        float_format=float_fmt,
        caption=caption if caption else stem.replace("_", " ").title(),
        label=f"tab:{label if label else stem}",
        na_rep="—",
        escape=True,
    )
    (out_dir / f"{stem}.tex").write_text(tex, encoding="utf-8")


def _save_csv(df: pd.DataFrame, out_dir: Path, stem: str):
    df.to_csv(out_dir / f"{stem}.csv", index=True)


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 0 — DESCRIPCIÓN GENERAL DE COHORTES
# ─────────────────────────────────────────────────────────────────────────────

def block0_cohort_overview(datasets: dict, odirs: OutDirs):
    """
    Tabla y figura resumen de las tres cohortes: n pacientes, n mediciones,
    rango temporal, distribución edad/sexo.
    Salida: cross/  (comparativa entre datasets)
    """
    out_dir = odirs.cross
    rows = []
    for ds_name, (pi, gl) in datasets.items():
        n_pat      = pi["Patient_ID"].nunique()
        n_meas     = len(gl.dropna(subset=["Measurement"]))
        ts_min     = gl["timestamp"].min()
        ts_max     = gl["timestamp"].max()
        age_known  = pi[pi["_age_num"].notna()]
        age_mean   = age_known["_age_num"].mean()
        age_std    = age_known["_age_num"].std()
        sex_f      = (pi["sex_label"] == "F").sum()
        sex_m      = (pi["sex_label"] == "M").sum()
        median_int = (
            gl.sort_values(["Patient_ID", "timestamp"])
            .groupby("Patient_ID")["timestamp"]
            .apply(lambda s: s.diff().median())
            .median()
        )
        rows.append({
            "Dataset":           DATASET_LABELS.get(ds_name, ds_name),
            "N patients":        n_pat,
            "N measurements":    n_meas,
            "Start date":        ts_min.date() if pd.notna(ts_min) else "—",
            "End date":          ts_max.date() if pd.notna(ts_max) else "—",
            "Age mean (SD)":     f"{age_mean:.1f} ({age_std:.1f})" if not np.isnan(age_mean) else "—",
            "Female N (%)":      f"{sex_f} ({100*sex_f/n_pat:.0f}%)" if n_pat else "—",
            "Male N (%)":        f"{sex_m} ({100*sex_m/n_pat:.0f}%)" if n_pat else "—",
            "Median interval":   str(median_int).split(".")[0] if pd.notna(median_int) else "—",
        })

    overview = pd.DataFrame(rows).set_index("Dataset")
    _save_csv(overview, out_dir, "B0_cohort_overview")
    _save_latex(overview, out_dir, "B0_cohort_overview",
                caption="Descriptive overview of the three T1D cohorts.",
                label="cohort_overview")

    # ── Figura: barras n pacientes + n mediciones (doble eje) ────────────────
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    ds_keys   = list(datasets.keys())
    ds_lab    = [DATASET_LABELS.get(k, k) for k in ds_keys]
    colors_ds = [DATASET_PALETTE[k] for k in ds_keys]
    x         = np.arange(len(ds_keys))
    width     = 0.35

    n_pats  = [datasets[k][0]["Patient_ID"].nunique() for k in ds_keys]
    n_meass = [len(datasets[k][1].dropna(subset=["Measurement"])) for k in ds_keys]

    bars1 = ax1.bar(x - width/2, n_pats,  width, color=colors_ds, alpha=0.85, label="N patients")
    bars2 = ax2.bar(x + width/2, n_meass, width, color=colors_ds, alpha=0.45, edgecolor=colors_ds,
                    linewidth=1.2, label="N measurements")

    ax1.set_ylabel("N patients")
    ax2.set_ylabel("N measurements")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ds_lab)
    ax1.set_title("Cohort Size Comparison")

    handles = [mpatches.Patch(color=colors_ds[i], label=ds_lab[i]) for i in range(len(ds_keys))]
    leg1 = ax1.legend(handles=handles, title="Dataset", loc="upper left")
    from matplotlib.lines import Line2D
    leg2 = ax2.legend(
        handles=[mpatches.Patch(color="gray", alpha=0.85, label="N patients"),
                 mpatches.Patch(color="gray", alpha=0.45, label="N measurements")],
        loc="upper right",
    )
    ax1.add_artist(leg1)

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                 f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    _savefig(fig, out_dir, "B0_cohort_size")
    print("  [B0] Cohort overview saved.")


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 1 — DISTRIBUCIÓN DEMOGRÁFICA (EDAD Y SEXO) POR DATASET
# ─────────────────────────────────────────────────────────────────────────────

def block1_demographic_distribution(datasets: dict, odirs: OutDirs):
    """
    Distribución de grupos de edad y sexo por dataset.
    Salida: cross/ (comparativas) + per-dataset/ (individuales)
    """
    # ── 1a. Figura individual por dataset: grupos de edad ─────────────────────
    age_counts = []
    for ds_name, (pi, _) in datasets.items():
        ct = (pi[pi["age_group"] != "Unknown"]
              .groupby("age_group", observed=True)["Patient_ID"]
              .nunique()
              .reindex(AGE_LABELS, fill_value=0))
        for grp, n in ct.items():
            age_counts.append({"Dataset": DATASET_LABELS[ds_name], "age_group": grp,
                                "n": n, "pct": 100*n/ct.sum() if ct.sum() else 0})

        # Figura individual para este dataset
        fig_ds, ax_ds = plt.subplots(figsize=(6, 4))
        sub_ds = pd.DataFrame([r for r in age_counts if r["Dataset"] == DATASET_LABELS[ds_name]])
        bars = ax_ds.bar(
            sub_ds["age_group"], sub_ds["n"],
            color=[AGE_PALETTE.get(g, "#999999") for g in sub_ds["age_group"]],
            edgecolor="white", linewidth=0.6
        )
        for bar, row in zip(bars, sub_ds.itertuples()):
            ax_ds.text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + 0.3,
                       f"n={int(row.n)}\n({row.pct:.0f}%)",
                       ha="center", va="bottom", fontsize=9)
        ax_ds.set_title(f"Age Group Distribution — {DATASET_LABELS[ds_name]}")
        ax_ds.set_xlabel("Age group")
        ax_ds.set_ylabel("N patients")
        plt.tight_layout()
        _savefig(fig_ds, odirs.per_ds[ds_name], "B1a_age_group_distribution")

    age_df = pd.DataFrame(age_counts)

    # ── 1a. Figura comparativa cross-dataset ──────────────────────────────────
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4.5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, (ds_name, (pi, _)) in zip(axes, datasets.items()):
        sub = age_df[age_df["Dataset"] == DATASET_LABELS[ds_name]]
        bars = ax.bar(
            sub["age_group"], sub["n"],
            color=[AGE_PALETTE.get(g, "#999999") for g in sub["age_group"]],
            edgecolor="white", linewidth=0.6
        )
        for bar, row in zip(bars, sub.itertuples()):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f"n={int(row.n)}\n({row.pct:.0f}%)",
                    ha="center", va="bottom", fontsize=8)
        ax.set_title(DATASET_LABELS[ds_name])
        ax.set_xlabel("Age group")
        ax.set_ylabel("N patients" if ax == axes[0] else "")
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle("Age Group Distribution by Dataset", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, odirs.cross, "B1a_age_group_distribution")

    # ── 1b. Distribución de sexo — figura individual por dataset ─────────────
    sex_rows = []
    for ds_name, (pi, _) in datasets.items():
        ct = pi[pi["sex_label"].isin(["F", "M"])].groupby("sex_label")["Patient_ID"].nunique()
        total = ct.sum()
        for sx, n in ct.items():
            sex_rows.append({"Dataset": DATASET_LABELS[ds_name], "sex": sx,
                             "n": n, "pct": 100*n/total if total else 0})

        # Individual
        fig_ds, ax_ds = plt.subplots(figsize=(5, 4))
        vals_f = ct.get("F", 0); vals_m = ct.get("M", 0)
        ax_ds.bar(["Female", "Male"], [vals_f, vals_m],
                  color=[SEX_PALETTE["F"], SEX_PALETTE["M"]], edgecolor="white", alpha=0.85)
        for xi, (v, tot_v) in enumerate([(vals_f, total), (vals_m, total)]):
            if tot_v:
                ax_ds.text(xi, v + 0.3, f"n={v}\n({100*v/tot_v:.0f}%)",
                           ha="center", va="bottom", fontsize=9)
        ax_ds.set_ylabel("N patients")
        ax_ds.set_title(f"Sex Distribution — {DATASET_LABELS[ds_name]}")
        plt.tight_layout()
        _savefig(fig_ds, odirs.per_ds[ds_name], "B1b_sex_distribution")

    sex_df = pd.DataFrame(sex_rows)

    # ── 1b. Comparativa cross-dataset ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ds_labs    = [DATASET_LABELS[k] for k in datasets]
    x          = np.arange(len(ds_labs))
    width      = 0.32
    for i, sx in enumerate(["F", "M"]):
        vals = [sex_df[(sex_df["Dataset"] == dl) & (sex_df["sex"] == sx)]["pct"].values
                for dl in ds_labs]
        vals = [v[0] if len(v) else 0 for v in vals]
        bars = ax.bar(x + (i - 0.5)*width, vals, width,
                      color=SEX_PALETTE[sx], label=f"{'Female' if sx=='F' else 'Male'} ({sx})",
                      alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_labs)
    ax.set_ylabel("% of patients")
    ax.set_ylim(0, 105)
    ax.set_title("Sex Distribution by Dataset")
    ax.legend()
    plt.tight_layout()
    _savefig(fig, odirs.cross, "B1b_sex_distribution")

    # ── 1c. Heatmap edad × sexo — individual por dataset + cross ─────────────
    for ds_name, (pi, _) in datasets.items():
        known = pi[(pi["age_group"].isin(AGE_LABELS)) & (pi["sex_label"].isin(["F", "M"]))]
        ct    = known.pivot_table(index="sex_label", columns="age_group",
                                  values="Patient_ID", aggfunc="nunique",
                                  fill_value=0).reindex(columns=AGE_LABELS)
        fig_ds, ax_ds = plt.subplots(figsize=(6, 3))
        im = ax_ds.imshow(ct.values, cmap="YlOrRd", aspect="auto")
        ax_ds.set_xticks(range(len(AGE_LABELS)))
        ax_ds.set_xticklabels(AGE_LABELS)
        ax_ds.set_yticks(range(len(ct.index)))
        ax_ds.set_yticklabels(ct.index)
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                v = ct.values[i, j]
                ax_ds.text(j, i, str(v), ha="center", va="center", fontsize=11,
                           color="white" if v > ct.values.max()*0.6 else "black")
        ax_ds.set_title(f"Patients by Age Group & Sex — {DATASET_LABELS[ds_name]}")
        ax_ds.set_xlabel("Age group"); ax_ds.set_ylabel("Sex")
        plt.colorbar(im, ax=ax_ds, shrink=0.8, label="N patients")
        plt.tight_layout()
        _savefig(fig_ds, odirs.per_ds[ds_name], "B1c_age_sex_heatmap")

    # Cross-dataset version
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]
    for ax, (ds_name, (pi, _)) in zip(axes, datasets.items()):
        known = pi[(pi["age_group"].isin(AGE_LABELS)) & (pi["sex_label"].isin(["F", "M"]))]
        ct    = known.pivot_table(index="sex_label", columns="age_group",
                                  values="Patient_ID", aggfunc="nunique",
                                  fill_value=0).reindex(columns=AGE_LABELS)
        im = ax.imshow(ct.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(AGE_LABELS))); ax.set_xticklabels(AGE_LABELS)
        ax.set_yticks(range(len(ct.index))); ax.set_yticklabels(ct.index)
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                v = ct.values[i, j]
                ax.text(j, i, str(v), ha="center", va="center", fontsize=10,
                        color="white" if v > ct.values.max()*0.6 else "black")
        ax.set_title(DATASET_LABELS[ds_name])
        ax.set_xlabel("Age group")
        ax.set_ylabel("Sex" if ax == axes[0] else "")
        plt.colorbar(im, ax=ax, shrink=0.7, label="N patients")
    fig.suptitle("Patient Count by Age Group and Sex", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, odirs.cross, "B1c_age_sex_heatmap")

    # ── Tabla LaTeX resumen demográfico (cross) ───────────────────────────────
    rows_tex = []
    for ds_name, (pi, _) in datasets.items():
        for ag in AGE_LABELS:
            sub   = pi[pi["age_group"] == ag]
            n_f   = (sub["sex_label"] == "F").sum()
            n_m   = (sub["sex_label"] == "M").sum()
            n_tot = len(sub)
            rows_tex.append({
                "Dataset":     DATASET_LABELS[ds_name],
                "Age group":   ag,
                "Total":       n_tot,
                "Female N (%)": f"{n_f} ({100*n_f/n_tot:.0f}%)" if n_tot else "0",
                "Male N (%)":   f"{n_m} ({100*n_m/n_tot:.0f}%)" if n_tot else "0",
            })
    tex_df = pd.DataFrame(rows_tex).set_index(["Dataset", "Age group"])
    _save_csv(tex_df, odirs.cross, "B1_demographic_summary")
    _save_latex(tex_df, odirs.cross, "B1_demographic_summary",
                caption="Patient distribution by age group and sex per dataset.",
                label="demographic_summary")
    print("  [B1] Demographic distribution saved.")


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 2 — DATOS DE GLUCOSA: COBERTURA Y CARGA DE MEDICIONES POR GRUPO
# ─────────────────────────────────────────────────────────────────────────────

def block2_data_coverage(datasets: dict, odirs: OutDirs):
    out_dir = odirs.cross  # comparativas cross-dataset
    """
    Cobertura temporal de datos y número de mediciones por grupo demográfico.
    """
    # ── 2a. Mediciones por paciente según grupo de edad ───────────────────────
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4.5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, (ds_name, (pi, gl)) in zip(axes, datasets.items()):
        meas_per_pat = gl.groupby("Patient_ID")["Measurement"].count().reset_index()
        meas_per_pat.columns = ["Patient_ID", "n_measurements"]
        meas_per_pat = meas_per_pat.merge(pi[["Patient_ID", "age_group"]], on="Patient_ID", how="left")
        meas_per_pat = meas_per_pat[meas_per_pat["age_group"].isin(AGE_LABELS)]

        parts = [meas_per_pat[meas_per_pat["age_group"] == ag]["n_measurements"].dropna()
                 for ag in AGE_LABELS]
        parts_nonempty = [(ag, p) for ag, p in zip(AGE_LABELS, parts) if len(p) > 0]

        if not parts_nonempty:
            ax.set_title(DATASET_LABELS[ds_name] + "\n(no age data)")
            continue

        bp = ax.boxplot(
            [p.values for _, p in parts_nonempty],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1),
            flierprops=dict(marker=".", markersize=3, alpha=0.4),
        )
        for patch, (ag, _) in zip(bp["boxes"], parts_nonempty):
            patch.set_facecolor(AGE_PALETTE.get(ag, "#999999"))
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(parts_nonempty)+1))
        ax.set_xticklabels([ag for ag, _ in parts_nonempty], rotation=0)
        ax.set_xlabel("Age group")
        ax.set_ylabel("N measurements per patient" if ax == axes[0] else "")
        ax.set_title(DATASET_LABELS[ds_name])

    fig.suptitle("Measurements per Patient by Age Group", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, out_dir, "B2a_measurements_per_patient_age")

    # ── 2b. Duración de seguimiento por grupo de edad ─────────────────────────
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4.5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    coverage_rows = []
    for ax, (ds_name, (pi, gl)) in zip(axes, datasets.items()):
        span = (gl.groupby("Patient_ID")["timestamp"]
                .agg(start="min", end="max")
                .assign(duration_days=lambda d: (d["end"]-d["start"]).dt.total_seconds()/86400)
                .reset_index())
        span = span.merge(pi[["Patient_ID", "age_group"]], on="Patient_ID", how="left")
        span = span[span["age_group"].isin(AGE_LABELS)]

        parts = [(ag, span[span["age_group"]==ag]["duration_days"].dropna())
                 for ag in AGE_LABELS]
        parts = [(ag, p) for ag, p in parts if len(p) > 0]

        if not parts:
            ax.set_title(DATASET_LABELS[ds_name] + "\n(no age data)")
            continue

        bp = ax.boxplot(
            [p.values for _, p in parts],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1),
            flierprops=dict(marker=".", markersize=3, alpha=0.4),
        )
        for patch, (ag, _) in zip(bp["boxes"], parts):
            patch.set_facecolor(AGE_PALETTE.get(ag, "#999999"))
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(parts)+1))
        ax.set_xticklabels([ag for ag, _ in parts])
        ax.set_xlabel("Age group")
        ax.set_ylabel("Follow-up duration (days)" if ax == axes[0] else "")
        ax.set_title(DATASET_LABELS[ds_name])

        for ag, p in parts:
            coverage_rows.append({
                "Dataset":          DATASET_LABELS[ds_name],
                "age_group":        ag,
                "median_days":      p.median(),
                "q25_days":         p.quantile(0.25),
                "q75_days":         p.quantile(0.75),
                "n_patients":       len(p),
            })

    fig.suptitle("Follow-up Duration by Age Group", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, out_dir, "B2b_followup_duration_age")

    # ── 2c. Mediciones por paciente según sexo ───────────────────────────────
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4.5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, (ds_name, (pi, gl)) in zip(axes, datasets.items()):
        meas_per_pat = gl.groupby("Patient_ID")["Measurement"].count().reset_index()
        meas_per_pat.columns = ["Patient_ID", "n_measurements"]
        meas_per_pat = meas_per_pat.merge(pi[["Patient_ID", "sex_label"]], on="Patient_ID", how="left")
        meas_per_pat = meas_per_pat[meas_per_pat["sex_label"].isin(["F", "M"])]

        if meas_per_pat.empty:
            ax.set_title(DATASET_LABELS[ds_name] + "\n(no sex data)")
            continue

        parts = [(sx, meas_per_pat[meas_per_pat["sex_label"]==sx]["n_measurements"])
                 for sx in ["F", "M"]]
        parts = [(sx, p) for sx, p in parts if len(p) > 0]

        bp = ax.boxplot(
            [p.values for _, p in parts],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1),
            flierprops=dict(marker=".", markersize=3, alpha=0.4),
        )
        for patch, (sx, _) in zip(bp["boxes"], parts):
            patch.set_facecolor(SEX_PALETTE[sx])
            patch.set_alpha(0.7)

        for (sx, p), pos in zip(parts, range(1, len(parts)+1)):
            ax.text(pos, ax.get_ylim()[1]*0.02 if ax.get_ylim()[1] > 0 else 5,
                    f"n={len(p)}", ha="center", fontsize=8)

        ax.set_xticks(range(1, len(parts)+1))
        ax.set_xticklabels([f"{'Female' if s=='F' else 'Male'}" for s, _ in parts])
        ax.set_xlabel("Sex")
        ax.set_ylabel("N measurements per patient" if ax == axes[0] else "")
        ax.set_title(DATASET_LABELS[ds_name])

    fig.suptitle("Measurements per Patient by Sex", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, out_dir, "B2c_measurements_per_patient_sex")

    # Tabla resumen cobertura
    if coverage_rows:
        cov_df = pd.DataFrame(coverage_rows).set_index(["Dataset", "age_group"])
        _save_csv(cov_df, out_dir, "B2_coverage_by_age")
        _save_latex(cov_df, out_dir, "B2_coverage_by_age",
                    caption="Follow-up duration (days) by age group and dataset.",
                    label="coverage_age", float_fmt="%.1f")
    print("  [B2] Data coverage saved.")


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 3 — PERFIL GLUCÉMICO POR GRUPO DEMOGRÁFICO
# ─────────────────────────────────────────────────────────────────────────────

def block3_glucose_profile(datasets: dict, odirs: OutDirs):
    out_dir = odirs.cross  # comparativas cross-dataset
    """
    Estadísticos glucémicos (media, SD, mediana, IQR) y distribución de rangos
    clínicos (TBR_2, TBR_1, TIR, TAR_1, TAR_2) por grupo de edad y sexo.
    """
    RANGE_ORDER = ["TBR_2", "TBR_1", "TIR", "TAR_1", "TAR_2"]

    # ── 3a. Glucosa media por grupo de edad — comparativa datasets ────────────
    stat_rows = []
    for ds_name, (pi, gl) in datasets.items():
        sub = gl[gl["age_group"].isin(AGE_LABELS) & gl["Measurement"].notna()]
        for ag in AGE_LABELS:
            vals = sub[sub["age_group"] == ag]["Measurement"]
            if len(vals) < 5:
                continue
            stat_rows.append({
                "Dataset":   DATASET_LABELS[ds_name],
                "age_group": ag,
                "mean":      vals.mean(),
                "std":       vals.std(),
                "median":    vals.median(),
                "q25":       vals.quantile(0.25),
                "q75":       vals.quantile(0.75),
                "n_obs":     len(vals),
            })
    stat_df = pd.DataFrame(stat_rows)

    if not stat_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        x      = np.arange(len(AGE_LABELS))
        width  = 0.25
        for i, ds_name in enumerate(datasets.keys()):
            sub = stat_df[stat_df["Dataset"] == DATASET_LABELS[ds_name]]
            means  = [sub[sub["age_group"]==ag]["mean"].values[0]
                      if len(sub[sub["age_group"]==ag]) else np.nan for ag in AGE_LABELS]
            stds   = [sub[sub["age_group"]==ag]["std"].values[0]
                      if len(sub[sub["age_group"]==ag]) else np.nan for ag in AGE_LABELS]
            bars = ax.bar(x + (i-1)*width, means, width,
                          color=DATASET_PALETTE[ds_name], alpha=0.8,
                          label=DATASET_LABELS[ds_name],
                          yerr=stds, capsize=3, error_kw=dict(linewidth=0.8))

        ax.axhline(70,  color="#7EA6FF", linestyle=":", linewidth=1, alpha=0.7, label="Hypo threshold (70)")
        ax.axhline(180, color="#F4A261", linestyle=":", linewidth=1, alpha=0.7, label="Hyper threshold (180)")
        ax.set_xticks(x)
        ax.set_xticklabels(AGE_LABELS)
        ax.set_xlabel("Age group")
        ax.set_ylabel("Mean glucose (mg/dL) ± SD")
        ax.set_title("Mean Glucose Level by Age Group")
        ax.legend(loc="upper right", ncol=2)
        plt.tight_layout()
        _savefig(fig, out_dir, "B3a_mean_glucose_by_age")

    # ── 3b. Distribución de rangos clínicos por grupo de edad — stacked bar ──
    for ds_name, (pi, gl) in datasets.items():
        sub = gl[gl["age_group"].isin(AGE_LABELS) & gl["glucose_range"].isin(RANGE_ORDER)]
        pivot = (sub.groupby(["age_group", "glucose_range"])["Measurement"]
                 .count()
                 .unstack(fill_value=0)
                 .reindex(columns=RANGE_ORDER, fill_value=0))
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(8, 4.5))
        bottom = np.zeros(len(pivot_pct))
        for rng in RANGE_ORDER:
            if rng not in pivot_pct.columns:
                continue
            vals = pivot_pct[rng].values
            ax.bar(pivot_pct.index, vals, bottom=bottom,
                   color=GLUCOSE_RANGE_PALETTE[rng], label=GLUCOSE_RANGE_LABELS[rng],
                   edgecolor="white", linewidth=0.4)
            # Etiqueta si > 3%
            for xi, (v, b) in enumerate(zip(vals, bottom)):
                if v > 3:
                    ax.text(xi, b + v/2, f"{v:.1f}%", ha="center", va="center",
                            fontsize=7.5, color="black")
            bottom += vals

        ax.axhline(70, color="#7DCB8A", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Age group")
        ax.set_ylabel("% of measurements")
        ax.set_ylim(0, 105)
        ax.set_title(f"Glucose Range Distribution by Age Group — {DATASET_LABELS[ds_name]}")
        ax.legend(loc="upper right", ncol=1, fontsize=8)
        plt.tight_layout()
        _savefig(fig, odirs.per_ds[ds_name], "B3b_glucose_ranges_age")

    # ── 3c. TIR (Time in Range) por sexo — comparativa datasets ─────────────
    tir_rows = []
    for ds_name, (pi, gl) in datasets.items():
        sub = gl[gl["sex_label"].isin(["F", "M"]) & gl["glucose_range"].notna()]
        for sx in ["F", "M"]:
            s = sub[sub["sex_label"] == sx]
            total = len(s)
            if total == 0:
                continue
            tir_pct  = 100 * (s["glucose_range"] == "TIR").sum()  / total
            tbr_pct  = 100 * s["glucose_range"].isin(["TBR_1","TBR_2"]).sum() / total
            tar_pct  = 100 * s["glucose_range"].isin(["TAR_1","TAR_2"]).sum() / total
            tir_rows.append({
                "Dataset": DATASET_LABELS[ds_name], "Sex": sx,
                "TIR %": tir_pct, "TBR %": tbr_pct, "TAR %": tar_pct,
                "n_measurements": total
            })
    tir_df = pd.DataFrame(tir_rows)

    if not tir_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        metrics_plot = [("TIR %", "#7DCB8A", "Time in Range (%)"),
                        ("TBR %", "#7EA6FF", "Time Below Range (%)"),
                        ("TAR %", "#F4A261", "Time Above Range (%)")]
        for ax, (metric, color, ylabel) in zip(axes, metrics_plot):
            x     = np.arange(len(datasets))
            width = 0.3
            for i, sx in enumerate(["F", "M"]):
                vals = [tir_df[(tir_df["Dataset"]==DATASET_LABELS[k]) & (tir_df["Sex"]==sx)][metric].values
                        for k in datasets]
                vals = [v[0] if len(v) else np.nan for v in vals]
                bars = ax.bar(x + (i-0.5)*width, vals, width,
                              color=SEX_PALETTE[sx],
                              label=f"{'Female' if sx=='F' else 'Male'}",
                              alpha=0.85, edgecolor="white")
                for bar, v in zip(bars, vals):
                    if not np.isnan(v):
                        ax.text(bar.get_x() + bar.get_width()/2,
                                bar.get_height() + 0.5,
                                f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels([DATASET_LABELS[k] for k in datasets], rotation=15, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.legend()
        fig.suptitle("Glycemic Time Metrics by Sex", fontsize=13, fontweight="bold")
        plt.tight_layout()
        _savefig(fig, out_dir, "B3c_tir_tbr_tar_by_sex")
        _save_csv(tir_df.set_index(["Dataset","Sex"]), out_dir, "B3_glycemic_metrics_sex")
        _save_latex(tir_df.set_index(["Dataset","Sex"]), out_dir, "B3_glycemic_metrics_sex",
                    caption="Glycemic time metrics (TIR, TBR, TAR) by sex and dataset.",
                    label="glycemic_sex")

    # ── 3d. Violinplot glucosa por grupo edad × dataset ───────────────────────
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5*len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, (ds_name, (pi, gl)) in zip(axes, datasets.items()):
        sub = gl[gl["age_group"].isin(AGE_LABELS) & gl["Measurement"].between(30, 500)]
        parts = [(ag, sub[sub["age_group"]==ag]["Measurement"].dropna())
                 for ag in AGE_LABELS]
        parts = [(ag, p) for ag, p in parts if len(p) >= 10]
        if not parts:
            continue
        vp = ax.violinplot([p.values for _, p in parts], positions=range(len(parts)),
                           showmedians=True, showextrema=False)
        for body, (ag, _) in zip(vp["bodies"], parts):
            body.set_facecolor(AGE_PALETTE.get(ag, "#999999"))
            body.set_alpha(0.65)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.5)
        ax.axhline(70,  color="#7EA6FF", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(180, color="#F4A261", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xticks(range(len(parts)))
        ax.set_xticklabels([ag for ag, _ in parts], rotation=0)
        ax.set_xlabel("Age group")
        ax.set_ylabel("Glucose (mg/dL)" if ax == axes[0] else "")
        ax.set_title(DATASET_LABELS[ds_name])
        ax.set_ylim(20, 500)
    fig.suptitle("Glucose Distribution by Age Group", fontsize=13, fontweight="bold")
    # Leyenda umbrales
    handles = [mpatches.Patch(color="#7EA6FF", alpha=0.7, label="Hypo threshold (70 mg/dL)"),
               mpatches.Patch(color="#F4A261", alpha=0.7, label="Hyper threshold (180 mg/dL)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    _savefig(fig, out_dir, "B3d_glucose_violin_age")

    # tabla estadísticos
    if not stat_df.empty:
        stat_out = stat_df.set_index(["Dataset","age_group"])[["mean","std","median","q25","q75","n_obs"]]
        _save_csv(stat_out, out_dir, "B3_glucose_stats_age")
        _save_latex(stat_out, out_dir, "B3_glucose_stats_age",
                    caption="Descriptive glucose statistics by age group and dataset.",
                    label="glucose_stats_age")
    print("  [B3] Glucose profiles saved.")


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 4 — DESBALANCEO DE CLASES POR GRUPO DEMOGRÁFICO
# ─────────────────────────────────────────────────────────────────────────────

def _class_label(v):
    if pd.isna(v): return "missing"
    if v < 70:     return "hypoglycemia"
    if v <= 180:   return "normoglycemia"
    return "hyperglycemia"

CLASS_ORDER  = ["hypoglycemia", "normoglycemia", "hyperglycemia"]
CLASS_COLORS = {"hypoglycemia": "#7EA6FF", "normoglycemia": "#7DCB8A", "hyperglycemia": "#F4A261"}
CLASS_LABELS = {"hypoglycemia": "Hypoglycemia", "normoglycemia": "Normoglycemia",
                "hyperglycemia": "Hyperglycemia"}

def block4_class_imbalance(datasets: dict, odirs: OutDirs):
    out_dir = odirs.cross  # comparativas cross-dataset
    """
    Distribución de clases glucémicas (hipo/normo/hiper) por grupo de edad y sexo.
    """
    RANGE_ORDER_CLASS = ["TBR_2", "TBR_1", "TIR", "TAR_1", "TAR_2"]

    # ── 4a. % clase por grupo de edad × dataset ───────────────────────────────
    imb_rows = []
    for ds_name, (pi, gl) in datasets.items():
        sub = gl[gl["age_group"].isin(AGE_LABELS) & gl["Measurement"].notna()].copy()
        sub["class_label"] = sub["Measurement"].apply(_class_label)
        sub = sub[sub["class_label"].isin(CLASS_ORDER)]
        for ag in AGE_LABELS:
            ag_sub = sub[sub["age_group"] == ag]
            total  = len(ag_sub)
            if total == 0:
                continue
            for cl in CLASS_ORDER:
                n = (ag_sub["class_label"] == cl).sum()
                imb_rows.append({
                    "Dataset": DATASET_LABELS[ds_name], "age_group": ag,
                    "class": cl, "n": n, "pct": 100*n/total
                })
    imb_df = pd.DataFrame(imb_rows)

    if not imb_df.empty:
        fig, axes = plt.subplots(len(datasets), 1, figsize=(9, 4*len(datasets)), sharex=False)
        if len(datasets) == 1:
            axes = [axes]
        for ax, ds_name in zip(axes, datasets.keys()):
            sub = imb_df[imb_df["Dataset"] == DATASET_LABELS[ds_name]]
            pivot = sub.pivot(index="age_group", columns="class", values="pct")\
                       .reindex(index=AGE_LABELS, columns=CLASS_ORDER, fill_value=0)
            bottom = np.zeros(len(pivot))
            for cl in CLASS_ORDER:
                if cl not in pivot.columns:
                    continue
                vals = pivot[cl].values
                bars = ax.bar(pivot.index, vals, bottom=bottom,
                              color=CLASS_COLORS[cl], label=CLASS_LABELS[cl],
                              edgecolor="white", linewidth=0.4, alpha=0.9)
                for xi, (v, b) in enumerate(zip(vals, bottom)):
                    if v > 2.5:
                        ax.text(xi, b + v/2, f"{v:.1f}%", ha="center", va="center",
                                fontsize=8, color="black")
                bottom += vals
            ax.set_ylabel("% of measurements")
            ax.set_ylim(0, 105)
            ax.set_title(f"Class Distribution by Age Group — {DATASET_LABELS[ds_name]}")
            ax.legend(loc="upper right", fontsize=8)
        fig.suptitle("Glycemic Class Distribution by Age Group", fontsize=13, fontweight="bold")
        plt.tight_layout()
        _savefig(fig, out_dir, "B4a_class_distribution_age")

    # ── 4b. Ratio de desbalanceo (normo/hipo) por grupo de edad ──────────────
    ratio_rows = []
    for ds_name, (pi, gl) in datasets.items():
        sub = gl[gl["age_group"].isin(AGE_LABELS) & gl["Measurement"].notna()].copy()
        sub["class_label"] = sub["Measurement"].apply(_class_label)
        pat_cls = sub.groupby(["Patient_ID", "age_group", "class_label"])["Measurement"]\
                     .count().unstack(fill_value=0).reset_index()
        if "hypoglycemia" not in pat_cls.columns:
            pat_cls["hypoglycemia"] = 0
        if "normoglycemia" not in pat_cls.columns:
            pat_cls["normoglycemia"] = 0
        pat_cls["imbalance_ratio"] = pat_cls["normoglycemia"] / pat_cls["hypoglycemia"].replace(0, np.nan)
        for ag in AGE_LABELS:
            sub_ag = pat_cls[(pat_cls["age_group"] == ag) & pat_cls["imbalance_ratio"].notna()]
            if len(sub_ag) < 2:
                continue
            ratio_rows.append({
                "Dataset":   DATASET_LABELS[ds_name],
                "age_group": ag,
                "median_ratio": sub_ag["imbalance_ratio"].median(),
                "q25_ratio":    sub_ag["imbalance_ratio"].quantile(0.25),
                "q75_ratio":    sub_ag["imbalance_ratio"].quantile(0.75),
                "n_patients":   len(sub_ag),
            })
    ratio_df = pd.DataFrame(ratio_rows)

    if not ratio_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x      = np.arange(len(AGE_LABELS))
        width  = 0.25
        for i, ds_name in enumerate(datasets.keys()):
            sub = ratio_df[ratio_df["Dataset"] == DATASET_LABELS[ds_name]]
            medians = [sub[sub["age_group"]==ag]["median_ratio"].values[0]
                       if len(sub[sub["age_group"]==ag]) else np.nan for ag in AGE_LABELS]
            q25s    = [sub[sub["age_group"]==ag]["q25_ratio"].values[0]
                       if len(sub[sub["age_group"]==ag]) else np.nan for ag in AGE_LABELS]
            q75s    = [sub[sub["age_group"]==ag]["q75_ratio"].values[0]
                       if len(sub[sub["age_group"]==ag]) else np.nan for ag in AGE_LABELS]
            yerr_lo = [m - q if not (np.isnan(m) or np.isnan(q)) else 0 for m, q in zip(medians, q25s)]
            yerr_hi = [q - m if not (np.isnan(m) or np.isnan(q)) else 0 for m, q in zip(medians, q75s)]
            ax.bar(x + (i-1)*width, medians, width,
                   color=DATASET_PALETTE[ds_name], label=DATASET_LABELS[ds_name],
                   yerr=[yerr_lo, yerr_hi], capsize=3,
                   error_kw=dict(linewidth=0.8), alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(AGE_LABELS)
        ax.set_xlabel("Age group")
        ax.set_ylabel("Imbalance ratio (normoglycemia / hypoglycemia)\nMedian ± IQR")
        ax.set_title("Class Imbalance Ratio by Age Group")
        ax.legend()
        plt.tight_layout()
        _savefig(fig, out_dir, "B4b_imbalance_ratio_age")

        _save_csv(ratio_df.set_index(["Dataset","age_group"]), out_dir, "B4_imbalance_ratio_age")
        _save_latex(ratio_df.set_index(["Dataset","age_group"]), out_dir, "B4_imbalance_ratio_age",
                    caption="Imbalance ratio (normoglycemia/hypoglycemia) per patient, by age group.",
                    label="imbalance_ratio_age", float_fmt="%.1f")

    # ── 4c. Distribución de clases por sexo ──────────────────────────────────
    sex_cls_rows = []
    for ds_name, (pi, gl) in datasets.items():
        sub = gl[gl["sex_label"].isin(["F","M"]) & gl["Measurement"].notna()].copy()
        sub["class_label"] = sub["Measurement"].apply(_class_label)
        sub = sub[sub["class_label"].isin(CLASS_ORDER)]
        for sx in ["F", "M"]:
            s = sub[sub["sex_label"] == sx]
            total = len(s)
            if total == 0:
                continue
            for cl in CLASS_ORDER:
                n = (s["class_label"] == cl).sum()
                sex_cls_rows.append({
                    "Dataset": DATASET_LABELS[ds_name], "Sex": sx,
                    "class": cl, "pct": 100*n/total
                })
    scls_df = pd.DataFrame(sex_cls_rows)

    if not scls_df.empty:
        fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4.5))
        if len(datasets) == 1:
            axes = [axes]
        for ax, ds_name in zip(axes, datasets.keys()):
            sub = scls_df[scls_df["Dataset"] == DATASET_LABELS[ds_name]]
            pivot = sub.pivot(index="Sex", columns="class", values="pct")\
                       .reindex(index=["F","M"], columns=CLASS_ORDER, fill_value=0)
            bottom = np.zeros(len(pivot))
            for cl in CLASS_ORDER:
                if cl not in pivot.columns:
                    continue
                vals = pivot[cl].values
                ax.bar(pivot.index, vals, bottom=bottom,
                       color=CLASS_COLORS[cl], label=CLASS_LABELS[cl],
                       edgecolor="white", linewidth=0.4, alpha=0.9)
                for xi, (v, b) in enumerate(zip(vals, bottom)):
                    if v > 2:
                        ax.text(xi, b + v/2, f"{v:.1f}%", ha="center", va="center", fontsize=9)
                bottom += vals
            ax.set_xticklabels(["Female", "Male"])
            ax.set_ylabel("% of measurements" if ax == axes[0] else "")
            ax.set_ylim(0, 105)
            ax.set_title(DATASET_LABELS[ds_name])
            ax.legend(loc="upper right", fontsize=8)
        fig.suptitle("Glycemic Class Distribution by Sex", fontsize=13, fontweight="bold")
        plt.tight_layout()
        _savefig(fig, out_dir, "B4c_class_distribution_sex")

    print("  [B4] Class imbalance analysis saved.")


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 5 — VARIABILIDAD GLUCÉMICA POR GRUPO DEMOGRÁFICO
# ─────────────────────────────────────────────────────────────────────────────

def _cv(s):
    m = s.mean()
    return (s.std() / m * 100) if m > 0 else np.nan

def block5_glycemic_variability(datasets: dict, odirs: OutDirs):
    out_dir = odirs.cross  # comparativas cross-dataset
    """
    Coeficiente de variación (CV), SD y rango intercuartílico de glucosa
    por paciente, desglosados por grupo de edad y sexo.
    """
    var_rows = []
    for ds_name, (pi, gl) in datasets.items():
        sub = gl[gl["Measurement"].between(30, 500) & gl["Measurement"].notna()]
        per_pat = (sub.groupby("Patient_ID")["Measurement"]
                   .agg(mean="mean", std="std", cv=_cv,
                        q25=lambda x: x.quantile(0.25),
                        q75=lambda x: x.quantile(0.75),
                        iqr=lambda x: x.quantile(0.75)-x.quantile(0.25))
                   .reset_index())
        per_pat = per_pat.merge(pi[["Patient_ID","age_group","sex_label"]], on="Patient_ID", how="left")
        per_pat["dataset"] = ds_name
        var_rows.append(per_pat)

    if not var_rows:
        print("  [B5] No data for variability block.")
        return

    var_df = pd.concat(var_rows, ignore_index=True)

    # ── 5a. CV por grupo de edad — multipanel datasets ────────────────────────
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4.5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds_name in zip(axes, datasets.keys()):
        sub = var_df[(var_df["dataset"]==ds_name) & (var_df["age_group"].isin(AGE_LABELS))]
        parts = [(ag, sub[sub["age_group"]==ag]["cv"].dropna())
                 for ag in AGE_LABELS]
        parts = [(ag, p) for ag, p in parts if len(p) >= 2]
        if not parts:
            continue
        bp = ax.boxplot([p.values for _, p in parts], patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5),
                        whiskerprops=dict(linewidth=1),
                        capprops=dict(linewidth=1),
                        flierprops=dict(marker=".", markersize=3, alpha=0.4))
        for patch, (ag, _) in zip(bp["boxes"], parts):
            patch.set_facecolor(AGE_PALETTE.get(ag, "#999"))
            patch.set_alpha(0.7)
        ax.axhline(36, color="red", linestyle="--", linewidth=0.9, alpha=0.6, label="CV=36% target")
        ax.set_xticks(range(1, len(parts)+1))
        ax.set_xticklabels([ag for ag, _ in parts])
        ax.set_xlabel("Age group")
        ax.set_ylabel("CV (%)" if ax == axes[0] else "")
        ax.set_title(DATASET_LABELS[ds_name])
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Glycemic Coefficient of Variation (CV) by Age Group", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, out_dir, "B5a_cv_by_age")

    # ── 5b. CV por sexo ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x      = np.arange(len(datasets))
    width  = 0.3
    for i, sx in enumerate(["F", "M"]):
        medians = []
        q25s, q75s = [], []
        for ds_name in datasets.keys():
            sub = var_df[(var_df["dataset"]==ds_name) & (var_df["sex_label"]==sx)]["cv"].dropna()
            medians.append(sub.median() if len(sub) else np.nan)
            q25s.append(sub.quantile(0.25) if len(sub) else np.nan)
            q75s.append(sub.quantile(0.75) if len(sub) else np.nan)
        yerr_lo = [m-q if not (np.isnan(m) or np.isnan(q)) else 0 for m,q in zip(medians,q25s)]
        yerr_hi = [q-m if not (np.isnan(m) or np.isnan(q)) else 0 for m,q in zip(medians,q75s)]
        ax.bar(x + (i-0.5)*width, medians, width, color=SEX_PALETTE[sx],
               label=f"{'Female' if sx=='F' else 'Male'}",
               yerr=[yerr_lo, yerr_hi], capsize=3, error_kw=dict(linewidth=0.8), alpha=0.85)
    ax.axhline(36, color="red", linestyle="--", linewidth=0.9, alpha=0.6, label="CV=36% target")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[k] for k in datasets])
    ax.set_ylabel("CV (%) — Median ± IQR per patient")
    ax.set_title("Glycemic Coefficient of Variation by Sex")
    ax.legend()
    plt.tight_layout()
    _savefig(fig, out_dir, "B5b_cv_by_sex")

    # ── 5c. Tabla estadísticos de variabilidad ────────────────────────────────
    var_agg_rows = []
    for ds_name in datasets.keys():
        for ag in AGE_LABELS:
            s = var_df[(var_df["dataset"]==ds_name) & (var_df["age_group"]==ag)]
            if len(s) < 2:
                continue
            var_agg_rows.append({
                "Dataset":    DATASET_LABELS[ds_name],
                "Age group":  ag,
                "N patients": len(s),
                "CV median":  s["cv"].median(),
                "CV IQR":     f"{s['cv'].quantile(0.25):.1f}–{s['cv'].quantile(0.75):.1f}",
                "SD median":  s["std"].median(),
                "IQR median": s["iqr"].median(),
            })
    if var_agg_rows:
        var_agg = pd.DataFrame(var_agg_rows).set_index(["Dataset","Age group"])
        _save_csv(var_agg, out_dir, "B5_variability_by_age")
        _save_latex(var_agg, out_dir, "B5_variability_by_age",
                    caption="Glycemic variability (CV, SD, IQR) per patient by age group.",
                    label="variability_age", float_fmt="%.1f")
    print("  [B5] Glycemic variability saved.")


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 6 — PATRONES CIRCADIANOS POR GRUPO DEMOGRÁFICO
# ─────────────────────────────────────────────────────────────────────────────

def block6_circadian_patterns(datasets: dict, odirs: OutDirs):
    out_dir = odirs.cross  # comparativas cross-dataset
    """
    Glucosa media por hora del día, desglosada por grupo de edad y sexo.
    """
    # ── 6a. Perfil circadiano por grupo de edad ───────────────────────────────
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5*len(datasets), 4.5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, (ds_name, (pi, gl)) in zip(axes, datasets.items()):
        sub = gl[gl["age_group"].isin(AGE_LABELS) & gl["Measurement"].between(30,500)].copy()
        sub["hour"] = sub["timestamp"].dt.hour
        hourly = sub.groupby(["age_group","hour"])["Measurement"].mean().reset_index()

        for ag in AGE_LABELS:
            h = hourly[hourly["age_group"]==ag].sort_values("hour")
            if h.empty:
                continue
            ax.plot(h["hour"], h["Measurement"],
                    color=AGE_PALETTE.get(ag,"#999"), linewidth=1.8, label=ag, alpha=0.9)

        ax.axhline(70,  color="#7EA6FF", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.axhline(180, color="#F4A261", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.set_xticks([0,6,12,18,23])
        ax.set_xticklabels(["00:00","06:00","12:00","18:00","23:00"])
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Mean glucose (mg/dL)" if ax == axes[0] else "")
        ax.set_title(DATASET_LABELS[ds_name])
        ax.legend(title="Age group", fontsize=8)

    fig.suptitle("Circadian Glucose Profile by Age Group", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, out_dir, "B6a_circadian_profile_age")

    # ── 6b. Perfil circadiano por sexo ────────────────────────────────────────
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5*len(datasets), 4.5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, (ds_name, (pi, gl)) in zip(axes, datasets.items()):
        sub = gl[gl["sex_label"].isin(["F","M"]) & gl["Measurement"].between(30,500)].copy()
        sub["hour"] = sub["timestamp"].dt.hour
        hourly = sub.groupby(["sex_label","hour"])["Measurement"].mean().reset_index()

        for sx in ["F", "M"]:
            h = hourly[hourly["sex_label"]==sx].sort_values("hour")
            if h.empty:
                continue
            ax.plot(h["hour"], h["Measurement"],
                    color=SEX_PALETTE[sx], linewidth=2,
                    label=f"{'Female' if sx=='F' else 'Male'}", alpha=0.9)

        ax.axhline(70,  color="#7EA6FF", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.axhline(180, color="#F4A261", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.set_xticks([0,6,12,18,23])
        ax.set_xticklabels(["00:00","06:00","12:00","18:00","23:00"])
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Mean glucose (mg/dL)" if ax == axes[0] else "")
        ax.set_title(DATASET_LABELS[ds_name])
        ax.legend(fontsize=9)

    fig.suptitle("Circadian Glucose Profile by Sex", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, out_dir, "B6b_circadian_profile_sex")
    print("  [B6] Circadian patterns saved.")


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 7 — HETEROGENEIDAD INTER-PACIENTE POR GRUPO DEMOGRÁFICO
# ─────────────────────────────────────────────────────────────────────────────

def block7_inter_patient_heterogeneity(datasets: dict, odirs: OutDirs):
    out_dir = odirs.cross  # comparativas cross-dataset
    """
    Scatterplot edad real vs. glucosa media y CV por paciente.
    Permite ver si los grupos de edad capturan gradientes continuos.
    """
    fig_rows = 2
    fig, axes = plt.subplots(fig_rows, len(datasets),
                             figsize=(5.5*len(datasets), 4.5*fig_rows))
    if len(datasets) == 1:
        axes = axes.reshape(-1, 1)

    for col, (ds_name, (pi, gl)) in enumerate(datasets.items()):
        per_pat = (gl[gl["Measurement"].between(30,500)]
                   .groupby("Patient_ID")["Measurement"]
                   .agg(mean="mean", cv=_cv)
                   .reset_index())
        per_pat = per_pat.merge(pi[["Patient_ID","_age_num","sex_label","age_group"]],
                                on="Patient_ID", how="left")
        per_pat = per_pat[per_pat["_age_num"].notna() & per_pat["age_group"].isin(AGE_LABELS)]

        for row, (metric, ylabel, title_suffix) in enumerate([
            ("mean", "Mean glucose (mg/dL)", "Mean Glucose"),
            ("cv",   "CV (%)",               "Glycemic Variability (CV)"),
        ]):
            ax = axes[row][col]
            for sx in ["F", "M"]:
                sub = per_pat[per_pat["sex_label"] == sx]
                ax.scatter(sub["_age_num"], sub[metric],
                           color=SEX_PALETTE[sx], alpha=0.55, s=28,
                           label=f"{'Female' if sx=='F' else 'Male'}", edgecolors="white", linewidths=0.3)
            # Línea tendencia global
            all_valid = per_pat[[metric,"_age_num"]].dropna()
            if len(all_valid) >= 5:
                z = np.polyfit(all_valid["_age_num"], all_valid[metric], 1)
                p = np.poly1d(z)
                xs = np.linspace(all_valid["_age_num"].min(), all_valid["_age_num"].max(), 100)
                ax.plot(xs, p(xs), color="black", linewidth=1.5, linestyle="--", alpha=0.7,
                        label=f"Trend (slope={z[0]:.2f})")
            # Líneas de grupo de edad
            for bound in [30, 45, 65]:
                ax.axvline(bound, color="#AAAAAA", linestyle=":", linewidth=0.8)
            if row == 0:
                ax.axhline(70,  color="#7EA6FF", linestyle=":", linewidth=0.8, alpha=0.6)
                ax.axhline(180, color="#F4A261", linestyle=":", linewidth=0.8, alpha=0.6)
            ax.set_xlabel("Age (years)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{DATASET_LABELS[ds_name]} — {title_suffix}")
            ax.legend(fontsize=8)

    fig.suptitle("Inter-patient Heterogeneity: Age vs. Glucose Metrics",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, out_dir, "B7_inter_patient_heterogeneity")
    print("  [B7] Inter-patient heterogeneity saved.")


# ─────────────────────────────────────────────────────────────────────────────
# BLOQUE 8 — TABLA RESUMEN CROSS-DATASET (LaTeX)
# ─────────────────────────────────────────────────────────────────────────────

def block8_summary_table(datasets: dict, odirs: OutDirs):
    out_dir = odirs.cross  # comparativas cross-dataset
    """
    Tabla maestra: para cada dataset × grupo de edad × sexo:
    n_patients, mean_glucose, CV_median, TIR%, TBR%, TAR%, imbalance_ratio
    """
    rows = []
    for ds_name, (pi, gl) in datasets.items():
        gl_valid = gl[gl["Measurement"].between(30,500) & gl["age_group"].isin(AGE_LABELS)].copy()
        gl_valid["class_label"] = gl_valid["Measurement"].apply(_class_label)

        for ag in AGE_LABELS:
            sub_ag = gl_valid[gl_valid["age_group"] == ag]
            if len(sub_ag) == 0:
                continue
            total = len(sub_ag[sub_ag["class_label"].isin(CLASS_ORDER)])
            n_pat = sub_ag["Patient_ID"].nunique()

            # Per-patient CV
            cv_vals = (sub_ag.groupby("Patient_ID")["Measurement"]
                       .apply(_cv).dropna())
            cv_med  = cv_vals.median() if len(cv_vals) else np.nan

            # Glycemic ranges
            rng_counts = sub_ag["glucose_range"].value_counts()
            tir  = 100*rng_counts.get("TIR",   0)/total if total else np.nan
            tbr  = 100*(rng_counts.get("TBR_1",0)+rng_counts.get("TBR_2",0))/total if total else np.nan
            tar  = 100*(rng_counts.get("TAR_1",0)+rng_counts.get("TAR_2",0))/total if total else np.nan

            # Imbalance
            n_hypo  = (sub_ag["class_label"]=="hypoglycemia").sum()
            n_normo = (sub_ag["class_label"]=="normoglycemia").sum()
            imb     = n_normo/n_hypo if n_hypo>0 else np.nan

            rows.append({
                "Dataset":         DATASET_LABELS[ds_name],
                "Age group":       ag,
                "N patients":      n_pat,
                "Mean glucose":    round(sub_ag["Measurement"].mean(), 1),
                "SD glucose":      round(sub_ag["Measurement"].std(),  1),
                "CV (%) median":   round(cv_med, 1) if not np.isnan(cv_med) else np.nan,
                "TIR (%)":         round(tir, 1)  if not np.isnan(tir)  else np.nan,
                "TBR (%)":         round(tbr, 1)  if not np.isnan(tbr)  else np.nan,
                "TAR (%)":         round(tar, 1)  if not np.isnan(tar)  else np.nan,
                "Imbalance ratio": round(imb, 1)  if not np.isnan(imb)  else np.nan,
            })

    if rows:
        summary = pd.DataFrame(rows).set_index(["Dataset","Age group"])
        _save_csv(summary, out_dir, "B8_master_summary")
        _save_latex(summary, out_dir, "B8_master_summary",
                    caption="Master summary table: glycemic and demographic characteristics by age group and dataset.",
                    label="master_summary", float_fmt="%.1f")
        print("  [B8] Master summary table saved.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    matplotlib.rcParams.update(MPL_RC)

    # -- Deteccion de ROOT
    root     = find_project_root(Path.cwd())
    data_dir = root / "data"

    cfg = _load_config(root)
    if cfg and hasattr(cfg, "OUTPUT_ROOT"):
        eda_root = Path(cfg.OUTPUT_ROOT).parent / "EDA"
    else:
        eda_root = root / "EDA"

    print(f"ROOT:    {root}")
    print(f"EDA OUT: {eda_root}")

    # -- Carga de datasets
    datasets_to_load = ["T1DiabetesGranada", "DIATREND", "REPLACE-BG"]
    datasets = {}
    for ds_name in datasets_to_load:
        try:
            pi, gl = load_dataset(data_dir, ds_name)
            pi, gl = enrich_demographics(pi, gl)
            datasets[ds_name] = (pi, gl)
            print(f"  OK {ds_name} loaded.")
        except FileNotFoundError as e:
            print(f"  SKIP {ds_name} not found. ({e})")
        except Exception as e:
            import traceback
            print(f"  ERR {ds_name}: {e}")
            traceback.print_exc()

    if not datasets:
        raise RuntimeError("No se pudo cargar ningun dataset.")

    # -- Construccion de directorios
    #   EDA/
    #   cross/              <- comparativas entre datasets
    #   T1DiabetesGranada/  <- figuras individuales
    #   DIATREND/
    #   REPLACE-BG/
    odirs = build_out_dirs(eda_root, list(datasets.keys()))
    print(f"  Subdirectorios: cross/ + {list(odirs.per_ds.keys())}")

    # -- Bloques
    print("\n--- Bloque 0: Cohort Overview ---")
    block0_cohort_overview(datasets, odirs)

    print("\n--- Bloque 1: Demographic Distribution ---")
    block1_demographic_distribution(datasets, odirs)

    print("\n--- Bloque 2: Data Coverage ---")
    block2_data_coverage(datasets, odirs)

    print("\n--- Bloque 3: Glucose Profiles ---")
    block3_glucose_profile(datasets, odirs)

    print("\n--- Bloque 4: Class Imbalance ---")
    block4_class_imbalance(datasets, odirs)

    print("\n--- Bloque 5: Glycemic Variability ---")
    block5_glycemic_variability(datasets, odirs)

    print("\n--- Bloque 6: Circadian Patterns ---")
    block6_circadian_patterns(datasets, odirs)

    print("\n--- Bloque 7: Inter-patient Heterogeneity ---")
    block7_inter_patient_heterogeneity(datasets, odirs)

    print("\n--- Bloque 8: Master Summary Table ---")
    block8_summary_table(datasets, odirs)

    # -- Resumen
    total = sum(len(list(d.iterdir())) for d in [odirs.cross] + list(odirs.per_ds.values()))
    print(f"\nEDA completo. Resultados en: {eda_root}")
    print(f"  cross/  -> {len(list(odirs.cross.iterdir()))} archivos")
    for ds_name, d in odirs.per_ds.items():
        print(f"  {ds_name}/  -> {len(list(d.iterdir()))} archivos")
    print(f"  Total: {total} archivos")


if __name__ == "__main__":
    main()