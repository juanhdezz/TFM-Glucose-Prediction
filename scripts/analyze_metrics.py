#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis comparativo de técnicas de balanceo para predicción de glucosa.

Extrae métricas de rendimiento (RMSE, MAE, MAPE) y número de puntos por zona
de los archivos generados por el pipeline de entrenamiento. Compara cada técnica
con el dataset original, genera visualizaciones profesionales y un informe detallado.

Estructura esperada de los archivos:
  - metrics_performance_by_range_*_ALL_aggregate_then_metric.csv
    Columnas: Range, A, B, C, D, E, A + B, RMSE, MSE, MAE, MAPE
  - metrics_performance_by_range_*_ALL_metric_then_aggregate.csv
    Columnas: Range, A, B, C, D, E, A + B, RMSE, MSE, MAE, MAPE, Fold
  - number_of_points_by_zones_*_ALL_aggregate_then_metric.csv
    Columnas: Range, A, B, C, D, E
  - number_of_points_by_zones_*_ALL_metric_then_aggregate.csv
    Columnas: Range, A, B, C, D, E, Fold

Uso: python3 /home/juanhdez/scripts/analyze_metrics.py
"""

import os
import re
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo profesional con seaborn
sns.set_theme(style="darkgrid", palette="Set2", font_scale=1.2)

# Rutas
BASE_OUTPUT = Path("/home/juanhdez/data/output")
RESULTS_DIR = Path("/home/juanhdez/results/metrics_comparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODES = ["aggregate_then_metric", "metric_then_aggregate"]


def find_experiment_dirs(base_dir):
    """
    Recorre la estructura y devuelve un diccionario:
    { (dataset, tecnica): path_to_EXP_folder }
    """
    experiments = {}
    for dataset_dir in base_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        for tecnica_dir in dataset_dir.iterdir():
            if not tecnica_dir.is_dir():
                continue
            tecnica = tecnica_dir.name
            exp_dirs = list(tecnica_dir.glob("EXP-*"))
            if not exp_dirs:
                continue
            exp_path = exp_dirs[0]
            experiments[(dataset, tecnica)] = exp_path
    return experiments


def load_metric_file(file_path, metric_type, mode):
    """
    Carga un archivo CSV y extrae las columnas de interés.
    - performance: columnas RMSE, MAE, MAPE (y MSE si existe)
    - points: suma de las columnas A, B, C, D, E -> total_points
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error al leer {file_path}: {e}")
        return None

    id_col = "Range"
    if id_col not in df.columns:
        id_col = df.columns[0]  # fallback

    if metric_type == "performance":
        # Métricas de error
        perf_metrics = ["RMSE", "MAE", "MAPE", "MSE"]
        metric_cols = [c for c in perf_metrics if c in df.columns]
        if not metric_cols:
            # Si no están, buscar columnas numéricas (excluyendo id y A,B,C,D,E,A+B)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if id_col in numeric_cols:
                numeric_cols.remove(id_col)
            exclude = ["A", "B", "C", "D", "E", "A + B", "Fold"]
            metric_cols = [c for c in numeric_cols if c not in exclude]
        if not metric_cols:
            print(f"Advertencia: no se encontraron métricas en {file_path}")
            return None

        # Eliminar columna Fold si existe
        if "Fold" in df.columns:
            df = df.drop(columns=["Fold"])

        keep_cols = [id_col] + metric_cols
        df_clean = df[keep_cols].copy()
        df_clean["dataset"] = "?"
        df_clean["tecnica"] = "?"
        df_clean["tipo"] = metric_type
        df_clean["modo"] = mode
        df_clean["id_col"] = id_col
        df_clean["metric_columns"] = ",".join(metric_cols)
        return df_clean, metric_cols, id_col

    elif metric_type == "points":
        # Columnas A, B, C, D, E representan conteos; las sumamos
        point_cols = ["A", "B", "C", "D", "E"]
        existing = [c for c in point_cols if c in df.columns]
        if not existing:
            # Si no, intentar con columnas numéricas excepto id y Fold
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if id_col in numeric_cols:
                numeric_cols.remove(id_col)
            if "Fold" in numeric_cols:
                numeric_cols.remove("Fold")
            existing = numeric_cols
        if not existing:
            print(f"Advertencia: no se encontraron columnas de conteo en {file_path}")
            return None

        # Crear columna total_points
        df["total_points"] = df[existing].sum(axis=1)
        keep_cols = [id_col, "total_points"]
        df_clean = df[keep_cols].copy()
        df_clean["dataset"] = "?"
        df_clean["tecnica"] = "?"
        df_clean["tipo"] = metric_type
        df_clean["modo"] = mode
        df_clean["id_col"] = id_col
        df_clean["metric_columns"] = "total_points"
        return df_clean, ["total_points"], id_col
    else:
        return None


def load_all_experiments(experiments):
    """
    Carga todos los archivos de métricas de los experimentos.
    """
    all_data = []
    for (dataset, tecnica), exp_path in experiments.items():
        metrics_dir = exp_path / "results" / "evaluation" / "metrics"
        if not metrics_dir.exists():
            continue

        for metric_type in ["performance", "points"]:
            for mode in MODES:
                # Patrón de archivo: *{metric_type}*_ALL_{mode}.csv
                pattern = f"*{metric_type}*_ALL_{mode}.csv"
                files = list(metrics_dir.glob(pattern))
                if not files:
                    continue
                file_path = files[0]
                result = load_metric_file(file_path, metric_type, mode)
                if result is None:
                    continue
                df_clean, metric_cols, id_col = result
                df_clean["dataset"] = dataset
                df_clean["tecnica"] = tecnica
                all_data.append(df_clean)

    if not all_data:
        print("No se encontraron datos para cargar.")
        return None
    return pd.concat(all_data, ignore_index=True)


def aggregate_metrics_by_id(df_combined):
    """
    Agrupa por (dataset, tecnica, modo, id) y agrega:
      - performance: media (promedio sobre folds)
      - points: suma (total de puntos por zona)
    """
    df_perf = df_combined[df_combined["tipo"] == "performance"].copy()
    df_points = df_combined[df_combined["tipo"] == "points"].copy()

    agg_list = []

    if not df_perf.empty:
        id_col = df_perf["id_col"].iloc[0]
        metric_cols = [c for c in df_perf.columns
                       if c not in ["dataset", "tecnica", "tipo", "modo", "id_col", "metric_columns", "index"]
                       and np.issubdtype(df_perf[c].dtype, np.number)]
        if metric_cols:
            grouped = df_perf.groupby(["dataset", "tecnica", "modo", id_col])[metric_cols].mean().reset_index()
            grouped.rename(columns={id_col: "id"}, inplace=True)
            grouped["tipo"] = "performance"
            agg_list.append(grouped)

    if not df_points.empty:
        id_col = df_points["id_col"].iloc[0]
        metric_cols = [c for c in df_points.columns
                       if c not in ["dataset", "tecnica", "tipo", "modo", "id_col", "metric_columns", "index"]
                       and np.issubdtype(df_points[c].dtype, np.number)]
        if metric_cols:
            # Sumamos el total de puntos por zona (ya viene sumado, pero por si hay varios folds)
            grouped = df_points.groupby(["dataset", "tecnica", "modo", id_col])[metric_cols].sum().reset_index()
            grouped.rename(columns={id_col: "id"}, inplace=True)
            grouped["tipo"] = "points"
            agg_list.append(grouped)

    if not agg_list:
        print("No se pudo agregar ningún dato.")
        return None
    return pd.concat(agg_list, ignore_index=True)


def compute_relative_improvement(df_agg):
    """
    Calcula la mejora relativa de cada técnica respecto al original.
    Para métricas de error (RMSE, MAE, MAPE): (orig - tec) / orig * 100.
    """
    original = df_agg[df_agg["tecnica"] == "original"].copy()
    tecnicas = df_agg[df_agg["tecnica"] != "original"].copy()

    if original.empty:
        print("No se encontraron datos 'original' para calcular mejoras.")
        return None

    merged = pd.merge(tecnicas, original,
                      on=["dataset", "tipo", "modo", "id"],
                      suffixes=("", "_orig"))

    metric_cols = [c for c in merged.columns
                   if c not in ["dataset", "tecnica", "modo", "id", "tipo", "tecnica_orig", "id_col", "metric_columns"]
                   and not c.endswith("_orig")
                   and np.issubdtype(merged[c].dtype, np.number)]

    for m in metric_cols:
        col_orig = f"{m}_orig"
        if col_orig in merged.columns:
            denom = np.abs(merged[col_orig].replace(0, np.nan))
            merged[f"improvement_{m}"] = ((merged[col_orig] - merged[m]) / denom) * 100

    return merged


def plot_performance_comparison(df_agg, improvements, output_dir):
    """
    Genera gráficos de barras comparativos para RMSE, MAE, MAPE.
    """
    perf = df_agg[df_agg["tipo"] == "performance"].copy()
    if perf.empty:
        return

    metric_cols = [c for c in perf.columns
                   if c not in ["dataset", "tecnica", "modo", "id", "tipo"]
                   and np.issubdtype(perf[c].dtype, np.number)]
    if not metric_cols:
        return

    datasets = perf["dataset"].unique()
    modos = perf["modo"].unique()

    for dataset in datasets:
        df_ds = perf[perf["dataset"] == dataset]
        n_metrics = len(metric_cols)
        n_modos = len(modos)
        fig, axes = plt.subplots(n_metrics, n_modos,
                                 figsize=(6*n_modos, 5*n_metrics),
                                 squeeze=False)
        fig.suptitle(f"Comparación de rendimiento - {dataset}", fontsize=16)

        for i, metric in enumerate(metric_cols):
            for j, modo in enumerate(modos):
                ax = axes[i, j]
                df_sub = df_ds[df_ds["modo"] == modo]
                if df_sub.empty:
                    ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                            transform=ax.transAxes)
                    continue
                pivot = df_sub.pivot(index="id", columns="tecnica", values=metric)
                if "original" not in pivot.columns:
                    ax.text(0.5, 0.5, "Original no disponible", ha="center", va="center",
                            transform=ax.transAxes)
                    continue
                cols = ["original"] + [c for c in pivot.columns if c != "original"]
                pivot = pivot[cols]
                pivot.plot(kind="bar", ax=ax, legend=(i==0 and j==0))
                ax.set_title(f"{metric} - {modo}")
                ax.set_xlabel("Rango")
                ax.set_ylabel(metric)
                ax.grid(axis="y", linestyle="--", alpha=0.7)
                if i == 0 and j == 0:
                    ax.legend(loc="upper right", fontsize="small")

        plt.tight_layout()
        fig.subplots_adjust(top=0.92)
        fname = f"performance_{dataset}_comparison.png"
        plt.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
        plt.savefig(output_dir / fname.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)

    # Gráficos de mejora relativa (usando matplotlib directamente para evitar errores con seaborn)
    if improvements is not None and not improvements.empty:
        imp_perf = improvements[improvements["tipo"] == "performance"].copy()
        if not imp_perf.empty:
            imp_cols = [c for c in imp_perf.columns if c.startswith("improvement_")]
            for imp_col in imp_cols:
                # Agrupar por dataset y técnica, calcular media y std
                grouped = imp_perf.groupby(["dataset", "tecnica"])[imp_col].agg(["mean", "std"]).reset_index()
                for dataset in datasets:
                    df_hm = grouped[grouped["dataset"] == dataset]
                    if df_hm.empty:
                        continue
                    # Ordenar por media descendente para mejor visualización
                    df_hm = df_hm.sort_values("mean", ascending=False)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    # Usar bar plot con barras de error
                    x = np.arange(len(df_hm))
                    ax.bar(x, df_hm["mean"], yerr=df_hm["std"], capsize=5,
                           color=sns.color_palette("Set2", n_colors=len(df_hm)))
                    ax.set_xticks(x)
                    ax.set_xticklabels(df_hm["tecnica"], rotation=45, ha="right")
                    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
                    ax.set_title(f"Mejora relativa promedio de {imp_col} - {dataset}")
                    ax.set_ylabel(f"{imp_col} (% mejora)")
                    ax.grid(axis="y", linestyle="--", alpha=0.5)
                    plt.tight_layout()
                    fname = f"improvement_{imp_col}_{dataset}.png"
                    plt.savefig(output_dir / fname, dpi=300)
                    plt.savefig(output_dir / fname.replace(".png", ".pdf"))
                    plt.close(fig)


def plot_points_comparison(df_agg, output_dir):
    """
    Genera gráficos de barras comparativos para el número total de puntos por zona.
    """
    points = df_agg[df_agg["tipo"] == "points"].copy()
    if points.empty:
        return

    count_cols = [c for c in points.columns
                  if c not in ["dataset", "tecnica", "modo", "id", "tipo"]
                  and np.issubdtype(points[c].dtype, np.number)]
    if not count_cols:
        return
    count_col = count_cols[0]

    datasets = points["dataset"].unique()
    modos = points["modo"].unique()

    for dataset in datasets:
        df_ds = points[points["dataset"] == dataset]
        for modo in modos:
            df_sub = df_ds[df_ds["modo"] == modo]
            if df_sub.empty:
                continue
            pivot = df_sub.pivot(index="id", columns="tecnica", values=count_col)
            if "original" not in pivot.columns:
                continue
            cols = ["original"] + [c for c in pivot.columns if c != "original"]
            pivot = pivot[cols]
            fig, ax = plt.subplots(figsize=(12, 6))
            pivot.plot(kind="bar", ax=ax)
            ax.set_title(f"Distribución de puntos totales por zona - {dataset} - {modo}")
            ax.set_xlabel("Zona")
            ax.set_ylabel("Número total de puntos")
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            ax.legend(title="Técnica")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fname = f"points_distribution_{dataset}_{modo}.png"
            plt.savefig(output_dir / fname, dpi=300)
            plt.savefig(output_dir / fname.replace(".png", ".pdf"))
            plt.close(fig)


def generate_summary_tables(df_agg, improvements, output_dir):
    """Guarda tablas resumen en CSV."""
    perf = df_agg[df_agg["tipo"] == "performance"]
    if not perf.empty:
        metric_cols = [c for c in perf.columns
                       if c not in ["dataset", "tecnica", "modo", "id", "tipo"]
                       and np.issubdtype(perf[c].dtype, np.number)]
        if metric_cols:
            summary = perf.groupby(["dataset", "tecnica", "modo"])[metric_cols].mean().reset_index()
            summary.to_csv(output_dir / "performance_summary.csv", index=False)

    points = df_agg[df_agg["tipo"] == "points"]
    if not points.empty:
        count_cols = [c for c in points.columns
                      if c not in ["dataset", "tecnica", "modo", "id", "tipo"]
                      and np.issubdtype(points[c].dtype, np.number)]
        if count_cols:
            summary = points.groupby(["dataset", "tecnica", "modo"])[count_cols].sum().reset_index()
            summary.to_csv(output_dir / "points_summary.csv", index=False)

    if improvements is not None and not improvements.empty:
        imp_cols = [c for c in improvements.columns if c.startswith("improvement_")]
        if imp_cols:
            summary = improvements.groupby(["dataset", "tecnica", "modo"])[imp_cols].mean().reset_index()
            summary.to_csv(output_dir / "improvement_summary.csv", index=False)


def generate_analysis_report(df_agg, improvements, output_dir):
    """Genera un informe de texto con el análisis detallado."""
    report_path = output_dir / "analysis_report.txt"
    with open(report_path, "w") as f:
        f.write("ANÁLISIS COMPARATIVO DE TÉCNICAS DE BALANCEO\n")
        f.write("="*60 + "\n\n")

        f.write("RESUMEN DE DATOS CARGADOS\n")
        f.write(f"Número total de registros: {len(df_agg)}\n")
        f.write(f"Datasets: {', '.join(df_agg['dataset'].unique())}\n")
        f.write(f"Técnicas: {', '.join(df_agg['tecnica'].unique())}\n")
        f.write(f"Modos de agregación: {', '.join(df_agg['modo'].unique())}\n\n")

        # Análisis de rendimiento
        perf = df_agg[df_agg["tipo"] == "performance"]
        if not perf.empty:
            f.write("RENDIMIENTO (RMSE, MAE, MAPE)\n")
            f.write("-"*40 + "\n")
            metric_cols = [c for c in perf.columns
                           if c not in ["dataset", "tecnica", "modo", "id", "tipo"]
                           and np.issubdtype(perf[c].dtype, np.number)]
            if metric_cols:
                for dataset in perf["dataset"].unique():
                    f.write(f"\nDataset: {dataset}\n")
                    for modo in perf["modo"].unique():
                        f.write(f"  Modo: {modo}\n")
                        df_sub = perf[(perf["dataset"] == dataset) & (perf["modo"] == modo)]
                        if df_sub.empty:
                            continue
                        mean_metrics = df_sub.groupby("tecnica")[metric_cols].mean()
                        for met in metric_cols:
                            best_tec = mean_metrics[met].idxmin()
                            best_val = mean_metrics[met].min()
                            f.write(f"    Mejor para {met}: {best_tec} con {met}={best_val:.4f}\n")
                        f.write("\n")

        # Análisis de puntos
        points = df_agg[df_agg["tipo"] == "points"]
        if not points.empty:
            f.write("\nNÚMERO DE PUNTOS TOTALES POR ZONA\n")
            f.write("-"*40 + "\n")
            count_cols = [c for c in points.columns
                          if c not in ["dataset", "tecnica", "modo", "id", "tipo"]
                          and np.issubdtype(points[c].dtype, np.number)]
            if count_cols:
                count_col = count_cols[0]
                for dataset in points["dataset"].unique():
                    f.write(f"\nDataset: {dataset}\n")
                    df_orig = points[(points["dataset"] == dataset) & (points["tecnica"] == "original")]
                    if df_orig.empty:
                        continue
                    for modo in points["modo"].unique():
                        f.write(f"  Modo: {modo}\n")
                        df_sub = points[(points["dataset"] == dataset) & (points["modo"] == modo)]
                        if df_sub.empty:
                            continue
                        pivot = df_sub.pivot(index="id", columns="tecnica", values=count_col)
                        if "original" not in pivot.columns:
                            continue
                        pivot_abs = pivot.drop(columns=["original"]).subtract(pivot["original"], axis=0).abs()
                        if not pivot_abs.empty:
                            avg_abs_diff = pivot_abs.mean()
                            best_tec = avg_abs_diff.idxmin()
                            best_diff = avg_abs_diff.min()
                            f.write(f"    Técnica más cercana al original: {best_tec} (diferencia media absoluta={best_diff:.2f})\n")

        # Mejoras relativas
        if improvements is not None and not improvements.empty:
            f.write("\nMEJORAS RELATIVAS (respecto a original)\n")
            f.write("-"*40 + "\n")
            imp_cols = [c for c in improvements.columns if c.startswith("improvement_")]
            if imp_cols:
                for col in imp_cols:
                    f.write(f"\n{col}:\n")
                    grouped = improvements.groupby(["dataset", "tecnica"])[col].mean().reset_index()
                    for dataset in grouped["dataset"].unique():
                        f.write(f"  Dataset: {dataset}\n")
                        df_ds = grouped[grouped["dataset"] == dataset]
                        df_ds_sorted = df_ds.sort_values(col, ascending=False)
                        for _, row in df_ds_sorted.iterrows():
                            f.write(f"    {row['tecnica']}: {row[col]:.2f}%\n")
        f.write("\nFIN DEL ANÁLISIS\n")


def main():
    print("Iniciando análisis de métricas...")
    experiments = find_experiment_dirs(BASE_OUTPUT)
    print(f"Se encontraron {len(experiments)} experimentos.")

    df_combined = load_all_experiments(experiments)
    if df_combined is None:
        print("No se pudieron cargar datos. Saliendo.")
        return

    df_agg = aggregate_metrics_by_id(df_combined)
    if df_agg is None or df_agg.empty:
        print("No se pudo agregar datos. Saliendo.")
        return
    print(f"Datos agregados: {len(df_agg)} filas.")

    improvements = compute_relative_improvement(df_agg)
    print("Mejoras relativas calculadas.")

    output_dir = RESULTS_DIR
    print(f"Generando gráficos en {output_dir}...")
    plot_performance_comparison(df_agg, improvements, output_dir)
    plot_points_comparison(df_agg, output_dir)

    generate_summary_tables(df_agg, improvements, output_dir)
    generate_analysis_report(df_agg, improvements, output_dir)

    print(f"Análisis completado. Resultados guardados en {output_dir}")


if __name__ == "__main__":
    main()