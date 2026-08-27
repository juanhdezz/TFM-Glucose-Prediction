#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — Orquestador RELIEF-T1D Analysis Pipeline (con análisis intra‑familia).
Uso:
    python main.py                    pipeline completo
    python main.py --only load        solo carga y validación
    python main.py --only stats       solo estadísticos
    python main.py --only viz         solo figuras
    python main.py --only tables      solo tablas LaTeX/CSV
    python main.py --only dashboards  solo dashboards
    python main.py --debug            verbose
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analysis.loader import load_all_experiments
from analysis.preprocessing import build_master_table, build_fold_table, compute_deltas
from analysis.stats import run_all_stats
from analysis.tables import generate_all_tables
from analysis.dashboards import generate_all_dashboards
from analysis.viz.heatmaps import plot_all_heatmaps
from analysis.viz.dumbbell import plot_all_dumbbells
from analysis.viz.distributions import plot_all_distributions
from analysis.viz.rankings import plot_all_rankings
from analysis.viz.profiles import plot_all_profiles
from analysis.config import DATASET_LABELS 
from analysis.config import DATASET_ORDER, TECHNIQUE_LABELS, DIMENSION_LABELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("analysis_run.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Pipeline de análisis RELIEF-T1D")
    p.add_argument("--only",
                   choices=["load","stats","viz","tables","dashboards"],
                   default=None)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    log.info("=" * 55)
    log.info("  RELIEF-T1D  —  Analysis Pipeline (incl. intra‑family)")
    log.info("=" * 55)

    # ── 1. Load ──
    log.info("[1] Cargando experimentos...")
    raw = load_all_experiments()
    if not raw:
        log.error("No se encontraron experimentos. Verifica OUTPUT_ROOT en analysis/config.py")
        sys.exit(1)

    master    = build_master_table(raw)
    fold_long = build_fold_table(raw)
    master    = compute_deltas(master)

    log.info(f"    master   : {master.shape[0]:,} filas  |  "
             f"{master['condition'].nunique()} condiciones  |  "
             f"{master['dataset'].nunique()} datasets")
    log.info(f"    fold_long: {fold_long.shape[0]:,} filas")

    if args.only == "load":
        log.info("\n=== SUMMARY ===")
        for ds in master["dataset"].unique():
            ds_data = master[(master["dataset"]==ds)&(master["metric"]=="RMSE")&(master["range"]=="ENTIRE")]
            log.info(f"  {ds}: {ds_data['condition'].nunique()} conditions, "
                     f"RMSE range [{ds_data['mean'].min():.2f} – {ds_data['mean'].max():.2f}]")
        return

    # ── 2. Stats ──
    stats_results = {}
    if args.only in (None, "stats"):
        log.info("[2] Análisis estadístico (global + intra‑familia)...")
        stats_results = run_all_stats(master, fold_long)

    if args.only == "stats":
        return

    borda_df      = stats_results.get("borda", None)
    borda_size    = stats_results.get("borda_size", None)
    borda_mechanism = stats_results.get("borda_mechanism", None)

    # ── 3. Viz ──
    if args.only in (None, "viz"):
        log.info("[3] Generando figuras (globales e intra‑familia)...")
        plot_all_heatmaps(master)
        plot_all_dumbbells(master)
        plot_all_distributions(fold_long)
        # Los rankings usan borda_df global y los nuevos borda_size/mechanism
        # En tu script principal, antes de plot_all_rankings
        print("=== DIAGNÓSTICO ===")
        print("1. DATASET_LABELS:", DATASET_LABELS)
        print("2. Valores únicos en master['dataset']:", master['dataset'].unique())
        print("3. Valores únicos en master['condition']:", master['condition'].unique())

        # Si hay algún valor incorrecto, lo verás aquí
        for ds in master['dataset'].unique():
            if ds not in DATASET_LABELS:
                print(f"⚠️ Dataset '{ds}' no está en DATASET_LABELS!")
        plot_all_rankings(master, fold_long, borda_df=borda_df)
        plot_all_profiles(master)

    if args.only == "viz":
        return

    # ── 4. Tables ──
    if args.only in (None, "tables"):
        log.info("[4] Generando tablas (globales e intra‑familia)...")
        generate_all_tables(master, fold_long)

    if args.only == "tables":
        return

    # ── 5. Dashboards ──
    if args.only in (None, "dashboards"):
        log.info("[5] Generando dashboards (globales e intra‑familia)...")
        generate_all_dashboards(master, fold_long, borda_df=borda_df)

    log.info("=" * 55)
    log.info("  Pipeline completado → outputs/")
    log.info("=" * 55)


if __name__ == "__main__":
    main()