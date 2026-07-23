# -*- coding: utf-8 -*-
"""
tables.py — Tablas LaTeX y CSV para el TFM.

T1  main_results_{metric}.tex / .csv     mean ± std por dataset × condición
T2  ranking_table.tex / .csv             posiciones de ranking por dataset
T3  delta_table_{metric}.tex             tabla de deltas porcentuales
T4  clarke_zones_table.tex               % puntos en zona A y A+B por condición

Versiones intra‑familia.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from analysis.config import (
    TABLES_DIR, DATASET_ORDER, DATASET_LABELS,
    TECHNIQUE_ORDER, TECHNIQUE_LABELS, RANGE_ORDER, RANGE_LABELS,
    ANALYSIS_METRICS, LOWER_IS_BETTER, PRIMARY_METRIC,
    FAMILY_SIZE_LABELS, FAMILY_MECHANISM_LABELS,
)

log = logging.getLogger(__name__)


def _cond_label(cond):
    if cond == "original": return "Original"
    parts = cond.replace("balanced_","").split("_",1)
    dim  = parts[0].capitalize()
    tech = TECHNIQUE_LABELS.get(parts[1] if len(parts)>1 else cond, cond)
    return f"{dim} · {tech}"

def _sort_conds(conds):
    def _k(c):
        if c == "original": return (0,0,0)
        parts = c.replace("balanced_","").split("_",1)
        dim = parts[0]; tech = parts[1] if len(parts)>1 else ""
        return (1, {"age":0,"sex":1}.get(dim,2),
                TECHNIQUE_ORDER.index(tech) if tech in TECHNIQUE_ORDER else 99)
    return sorted(conds, key=_k)


# ── T1: Main results table ────────────────────────────────────────────────────

def generate_main_results_table(master: pd.DataFrame, metric: str = "RMSE",
                                suffix: str = "") -> None:
    sub = master[master["metric"] == metric].copy()
    if sub.empty:
        log.warning(f"main_results_table: sin datos para {metric}")
        return

    ranges   = [r for r in RANGE_ORDER if r in sub["range"].unique()]
    datasets = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
    conds    = _sort_conds(sub["condition"].unique().tolist())

    # ── CSV ──────────────────────────────────────────────────────────
    rows = []
    for cond in conds:
        row = {"Condition": _cond_label(cond)}
        for ds in datasets:
            for rng in ranges:
                cell = sub[(sub["condition"]==cond)&(sub["dataset"]==ds)&(sub["range"]==rng)]
                if cell.empty:
                    row[f"{DATASET_LABELS.get(ds,ds)}_{RANGE_LABELS.get(rng,rng)}"] = "—"
                else:
                    m = cell["mean"].values[0]
                    s = cell["std"].values[0]
                    row[f"{DATASET_LABELS.get(ds,ds)}_{RANGE_LABELS.get(rng,rng)}"] = \
                        f"{m:.2f} ± {s:.2f}" if not np.isnan(m) else "—"
        rows.append(row)

    df_csv = pd.DataFrame(rows)
    csv_name = f"main_results_{metric}{'_'+suffix if suffix else ''}.csv"
    df_csv.to_csv(TABLES_DIR / csv_name, index=False)
    log.info(f"  CSV: {csv_name}")

    # ── LaTeX ────────────────────────────────────────────────────────
    n_ranges = len(ranges)
    tex_name = f"main_results_{metric}{'_'+suffix if suffix else ''}.tex"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Results by condition — " + metric + r" (mean $\pm$ std across 5 folds)}",
        r"\label{tab:results_" + metric.lower() + (suffix if suffix else "") + r"}",
        r"\begin{tabular}{l" + "r" * (len(datasets) * n_ranges) + r"}",
        r"\toprule",
    ]

    # Header row 1: dataset names spanning range columns
    ds_headers = [""]
    for ds in datasets:
        ds_headers.append(
            r"\multicolumn{" + str(n_ranges) + r"}{c}{\textbf{" + DATASET_LABELS.get(ds, ds) + r"}}"
        )
    lines.append(" & ".join(ds_headers) + r" \\")

    # Cmidrule per dataset block
    cmidrules = []
    col_start = 2
    for _ in datasets:
        cmidrules.append(f"\\cmidrule(lr){{{col_start}-{col_start + n_ranges - 1}}}")
        col_start += n_ranges
    lines.append("".join(cmidrules))

    # Header row 2: range names
    rng_headers = [r"\textbf{Condition}"]
    for ds in datasets:
        for rng in ranges:
            rng_headers.append(r"\textbf{" + RANGE_LABELS.get(rng, rng) + r"}")
    lines.append(" & ".join(rng_headers) + r" \\")
    lines.append(r"\midrule")

    # Find best (non-original) value per (dataset, range) for bold
    asc = metric in LOWER_IS_BETTER
    best_vals = {}
    for ds in datasets:
        for rng in ranges:
            non_orig = sub[
                (~sub["is_original"]) & (sub["dataset"] == ds) & (sub["range"] == rng)
            ]["mean"].dropna()
            if not non_orig.empty:
                best_vals[(ds, rng)] = non_orig.min() if asc else non_orig.max()

    # Data rows
    prev_dim = None
    for cond in conds:
        # Dimension separator
        if cond != "original":
            parts = cond.replace("balanced_", "").split("_", 1)
            dim = parts[0] if len(parts) > 0 else ""
            if dim != prev_dim:
                if prev_dim is not None:
                    lines.append(r"\midrule")
                prev_dim = dim

        cells = [
            r"\textit{" + _cond_label(cond) + r"}" if cond == "original" else _cond_label(cond)
        ]

        for ds in datasets:
            for rng in ranges:
                cell = sub[
                    (sub["condition"] == cond) &
                    (sub["dataset"] == ds) &
                    (sub["range"] == rng)
                ]
                if cell.empty:
                    cells.append("—")
                    continue
                m = cell["mean"].values[0]
                s = cell["std"].values[0]
                if np.isnan(m):
                    cells.append("—")
                    continue
                val_str = f"{m:.2f} \\pm {s:.2f}"
                # Bold if best non-original
                if (ds, rng) in best_vals and not cell["is_original"].values[0]:
                    if abs(m - best_vals[(ds, rng)]) < 1e-6:
                        val_str = r"\mathbf{" + val_str + r"}"
                cells.append(f"${val_str}$")

        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex_path = TABLES_DIR / tex_name
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  LaTeX: {tex_name}")


# ── T2: Ranking table ─────────────────────────────────────────────────────────

def generate_ranking_table(master: pd.DataFrame, metric: str = "RMSE",
                           rng: str = "ENTIRE", suffix: str = "") -> None:
    sub = master[(master["metric"]==metric)&(master["range"]==rng)].copy()
    if sub.empty: return
    datasets = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
    conds    = _sort_conds(sub["condition"].unique().tolist())
    asc      = metric in LOWER_IS_BETTER

    rank_data = {}
    for ds in datasets:
        ds_sub = sub[sub["dataset"]==ds].sort_values("mean", ascending=asc)
        rank_data[ds] = {row["condition"]: (i+1, row["mean"], row["std"])
                         for i, (_, row) in enumerate(ds_sub.iterrows())}

    # ── CSV ──────────────────────────────────────────────────────────
    rows = []
    for cond in conds:
        row = {"Condition": _cond_label(cond)}
        for ds in datasets:
            if cond in rank_data.get(ds,{}):
                rank, mean, std = rank_data[ds][cond]
                row[DATASET_LABELS.get(ds,ds)] = f"#{rank}  ({mean:.2f}±{std:.2f})"
            else:
                row[DATASET_LABELS.get(ds,ds)] = "—"
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = TABLES_DIR / f"ranking_{metric}_{rng}{'_'+suffix if suffix else ''}.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"  CSV: {out_csv.name}")

    # ── LaTeX ────────────────────────────────────────────────────────
    tex_name = f"ranking_{metric}_{rng}{'_'+suffix if suffix else ''}.tex"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Ranking by condition — " + f"{metric} · {RANGE_LABELS.get(rng, rng)}" + r"  (1 = best)}",
        r"\label{tab:ranking_" + f"{metric}_{rng}{'_'+suffix if suffix else ''}".lower() + r"}",
        r"\begin{tabular}{l" + "c" * len(datasets) + r"}",
        r"\toprule",
        r"\textbf{Condition} & " + " & ".join(r"\textbf{" + DATASET_LABELS.get(d,d) + r"}" for d in datasets) + r" \\",
        r"\midrule",
    ]

    prev_dim = None
    for cond in conds:
        if cond != "original":
            parts = cond.replace("balanced_","").split("_",1)
            dim = parts[0] if len(parts)>0 else ""
            if dim != prev_dim and prev_dim is not None:
                lines.append(r"\midrule")
            prev_dim = dim

        cells = [_cond_label(cond)]
        for ds in datasets:
            if cond in rank_data.get(ds, {}):
                rank, mean, std = rank_data[ds][cond]
                cells.append(f"\\#{rank} ({mean:.2f}$\\pm${std:.2f})")
            else:
                cells.append("—")
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = TABLES_DIR / tex_name
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  LaTeX: {tex_name}")


# ── T3: Delta table ───────────────────────────────────────────────────────────

def generate_delta_table(master: pd.DataFrame, metric: str = "RMSE",
                         rng: str = "ENTIRE", suffix: str = "") -> None:
    sub = master[(~master["is_original"])&(master["metric"]==metric)&(master["range"]==rng)].copy()
    if sub.empty: return

    pivot = sub.pivot_table(index="condition", columns="dataset",
                            values="improvement_pct", aggfunc="mean")
    datasets = [d for d in DATASET_ORDER if d in pivot.columns]
    conds    = _sort_conds(list(pivot.index))
    pivot    = pivot[datasets]

    # ── CSV ──────────────────────────────────────────────────────────
    records = []
    for cond in conds:
        row = {"Condition": _cond_label(cond)}
        for ds in datasets:
            v = pivot.loc[cond, ds] if cond in pivot.index else np.nan
            row[DATASET_LABELS.get(ds,ds)] = f"{v:+.1f}%" if not np.isnan(v) else "—"
        records.append(row)
    df = pd.DataFrame(records)
    out_csv = TABLES_DIR / f"delta_{metric}_{rng}{'_'+suffix if suffix else ''}.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"  CSV: {out_csv.name}")

    # ── LaTeX ────────────────────────────────────────────────────────
    tex_name = f"delta_{metric}_{rng}{'_'+suffix if suffix else ''}.tex"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Improvement (\%) vs. baseline  |  "
        + f"{metric} · {RANGE_LABELS.get(rng, rng)}"
        + r"  $\uparrow$ positive = improvement}",
        r"\label{tab:delta_" + f"{metric}_{rng}{'_'+suffix if suffix else ''}".lower() + r"}",
        r"\begin{tabular}{l" + "r" * len(datasets) + r"}",
        r"\toprule",
        r"\textbf{Condition} & " + " & ".join(r"\textbf{" + DATASET_LABELS.get(d,d) + r"}" for d in datasets) + r" \\",
        r"\midrule",
    ]

    prev_dim = None
    for cond in conds:
        parts = cond.replace("balanced_","").split("_",1)
        dim = parts[0] if len(parts)>0 else ""
        if dim != prev_dim and prev_dim is not None:
            lines.append(r"\midrule")
        prev_dim = dim

        cells = [_cond_label(cond)]
        for ds in datasets:
            v = pivot.loc[cond, ds] if cond in pivot.index else np.nan
            if np.isnan(v):
                cells.append("—")
            else:
                sign = "+" if v > 0 else ""
                col = "OliveGreen" if v > 0 else ("BrickRed" if v < 0 else "black")
                cells.append(r"\textcolor{" + col + r"}{" + f"{sign}{v:.1f}\\%}}")
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = TABLES_DIR / tex_name
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  LaTeX: {tex_name}")


# ── T4: Clarke zones table ────────────────────────────────────────────────────

def generate_clarke_table(master: pd.DataFrame, suffix: str = "") -> None:
    sub = master[(master["metric"].isin(["A","A + B"]))&(master["range"]=="ENTIRE")].copy()
    if sub.empty: return
    datasets = [d for d in DATASET_ORDER if d in sub["dataset"].unique()]
    conds    = _sort_conds(sub["condition"].unique().tolist())

    # ── CSV ──────────────────────────────────────────────────────────
    rows = []
    for cond in conds:
        row = {"Condition": _cond_label(cond)}
        for ds in datasets:
            for zone in ["A","A + B"]:
                cell = sub[(sub["condition"]==cond)&(sub["dataset"]==ds)&(sub["metric"]==zone)]
                if cell.empty:
                    row[f"{DATASET_LABELS.get(ds,ds)} Zone {zone}"] = "—"
                else:
                    m = cell["mean"].values[0]; s = cell["std"].values[0]
                    row[f"{DATASET_LABELS.get(ds,ds)} Zone {zone}"] = f"{m:.1f} ± {s:.1f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = TABLES_DIR / f"clarke_zones{'_'+suffix if suffix else ''}.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"  CSV: {out_csv.name}")

    # ── LaTeX ────────────────────────────────────────────────────────
    tex_name = f"clarke_zones{'_'+suffix if suffix else ''}.tex"
    n_cols = len(datasets) * 2  # A y A+B por dataset
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Clarke Error Grid — \% points in Zone A and A+B by condition}",
        r"\label{tab:clarke" + (suffix if suffix else "") + r"}",
        r"\begin{tabular}{l" + "r" * n_cols + r"}",
        r"\toprule",
    ]

    # Header row
    ds_headers = [r"\textbf{Condition}"]
    for ds in datasets:
        ds_headers.append(r"\multicolumn{2}{c}{\textbf{" + DATASET_LABELS.get(ds, ds) + r"}}")
    lines.append(" & ".join(ds_headers) + r" \\")
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")

    zone_headers = [""]
    for ds in datasets:
        zone_headers.append(r"\textbf{Zone A}")
        zone_headers.append(r"\textbf{Zone A+B}")
    lines.append(" & ".join(zone_headers) + r" \\")
    lines.append(r"\midrule")

    prev_dim = None
    for cond in conds:
        if cond != "original":
            parts = cond.replace("balanced_","").split("_",1)
            dim = parts[0] if len(parts)>0 else ""
            if dim != prev_dim and prev_dim is not None:
                lines.append(r"\midrule")
            prev_dim = dim

        cells = [_cond_label(cond)]
        for ds in datasets:
            for zone in ["A", "A + B"]:
                cell = sub[(sub["condition"]==cond)&(sub["dataset"]==ds)&(sub["metric"]==zone)]
                if cell.empty:
                    cells.append("—")
                else:
                    m = cell["mean"].values[0]
                    s = cell["std"].values[0]
                    cells.append(f"{m:.1f} \\pm {s:.1f}")
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = TABLES_DIR / tex_name
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  LaTeX: {tex_name}")


# ── Intra‑family tables ──────────────────────────────────────────────────────

def generate_all_tables(master: pd.DataFrame, fold_long: pd.DataFrame = None) -> None:
    log.info("  Generando tablas...")

    # Globales
    for metric in ["RMSE", "MAE"]:
        generate_main_results_table(master, metric=metric)
        generate_ranking_table(master, metric=metric, rng="ENTIRE")
        generate_ranking_table(master, metric=metric, rng="TBR_2")
        generate_delta_table(master, metric=metric, rng="ENTIRE")
        generate_delta_table(master, metric=metric, rng="TBR_1")
    generate_clarke_table(master)

    # Intra‑familia
    for family_col in ["family_size", "family_mechanism"]:
        for fam in master[family_col].dropna().unique():
            sub = master[master[family_col] == fam]
            if sub.empty: continue
            suf = f"{family_col}_{fam}"
            for metric in ["RMSE", "MAE"]:
                generate_main_results_table(sub, metric=metric, suffix=suf)
                generate_ranking_table(sub, metric=metric, rng="ENTIRE", suffix=suf)
                generate_ranking_table(sub, metric=metric, rng="TBR_2", suffix=suf)
                generate_delta_table(sub, metric=metric, rng="ENTIRE", suffix=suf)
                generate_delta_table(sub, metric=metric, rng="TBR_1", suffix=suf)
            generate_clarke_table(sub, suffix=suf)

    log.info("  Tablas completadas.")