# ─────────────────────────────────────────────────────────────────────────────
# 2b-trainset_balancing.py
#
# Genera datasets balanceados por dimensión demográfica (edad / sexo), uno por
# cada combinación (fold, grupo, técnica).
#
# Metodología por fold i:
#   1. Tomar el dataset completo de ventanas.
#   2. Quedarse solo con las filas cuyo fold_i ∈ {train, val, test}.
#   3. Separar train / val+test.
#   4. Añadir columnas demográficas a train (join con patient_info).
#   5. Balancear train según la técnica elegida sobre la columna de grupo.
#   6. Concatenar train_balanceado + val+test intactos.
#   7. Eliminar columnas auxiliares demográficas y TODAS las columnas fold_*.
#   8. Añadir columna "split" con valor "train" / "val" / "test".
#   9. Guardar como parquet.
#
# Formato de salida (un archivo por fold × grupo × técnica):
#   data/<DATASET>/balanced_outputs/
#     <windows_stem>_fold<i>_<group>_<technique>.parquet
#
# Columnas de salida: x0..x7, y, x_date_7, x_time_7, patient_id, split
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# ─── Rutas y constantes globales ─────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

DEFAULT_SEED      = 42
DEFAULT_HORIZONS  = [4]
DEFAULT_FOLDS     = [0, 1, 2, 3, 4]
DEFAULT_GROUPS    = ["sex", "age"]
DEFAULT_TECHNIQUES = [
    "undersampling",
    "oversampling",
    "patient_aware_undersampling",
    "smote",
    "jittering"
    "undersampling_oversampling", # Mismo tamaño , buscamos proporcion equitativa 50% / 50% entre sexos o 25% entre grupos de edad. Reduccion aleatroia , aumentando aleatoriamente.
    "undersampling_smote", # Mismo tamaño , buscamos proporcion equitativa 50% / 50% entre sexos o 25% entre grupos de edad. reduccion aleatoria , aumentando con smote.
]
DEFAULT_SENSOR_LIMITS = {
    "DIATREND":           (39.0, 401.0),
    "REPLACE-BG":         (39.0, 401.0),
    "T1DiabetesGranada":  (40.0, 500.0),
}

AGE_BINS   = [-np.inf, 18, 30, 45, 60, np.inf]
AGE_LABELS = ["<=18", "19-30", "31-45", "46-60", ">60"]
SEX_MAP    = {"F": "F", "M": "M"}

# Columnas que se conservan en el archivo de salida (sin fold_*)
KEEP_COLUMNS = [f"x{i}" for i in range(8)] + ["y", "x_date_7", "x_time_7", "patient_id"]


# ─── Dataclass de assets ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class DatasetAssets:
    dataset_name:      str
    dataset_dir:       Path
    patient_info_file: Path
    output_dir:        Path


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera parquets balanceados (un archivo por fold × grupo × técnica)."
    )
    parser.add_argument("--datasets",   nargs="*", default=None,
                        help="Carpetas de dataset a procesar (por defecto todas bajo data/).")
    parser.add_argument("--groups",     nargs="*", default=DEFAULT_GROUPS,
                        choices=DEFAULT_GROUPS,
                        help="Dimensiones demográficas a balancear.")
    parser.add_argument("--techniques", nargs="*", default=DEFAULT_TECHNIQUES,
                        choices=DEFAULT_TECHNIQUES,
                        help="Técnicas de balanceo a aplicar.")
    parser.add_argument("--horizons",   nargs="*", type=int, default=DEFAULT_HORIZONS,
                        help="Horizontes de predicción a procesar.")
    parser.add_argument("--folds",      nargs="*", type=int, default=DEFAULT_FOLDS,
                        help="Índices de fold a balancear.")
    parser.add_argument("--seed",       type=int, default=DEFAULT_SEED)
    
    parser.add_argument("--output-root", type=str, default=str(DATA_DIR),
                        help="Raíz que contiene las carpetas de dataset.")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Sobreescribir archivos existentes.")
    return parser.parse_args()


# ─── Helpers de disco ────────────────────────────────────────────────────────
def discover_dataset_dirs(root: Path, dataset_names: Optional[Sequence[str]]) -> List[Path]:
    if dataset_names:
        return [root / name for name in dataset_names]
    return [p for p in sorted(root.iterdir()) if p.is_dir()]


def resolve_patient_info(dataset_dir: Path) -> Path:
    for candidate in [
        dataset_dir / "patient_info.parquet",
        dataset_dir / "Patient_info.parquet",
        dataset_dir / "patient_info.csv",
        dataset_dir / "Patient_info.csv",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No patient info file found in {dataset_dir}")


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


# ─── Helpers de columnas ─────────────────────────────────────────────────────
def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def find_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    normalized = {normalize_name(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_name(cand)
        if key in normalized:
            return normalized[key]
    raise KeyError(f"None of these columns were found: {candidates}")


def normalize_patient_key(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


# ─── Demografía ──────────────────────────────────────────────────────────────
def infer_age_series(patient_info: pd.DataFrame) -> pd.Series:
    """Devuelve una Serie con la edad numérica, inferida desde 'Age' o 'Birth_year'."""
    for candidate in ["Age", "age"]:
        try:
            age_col = find_column(patient_info, [candidate])
            return pd.to_numeric(patient_info[age_col], errors="coerce")
        except KeyError:
            continue

    birth_col = None
    for candidate in ["Birth_year", "birth_year", "Birth Year", "birth year"]:
        try:
            birth_col = find_column(patient_info, [candidate])
            break
        except KeyError:
            continue

    if birth_col is None:
        raise KeyError("No se pudo inferir la edad: falta 'Age' y 'Birth_year' en patient_info.")

    birth_year = pd.to_numeric(patient_info[birth_col], errors="coerce")
    ref_year   = pd.Series(datetime.now().year, index=patient_info.index, dtype="float64")

    for candidate in [
        "Final_measurement_date", "final_measurement_date",
        "Initial_measurement_date", "initial_measurement_date",
    ]:
        try:
            date_col = find_column(patient_info, [candidate])
        except KeyError:
            continue
        parsed = pd.to_datetime(patient_info[date_col], errors="coerce")
        if parsed.notna().any():
            ref_year = parsed.dt.year.astype("float64")
            break

    return ref_year - birth_year


def build_demographic_lookup(patient_info: pd.DataFrame) -> pd.DataFrame:
    """Devuelve DataFrame indexado por _patient_key con sex_group, age, age_group."""
    meta = patient_info.copy()
    meta_patient_col = find_column(meta, ["patient_id", "Patient_ID", "patientid"])
    meta["_patient_key"] = normalize_patient_key(meta[meta_patient_col])

    sex_col = find_column(meta, ["Sex", "sex"])
    meta["sex_group"] = (
        meta[sex_col].astype(str).str.strip().str.upper().map(SEX_MAP).fillna("Unknown")
    )
    meta["age"]       = infer_age_series(meta)
    meta["age_group"] = (
        pd.cut(meta["age"], bins=AGE_BINS, labels=AGE_LABELS,
               include_lowest=True, right=True)
        .astype("object")
        .fillna("Unknown")
    )

    return (
        meta[["_patient_key", "sex_group", "age", "age_group"]]
        .drop_duplicates(subset=["_patient_key"])
        .set_index("_patient_key")
    )


def attach_demographics(train_df: pd.DataFrame, demo_lookup: pd.DataFrame) -> pd.DataFrame:
    """Une el DataFrame de train con las columnas demográficas."""
    windows_patient_col = find_column(train_df, ["patient_id", "Patient_ID", "patientid"])
    keys = normalize_patient_key(train_df[windows_patient_col])
    result = train_df.copy()
    result["sex_group"] = keys.map(demo_lookup["sex_group"]).fillna("Unknown").values
    result["age"]       = keys.map(demo_lookup["age"]).values
    result["age_group"] = keys.map(demo_lookup["age_group"]).fillna("Unknown").values
    return result


# ─── Técnicas de balanceo ────────────────────────────────────────────────────
def clip_feature_columns(
    df: pd.DataFrame, feature_columns: Sequence[str], low: float, high: float
) -> pd.DataFrame:
    clipped = df.copy()
    for col in feature_columns:
        clipped[col] = pd.to_numeric(clipped[col], errors="coerce").clip(lower=low, upper=high)
    return clipped


def undersample_group(
    group_df: pd.DataFrame, target: int, rng: np.random.Generator
) -> pd.DataFrame:
    if len(group_df) <= target:
        return group_df.copy()
    chosen = rng.choice(group_df.index.to_numpy(), size=target, replace=False)
    return group_df.loc[chosen].copy()


def oversample_group(
    group_df: pd.DataFrame, target: int, rng: np.random.Generator
) -> pd.DataFrame:
    if len(group_df) >= target:
        return group_df.copy()
    chosen = rng.choice(group_df.index.to_numpy(), size=target, replace=True)
    return group_df.loc[chosen].copy()


def patient_aware_undersample_group(
    group_df: pd.DataFrame, target: int, rng: np.random.Generator
) -> pd.DataFrame:
    if len(group_df) <= target:
        return group_df.copy()

    patient_counts = group_df.groupby("patient_id").size().sort_values(ascending=False)
    patient_cap    = max(1, int(math.ceil(target / len(patient_counts))))
    sampled_parts: List[pd.DataFrame] = []

    for _, patient_df in group_df.groupby("patient_id", sort=False):
        take   = min(len(patient_df), patient_cap)
        chosen = rng.choice(patient_df.index.to_numpy(), size=take, replace=False)
        sampled_parts.append(group_df.loc[chosen].copy())

    sampled = pd.concat(sampled_parts, ignore_index=False)
    if len(sampled) > target:
        chosen  = rng.choice(sampled.index.to_numpy(), size=target, replace=False)
        sampled = sampled.loc[chosen].copy()
    return sampled.copy()


def smote_group(
    group_df: pd.DataFrame,
    target: int,
    rng: np.random.Generator,
    feature_columns: Sequence[str],
    sensor_limits: Tuple[float, float],
) -> pd.DataFrame:
    current = len(group_df)
    if current >= target:
        return group_df.copy()

    needed = target - current
    if current == 1:
        synthetic = pd.concat([group_df.copy()] * needed, ignore_index=True)
        return clip_feature_columns(synthetic, feature_columns, *sensor_limits)

    features     = group_df[list(feature_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    feature_mean = np.nanmean(features, axis=0)
    feature_std  = np.nanstd(features, axis=0)
    feature_std[feature_std == 0] = 1.0
    scaled       = (features - feature_mean) / feature_std

    n_neighbors = min(5, current - 1)
    nn          = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(scaled)
    neighbors   = nn.kneighbors(scaled, return_distance=False)
    base_rows   = group_df.reset_index(drop=True)

    synthetic_rows: List[pd.Series] = []
    for _ in range(needed):
        base_idx           = int(rng.integers(0, current))
        candidate_nbrs     = neighbors[base_idx][1:]
        neighbor_idx       = base_idx if len(candidate_nbrs) == 0 else int(rng.choice(candidate_nbrs))
        lam                = float(rng.random())
        synthetic_vector   = features[base_idx] + lam * (features[neighbor_idx] - features[base_idx])

        synthetic_row = base_rows.iloc[base_idx].copy()
        for idx, col in enumerate(feature_columns):
            synthetic_row[col] = synthetic_vector[idx]
        synthetic_rows.append(synthetic_row)

    synthetic_df = pd.DataFrame(synthetic_rows)
    synthetic_df = clip_feature_columns(synthetic_df, feature_columns, *sensor_limits)
    return pd.concat([group_df, synthetic_df], ignore_index=True)


def jitter_group(
    group_df: pd.DataFrame,
    target: int,
    rng: np.random.Generator,
    feature_columns: Sequence[str],
    sensor_limits: Tuple[float, float],
    jitter_scale: float = 0.20,
) -> pd.DataFrame:
    current = len(group_df)
    if current >= target:
        return group_df.copy()

    needed   = target - current
    features = group_df[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    stds     = features.std(axis=0, ddof=0).replace(0, 1.0).to_numpy(dtype=float)
    base_rows = group_df.reset_index(drop=True)

    synthetic_rows: List[pd.Series] = []
    for _ in range(needed):
        base_idx = int(rng.integers(0, current))
        base_row = base_rows.iloc[base_idx].copy()
        noise    = rng.normal(loc=0.0, scale=stds * jitter_scale, size=len(feature_columns))
        for idx, col in enumerate(feature_columns):
            base_row[col] = pd.to_numeric(base_row[col], errors="coerce") + float(noise[idx])
        synthetic_rows.append(base_row)

    synthetic_df = pd.DataFrame(synthetic_rows)
    synthetic_df = clip_feature_columns(synthetic_df, feature_columns, *sensor_limits)
    return pd.concat([group_df, synthetic_df], ignore_index=True)


# ─── Selección de targets de conteo ──────────────────────────────────────────
def select_target_counts(
    counts: pd.Series,
    technique: str,
    total_rows: int,
    reference_props: Optional[Mapping[str, float]] = None,
) -> Dict[str, int]:
    if technique in {"undersampling", "patient_aware_undersampling"}:
        target = int(counts.min())
        return {group: target for group in counts.index}

    if technique in {"oversampling", "smote", "jittering"}:
        target = int(counts.max())
        return {group: target for group in counts.index}

    if technique == "reference_proportional":
        if not reference_props:
            raise ValueError("reference_proportional requiere un archivo de referencia.")

        present_groups    = [g for g in counts.index if g in reference_props]
        missing_from_ref  = [g for g in counts.index if g not in reference_props]

        if missing_from_ref:
            print(
                f"    [WARN] reference_proportional: grupos {missing_from_ref} no están en "
                "el archivo de referencia — se mantienen en su tamaño actual."
            )

        present_sum = sum(reference_props[g] for g in present_groups)
        if present_sum == 0:
            raise ValueError("La suma de proporciones de referencia para los grupos presentes es 0.")

        raw_targets = {g: (reference_props[g] / present_sum) * total_rows for g in present_groups}
        floors      = {g: int(math.floor(v)) for g, v in raw_targets.items()}
        remainder   = total_rows - sum(floors.values())
        ranked      = sorted(present_groups, key=lambda g: raw_targets[g] - floors[g], reverse=True)
        targets     = floors.copy()
        for g in ranked[:remainder]:
            targets[g] += 1
        for g in missing_from_ref:
            targets[g] = int(counts[g])

        print(f"    [reference_proportional] Distribución objetivo:")
        for g, t in sorted(targets.items()):
            actual    = int(counts.get(g, 0))
            direction = "↑ oversample" if t > actual else ("↓ undersample" if t < actual else "= sin cambio")
            print(f"      {g}: {actual:>8,} → {t:>8,} ({100 * t / total_rows:.1f}%)  {direction}")

        return targets

    raise ValueError(f"Técnica desconocida: {technique}")


def resample_group(
    group_df: pd.DataFrame,
    target: int,
    technique: str,
    rng: np.random.Generator,
    feature_columns: Sequence[str],
    sensor_limits: Tuple[float, float],
) -> pd.DataFrame:
    if technique == "undersampling":
        return undersample_group(group_df, target, rng)
    if technique == "oversampling":
        return oversample_group(group_df, target, rng)
    if technique == "patient_aware_undersampling":
        return patient_aware_undersample_group(group_df, target, rng)
    if technique == "smote":
        return smote_group(group_df, target, rng, feature_columns, sensor_limits)
    if technique == "jittering":
        return jitter_group(group_df, target, rng, feature_columns, sensor_limits)
    if technique == "reference_proportional":
        current = len(group_df)
        if target == current:
            return group_df.copy()
        elif target > current:
            return oversample_group(group_df, target, rng)
        else:
            return undersample_group(group_df, target, rng)
    raise ValueError(f"Técnica no soportada: {technique}")


def resample_train_fold(
    train_df: pd.DataFrame,
    group_col: str,
    technique: str,
    rng: np.random.Generator,
    sensor_limits: Tuple[float, float],
    reference_props: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    counts         = train_df[group_col].value_counts(dropna=False)
    target_counts  = select_target_counts(counts, technique, len(train_df), reference_props)
    feature_columns = [f"x{i}" for i in range(8)] + ["y"]

    parts = []
    for group_value, target in target_counts.items():
        group_df = train_df[train_df[group_col] == group_value].copy()
        parts.append(resample_group(group_df, target, technique, rng, feature_columns, sensor_limits))

    balanced = pd.concat(parts, ignore_index=True)
    shuffled = balanced.sample(
        frac=1.0, random_state=int(rng.integers(0, 2**32 - 1))
    ).reset_index(drop=True)
    return shuffled


# ─── Carga de proporciones de referencia ─────────────────────────────────────
def load_reference_proportions(reference_file: Optional[str]) -> Optional[Dict[str, float]]:
    if not reference_file:
        return None

    path = Path(reference_file)
    if not path.exists():
        raise FileNotFoundError(f"Archivo de referencia no encontrado: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return {str(k): float(v) for k, v in payload.items()}

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows   = list(reader)
        if not rows:
            raise ValueError(f"CSV de referencia vacío: {path}")
        if "group" not in rows[0] or "proportion" not in rows[0]:
            raise ValueError("El CSV de referencia debe tener columnas 'group' y 'proportion'.")
        return {str(row["group"]): float(row["proportion"]) for row in rows}

    raise ValueError("El archivo de referencia debe ser CSV o JSON.")


# ─── Nombre de archivo de salida ─────────────────────────────────────────────
def build_output_name(windows_stem: str, fold_idx: int, group_name: str, technique: str) -> str:
    return f"{windows_stem}_fold{fold_idx}_{group_name}_{technique}.parquet"


# ─── Procesado de un dataset ─────────────────────────────────────────────────
def process_dataset(
    assets:          DatasetAssets,
    horizons:        Sequence[int],
    groups:          Sequence[str],
    techniques:      Sequence[str],
    folds:           Sequence[int],
    seed:            int,
    reference_props: Optional[Mapping[str, float]],
    overwrite:       bool,
) -> None:
    patient_info  = load_table(assets.patient_info_file)
    demo_lookup   = build_demographic_lookup(patient_info)
    sensor_limits = DEFAULT_SENSOR_LIMITS.get(assets.dataset_name, (39.0, 401.0))

    for horizon in horizons:
        # Buscar archivo de ventanas
        windows_candidates = sorted(
            assets.dataset_dir.glob(
                f"windows_with_5folds_{assets.dataset_name}*PH{horizon}*.parquet"
            )
        )
        if not windows_candidates:
            windows_candidates = sorted(
                assets.dataset_dir.glob(
                    f"windows_with_5folds_{assets.dataset_name}*.parquet"
                )
            )
        if not windows_candidates:
            print(f"[WARN] No se encontró archivo de ventanas para {assets.dataset_name} "
                  f"y horizonte {horizon}; omitiendo.")
            continue

        windows_file = windows_candidates[0]
        windows_stem = windows_file.stem

        print(f"\n=== Dataset {assets.dataset_name} | PH={horizon} ===")
        print(f"Ventanas : {windows_file.name}")
        print(f"Pat. info: {assets.patient_info_file.name}")

        df_windows = load_table(windows_file)

        # Columnas fold disponibles
        fold_columns_available = {
            int(c.split("_")[-1]): c
            for c in df_windows.columns
            if c.startswith("fold_") and c.split("_")[-1].isdigit()
        }

        for group_name in groups:
            group_col = f"{group_name}_group"

            for fold_idx in folds:
                if fold_idx not in fold_columns_available:
                    print(f"  [WARN] fold_{fold_idx} no encontrado en {windows_file.name}; omitiendo.")
                    continue

                fold_col = fold_columns_available[fold_idx]

                # ── Separar train / val+test usando solo este fold ────────────
                fold_values   = df_windows[fold_col].astype(str).str.lower()
                train_mask    = fold_values == "train"
                val_mask      = fold_values == "val"
                test_mask     = fold_values == "test"

                # Filas que pertenecen a este fold (train + val + test)
                in_fold_mask  = train_mask | val_mask | test_mask

                raw_train     = df_windows.loc[train_mask].copy()
                raw_untouched = df_windows.loc[val_mask | test_mask].copy()

                print(
                    f"\n  Fold {fold_idx} | grupo '{group_name}': "
                    f"train={len(raw_train):,}, val+test={len(raw_untouched):,}"
                )

                # ── Añadir demografía al train ────────────────────────────────
                train_with_demo = attach_demographics(raw_train, demo_lookup)

                if group_col not in train_with_demo.columns:
                    print(f"    [WARN] Columna de grupo '{group_col}' no disponible; omitiendo.")
                    continue

                group_dist_before = train_with_demo[group_col].value_counts(dropna=False).to_dict()
                print(f"    Distribución antes: {group_dist_before}")

                for technique in techniques:
                    if technique == "reference_proportional" and reference_props is None:
                        print("    [WARN] reference_proportional solicitado sin archivo de referencia; omitiendo.")
                        continue

                    output_name = build_output_name(windows_stem, fold_idx, group_name, technique)
                    output_path = assets.output_dir / output_name
                    if output_path.exists() and not overwrite:
                        print(f"    ⏭  Existe: {output_name}")
                        continue

                    rng = np.random.default_rng(
                        seed
                        + fold_idx * 1000
                        + abs(hash((assets.dataset_name, group_name, technique))) % 1000
                    )

                    # ── Balancear train ───────────────────────────────────────
                    balanced_train = resample_train_fold(
                        train_df        = train_with_demo,
                        group_col       = group_col,
                        technique       = technique,
                        rng             = rng,
                        sensor_limits   = sensor_limits,
                        reference_props = reference_props,
                    )

                    # ── Añadir columna 'split' ────────────────────────────────
                    balanced_train  = balanced_train.copy()
                    balanced_train["split"] = "train"

                    untouched_out = raw_untouched.copy()
                    # Reasignar split desde la columna fold
                    untouched_out["split"] = (
                        df_windows.loc[val_mask | test_mask, fold_col]
                        .astype(str).str.lower()
                        .values
                    )

                    # ── Construir salida: solo columnas de interés + split ────
                    keep = [c for c in KEEP_COLUMNS if c in balanced_train.columns]
                    balanced_train_out = balanced_train[keep + ["split"]].copy()

                    keep2 = [c for c in KEEP_COLUMNS if c in untouched_out.columns]
                    untouched_final = untouched_out[keep2 + ["split"]].copy()

                    out_df = pd.concat(
                        [balanced_train_out, untouched_final], ignore_index=True
                    )

                    # ── Log de distribución final ─────────────────────────────
                    group_dist_after = (
                        balanced_train[group_col].value_counts(dropna=False).to_dict()
                    )

                    out_df.to_parquet(output_path, index=False)

                    print(f"    ✓ Guardado {output_name}")
                    print(f"      antes  : {group_dist_before}")
                    print(f"      después: {group_dist_after}")
                    print(
                        f"      filas  : train={len(balanced_train_out):,}  "
                        f"val+test={len(untouched_final):,}  "
                        f"total={len(out_df):,}"
                    )

                    if technique == "reference_proportional" and reference_props:
                        total_bal = len(balanced_train)
                        print("      check reference_proportional:")
                        for g, n in sorted(group_dist_after.items()):
                            actual_pct = 100 * n / total_bal if total_bal > 0 else 0
                            ref_pct    = 100 * reference_props.get(str(g), 0)
                            deviation  = actual_pct - ref_pct
                            print(
                                f"        {g}: conseguido {actual_pct:.2f}%  |  "
                                f"objetivo {ref_pct:.2f}%  |  Δ = {deviation:+.2f}pp"
                            )


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args             = parse_args()
    root             = Path(args.output_root)
    reference_props  = load_reference_proportions(args.reference_file)

    dataset_dirs = discover_dataset_dirs(root, args.datasets)

    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            print(f"[WARN] Carpeta de dataset no encontrada: {dataset_dir}")
            continue

        try:
            assets = DatasetAssets(
                dataset_name      = dataset_dir.name,
                dataset_dir       = dataset_dir,
                patient_info_file = resolve_patient_info(dataset_dir),
                output_dir        = dataset_dir / "balanced_outputs",
            )
            assets.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"[WARN] No se pudo preparar el dataset {dataset_dir.name}: {exc}")
            continue

        process_dataset(
            assets          = assets,
            horizons        = args.horizons,
            groups          = args.groups,
            techniques      = args.techniques,
            folds           = args.folds,
            seed            = args.seed,
            reference_props = reference_props,
            overwrite       = args.overwrite,
        )


if __name__ == "__main__":
    main()