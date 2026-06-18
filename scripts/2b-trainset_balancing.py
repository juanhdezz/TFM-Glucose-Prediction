# Aqui se debe aplicar los cruces entre los datasets con informacion demografica como sexo y edad, par aque asi 
# al coger el dataset que contiene las ventanas deslizantes, y los fold en los que aparece cada ventana,
# una vez filtradas las ventanas que sean de train , y haciendo el join de esas ventanas con sus datois demograficos 
# podamos empezar a aplicar las tecnicas de balanceo

# los datasets con toda la informacion demografica vienen en C:\Users\User\Desktop\TFM-Glucose-Prediction\data\{DIATREND/REPLACE/T1DIA}\patient_info.parquet
# los datasets que contienen las ventanas deslizantes vienen en C:\Users\User\Desktop\TFM-Glucose-Prediction\data\{DIATREND/REPLACE/T1DIA}\windows_with_5folds_{DIATREND/REPLACE/T1DIA}_2026-03-27_PH4.parquet
# los datasets de output de los balanceos se guardaran en C:\Users\User\Desktop\TFM-Glucose-Prediction\data\{DIATREND/REPLACE/T1DIA}\balanced_outputs

"""Balance training windows by demographic group.

This script joins window parquet files with patient metadata and applies the
balancing techniques described in docs/tecnicas.md only to the rows marked as
train in each fold column. Validation and test rows are preserved.

Outputs are written under data/<dataset>/balanced_outputs as one parquet per
fold, grouping and balancing technique.
"""

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


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DEFAULT_SEED = 42
DEFAULT_HORIZONS = [4]
DEFAULT_FOLDS = [0, 1, 2, 3, 4]
DEFAULT_GROUPS = ["sex", "age"]
DEFAULT_TECHNIQUES = [
	"undersampling",
	"oversampling",
	"patient_aware_undersampling",
	"smote",
	"jittering",
	"reference_proportional",
]
DEFAULT_SENSOR_LIMITS = {
	"DIATREND": (39.0, 401.0),
	"REPLACE-BG": (39.0, 401.0),
	"T1DiabetesGranada": (40.0, 500.0),
}
AGE_BINS = [-np.inf, 18, 30, 45, 60, np.inf]
AGE_LABELS = ["<=18", "19-30", "31-45", "46-60", ">60"]
SEX_MAP = {"F": "F", "M": "M"}


@dataclass(frozen=True)
class DatasetAssets:
	dataset_name: str
	dataset_dir: Path
	patient_info_file: Path
	output_dir: Path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Build balanced train-set parquets from window folds.")
	parser.add_argument(
		"--datasets",
		nargs="*",
		default=None,
		help="Dataset folders to process. Defaults to all folders under data/.",
	)
	parser.add_argument(
		"--groups",
		nargs="*",
		default=DEFAULT_GROUPS,
		choices=DEFAULT_GROUPS,
		help="Demographic grouping columns to balance.",
	)
	parser.add_argument(
		"--techniques",
		nargs="*",
		default=DEFAULT_TECHNIQUES,
		choices=DEFAULT_TECHNIQUES,
		help="Balancing techniques to apply.",
	)
	parser.add_argument(
		"--horizons",
		nargs="*",
		type=int,
		default=DEFAULT_HORIZONS,
		help="Prediction horizons to process.",
	)
	parser.add_argument(
		"--folds",
		nargs="*",
		type=int,
		default=DEFAULT_FOLDS,
		help="Fold indices to balance.",
	)
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
	parser.add_argument(
		"--reference-file",
		type=str,
		default=None,
		help="CSV or JSON file with target proportions for reference_proportional balancing.",
	)
	parser.add_argument(
		"--output-root",
		type=str,
		default=str(DATA_DIR),
		help="Root directory that contains the dataset folders.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing output files.",
	)
	return parser.parse_args()


def discover_dataset_dirs(root: Path, dataset_names: Optional[Sequence[str]]) -> List[Path]:
	if dataset_names:
		return [root / name for name in dataset_names]
	return [path for path in sorted(root.iterdir()) if path.is_dir()]


def resolve_patient_info(dataset_dir: Path) -> Path:
	candidates = [
		dataset_dir / "patient_info.parquet",
		dataset_dir / "Patient_info.parquet",
		dataset_dir / "patient_info.csv",
		dataset_dir / "Patient_info.csv",
	]
	for candidate in candidates:
		if candidate.exists():
			return candidate
	raise FileNotFoundError(f"No patient info file found in {dataset_dir}")


def load_table(path: Path) -> pd.DataFrame:
	if path.suffix.lower() == ".parquet":
		return pd.read_parquet(path)
	if path.suffix.lower() == ".csv":
		return pd.read_csv(path)
	raise ValueError(f"Unsupported table format: {path}")


def normalize_name(name: str) -> str:
	return str(name).strip().lower().replace(" ", "_")


def find_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
	normalized = {normalize_name(column): column for column in df.columns}
	for candidate in candidates:
		key = normalize_name(candidate)
		if key in normalized:
			return normalized[key]
	raise KeyError(f"None of these columns were found: {candidates}")


def normalize_patient_key(series: pd.Series) -> pd.Series:
	return series.astype(str).str.strip()


def infer_age_series(patient_info: pd.DataFrame) -> pd.Series:
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
		raise KeyError("Could not infer age: neither Age nor Birth_year exists in patient info.")

	birth_year = pd.to_numeric(patient_info[birth_col], errors="coerce")
	ref_year = pd.Series(datetime.now().year, index=patient_info.index, dtype="float64")

	for candidate in ["Final_measurement_date", "final_measurement_date", "Initial_measurement_date", "initial_measurement_date"]:
		try:
			date_col = find_column(patient_info, [candidate])
		except KeyError:
			continue
		parsed = pd.to_datetime(patient_info[date_col], errors="coerce")
		if parsed.notna().any():
			ref_year = parsed.dt.year.astype("float64")
			break

	return ref_year - birth_year


def add_demographic_columns(df_windows: pd.DataFrame, patient_info: pd.DataFrame) -> pd.DataFrame:
	windows = df_windows.copy()
	meta = patient_info.copy()

	windows_patient_col = find_column(windows, ["patient_id", "Patient_ID", "patientid"])
	meta_patient_col = find_column(meta, ["patient_id", "Patient_ID", "patientid"])

	windows["_patient_key"] = normalize_patient_key(windows[windows_patient_col])
	meta["_patient_key"] = normalize_patient_key(meta[meta_patient_col])

	sex_col = find_column(meta, ["Sex", "sex"])
	meta["sex_group"] = meta[sex_col].astype(str).str.strip().str.upper().map(SEX_MAP).fillna("Unknown")
	meta["age"] = infer_age_series(meta)
	meta["age_group"] = pd.cut(
		meta["age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True, right=True
	).astype("object").fillna("Unknown")

	meta = meta[["_patient_key", "sex_group", "age", "age_group"]].drop_duplicates(subset=["_patient_key"])
	merged = windows.merge(meta, on="_patient_key", how="left")
	return merged.drop(columns=["_patient_key"])


def clip_feature_columns(df: pd.DataFrame, feature_columns: Sequence[str], low: float, high: float) -> pd.DataFrame:
	clipped = df.copy()
	for column in feature_columns:
		clipped[column] = pd.to_numeric(clipped[column], errors="coerce").clip(lower=low, upper=high)
	return clipped


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
			raise ValueError("reference_proportional requires a reference distribution file.")
		missing = [group for group in counts.index if group not in reference_props]
		if missing:
			raise KeyError(f"Reference proportions are missing groups: {missing}")

		raw_targets = {group: float(reference_props[group]) * total_rows for group in counts.index}
		floors = {group: int(math.floor(value)) for group, value in raw_targets.items()}
		remainder = total_rows - sum(floors.values())
		ranked = sorted(counts.index, key=lambda group: raw_targets[group] - floors[group], reverse=True)
		targets = floors.copy()
		for group in ranked[:remainder]:
			targets[group] += 1
		return targets

	raise ValueError(f"Unknown technique: {technique}")


def undersample_group(group_df: pd.DataFrame, target: int, rng: np.random.Generator) -> pd.DataFrame:
	if len(group_df) <= target:
		return group_df.copy()
	chosen = rng.choice(group_df.index.to_numpy(), size=target, replace=False)
	return group_df.loc[chosen].copy()


def oversample_group(group_df: pd.DataFrame, target: int, rng: np.random.Generator) -> pd.DataFrame:
	if len(group_df) >= target:
		return group_df.copy()
	chosen = rng.choice(group_df.index.to_numpy(), size=target, replace=True)
	return group_df.loc[chosen].copy()


def patient_aware_undersample_group(group_df: pd.DataFrame, target: int, rng: np.random.Generator) -> pd.DataFrame:
	if len(group_df) <= target:
		return group_df.copy()

	patient_counts = group_df.groupby("patient_id").size().sort_values(ascending=False)
	patient_cap = max(1, int(math.ceil(target / len(patient_counts))))
	sampled_parts: List[pd.DataFrame] = []

	for _, patient_df in group_df.groupby("patient_id", sort=False):
		take = min(len(patient_df), patient_cap)
		chosen = rng.choice(patient_df.index.to_numpy(), size=take, replace=False)
		sampled_parts.append(group_df.loc[chosen].copy())

	sampled = pd.concat(sampled_parts, ignore_index=False)
	if len(sampled) > target:
		chosen = rng.choice(sampled.index.to_numpy(), size=target, replace=False)
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
		synthetic = pd.concat([group_df.copy() for _ in range(needed)], ignore_index=True)
		return clip_feature_columns(synthetic, feature_columns, *sensor_limits)

	features = group_df.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
	feature_mean = np.nanmean(features, axis=0)
	feature_std = np.nanstd(features, axis=0)
	feature_std[feature_std == 0] = 1.0
	scaled = (features - feature_mean) / feature_std

	n_neighbors = min(5, current - 1)
	nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
	nn.fit(scaled)
	neighbors = nn.kneighbors(scaled, return_distance=False)
	base_rows = group_df.reset_index(drop=True)

	synthetic_rows: List[pd.Series] = []
	for _ in range(needed):
		base_idx = int(rng.integers(0, current))
		candidate_neighbors = neighbors[base_idx][1:]
		if len(candidate_neighbors) == 0:
			neighbor_idx = base_idx
		else:
			neighbor_idx = int(rng.choice(candidate_neighbors))

		lam = float(rng.random())
		base_vector = features[base_idx]
		neighbor_vector = features[neighbor_idx]
		synthetic_vector = base_vector + lam * (neighbor_vector - base_vector)

		synthetic_row = base_rows.iloc[base_idx].copy()
		for idx, column in enumerate(feature_columns):
			synthetic_row[column] = synthetic_vector[idx]
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

	needed = target - current
	features = group_df.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
	stds = features.std(axis=0, ddof=0).replace(0, 1.0).to_numpy(dtype=float)
	base_rows = group_df.reset_index(drop=True)

	synthetic_rows: List[pd.Series] = []
	for _ in range(needed):
		base_idx = int(rng.integers(0, current))
		base_row = base_rows.iloc[base_idx].copy()
		noise = rng.normal(loc=0.0, scale=stds * jitter_scale, size=len(feature_columns))
		for idx, column in enumerate(feature_columns):
			base_row[column] = pd.to_numeric(base_row[column], errors="coerce") + float(noise[idx])
		synthetic_rows.append(base_row)

	synthetic_df = pd.DataFrame(synthetic_rows)
	synthetic_df = clip_feature_columns(synthetic_df, feature_columns, *sensor_limits)
	return pd.concat([group_df, synthetic_df], ignore_index=True)


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
	raise ValueError(f"Unsupported technique: {technique}")


def resample_train_fold(
	train_df: pd.DataFrame,
	group_col: str,
	technique: str,
	rng: np.random.Generator,
	sensor_limits: Tuple[float, float],
	reference_props: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
	counts = train_df[group_col].value_counts(dropna=False)
	target_counts = select_target_counts(counts, technique, len(train_df), reference_props)
	feature_columns = [f"x{i}" for i in range(8)] + ["y"]

	parts = []
	for group_value, target in target_counts.items():
		group_df = train_df[train_df[group_col] == group_value].copy()
		parts.append(resample_group(group_df, target, technique, rng, feature_columns, sensor_limits))

	balanced = pd.concat(parts, ignore_index=True)
	shuffled = balanced.sample(frac=1.0, random_state=int(rng.integers(0, 2**32 - 1))).reset_index(drop=True)
	return shuffled


def load_reference_proportions(reference_file: Optional[str]) -> Optional[Dict[str, float]]:
	if not reference_file:
		return None

	path = Path(reference_file)
	if not path.exists():
		raise FileNotFoundError(f"Reference file not found: {path}")

	if path.suffix.lower() == ".json":
		with path.open("r", encoding="utf-8") as handle:
			payload = json.load(handle)
		return {str(key): float(value) for key, value in payload.items()}

	if path.suffix.lower() == ".csv":
		with path.open("r", encoding="utf-8", newline="") as handle:
			reader = csv.DictReader(handle)
			rows = list(reader)
		if not rows:
			raise ValueError(f"Empty reference CSV: {path}")
		if "group" not in rows[0] or "proportion" not in rows[0]:
			raise ValueError("Reference CSV must contain 'group' and 'proportion' columns.")
		return {str(row["group"]): float(row["proportion"]) for row in rows}

	raise ValueError("Reference file must be CSV or JSON.")


def build_output_name(windows_stem: str, fold_idx: int, group_name: str, technique: str) -> str:
	return f"{windows_stem}_fold{fold_idx}_{group_name}_{technique}.parquet"


def process_dataset(
	assets: DatasetAssets,
	horizons: Sequence[int],
	groups: Sequence[str],
	techniques: Sequence[str],
	folds: Sequence[int],
	seed: int,
	reference_props: Optional[Mapping[str, float]],
	overwrite: bool,
) -> None:
	patient_info = load_table(assets.patient_info_file)
	sensor_limits = DEFAULT_SENSOR_LIMITS.get(assets.dataset_name, (39.0, 401.0))

	for horizon in horizons:
		windows_candidates = sorted(assets.dataset_dir.glob(f"windows_with_5folds_{assets.dataset_name}*PH{horizon}*.parquet"))
		if not windows_candidates:
			windows_candidates = sorted(assets.dataset_dir.glob(f"windows_with_5folds_{assets.dataset_name}*.parquet"))
		if not windows_candidates:
			print(f"[WARN] No windows file found for {assets.dataset_name} and horizon {horizon}; skipping.")
			continue

		windows_file = windows_candidates[0]
		windows_stem = windows_file.stem

		print(f"\n=== Dataset {assets.dataset_name} | PH={horizon} ===")
		print(f"Windows file: {windows_file.name}")
		print(f"Patient info: {assets.patient_info_file.name}")

		df_windows = load_table(windows_file)
		df_joined = add_demographic_columns(df_windows, patient_info)

		fold_columns = [f"fold_{idx}" for idx in folds if f"fold_{idx}" in df_joined.columns]
		if not fold_columns:
			raise KeyError(f"No requested fold columns were found in {windows_file}")

		for group_name in groups:
			group_col = f"{group_name}_group"
			if group_col not in df_joined.columns:
				print(f"[WARN] Missing group column {group_col}; skipping.")
				continue

			for fold_col in fold_columns:
				fold_idx = int(fold_col.split("_")[-1])
				train_mask = df_joined[fold_col].astype(str) == "train"
				train_df = df_joined.loc[train_mask].copy()
				untouched_df = df_joined.loc[~train_mask].copy()

				print(
					f"  Fold {fold_idx} | {group_name}: train={len(train_df):,}, untouched={len(untouched_df):,}"
				)

				for technique in techniques:
					if technique == "reference_proportional" and reference_props is None:
						print("    [WARN] reference_proportional requested but no reference file was provided; skipping.")
						continue

					rng = np.random.default_rng(seed + fold_idx * 1000 + abs(hash((assets.dataset_name, group_name, technique))) % 1000)
					balanced_train = resample_train_fold(
						train_df=train_df,
						group_col=group_col,
						technique=technique,
						rng=rng,
						sensor_limits=sensor_limits,
						reference_props=reference_props,
					)

					out_df = pd.concat([balanced_train, untouched_df], ignore_index=True)
					out_df = out_df.drop(columns=["sex_group", "age", "age_group"], errors="ignore")

					output_name = build_output_name(windows_stem, fold_idx, group_name, technique)
					output_path = assets.output_dir / output_name
					if output_path.exists() and not overwrite:
						print(f"    Skipping existing file: {output_name}")
						continue

					out_df.to_parquet(output_path, index=False)

					before_counts = train_df[group_col].value_counts(dropna=False).to_dict()
					after_counts = balanced_train[group_col].value_counts(dropna=False).to_dict()
					print(f"    ✓ Saved {output_name}")
					print(f"      before: {before_counts}")
					print(f"      after : {after_counts}")


def main() -> None:
	args = parse_args()
	root = Path(args.output_root)
	reference_props = load_reference_proportions(args.reference_file)

	dataset_dirs = discover_dataset_dirs(root, args.datasets)
	for dataset_dir in dataset_dirs:
		if not dataset_dir.exists():
			print(f"[WARN] Dataset folder not found: {dataset_dir}")
			continue

		try:
			assets = DatasetAssets(
				dataset_name=dataset_dir.name,
				dataset_dir=dataset_dir,
				patient_info_file=resolve_patient_info(dataset_dir),
				output_dir=dataset_dir / "balanced_outputs",
			)
			assets.output_dir.mkdir(parents=True, exist_ok=True)
		except Exception as exc:
			print(f"[WARN] Could not prepare dataset {dataset_dir.name}: {exc}")
			continue

		process_dataset(
			assets=assets,
			horizons=args.horizons,
			groups=args.groups,
			techniques=args.techniques,
			folds=args.folds,
			seed=args.seed,
			reference_props=reference_props,
			overwrite=args.overwrite,
		)


if __name__ == "__main__":
	main()