# -*- coding: utf-8 -*-
"""
RELIEF-T1D — Training & Prediction Script
Adapted for server execution with automatic discovery of balanced parquet files.

Discovery logic:
  - Scans /data/input/{DATASET}/ for parquet files to process.
  - IGNORES everything inside balanced_outputs/ subdirectories.
  - PROCESSES:
      a) Original fold file:  windows_with_5folds_{DATASET}_*_PH4.parquet
      b) Compacted balanced:  {DATASET}_balanced_{age|sex}_{method}_PH4.parquet

Fixed parameters for this run:
  - Algorithm : LSTM
  - PH        : 4
  - H         : 8
  - Loss      : RMSE

Output structure per experiment:
  /data/output/{DATASET}/{balancing_condition}/
    EXP-{date}-prediction-RMSE-LSTM-H8-PH4-{context}-v{N}/
      models/
      plots/training/
      results/predictions/
      results/evaluation/metrics/
      results/evaluation/figures/
      summaries/
      readmes/
      experiment_log.out
      README.md
"""

# ----------------------- Imports -----------------------
from test_predictions import test_by_range, get_train_plots_loss, fold_results_aggregation
from loss_functions import RMSE, ClinicalPenaltyMetric
import matplotlib
matplotlib.use('Agg')  
import argparse
from keras.layers import Dense, LSTM, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import keras
import numpy as np
import tensorflow as tf
import random
import os
import sys
import re
import json
import time
import datetime
import platform
import socket
import psutil
import subprocess
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
# ----------------------- End imports -----------------------

# ----------------------- Reproducibility -----------------------
np.random.seed(50)
tf.random.set_seed(50)
random.seed(50)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
# ----------------------- End reproducibility -----------------------

# ----------------------- Fixed experimental parameters -----------------------
HISTORY_LENGTH    = 8
HORIZON           = 4
ALGORITHM         = 'LSTM'
LOSS_NAME         = 'RMSE'
BATCH_SIZE        = 4096
PATIENCE          = 10
MAX_EPOCH         = 500
DROPOUT           = 0.1
RECURRENT_DROPOUT = 0.0
DO_CROSS_VALIDATION = True
K_FOLDS           = 5

INPUT_ROOT  = Path('/home/juanhdez/data/input')
OUTPUT_ROOT = Path('/home/juanhdez/data/output')
# ----------------------- End fixed parameters -----------------------

# ----------------------- Sensor limits per dataset -----------------------
SENSOR_LIMITS = {
    'DIATREND':          (39.0, 401.0),
    'REPLACE-BG':        (39.0, 401.0),
    'T1DiabetesGranada': (40.0, 500.0),
}
DEFAULT_SENSOR_LIMITS = (39.0, 401.0)
# ----------------------- End sensor limits -----------------------

# ----------------------- Parquet discovery -----------------------
@dataclass
class ExperimentTarget:
    """
    Describes one full cross-validation experiment (5 folds).

    Hay dos modos de uso:
      - Original: un único parquet con columnas fold_0..fold_4  → fold_files=[], parquet_path apunta al archivo.
      - Balanceado: 5 parquets (uno por fold) sin columnas fold_*,
                    con columna 'split' (train/val/test)          → fold_files lista los 5, parquet_path=None.
    """
    dataset_name:        str
    balancing_condition: str   # e.g. 'original', 'balanced_age_smote', 'balanced_sex_oversampling'
    min_sensor:          float
    max_sensor:          float
    # Modo original: un único archivo con todas las columnas fold_i
    parquet_path: Optional[Path] = None
    # Modo balanceado: lista de K_FOLDS archivos, uno por fold, ordenados fold 0..4
    fold_files:   list = None  # List[Path]

    def __post_init__(self):
        if self.fold_files is None:
            self.fold_files = []

    def __str__(self):
        if self.parquet_path:
            return f"{self.dataset_name} | {self.balancing_condition} | {self.parquet_path.name}"
        return (
            f"{self.dataset_name} | {self.balancing_condition} | "
            f"{len(self.fold_files)} fold-files"
        )


def detect_dataset_name(folder_name: str) -> str:
    """
    Map the directory name to the canonical dataset name used in filenames.
    The folder names in /data/input/ are the canonical names themselves.
    """
    return folder_name


def discover_targets(dataset_filter: list[str] | None = None) -> list[ExperimentTarget]:
    """
    Recorre INPUT_ROOT y recoge todos los experimentos a ejecutar.

    Modo ORIGINAL:
      Busca en <dataset_dir>/ el archivo:
        windows_with_5folds_<DATASET>_*_PH<HORIZON>.parquet
      Este archivo contiene columnas fold_0..fold_4 y se procesa con la
      lógica clásica (filtrado por columna fold_i).

    Modo BALANCEADO:
      Busca en <dataset_dir>/balanced_outputs/ archivos con el patrón:
        *_fold<i>_<group>_<technique>.parquet   (i = 0..K_FOLDS-1)
      Agrupa los K_FOLDS archivos de cada (grupo, técnica) en un único
      ExperimentTarget cuya lista fold_files contiene los 5 paths ordenados.
      Cada archivo ya lleva la columna 'split' (train/val/test) y NO tiene
      columnas fold_*.

    Solo se incluye un target balanceado cuando los K_FOLDS archivos están
    presentes (para no lanzar experimentos incompletos).
    """
    targets = []

    original_pattern = re.compile(
        r'^windows_with_5folds_.+_PH' + str(HORIZON) + r'\.parquet$'
    )
    # Patrón de los archivos generados por 2b-trainset_balancing.py:
    #   <windows_stem>_fold<i>_<group>_<technique>.parquet
    balanced_fold_pattern = re.compile(
        r'^(?P<stem>.+)_fold(?P<fold>\d+)_(?P<group>age|sex)_(?P<technique>.+)\.parquet$'
    )

    for dataset_dir in sorted(INPUT_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = detect_dataset_name(dataset_dir.name)
        if dataset_filter and dataset_name not in dataset_filter:
            continue

        min_s, max_s = SENSOR_LIMITS.get(dataset_name, DEFAULT_SENSOR_LIMITS)

        # ── Modo original ──────────────────────────────────────────────────
        for parquet_file in sorted(dataset_dir.glob('*.parquet')):
            if original_pattern.match(parquet_file.name):
                targets.append(ExperimentTarget(
                    dataset_name        = dataset_name,
                    balancing_condition = 'original',
                    min_sensor          = min_s,
                    max_sensor          = max_s,
                    parquet_path        = parquet_file,
                    fold_files          = [],
                ))

        # ── Modo balanceado ────────────────────────────────────────────────
        balanced_dir = dataset_dir / 'balanced_outputs'
        if not balanced_dir.exists():
            continue

        # Agrupar archivos por (stem_base, group, technique)
        # stem_base = todo hasta "_fold<i>_"
        from collections import defaultdict
        groups_map: dict[tuple[str, str, str], dict[int, Path]] = defaultdict(dict)

        for parquet_file in sorted(balanced_dir.glob('*.parquet')):
            m = balanced_fold_pattern.match(parquet_file.name)
            if not m:
                continue
            stem      = m.group('stem')
            fold_idx  = int(m.group('fold'))
            group     = m.group('group')
            technique = m.group('technique')
            key       = (stem, group, technique)
            groups_map[key][fold_idx] = parquet_file

        for (stem, group, technique), fold_dict in sorted(groups_map.items()):
            # Verificar que los K_FOLDS archivos existen
            missing = [i for i in range(K_FOLDS) if i not in fold_dict]
            if missing:
                print(
                    f"[WARN] {dataset_name} | {group} | {technique}: "
                    f"faltan folds {missing}; experimento omitido."
                )
                continue

            fold_files  = [fold_dict[i] for i in range(K_FOLDS)]
            condition   = f"balanced_{group}_{technique}"

            targets.append(ExperimentTarget(
                dataset_name        = dataset_name,
                balancing_condition = condition,
                min_sensor          = min_s,
                max_sensor          = max_s,
                parquet_path        = None,
                fold_files          = fold_files,
            ))

    return targets
# ----------------------- End discovery -----------------------

# ----------------------- Console Tee -----------------------
class Tee:
    def __init__(self, *files):
        self.files = [f for f in files if f]

    def write(self, obj):
        for f in self.files:
            if not f.closed:
                try:
                    f.write(obj)
                    f.flush()
                except (IOError, ValueError):
                    pass

    def flush(self):
        for f in self.files:
            if not f.closed:
                try:
                    f.flush()
                except (IOError, ValueError):
                    pass

RUN_TS       = datetime.datetime.now()
GLOBAL_RUN_ID = RUN_TS.strftime("%Y%m%d_%H%M%S")
original_stdout = sys.stdout
original_stderr = sys.stderr
# ----------------------- End Tee -----------------------

# ----------------------- System helpers -----------------------
def print_system_info():
    print("Date & Time")
    print("-----------------------------------")
    print(f"Script Execution Start : {RUN_TS.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current Time           : {datetime.datetime.now()}")
    print("\nSystem & Hardware")
    print("-----------------------------------")
    print(f"Hostname    : {socket.gethostname()}")
    print(f"OS          : {platform.system()} {platform.release()}")
    print(f"CPU         : {platform.processor()}")
    ram = psutil.virtual_memory()
    print(f"RAM Total   : {round(ram.total / (1024 ** 3), 2)} GB")
    print(f"RAM Available: {round(ram.available / (1024 ** 3), 2)} GB")
    print(f"\nPython      : {platform.python_version()}")
    print(f"TensorFlow  : {tf.__version__}")
    print("\nGPU (TensorFlow)")
    print("-----------------------------------")
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {getattr(gpu, 'name', str(gpu))}")
        else:
            print("  No GPUs detected by TensorFlow.")
    except Exception as e:
        print(f"  GPU detection error: {e}")
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version",
             "--format=csv,noheader"],
            stderr=subprocess.STDOUT
        )
        for i, line in enumerate(out.decode(errors="ignore").strip().split("\n")):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                name, mem_total, mem_used, util, driver = parts[:5]
                print(f"  [{i}] {name} | {mem_total} total | {mem_used} used | {util} util | driver {driver}")
    except Exception:
        pass


def get_git_info() -> dict:
    info = {"commit": "N/A", "branch": "N/A", "remote": "N/A"}
    try:
        info["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        info["remote"] = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"]).decode().strip()
    except Exception:
        pass
    return info
# ----------------------- End system helpers -----------------------

# ----------------------- Experiment directory helpers -----------------------
def safe_token(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.\-]+', '_', str(s))


def safe_relpath(p, start=None) -> str:
    try:
        return os.path.relpath(p, start=start) if start else os.path.relpath(p)
    except Exception:
        return str(p)


def next_versioned_dir(base_path: Path) -> Path:
    """Return base_path-v1, base_path-v2, … choosing the first that does not exist."""
    vnum = 1
    while True:
        candidate = base_path.parent / f"{base_path.name}-v{vnum}"
        if not candidate.exists():
            return candidate
        vnum += 1


def build_experiment_dir(target: ExperimentTarget) -> Path:
    """
    Build and create the experiment output directory.

    Structure:
      OUTPUT_ROOT / {dataset_name} / {balancing_condition} /
        EXP-{date}-RMSE-LSTM-H{H}-PH{PH}-{condition}-v{N}
    """
    date_str = RUN_TS.strftime("%Y%m%d")
    base_name = (
        f"EXP-{date_str}"
        f"-prediction-{safe_token(LOSS_NAME)}"
        f"-{safe_token(ALGORITHM)}"
        f"-H{HISTORY_LENGTH}"
        f"-PH{HORIZON}"
        f"-{safe_token(target.balancing_condition)}"
    )
    parent = OUTPUT_ROOT / safe_token(target.dataset_name) / safe_token(target.balancing_condition)
    exp_dir = next_versioned_dir(parent / base_name)

    for sub in [
        "models",
        "plots/training",
        "results/predictions",
        "results/evaluation/metrics",
        "results/evaluation/figures",
        "summaries",
        "readmes",
    ]:
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)

    return exp_dir
# ----------------------- End directory helpers -----------------------

# ----------------------- Model -----------------------
class LSTMModel:
    def __init__(self, input_shape, nb_output_units,
                 nb_hidden_units=128,
                 dropout_rate=DROPOUT,
                 recurrent_dropout_rate=RECURRENT_DROPOUT):
        self.input_shape      = input_shape
        self.nb_output_units  = nb_output_units
        self.nb_hidden_units  = nb_hidden_units
        self.dropout          = dropout_rate
        self.recurrent_dropout = recurrent_dropout_rate

    def __repr__(self):
        return (f"LSTM_{self.nb_hidden_units}"
                f"_drop{self.dropout}"
                f"_recDrop{self.recurrent_dropout}")

    def build(self) -> Model:
        i = Input(shape=self.input_shape)
        x = LSTM(self.nb_hidden_units,
                 dropout=self.dropout,
                 recurrent_dropout=self.recurrent_dropout)(i)
        x = Dense(self.nb_output_units, activation=None)(x)
        return Model(inputs=[i], outputs=[x])


def build_lstm(history_len: int, weights: str = '', loss_fn=None) -> Model:
    if loss_fn is None:
        loss_fn = RMSE
    m = LSTMModel(input_shape=(history_len, 1), nb_output_units=1).build()
    m.compile(
        loss=loss_fn,
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.RootMeanSquaredError(), ClinicalPenaltyMetric()],
    )
    if weights:
        print(f"Loading weights from: {weights}")
        m.load_weights(weights)
    return m
# ----------------------- End model -----------------------

# ----------------------- Callbacks -----------------------
def make_callbacks(filepath_prefix: str, early_stopping_patience: int):
    return [
        ModelCheckpoint(
            filepath=f"{filepath_prefix}.weights.h5",
            monitor='val_loss', mode='min',
            save_best_only=True, save_weights_only=True, verbose=1,
        ),
        EarlyStopping(
            monitor='val_loss', mode='min',
            patience=early_stopping_patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=f"{filepath_prefix}_logs",
            histogram_freq=0, write_graph=True,
            write_images=False, update_freq='epoch', profile_batch=0,
        ),
        keras.callbacks.CSVLogger(
            filename=f"{filepath_prefix}_history.csv",
            separator=',', append=False,
        ),
    ]
# ----------------------- End callbacks -----------------------

# ----------------------- Train function -----------------------
def train_fold(x_train, y_train, x_val, y_val, history_len: int, save_prefix: str):
    model = build_lstm(history_len)
    # LSTM expects (samples, timesteps, features)
    x_tr = x_train.reshape(x_train.shape[0], history_len, 1)
    x_v  = x_val.reshape(x_val.shape[0],   history_len, 1)
    hist = model.fit(
        x_tr, y_train,
        batch_size=BATCH_SIZE,
        validation_data=(x_v, y_val),
        epochs=MAX_EPOCH,
        callbacks=make_callbacks(save_prefix, PATIENCE),
    )
    return hist, model
# ----------------------- End train function -----------------------

# ----------------------- Artifact writers -----------------------
def write_best_json(save_prefix: str, loss_name: str, ph: int,
                    history_len: int, fold_idx: int) -> str:
    hist_csv = f"{save_prefix}_history.csv"
    best_json = f"{save_prefix}.best.json"
    try:
        df_hist = pd.read_csv(hist_csv)
        col     = 'val_loss' if 'val_loss' in df_hist.columns else 'loss'
        best_idx = int(df_hist[col].idxmin())
        payload  = {
            "monitor": col,
            "epoch_idx": best_idx,
            "epoch_count": int(df_hist.shape[0]),
            "metrics": df_hist.iloc[best_idx].to_dict(),
            "context": {
                "loss": loss_name, "algorithm": ALGORITHM,
                "prediction_horizon": ph, "history_length": history_len,
                "fold": fold_idx, "run_id": GLOBAL_RUN_ID,
            },
            "artifacts": {
                "weights_path": os.path.abspath(f"{save_prefix}.weights.h5"),
                "history_csv_path": os.path.abspath(hist_csv),
            },
            "created_at": datetime.datetime.now().isoformat(),
        }
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"WARNING: could not write best.json: {e}")
    return best_json


def write_fold_summary(
    exp_dir: Path, loss_name: str, ph: int, fold_idx: int,
    fold_started_at: datetime.datetime,
    train_sec: float, pred_sec: float,
    weights_path: str, hist_csv_path: str,
    best_json_path: str, preds_parquet_path: str,
    training_plot_prefix: str, history_len: int,
    min_sensor: float, max_sensor: float,
    data_file_path: str, data_counts: dict,
    balancing_condition: str,
):
    payload = {
        "experiment": {
            "loss": loss_name, "algorithm": ALGORITHM,
            "prediction_horizon": ph, "fold": fold_idx,
            "history_length": history_len,
            "balancing_condition": balancing_condition,
            "sensor_limits": {"min": min_sensor, "max": max_sensor},
            "started_at": fold_started_at.isoformat(),
            "run_id": GLOBAL_RUN_ID,
        },
        "data": {"file": os.path.abspath(data_file_path), "counts": data_counts},
        "timing": {
            "train_seconds": round(train_sec, 3),
            "train_elapsed": str(datetime.timedelta(seconds=train_sec)),
            "predict_seconds": round(pred_sec, 3),
            "predict_elapsed": str(datetime.timedelta(seconds=pred_sec)),
        },
        "artifacts": {
            "weights_path": safe_relpath(weights_path, start=str(exp_dir)),
            "history_csv_path": safe_relpath(hist_csv_path, start=str(exp_dir)),
            "best_json_path": safe_relpath(best_json_path, start=str(exp_dir)),
            "predictions_parquet_path": safe_relpath(preds_parquet_path, start=str(exp_dir)),
            "training_plot_prefix": safe_relpath(training_plot_prefix, start=str(exp_dir)),
        },
        "git": get_git_info(),
        "system": {
            "hostname": socket.gethostname(),
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
        },
    }

    json_path = exp_dir / "summaries" / f"summary_{loss_name}_{ALGORITHM}_H{ph}_Fold{fold_idx}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md_path = exp_dir / "readmes" / f"README_{loss_name}_{ALGORITHM}_H{ph}_Fold{fold_idx}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Fold {fold_idx} — {loss_name}/{ALGORITHM}/PH={ph}\n\n")
        f.write(f"**Balancing condition**: `{balancing_condition}`\n\n")
        f.write(f"**Data file**: `{os.path.abspath(data_file_path)}`\n\n")
        f.write("| Split | Patients | Windows | % Patients | % Windows |\n")
        f.write("|-------|----------|---------|------------|----------|\n")
        for split in ["train", "val", "test"]:
            f.write(
                f"| {split.capitalize()} "
                f"| {data_counts['patients'][split]} "
                f"| {data_counts['windows'][split]} "
                f"| {data_counts['percent']['patients'][split]:.2f}% "
                f"| {data_counts['percent']['windows'][split]:.2f}% |\n"
            )
        f.write(f"\n## Timings\n")
        f.write(f"- Training : {datetime.timedelta(seconds=train_sec)}\n")
        f.write(f"- Prediction: {datetime.timedelta(seconds=pred_sec)}\n")


def write_experiment_readme(exp_dir: Path, loss_name: str, ph: int,
                             history_len: int, balancing_condition: str):
    exp_name = exp_dir.name
    git_info = get_git_info()
    lines = [
        f"# Experiment: `{exp_name}`\n",
        f"- **Run ID**: `{GLOBAL_RUN_ID}`",
        f"- **Date**: `{RUN_TS.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- **Git Commit**: `{git_info['commit']}`\n",
        "## Configuration\n",
        f"- **Algorithm**: `{ALGORITHM}`",
        f"- **Loss**: `{loss_name}`",
        f"- **Prediction Horizon (PH)**: `{ph}`",
        f"- **History Length (H)**: `{history_len}`",
        f"- **Balancing Condition**: `{balancing_condition}`\n",
        "## Outputs\n",
        f"- Aggregated predictions: `results/predictions/`",
        f"- Metrics: `results/evaluation/metrics/`",
        f"- Clarke Error Grid: `results/evaluation/figures/`",
        f"- Log: `experiment_log.out`\n",
        f"---\n\n## Fold Details ({K_FOLDS} folds)\n",
    ]
    for i in range(1, K_FOLDS + 1):
        p = f"{loss_name}_{ALGORITHM}_H{ph}_Fold{i}"
        lines += [
            f"### Fold {i}\n",
            f"- Weights    : `models/{p}.weights.h5`",
            f"- History    : `models/{p}_history.csv`",
            f"- Predictions: `results/predictions/df_test_results_vectors_{p}.parquet`",
            f"- Summary    : `summaries/summary_{p}.json`\n",
        ]
    with open(exp_dir / "README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
# ----------------------- End artifact writers -----------------------

# ----------------------- Single experiment runner -----------------------
def run_experiment(target: ExperimentTarget) -> None:
    """
    Full training pipeline for one parquet file (all 5 folds).
    Creates its own experiment directory and log file.
    """
    ph          = HORIZON
    loss_name   = LOSS_NAME
    loss_fn     = RMSE

    EXP_DIR     = build_experiment_dir(target)
    MODELS_DIR  = EXP_DIR / "models"
    PRED_DIR    = EXP_DIR / "results" / "predictions"
    PLOTS_TRAIN = EXP_DIR / "plots" / "training"
    METRICS_DIR = EXP_DIR / "results" / "evaluation" / "metrics"
    FIGURES_DIR = EXP_DIR / "results" / "evaluation" / "figures"

    exp_log_file = None
    try:
        exp_log_path = EXP_DIR / "experiment_log.out"
        exp_log_file = open(exp_log_path, "a", encoding="utf-8")
        sys.stdout = sys.stderr = Tee(original_stdout, exp_log_file)

        print(f"\n{'='*60}")
        print(f"EXPERIMENT START")
        print(f"  Dataset            : {target.dataset_name}")
        print(f"  Balancing condition: {target.balancing_condition}")
        if target.parquet_path:
            print(f"  Parquet file       : {target.parquet_path}")
        else:
            print(f"  Fold files (x{len(target.fold_files)})   : {target.fold_files[0].parent.name}/")
        print(f"  Output dir         : {EXP_DIR}")
        print(f"{'='*60}\n")
        print_system_info()

        used_columns = [f"x{i}" for i in range(HISTORY_LENGTH)] + ["y"]
        info_columns = ['patient_id', 'x_time_7', 'x_date_7']

        aggregated_results          = []
        aggregated_results_perf     = []
        aggregated_results_npoints  = []
        aggregated_results_outrange = []

        tf.keras.backend.clear_session()
        np.random.seed(50)
        tf.random.set_seed(50)
        random.seed(50)

        # Determinar si estamos en modo original (un archivo) o balanceado (un archivo por fold)
        is_balanced_mode = bool(target.fold_files)

        for fold_i in range(K_FOLDS):
            current_fold = fold_i + 1
            current_name = f"{loss_name}_{ALGORITHM}_H{ph}_Fold{current_fold}"

            print(f"\n---- Fold {current_fold}/{K_FOLDS} ----")

            if is_balanced_mode:
                # ── Modo balanceado: cada fold tiene su propio archivo ─────────
                fold_path = target.fold_files[fold_i]
                print(f"Loading (balanced fold file): {fold_path.name}")
                df = pd.read_parquet(fold_path)

                # El archivo ya viene con columna 'split' en lugar de fold_*
                split_col = 'split'
                if split_col not in df.columns:
                    raise KeyError(
                        f"El archivo balanceado '{fold_path.name}' no contiene la columna 'split'. "
                        "Asegúrate de usar la versión actualizada de 2b-trainset_balancing.py."
                    )

                split_values = df[split_col].astype(str).str.lower()
                df_train_raw = df[split_values == 'train']
                df_val_raw   = df[split_values == 'val']
                df_test_raw  = df[split_values == 'test']

            else:
                # ── Modo original: un único archivo con columnas fold_i ────────
                print(f"Loading (original): {target.parquet_path}")
                df = pd.read_parquet(target.parquet_path)

                fold_col     = f'fold_{fold_i}'
                fold_values  = df[fold_col].astype(str).str.lower()
                df_train_raw = df[fold_values == 'train']
                df_val_raw   = df[fold_values == 'val']
                df_test_raw  = df[fold_values == 'test']

            n_pat_train = df_train_raw['patient_id'].nunique()
            df_train = df_train_raw.reset_index(drop=True)[used_columns]

            n_pat_val = df_val_raw['patient_id'].nunique()
            df_val = df_val_raw.reset_index(drop=True)[used_columns]

            n_pat_test = df_test_raw['patient_id'].nunique()
            df_test_info = df_test_raw.reset_index(drop=True)[info_columns]
            df_test = df_test_raw.reset_index(drop=True)[used_columns]

            total_patients = n_pat_train + n_pat_val + n_pat_test
            total_windows  = len(df_train) + len(df_val) + len(df_test)

            data_counts = {
                "patients": {
                    "train": int(n_pat_train),
                    "val":   int(n_pat_val),
                    "test":  int(n_pat_test),
                },
                "windows": {
                    "train": len(df_train),
                    "val":   len(df_val),
                    "test":  len(df_test),
                },
                "percent": {
                    "patients": {
                        "train": n_pat_train / total_patients * 100,
                        "val":   n_pat_val   / total_patients * 100,
                        "test":  n_pat_test  / total_patients * 100,
                    },
                    "windows": {
                        "train": len(df_train) / total_windows * 100,
                        "val":   len(df_val)   / total_windows * 100,
                        "test":  len(df_test)  / total_windows * 100,
                    },
                },
            }

            print(f"  Train : {n_pat_train} patients, {len(df_train):,} windows")
            print(f"  Val   : {n_pat_val}   patients, {len(df_val):,} windows")
            print(f"  Test  : {n_pat_test}  patients, {len(df_test):,} windows")

            x_train = df_train.iloc[:, :-1].to_numpy()
            y_train = df_train.iloc[:, -1:].to_numpy()
            x_val   = df_val.iloc[:, :-1].to_numpy()
            y_val   = df_val.iloc[:, -1:].to_numpy()
            x_test  = df_test.iloc[:, :-1].to_numpy()
            y_test  = df_test.iloc[:, -1:].to_numpy()

            del df, df_train_raw, df_val_raw, df_test_raw, df_train, df_val, df_test

            save_prefix = str(MODELS_DIR / current_name)

            print(f"\nTraining LSTM — Fold {current_fold}...")
            fold_started_at = datetime.datetime.now()
            t0 = time.time()
            hist, _ = train_fold(x_train, y_train, x_val, y_val, HISTORY_LENGTH, save_prefix)
            train_sec = time.time() - t0
            print(f"Training done ({datetime.timedelta(seconds=train_sec)})")

            # Prediction using best saved weights
            weights_file = f"{save_prefix}.weights.h5"
            model_load   = build_lstm(HISTORY_LENGTH, weights=weights_file, loss_fn=loss_fn)
            x_test_3d    = x_test.reshape(x_test.shape[0], HISTORY_LENGTH, 1)

            t0 = time.time()
            y_pred = model_load.predict(x_test_3d)
            pred_sec = time.time() - t0
            print(f"Prediction done ({datetime.timedelta(seconds=pred_sec)})")

            df_res = pd.DataFrame({'y_test': y_test.ravel(), 'y_predict': y_pred.ravel()})
            df_res = pd.concat([df_res, df_test_info.reset_index(drop=True)], axis=1)

            preds_path = str(PRED_DIR / f'df_test_results_vectors_{current_name}.parquet')
            df_res.to_parquet(preds_path)
            aggregated_results.append(df_res)

            training_plot_prefix = str(PLOTS_TRAIN / current_name)
            get_train_plots_loss(hist, training_plot_prefix)

            hist_csv_path = f"{save_prefix}_history.csv"
            best_json_path = write_best_json(save_prefix, loss_name, ph, HISTORY_LENGTH, current_fold)

            write_fold_summary(
                exp_dir=EXP_DIR,
                loss_name=loss_name, ph=ph, fold_idx=current_fold,
                fold_started_at=fold_started_at,
                train_sec=train_sec, pred_sec=pred_sec,
                weights_path=weights_file,
                hist_csv_path=hist_csv_path,
                best_json_path=best_json_path,
                preds_parquet_path=preds_path,
                training_plot_prefix=training_plot_prefix,
                history_len=HISTORY_LENGTH,
                min_sensor=target.min_sensor,
                max_sensor=target.max_sensor,
                data_file_path=str(target.fold_files[fold_i] if target.fold_files else target.parquet_path),
                data_counts=data_counts,
                balancing_condition=target.balancing_condition,
            )

        # -------- Metric-Then-Aggregate --------
        print("\nCalculating per-fold metrics (Metric-Then-Aggregate)...")
        for fold_n, df_fold in enumerate(aggregated_results, start=1):
            tag = f'_{loss_name}_{ALGORITHM}_H{ph}_Fold{fold_n}'
            df_perf, df_npoints, df_out = test_by_range(
                df_fold, tag,
                show_plot=False, plot_dir=str(FIGURES_DIR),
                minimum_sensor_reading=target.min_sensor,
                maximum_sensor_reading=target.max_sensor,
            )
            df_perf['Fold']    = fold_n
            df_npoints['Fold'] = fold_n
            df_out['Fold']     = fold_n
            df_perf.round(2).to_csv(
                METRICS_DIR / f'metrics_performance_by_range{tag}.csv', index=False)
            df_npoints.to_csv(
                METRICS_DIR / f'number_of_points_by_zones{tag}.csv', index=False)
            df_out.to_csv(
                METRICS_DIR / f'predictions_out_of_limits{tag}.csv', index=False)
            aggregated_results_perf.append(df_perf)
            aggregated_results_npoints.append(df_npoints)
            aggregated_results_outrange.append(df_out)
            print(f"  Fold {fold_n} metrics saved.")

        suffix_mta = f'_{loss_name}_{ALGORITHM}_H{ph}_ALL'

        df_perf_all = pd.concat(aggregated_results_perf, ignore_index=True)
        df_perf_all.round(2).to_csv(
            METRICS_DIR / f'metrics_performance_by_range{suffix_mta}_metric_then_aggregate.csv',
            index=False)
        df_perf_no_fold = df_perf_all.drop(columns=['Fold'])
        df_flat, df_report = fold_results_aggregation(df_perf_no_fold)
        df_flat.round(2).to_csv(
            METRICS_DIR / f'metrics_performance_by_range{suffix_mta}_mta_flat.csv', index=False)
        df_report.round(2).to_csv(
            METRICS_DIR / f'metrics_performance_by_range{suffix_mta}_mta_report.csv', index=False)

        df_npts_all = pd.concat(aggregated_results_npoints, ignore_index=True)
        df_npts_all.to_csv(
            METRICS_DIR / f'number_of_points_by_zones{suffix_mta}_metric_then_aggregate.csv',
            index=False)
        df_npts_flat, df_npts_report = fold_results_aggregation(df_npts_all.drop(columns=['Fold']))
        df_npts_flat.to_csv(
            METRICS_DIR / f'number_of_points_by_zones{suffix_mta}_mta_flat.csv', index=False)
        df_npts_report.to_csv(
            METRICS_DIR / f'number_of_points_by_zones{suffix_mta}_mta_report.csv', index=False)

        df_out_all = pd.concat(aggregated_results_outrange, ignore_index=True)
        df_out_all.to_csv(
            METRICS_DIR / f'predictions_out_of_limits{suffix_mta}_metric_then_aggregate.csv',
            index=False)
        df_out_flat, df_out_report = fold_results_aggregation(df_out_all.drop(columns=['Fold']))
        df_out_flat.to_csv(
            METRICS_DIR / f'predictions_out_of_limits{suffix_mta}_mta_flat.csv', index=False)
        df_out_report.to_csv(
            METRICS_DIR / f'predictions_out_of_limits{suffix_mta}_mta_report.csv', index=False)

        # -------- Aggregate-Then-Metric --------
        print("\nCalculating aggregated metrics (Aggregate-Then-Metric)...")
        df_all_folds = pd.concat(aggregated_results, ignore_index=True)
        df_all_folds.to_parquet(
            PRED_DIR / f'df_test_results_vectors{suffix_mta}.parquet')
        print(f"  Total rows across all folds: {len(df_all_folds):,}")

        df_perf_atm, df_npts_atm, df_out_atm = test_by_range(
            df_all_folds, suffix_mta,
            show_plot=True, plot_dir=str(FIGURES_DIR),
            minimum_sensor_reading=target.min_sensor,
            maximum_sensor_reading=target.max_sensor,
        )
        df_perf_atm.round(2).to_csv(
            METRICS_DIR / f'metrics_performance_by_range{suffix_mta}_aggregate_then_metric.csv',
            index=False)
        df_npts_atm.to_csv(
            METRICS_DIR / f'number_of_points_by_zones{suffix_mta}_aggregate_then_metric.csv',
            index=False)
        df_out_atm.to_csv(
            METRICS_DIR / f'predictions_out_of_limits{suffix_mta}_aggregate_then_metric.csv',
            index=False)

        write_experiment_readme(
            EXP_DIR, loss_name, ph, HISTORY_LENGTH, target.balancing_condition)

        print(f"\n{'='*60}")
        print(f"EXPERIMENT COMPLETE: {EXP_DIR.name}")
        print(f"{'='*60}\n")

    finally:
        if exp_log_file:
            exp_log_file.close()
        sys.stdout = sys.stderr = Tee(original_stdout)
# ----------------------- End single experiment runner -----------------------

# ----------------------- Main -----------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline de entrenamiento LSTM sobre datasets balanceados o el original."
    )
    parser.add_argument(
        '--datasets', nargs='*', default=None,
        help='Limitar el procesamiento a estas carpetas de dataset (nombres).'
    )

    parser.add_argument(
    '--technique', type=str, default=None,
    help=(
        'Ejecutar solo la balancing_condition indicada (ej: balanced_age_smote) '
        'para todos los datasets. Los archivos se buscan en '
        'balanced_outputs/ con el patrón *_fold{i}_{group}_{technique}.parquet. '
        'Formato esperado: balanced_{group}_{technique}  (ej: balanced_age_smote).'
    )
)
    parser.add_argument(
        '--file', type=str, default=None,
        help=(
            'Modo archivo único (original con columnas fold_*): ruta al parquet. '
            'El experimento se ejecuta sobre los K_FOLDS folds de ese archivo.'
        )
    )
    parser.add_argument(
        '--fold-dir', type=str, default=None,
        help=(
            'Modo carpeta de folds balanceados: ruta a una carpeta que contenga '
            'exactamente K_FOLDS parquets con patrón *_fold{i}_*.parquet. '
            'Se infieren dataset_name y balancing_condition del nombre de los archivos.'
        )
    )

    args = parser.parse_args()

    sys.stdout = Tee(original_stdout)

    print(f"Run ID  : {GLOBAL_RUN_ID}")
    print(f"Input   : {INPUT_ROOT}")
    print(f"Output  : {OUTPUT_ROOT}")

    # ──────────────────────────────────────────────────────────────────────────
    # MODO 1: archivo original único (columnas fold_0..fold_4)
    # ──────────────────────────────────────────────────────────────────────────
    if args.file:
        path = Path(args.file)
        if not path.exists():
            raise FileNotFoundError(f"Parquet no encontrado: {path}")

        dataset_name = path.parent.name
        target = ExperimentTarget(
            dataset_name        = dataset_name,
            balancing_condition = 'original',
            min_sensor          = SENSOR_LIMITS.get(dataset_name, DEFAULT_SENSOR_LIMITS)[0],
            max_sensor          = SENSOR_LIMITS.get(dataset_name, DEFAULT_SENSOR_LIMITS)[1],
            parquet_path        = path,
            fold_files          = [],
        )
        print("\nMODO: archivo original único")
        print(target)
        run_experiment(target)
        return

    # ──────────────────────────────────────────────────────────────────────────
    # MODO 2: carpeta con K_FOLDS archivos balanceados (uno por fold)
    # ──────────────────────────────────────────────────────────────────────────
    if args.fold_dir:
        fold_dir = Path(args.fold_dir)
        if not fold_dir.exists():
            raise FileNotFoundError(f"Carpeta de folds no encontrada: {fold_dir}")

        fold_pattern = re.compile(r'_fold(?P<fold>\d+)_(?P<group>age|sex)_(?P<technique>.+)\.parquet$')
        fold_dict: dict[int, Path] = {}
        group_name = technique_name = None

        for f in sorted(fold_dir.glob('*.parquet')):
            m = fold_pattern.search(f.name)
            if m:
                fold_idx = int(m.group('fold'))
                fold_dict[fold_idx] = f
                group_name     = group_name     or m.group('group')
                technique_name = technique_name or m.group('technique')

        missing = [i for i in range(K_FOLDS) if i not in fold_dict]
        if missing:
            raise ValueError(f"Faltan los folds {missing} en {fold_dir}")

        fold_files   = [fold_dict[i] for i in range(K_FOLDS)]
        dataset_name = fold_dir.parent.parent.name  # …/<dataset>/balanced_outputs/<aquí>
        condition    = f"balanced_{group_name}_{technique_name}"

        target = ExperimentTarget(
            dataset_name        = dataset_name,
            balancing_condition = condition,
            min_sensor          = SENSOR_LIMITS.get(dataset_name, DEFAULT_SENSOR_LIMITS)[0],
            max_sensor          = SENSOR_LIMITS.get(dataset_name, DEFAULT_SENSOR_LIMITS)[1],
            parquet_path        = None,
            fold_files          = fold_files,
        )
        print("\nMODO: carpeta de folds balanceados")
        print(target)
        run_experiment(target)
        return
    
    # ──────────────────────────────────────────────────────────────────────────
    # MODO 4: --technique  → ejecutar una balancing_condition concreta
    #          para todos los datasets
    # ──────────────────────────────────────────────────────────────────────────
    if args.technique:
        condition = args.technique  # ej: 'balanced_age_smote'

        # Validar formato: debe empezar por 'balanced_age_' o 'balanced_sex_'
        m = re.match(r'^balanced_(?P<group>age|sex)_(?P<technique>.+)$', condition)
        if not m:
            raise ValueError(
                f"--technique debe tener el formato 'balanced_{{age|sex}}_{{method}}'. "
                f"Recibido: '{condition}'"
            )
        group_name     = m.group('group')      # 'age' o 'sex'
        technique_name = m.group('technique')  # ej: 'smote'

        fold_pattern = re.compile(
            r'^(?P<stem>.+)_fold(?P<fold>\d+)_'
            + re.escape(group_name) + r'_'
            + re.escape(technique_name) + r'\.parquet$'
        )

        targets = []
        dataset_dirs = sorted(INPUT_ROOT.iterdir()) if not args.datasets else [
            INPUT_ROOT / d for d in args.datasets
        ]

        for dataset_dir in dataset_dirs:
            if not dataset_dir.is_dir():
                continue

            dataset_name = detect_dataset_name(dataset_dir.name)
            min_s, max_s = SENSOR_LIMITS.get(dataset_name, DEFAULT_SENSOR_LIMITS)
            balanced_dir = dataset_dir / 'balanced_outputs'

            if not balanced_dir.exists():
                print(f"[WARN] {dataset_name}: no existe balanced_outputs/, se omite.")
                continue

            fold_dict: dict[int, Path] = {}
            for f in sorted(balanced_dir.glob('*.parquet')):
                fm = fold_pattern.match(f.name)
                if fm:
                    fold_dict[int(fm.group('fold'))] = f

            missing = [i for i in range(K_FOLDS) if i not in fold_dict]
            if missing:
                print(
                    f"[WARN] {dataset_name} | {condition}: "
                    f"faltan folds {missing}; experimento omitido."
                )
                continue

            targets.append(ExperimentTarget(
                dataset_name        = dataset_name,
                balancing_condition = condition,
                min_sensor          = min_s,
                max_sensor          = max_s,
                parquet_path        = None,
                fold_files          = [fold_dict[i] for i in range(K_FOLDS)],
            ))

        if not targets:
            print(f"No se encontraron archivos para la condición '{condition}'.")
            return

        print(f"\nMODO: --technique '{condition}' → {len(targets)} dataset(s)")
        for i, target in enumerate(targets, 1):
            print(f"\n[{i}/{len(targets)}] {target}")
            try:
                run_experiment(target)
            except Exception as exc:
                print(f"ERROR en {target.dataset_name}: {exc}")
                import traceback
                traceback.print_exc()

        print("DONE")
        return

    # ──────────────────────────────────────────────────────────────────────────
    # MODO 3: descubrimiento automático (por defecto)
    # ──────────────────────────────────────────────────────────────────────────
    targets = discover_targets(dataset_filter=args.datasets)

    if not targets:
        print("No se encontraron archivos parquet para procesar.")
        return

    print(f"Encontrados {len(targets)} targets")

    for i, target in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] {target}")
        try:
            run_experiment(target)
        except Exception as exc:
            print(f"ERROR: {exc}")
            import traceback
            traceback.print_exc()

    print("DONE")

if __name__ == "__main__":
    main()