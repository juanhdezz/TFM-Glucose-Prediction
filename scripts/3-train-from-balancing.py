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
    """Describes one parquet file to be used as input for a training run."""
    parquet_path: Path
    dataset_name: str
    balancing_condition: str   # e.g. 'original', 'balanced_age_smote', 'balanced_sex_oversampling'
    min_sensor: float
    max_sensor: float

    def __str__(self):
        return f"{self.dataset_name} | {self.balancing_condition} | {self.parquet_path.name}"


def detect_dataset_name(folder_name: str) -> str:
    """
    Map the directory name to the canonical dataset name used in filenames.
    The folder names in /data/input/ are the canonical names themselves.
    """
    return folder_name


def discover_targets(dataset_filter: list[str] | None = None) -> list[ExperimentTarget]:
    """
    Walk INPUT_ROOT and collect all parquet files that should be processed.

    Rules:
      - Skip anything inside a 'balanced_outputs' directory.
      - Accept windows_with_5folds_{DATASET}_*_PH{HORIZON}.parquet  (original)
      - Accept {DATASET}_balanced_{group}_{method}_PH{HORIZON}.parquet (compacted)
      - Reject everything else.
    """
    targets = []

    original_pattern  = re.compile(
        r'^windows_with_5folds_.+_PH' + str(HORIZON) + r'\.parquet$'
    )
    balanced_pattern  = re.compile(
        r'^(?P<dataset>.+?)_balanced_(?P<group>age|sex)_(?P<method>[^_].*?)_PH'
        + str(HORIZON) + r'\.parquet$'
    )

    for dataset_dir in sorted(INPUT_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = detect_dataset_name(dataset_dir.name)
        
        # Filtro por dataset si se especifica
        if dataset_filter and dataset_name not in dataset_filter:
            continue

        dataset_name = detect_dataset_name(dataset_dir.name)
        min_s, max_s = SENSOR_LIMITS.get(dataset_name, DEFAULT_SENSOR_LIMITS)

        for parquet_file in sorted(dataset_dir.glob('*.parquet')):
            # Skip anything inside balanced_outputs/ — we only look at the top
            # level of the dataset directory, so glob('*.parquet') already
            # excludes subdirectories.  The check below is a belt-and-suspenders
            # guard in case the layout ever changes.
            if 'balanced_outputs' in str(parquet_file):
                continue

            fname = parquet_file.name

            # --- Original fold file ---
            if original_pattern.match(fname):
                targets.append(ExperimentTarget(
                    parquet_path=parquet_file,
                    dataset_name=dataset_name,
                    balancing_condition='original',
                    min_sensor=min_s,
                    max_sensor=max_s,
                ))
                continue

            # --- Compacted balanced file ---
            m = balanced_pattern.match(fname)
            if m:
                condition = f"balanced_{m.group('group')}_{m.group('method')}"
                targets.append(ExperimentTarget(
                    parquet_path=parquet_file,
                    dataset_name=dataset_name,
                    balancing_condition=condition,
                    min_sensor=min_s,
                    max_sensor=max_s,
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
        print(f"  Parquet file       : {target.parquet_path}")
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

        for fold_i in range(K_FOLDS):
            current_fold = fold_i + 1
            current_name = f"{loss_name}_{ALGORITHM}_H{ph}_Fold{current_fold}"

            print(f"\n---- Fold {current_fold}/{K_FOLDS} ----")
            print(f"Loading: {target.parquet_path}")

            df = pd.read_parquet(target.parquet_path)

            df_train = df[df[f'fold_{fold_i}'] == 'train']
            n_pat_train = df_train['patient_id'].nunique()
            df_train = df_train.reset_index(drop=True)[used_columns]

            df_val = df[df[f'fold_{fold_i}'] == 'val']
            n_pat_val = df_val['patient_id'].nunique()
            df_val = df_val.reset_index(drop=True)[used_columns]

            df_test = df[df[f'fold_{fold_i}'] == 'test']
            n_pat_test = df_test['patient_id'].nunique()
            df_test_info = df_test.reset_index(drop=True)[info_columns]
            df_test = df_test.reset_index(drop=True)[used_columns]

            total_patients = n_pat_train + n_pat_val + n_pat_test
            total_windows  = len(df)

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

            del df, df_train, df_val, df_test

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
                data_file_path=str(target.parquet_path),
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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--datasets', nargs='*', default=None,
        help='Limit processing to dataset folders'
    )


    args = parser.parse_args()

    sys.stdout = Tee(original_stdout)

    print(f"Run ID  : {GLOBAL_RUN_ID}")
    print(f"Input   : {INPUT_ROOT}")
    print(f"Output  : {OUTPUT_ROOT}")

    # ---------------------------
    # NEW MODE: single file run
    # ---------------------------
    if args.file:
        path = Path(args.file)

        if not path.exists():
            raise FileNotFoundError(f"Parquet not found: {path}")

        target = ExperimentTarget(
            parquet_path=path,
            dataset_name=path.parent.name,
            balancing_condition=path.stem,
            min_sensor=SENSOR_LIMITS.get(path.parent.name, DEFAULT_SENSOR_LIMITS)[0],
            max_sensor=SENSOR_LIMITS.get(path.parent.name, DEFAULT_SENSOR_LIMITS)[1],
        )

        print("\nSINGLE FILE MODE")
        print(target)

        run_experiment(target)

        return

    # ---------------------------
    # default mode (your current logic)
    # ---------------------------
    targets = discover_targets(dataset_filter=args.datasets)

    if not targets:
        print("No parquet files found.")
        return

    print(f"Found {len(targets)} targets")

    for i, target in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] {target}")
        try:
            run_experiment(target)
        except Exception as exc:
            print(f"ERROR: {exc}")
            import traceback
            traceback.print_exc()

    print("DONE")