
# 🐍 3-train.py

## Objetivo

RELIEF-T1D — Training & Prediction Script (English)

Directory layout per (loss, algo, PH):
models/
  <prefix>.weights.h5
  <prefix>_history.csv
  <prefix>.best.json
  <prefix>_logs/  # TensorBoard (tfevents)
plots/
  training/
    <prefix>_Loss.pdf
results/
  predictions/
    df_test_results_vectors_<prefix>.parquet
    df_test_results_vectors_<loss>_<algo>_H<PH>_ALL.parquet
  evaluation/
    metrics/
      metrics_performance_by_range_<loss>_<algo>_H<PH>_ALL.csv
      number_of_points_by_zones_<loss>_<algo>_H<PH>_ALL.csv
      predictions_out_of_limits_<loss>_<algo>_H<PH>_ALL.csv
    figures/
      Clarke_Error_Grid_<...>_ALL.png

Each experiment directory contains a self-contained log file (experiment_log.out).
No global log file is created.

---

## Pipeline Position

**Paso 5** del pipeline de procesamiento.

    Raw Data
        ↓
    1-generate_windows
        ↓
    2-folds_creation
        ↓
    2b-trainset_balancing
        ↓
    2c-compact_folds
        ↓
    3-train

---

## Inputs

| Input | Descripción |
|-------|-------------|
| ... | ... |

---

## Outputs

| Output | Descripción |
|--------|-------------|
| ... | ... |

---

## Funciones Principales

### `main()`
Función principal del script.

### Otras funciones
[Documentar funciones clave del script]

---

## Dependencias

- pandas
- numpy
- scikit-learn
- [...]

---

## Uso

    python 3-train.py [--args]

---

## Enlaces

- [[MOC_Scripts]]
- [[Pipeline]]
