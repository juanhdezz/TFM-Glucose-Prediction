
# 🐍 1-generate_windows.py

## Objetivo

:param bgl_measurement_dict: Dictionary with patient_id as key and numpy array of temporal series of BGL measurements
      :param history_length: Number of samples used to predict (8 for 120 minutes)
      :param horizon: Prediction horizon in number of windows (2 for 30 minutes, 4 for 60 minutes, and so on)
      :return: Get windows without missing values from one step sliding windows

---

## Pipeline Position

**Paso 1** del pipeline de procesamiento.

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

    python 1-generate_windows.py [--args]

---

## Enlaces

- [[MOC_Scripts]]
- [[Pipeline]]
