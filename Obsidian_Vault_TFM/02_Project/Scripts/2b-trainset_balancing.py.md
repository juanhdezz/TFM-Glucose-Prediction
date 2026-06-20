
# 🐍 2b-trainset_balancing.py

## Objetivo

Balance training windows by demographic group.

This script joins window parquet files with patient metadata and applies the
balancing techniques described in docs/tecnicas.md only to the rows marked as
train in each fold column. Validation and test rows are preserved.

Outputs are written under data/<dataset>/balanced_outputs as one parquet per
fold, grouping and balancing technique.

---

## Pipeline Position

**Paso 3** del pipeline de procesamiento.

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

    python 2b-trainset_balancing.py [--args]

---

## Enlaces

- [[MOC_Scripts]]
- [[Pipeline]]
