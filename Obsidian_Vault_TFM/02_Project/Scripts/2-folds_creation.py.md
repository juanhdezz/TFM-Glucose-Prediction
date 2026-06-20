
# 🐍 2-folds_creation.py

## Objetivo

Balanced *k*-fold split (patient‑wise).

    • **Test assignment (unchanged)** – patients are sorted globally by window
      count ↓, then taken in *k*-sized blocks; one patient from each block is
      randomly assigned to each test fold so that big patients are spread
      evenly.  Remaining (< *k*) patients go to the fold with the lightest
      load.

    • **Train / Val assignment (NEW)** – the remaining patients of each fold
      are also processed in blocks of *k* **(8 → 7‑train / 1‑val)**.  The last
      block, even if smaller than *k*, still sends **one random patient to
      validation** and the rest to training.  This guarantees *at least one*
      validation patient per fold.

---

## Pipeline Position

**Paso 2** del pipeline de procesamiento.

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

    python 2-folds_creation.py [--args]

---

## Enlaces

- [[MOC_Scripts]]
- [[Pipeline]]
