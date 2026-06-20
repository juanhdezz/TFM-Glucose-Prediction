
# 🗺️ MOC: Scripts

## Pipeline de Procesamiento

    Raw Data
        ↓
    1-generate_windows.py   # Generación de ventanas deslizantes
        ↓
    2-folds_creation.py    # Creación de 5-fold CV
        ↓
    2b-trainset_balancing.py  # Balanceo demográfico del train
        ↓
    2c-compact_folds.py    # Compactación de folds
        ↓
    3-train.py            # Entrenamiento del modelo LSTM

---

## Scripts Detallados

### [[1-generate_windows.py]]
Genera ventanas deslizantes a partir de series de glucosa.

### [[2-folds_creation.py]]
Asigna cada ventana a un fold (train/val/test).

### [[2b-trainset_balancing.py]]
Aplica balanceo demográfico solo al conjunto de entrenamiento.

### [[2c-compact_folds.py]]
Compacta los folds balanceados en un único archivo.

### [[3-train.py]]
Entrena el modelo LSTM.

### [[loss_functions.py]]
Funciones de pérdida personalizadas.

### [[test_predictions.py]]
Evaluación de predicciones.

---

## Módulos y Dependencias

### Librerías Principales
- `pandas`: Manipulación de datos
- `numpy`: Operaciones numéricas
- `sklearn`: NearestNeighbors para SMOTE
- `torch`: PyTorch para LSTM

---

## Enlaces
- [[Dashboard]]
- [[Pipeline]]
