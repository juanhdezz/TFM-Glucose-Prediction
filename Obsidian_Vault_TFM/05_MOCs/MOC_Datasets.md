
# 🗺️ MOC: Datasets

## Índice de Datasets

### Principales
- [[DIATREND]]
- [[REPLACE-BG]]
- [[T1DiabetesGranada]]

---

## Archivos Relacionados

### Datos de Glucemia
- `Glucose_measurements_*_FILTERED_*.parquet`

### Datos Demográficos
- `Patient_info.parquet`
- `Patient_info.csv`

### Ventanas
- `windows_with_5folds_*_PH4.parquet`

### Balanceados
- `*_balanced_*_*_PH4.parquet`

---

## Estructura de Carpetas

    data/
    ├── DIATREND/
    │   ├── Patient_info.parquet
    │   ├── Glucose_measurements_*_FILTERED_*.parquet
    │   ├── windows_with_5folds_*.parquet
    │   └── balanced_outputs/
    │       └── windows_with_5folds_*_foldX_*_*.parquet
    ├── REPLACE-BG/
    │   └── ...
    └── T1DiabetesGranada/
        └── ...

---

## Variables Clave

### Demográficas
- **Sexo:** `M` / `F`
- **Edad:** Continua (discretizada en 5 grupos)

### Glucemia
- **CGM:** Concentración de glucosa en mg/dL
- **PH4:** Predicción a 60 minutos (4 mediciones a 15 min)

---

## Comparativa de Datasets

| Dataset | Pacientes | Muestreo | Seguimiento | Sexo | Edad |
|---------|-----------|----------|-------------|------|------|
| DIATREND | 54 | 5 min | ≤7 años | ✅ | ✅ |
| REPLACE-BG | 226 | 5 min | 26 sem | ✅ | ✅ |
| T1DiabetesGranada | 643 | 15 min | ≤4 años | ✅ | ✅ |

---

## Enlaces
- [[Dashboard]]
- [[Pregunta_Investigacion]]
