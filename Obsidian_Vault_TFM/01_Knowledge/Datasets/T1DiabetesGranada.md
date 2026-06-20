
# 📊 T1DiabetesGranada

## Descripción

[[T1DiabetesGranada]] es uno de los tres datasets utilizados en este TFM.

---

## Origen de los Datos

### Datos de Glucemia
- **Archivo:** `Glucose_measurements_T1DiabetesGranada_FILTERED_*.parquet`
- **Medición:** Monitorización continua de glucosa (CGM)
- **Frecuencia de muestreo:** 5-15 minutos

### Datos Demográficos
- **Archivo:** `Patient_info.parquet` (o `.csv`)
- **Variables disponibles:**
  - `Patient_ID`: Identificador del paciente
  - `Sex`: Sexo del paciente (M/F)
  - `Age`: Edad del paciente

---

## Características

| Propiedad | Valor |
|-----------|-------|
| Pacientes | ... |
| Periodo de seguimiento | ... |
| Frecuencia de muestreo | ... |

---

## Experimentos Generados

**Total experimentos: 50**
- **Por sexo:** 20
- **Por edad:** 20

### Técnicas aplicadas
- [[Oversampling]]: 10 experimentos
- [[Undersampling]]: 20 experimentos
- [[Smote]]: 10 experimentos
- [[Jittering]]: 10 experimentos


---

## Archivos Relacionados

### Archivos Originales
- **Glucemia:** `[[No encontrado]]`
- **Demografía:** `[[Patient_info.csv]]`

### Archivos Generados
- **Ventanas:** `windows_with_5folds_T1DiabetesGranada_*.parquet`
- **Balanceados (compactos):**
  - `T1DiabetesGranada_balanced_age_jittering_PH4.parquet`
  - `T1DiabetesGranada_balanced_age_patient_aware_undersampling_PH4.parquet`
  - `T1DiabetesGranada_balanced_age_smote_PH4.parquet`
  - `T1DiabetesGranada_balanced_age_undersampling_PH4.parquet`
  - `T1DiabetesGranada_balanced_sex_jittering_PH4.parquet`
  - `T1DiabetesGranada_balanced_sex_oversampling_PH4.parquet`
  - `T1DiabetesGranada_balanced_sex_patient_aware_undersampling_PH4.parquet`
  - `T1DiabetesGranada_balanced_sex_smote_PH4.parquet`
  - `T1DiabetesGranada_balanced_sex_undersampling_PH4.parquet`

---

## Enlaces

- [[MOC_Datasets]]
- [[MOC_Experiments]]
- [[Pregunta_Investigacion]]
