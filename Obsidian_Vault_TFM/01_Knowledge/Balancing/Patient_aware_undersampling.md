
# 🧪 Patient_aware_undersampling

## Definición

Elimina muestras con un límite máximo por paciente, evitando que un único paciente domine el dataset.

---

## Fundamento Matemático

### Patient_aware_undersampling

[Descripción matemática de la técnica]

---

## Implementación en este TFM

- **Script:** [[2b-trainset_balancing.py]]
- **Función principal:** `resample_group()`

### Pseudocódigo

    def patient_aware_undersampling_group(group_df, target, rng):
        # Implementación de patient_aware_undersampling
        pass

---

## Aplicación por Dimensión Demográfica

### [[Sexo]]
- **Aplicable:** ✅ Sí
- **Procedimiento:** Se aplica la técnica sobre los grupos "M" y "F".

### [[Edad]]
- **Aplicable:** ✅ Sí
- **Procedimiento:** Se discretiza la edad en 5 grupos y se aplica la técnica sobre estos grupos.

---

## Riesgos y Limitaciones

### Riesgo de Sesgo
**Pseudo-replicación:** El oversampling puede generar redundancia exacta.

### Riesgo de Sobreajuste
⚠️ **Sobreajuste:** Las técnicas sintéticas pueden hacer que el modelo memorice patrones artificiales.

### Compatibilidad con Series Temporales
⚠️ **Violación de autocorrelación:** SMOTE y Jittering pueden generar secuencias que no respetan la dinámica temporal de la glucosa.

---

## Experimentos

### Archivos Compactos Generados
**Total: 6**

#### DIATREND
- `DIATREND_balanced_age_patient_aware_undersampling_PH4.parquet`
- `DIATREND_balanced_sex_patient_aware_undersampling_PH4.parquet`

#### REPLACE-BG
- `REPLACE-BG_balanced_age_patient_aware_undersampling_PH4.parquet`
- `REPLACE-BG_balanced_sex_patient_aware_undersampling_PH4.parquet`

#### T1DiabetesGranada
- `T1DiabetesGranada_balanced_age_patient_aware_undersampling_PH4.parquet`
- `T1DiabetesGranada_balanced_sex_patient_aware_undersampling_PH4.parquet`


---

## Referencias

- [[Kamiran2012Data]]: Data preprocessing techniques for classification without discrimination
- [[Chawla2002SMOTE]]: SMOTE: Synthetic Minority Over-sampling Technique

---

## Enlaces

- [[MOC_Balancing]]
- [[Pregunta_Investigacion]]
- [[2b-trainset_balancing]]
