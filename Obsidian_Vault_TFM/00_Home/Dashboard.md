
# 🏠 TFM Dashboard

## Pregunta de Investigación

**¿Mejora la predicción de glucosa al equilibrar la representación demográfica (sexo y edad) durante el entrenamiento?**

---

## Objetivo del TFM

Evaluar el impacto del balanceo demográfico en el rendimiento predictivo de un modelo LSTM para predicción de glucosa en pacientes con Diabetes Mellitus Tipo 1, utilizando datos de monitorización continua (CGM).

---

## 📊 Datasets

| Dataset | Pacientes | Fuente |
|---------|-----------|--------|
| [[DIATREND]] | 54 | Sun et al., 2023 |
| [[REPLACE-BG]] | 226 | Aleppo et al., 2017 |
| [[T1DiabetesGranada]] | 643 | Diaz et al., 2023 |

---

## 🧪 Técnicas de Balanceo

| Técnica | Tipo | Descripción |
|---------|------|-------------|
| [[Oversampling]] | Remuestreo | Duplica muestras de grupos minoritarios |
| [[Undersampling]] | Remuestreo | Elimina muestras de grupos mayoritarios |
| [[Patient-aware Undersampling]] | Remuestreo | Undersampling con límite por paciente |
| [[SMOTE]] | Sintético | Genera muestras sintéticas por interpolación |
| [[Jittering]] | Sintético | Añade ruido a muestras existentes |
| [[Reference Proportional]] | Remuestreo | Ajusta a distribución de referencia |

---

## 🔬 Experimentos Detectados

**Total: 150 experimentos**

| Dataset | Sexo | Edad | Total |
|---------|------|------|-------|
| DIATREND | 20 | 20 | 50 |
| REPLACE-BG | 20 | 20 | 50 |
| T1DiabetesGranada | 20 | 20 | 50 |


## 📁 Estructura del Proyecto

    TFM-Glucose-Prediction/
    ├── data/           # Datasets y archivos generados
    ├── docs/           # Documentación
    ├── scripts/        # Scripts del pipeline
    ├── notebooks/      # EDA y análisis
    └── memoria/        # Memoria LaTeX

---

## 🔗 Quick Links

- [[Roadmap_TFM]] - Hoja de ruta del proyecto
- [[Pregunta_Investigacion]] - Pregunta de investigación detallada
- [[MOC_Datasets]] - Mapa de conocimiento: Datasets
- [[MOC_Balancing]] - Mapa de conocimiento: Balanceo
- [[MOC_Scripts]] - Mapa de conocimiento: Scripts
- [[MOC_Experiments]] - Mapa de conocimiento: Experimentos
- [[MOC_Memoria]] - Mapa de conocimiento: Memoria

---

## 📈 Estado del Proyecto

- ✅ Generación de ventanas
- ✅ Creación de folds (5-fold CV)
- ✅ Balanceo demográfico
- ✅ Compactación de folds
- ⏳ Entrenamiento LSTM
- ⏳ Evaluación de resultados
- ⏳ Redacción de memoria

---

*Última actualización: 2026-06-20 14:53*
