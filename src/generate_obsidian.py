#!/usr/bin/env python3
"""
Obsidian Vault Generator for TFM "Glucose Prediction"
====================================================

This script generates a complete Obsidian knowledge base (Vault) for the
TFM project. It automatically creates:

1. A hierarchical folder structure with MOCs (Maps of Content)
2. Markdown notes with frontmatter and wikilinks
3. Technical documentation extracted from the project structure
4. Experiment discovery from balanced_outputs directories
5. Integration with LaTeX memory chapters
6. Dashboard and roadmap pages

Usage:
    python generate_obsidian_vault.py

The vault will be created at: Obsidian_Vault_TFM/
"""

import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field

# ============================================================================
# CONFIGURATION
# ============================================================================

VAULT_ROOT = Path("Obsidian_Vault_TFM")
PROJECT_ROOT = Path(__file__).resolve().parent

# Project structure paths (relative to project root)
PATHS = {
    "data": PROJECT_ROOT / "data",
    "docs": PROJECT_ROOT / "docs",
    "scripts": PROJECT_ROOT / "scripts",
    "notebooks": PROJECT_ROOT / "notebooks",
    "memory": PROJECT_ROOT / "memoria" / "capitulos",
}

# Datasets discovered from data directory
DATASETS = ["DIATREND", "REPLACE-BG", "T1DiabetesGranada"]

# Balancing techniques
BALANCING_TECHNIQUES = [
    "oversampling",
    "undersampling",
    "patient_aware_undersampling",
    "smote",
    "jittering",
    "reference_proportional",
]

# Demographic dimensions
DEMOGRAPHIC_DIMENSIONS = ["sex", "age"]

# Scripts to document
SCRIPTS = [
    "1-generate_windows.py",
    "2-folds_creation.py",
    "2b-trainset_balancing.py",
    "2c-compact_folds.py",
    "3-train.py",
    "loss_functions.py",
    "test_predictions.py",
]

# LaTeX chapters
LATEX_CHAPTERS = {
    "01_Introduccion.tex": "Introducción",
    "02_Estado_del_arte.tex": "Estado del Arte",
    "03_Desarrollo_del_TFM.tex": "Desarrollo del TFM",
    "04_Resultados_y_Discusion.tex": "Resultados y Discusión",
    "05_Conclusiones.tex": "Conclusiones",
}

# Notebooks to document
NOTEBOOKS = [
    "01_EDA/00_Quick_Inspection.ipynb",
    "01_EDA/01_Inspection_DIATREND.ipynb",
    "01_EDA/01_Inspection_ReplaceBG.ipynb",
    "01_EDA/01_Inspection_T1DiabetesGranada.ipynb",
]

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Experiment:
    """An experiment discovered from the balanced_outputs structure."""
    dataset: str
    demographic: str
    technique: str
    fold: int
    prediction_horizon: str
    path: Path

    @property
    def name(self) -> str:
        return f"{self.dataset}_{self.demographic}_{self.technique}_fold{self.fold}"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def write_md(path: Path, content: str, frontmatter: Optional[Dict] = None) -> None:
    """Write a markdown file with optional YAML frontmatter."""
    ensure_dir(path.parent)
    
    lines = []
    if frontmatter:
        lines.append("---")
        for key, value in frontmatter.items():
            if isinstance(value, list):
                lines.append(f'{key}: [{", ".join(value)}]')
            elif isinstance(value, bool):
                lines.append(f'{key}: {"true" if value else "false"}')
            else:
                lines.append(f'{key}: {value}')
        lines.append("---")
        lines.append("")
    
    lines.append(content)
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def discover_experiments() -> List[Experiment]:
    """Discover all experiments from balanced_outputs directories."""
    experiments = []
    
    for dataset in DATASETS:
        balanced_dir = PATHS["data"] / dataset / "balanced_outputs"
        if not balanced_dir.exists():
            continue
        
        # Pattern: windows_with_5folds_*_PH4_fold{X}_{demographic}_{technique}.parquet
        pattern = r"windows_with_5folds_.*_PH(\d+)_fold(\d+)_(\w+)_(\w+)\.parquet"
        
        for file_path in balanced_dir.glob("*.parquet"):
            match = re.search(pattern, file_path.name)
            if match:
                horizon, fold, demographic, technique = match.groups()
                experiments.append(Experiment(
                    dataset=dataset,
                    demographic=demographic,
                    technique=technique,
                    fold=int(fold),
                    prediction_horizon=f"PH{horizon}",
                    path=file_path,
                ))
    
    return experiments


def discover_compact_files() -> Dict[str, List[Path]]:
    """Discover compact balanced files at dataset root level."""
    compact = {}
    for dataset in DATASETS:
        dataset_dir = PATHS["data"] / dataset
        if not dataset_dir.exists():
            continue
        # Pattern: {dataset}_balanced_{demographic}_{technique}_PH4.parquet
        pattern = rf"{dataset}_balanced_(\w+)_(\w+)_PH4\.parquet"
        compact[dataset] = []
        for f in dataset_dir.glob("*.parquet"):
            if re.search(pattern, f.name):
                compact[dataset].append(f)
    return compact


def discover_patient_info() -> Dict[str, Path]:
    """Discover patient info files."""
    patient_info = {}
    for dataset in DATASETS:
        dataset_dir = PATHS["data"] / dataset
        if not dataset_dir.exists():
            continue
        for pattern in ["Patient_info.parquet", "Patient_info.csv", "patient_info.parquet"]:
            p = dataset_dir / pattern
            if p.exists():
                patient_info[dataset] = p
                break
    return patient_info


def discover_glucose_files() -> Dict[str, Path]:
    """Discover glucose measurement files."""
    glucose_files = {}
    for dataset in DATASETS:
        dataset_dir = PATHS["data"] / dataset
        if not dataset_dir.exists():
            continue
        pattern = f"Glucose_measurements_*_{dataset}_FILTERED*.parquet"
        matches = list(dataset_dir.glob(pattern))
        if matches:
            glucose_files[dataset] = matches[0]
    return glucose_files


# ============================================================================
# NOTE GENERATORS
# ============================================================================

def generate_dashboard() -> str:
    """Generate the main dashboard page."""
    experiments = discover_experiments()
    compact = discover_compact_files()
    
    exp_by_dataset = {}
    for exp in experiments:
        exp_by_dataset.setdefault(exp.dataset, []).append(exp)
    
    total_experiments = len(experiments)
    
    content = f"""
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

**Total: {total_experiments} experimentos**

| Dataset | Sexo | Edad | Total |
|---------|------|------|-------|
"""

    for dataset in DATASETS:
        dataset_exps = exp_by_dataset.get(dataset, [])
        sex_count = len([e for e in dataset_exps if e.demographic == "sex"])
        age_count = len([e for e in dataset_exps if e.demographic == "age"])
        content += f"| {dataset} | {sex_count} | {age_count} | {len(dataset_exps)} |\n"

    content += f"""

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

*Última actualización: {datetime.now().strftime("%Y-%m-%d %H:%M")}*
"""
    return content


def generate_roadmap() -> str:
    """Generate the project roadmap page."""
    return f"""
# 🗺️ Roadmap del TFM

## Fase 1: Preparación de Datos
- [x] Curación de datos de glucemia
- [x] Curación de datos demográficos (sexo, edad)
- [x] Generación de ventanas deslizantes (HW=8, PH=4)
- [x] Creación de 5-fold cross-validation

## Fase 2: Balanceo Demográfico
- [x] Implementación de [[Undersampling]]
- [x] Implementación de [[Oversampling]]
- [x] Implementación de [[Patient-aware Undersampling]]
- [x] Implementación de [[SMOTE]]
- [x] Implementación de [[Jittering]]
- [x] Generación de archivos balanceados por fold

## Fase 3: Entrenamiento y Evaluación
- [ ] Entrenamiento de modelo LSTM baseline
- [ ] Entrenamiento con balanceo por sexo
- [ ] Entrenamiento con balanceo por edad
- [ ] Evaluación de disparidad por subgrupo

## Fase 4: Análisis y Memoria
- [ ] Análisis de resultados
- [ ] Redacción de capítulo de resultados
- [ ] Redacción de conclusiones

---

## 📅 Cronograma Estimado

| Fase | Duración | Estado |
|------|----------|--------|
| Preparación de Datos | 2 semanas | ✅ Completado |
| Balanceo Demográfico | 2 semanas | ✅ Completado |
| Entrenamiento | 3 semanas | ⏳ En progreso |
| Análisis y Memoria | 3 semanas | ⏳ Pendiente |

---

## 🎯 Próximos Pasos

1. Ejecutar entrenamiento para todas las combinaciones
2. Comparar métricas entre experimentos
3. Analizar disparidad por subgrupo demográfico
4. Documentar resultados en la memoria
"""
    return content


def generate_research_question() -> str:
    """Generate the research question page."""
    return """
# ❓ Pregunta de Investigación

## Pregunta Principal

**¿Cómo afecta la composición demográfica —sexo y edad— del conjunto de entrenamiento al rendimiento predictivo de un modelo LSTM para predicción de glucosa, y puede corregirse mediante técnicas de balanceo aplicadas a las ventanas de entrenamiento?**

---

## Hipótesis

1. **Hipótesis nula (H₀):** El balanceo demográfico del conjunto de entrenamiento no tiene un efecto significativo en el rendimiento predictivo del modelo LSTM en subgrupos demográficos infrarrepresentados.

2. **Hipótesis alternativa (H₁):** El balanceo demográfico del conjunto de entrenamiento mejora significativamente el rendimiento predictivo del modelo LSTM en subgrupos demográficos infrarrepresentados, reduciendo la disparidad de rendimiento entre grupos.

---

## Sub-preguntas

1. ¿Existe disparidad de rendimiento por sexo en modelos LSTM entrenados sin balanceo demográfico?

2. ¿Existe disparidad de rendimiento por edad en modelos LSTM entrenados sin balanceo demográfico?

3. ¿Qué técnica de balanceo ([[Oversampling]], [[Undersampling]], [[Patient-aware Undersampling]], [[SMOTE]], [[Jittering]]) es más efectiva para mitigar la disparidad de rendimiento?

4. ¿El efecto del balanceo es consistente a través de los tres datasets ([[DIATREND]], [[REPLACE-BG]], [[T1DiabetesGranada]])?

5. ¿La paridad demográfica en la composición del dataset garantiza la paridad en el rendimiento del modelo?

---

## Justificación Fisiológica

- [[Sexo]]: Las hormonas sexuales y la distribución de masa corporal modulan la homeostasis de la glucosa.
- [[Edad]]: La dinámica glucémica difiere sustancialmente entre grupos etarios.

## Justificación Metodológica

- Los modelos LSTM minimizan una pérdida agregada sobre el conjunto de entrenamiento.
- Si el dataset está dominado por un subgrupo demográfico, el modelo optimiza para ese subgrupo.
- El balanceo corrige esta distorsión sin modificar la arquitectura del modelo.

---

## Restricciones Metodológicas

1. El balanceo se aplica **exclusivamente** al conjunto de entrenamiento.
2. Validation y Test permanecen intactos (no se balancean).
3. El modelo LSTM no se modifica (solo se transforman los datos de entrada).
4. Validación cruzada patient-wise (un paciente no aparece en train y test simultáneamente).

---

## Referencias Clave

- [[Wang2024Disparate]]: Disparidad por sexo y edad en enfermedades crónicas
- [[Ovalle2024Racial]]: Disparidad racial en predicción de glucosa con LSTM
- [[Straw2022SexBias]]: Sesgo por sexo en modelos clínicos
- [[Ricci2024Sociodemographic]]: Revisión de sesgo sociodemográfico en ML clínico
"""


def generate_dataset_note(dataset_name: str) -> str:
    """Generate a note for a specific dataset."""
    patient_info = discover_patient_info().get(dataset_name)
    glucose_file = discover_glucose_files().get(dataset_name)
    
    # Try to count experiments for this dataset
    experiments = [e for e in discover_experiments() if e.dataset == dataset_name]
    sex_exps = len([e for e in experiments if e.demographic == "sex"])
    age_exps = len([e for e in experiments if e.demographic == "age"])
    
    content = f"""
# 📊 {dataset_name}

## Descripción

[[{dataset_name}]] es uno de los tres datasets utilizados en este TFM.

---

## Origen de los Datos

### Datos de Glucemia
- **Archivo:** `Glucose_measurements_{dataset_name}_FILTERED_*.parquet`
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

**Total experimentos: {len(experiments)}**
- **Por sexo:** {sex_exps}
- **Por edad:** {age_exps}

### Técnicas aplicadas
"""

    for technique in BALANCING_TECHNIQUES:
        technique_exps = [e for e in experiments if e.technique == technique]
        if technique_exps:
            content += f"- [[{technique.capitalize()}]]: {len(technique_exps)} experimentos\n"

    content += f"""

---

## Archivos Relacionados

### Archivos Originales
- **Glucemia:** `[[{glucose_file.name if glucose_file else "No encontrado"}]]`
- **Demografía:** `[[{patient_info.name if patient_info else "No encontrado"}]]`

### Archivos Generados
- **Ventanas:** `windows_with_5folds_{dataset_name}_*.parquet`
- **Balanceados (compactos):**
"""

    compact_files = discover_compact_files().get(dataset_name, [])
    for cf in compact_files:
        content += f"  - `{cf.name}`\n"

    content += f"""
---

## Enlaces

- [[MOC_Datasets]]
- [[MOC_Experiments]]
- [[Pregunta_Investigacion]]
"""
    return content


def generate_technique_note(technique: str) -> str:
    """Generate a note for a balancing technique."""
    tech_name = technique.capitalize()
    tech_desc = {
        "oversampling": "Duplica aleatoriamente muestras del grupo minoritario.",
        "undersampling": "Elimina aleatoriamente muestras del grupo mayoritario.",
        "patient_aware_undersampling": "Elimina muestras con un límite máximo por paciente, evitando que un único paciente domine el dataset.",
        "smote": "Synthetic Minority Oversampling Technique: genera muestras sintéticas mediante interpolación entre vecinos cercanos.",
        "jittering": "Añade ruido gaussiano a muestras existentes para crear variaciones sintéticas.",
        "reference_proportional": "Re-muestrea para alcanzar una distribución de referencia poblacional.",
    }
    
    desc = tech_desc.get(technique, "Técnica de balanceo demográfico.")
    
    # Check if this technique appears in compact files
    compact_count = 0
    for dataset in DATASETS:
        compact_files = discover_compact_files().get(dataset, [])
        for f in compact_files:
            if technique in f.name:
                compact_count += 1
    
    content = f"""
# 🧪 {tech_name}

## Definición

{desc}

---

## Fundamento Matemático

### {tech_name}

[Descripción matemática de la técnica]

---

## Implementación en este TFM

- **Script:** [[2b-trainset_balancing.py]]
- **Función principal:** `resample_group()`

### Pseudocódigo

    def {technique}_group(group_df, target, rng):
        # Implementación de {technique}
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
{"" if technique == "patient_aware_undersampling" else "⚠️ "}**Pseudo-replicación:** El oversampling puede generar redundancia exacta.

### Riesgo de Sobreajuste
⚠️ **Sobreajuste:** Las técnicas sintéticas pueden hacer que el modelo memorice patrones artificiales.

### Compatibilidad con Series Temporales
⚠️ **Violación de autocorrelación:** SMOTE y Jittering pueden generar secuencias que no respetan la dinámica temporal de la glucosa.

---

## Experimentos

### Archivos Compactos Generados
**Total: {compact_count}**

"""

    for dataset in DATASETS:
        compact_files = discover_compact_files().get(dataset, [])
        dataset_exps = [f for f in compact_files if technique in f.name]
        if dataset_exps:
            content += f"#### {dataset}\n"
            for f in dataset_exps:
                content += f"- `{f.name}`\n"
            content += "\n"

    content += f"""
---

## Referencias

- [[Kamiran2012Data]]: Data preprocessing techniques for classification without discrimination
- [[Chawla2002SMOTE]]: SMOTE: Synthetic Minority Over-sampling Technique

---

## Enlaces

- [[MOC_Balancing]]
- [[Pregunta_Investigacion]]
- [[2b-trainset_balancing]]
"""
    return content


def generate_script_note(script_name: str) -> str:
    """Generate a note for a script."""
    script_path = PATHS["scripts"] / script_name
    if not script_path.exists():
        return f"# {script_name}\n\n**Archivo no encontrado en el sistema de archivos.**"

    # Try to read the script and extract docstring
    docstring = ""
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Extract docstring (between triple quotes)
            import re
            match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            if match:
                docstring = match.group(1).strip()
    except:
        pass

    # Determine script position in pipeline
    pipeline_order = ["1-generate_windows.py", "2-folds_creation.py", 
                      "2b-trainset_balancing.py", "2c-compact_folds.py", 
                      "3-train.py", "loss_functions.py", "test_predictions.py"]
    
    pos = pipeline_order.index(script_name) + 1 if script_name in pipeline_order else 0
    
    content = f"""
# 🐍 {script_name}

## Objetivo

{docstring if docstring else "Script del pipeline de procesamiento."}

---

## Pipeline Position

**Paso {pos}** del pipeline de procesamiento.

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

    python {script_name} [--args]

---

## Enlaces

- [[MOC_Scripts]]
- [[Pipeline]]
"""
    return content


def generate_moc_datasets() -> str:
    """Generate MOC for datasets."""
    content = """
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
"""
    return content


def generate_moc_balancing() -> str:
    """Generate MOC for balancing techniques."""
    content = """
# 🗺️ MOC: Técnicas de Balanceo

## Índice de Técnicas

### Remuestreo
- [[Oversampling]]
- [[Undersampling]]
- [[Patient-aware Undersampling]]

### Generación Sintética
- [[SMOTE]]
- [[Jittering]]

### Basado en Referencia
- [[Reference Proportional]]

---

## Dimensiones Demográficas

### [[Sexo]]
- Balanceo entre grupos `M` y `F`

### [[Edad]]
- Balanceo entre 5 grupos etarios:
  - `<=18`
  - `19-30`
  - `31-45`
  - `46-60`
  - `>60`

---

## Matriz de Experimentos

| Dataset | Técnica | Sexo | Edad |
|---------|---------|------|------|
| DIATREND | Oversampling | ✅ | ✅ |
| DIATREND | Undersampling | ✅ | ✅ |
| DIATREND | Patient-aware Undersampling | ✅ | ✅ |
| DIATREND | SMOTE | ✅ | ✅ |
| DIATREND | Jittering | ✅ | ✅ |
| ... | ... | ... | ... |

---

## Decisiones de Diseño

### ¿Por qué balancear por ventana y no por paciente?
- Un paciente con una serie larga genera más ventanas.
- El modelo se entrena sobre ventanas, no sobre pacientes.
- Balancear por paciente no garantiza balance por ventana.

### ¿Por qué no modificar el modelo?
- Restricción metodológica: el balanceo debe ser un preprocesado.
- Permite comparar directamente el efecto del balanceo.

---

## Enlaces
- [[Dashboard]]
- [[Pregunta_Investigacion]]
"""
    return content


def generate_moc_experiments() -> str:
    """Generate MOC for experiments."""
    experiments = discover_experiments()
    
    content = f"""
# 🗺️ MOC: Experimentos

## Resumen

**Total de experimentos: {len(experiments)}**

| Dataset | Sexo | Edad | Total |
|---------|------|------|-------|
"""

    exp_by_dataset = {}
    for exp in experiments:
        exp_by_dataset.setdefault(exp.dataset, []).append(exp)
    
    for dataset in DATASETS:
        dataset_exps = exp_by_dataset.get(dataset, [])
        sex_count = len([e for e in dataset_exps if e.demographic == "sex"])
        age_count = len([e for e in dataset_exps if e.demographic == "age"])
        content += f"| {dataset} | {sex_count} | {age_count} | {len(dataset_exps)} |\n"

    content += """
---

## Experimentos por Dataset

### DIATREND
"""

    for exp in exp_by_dataset.get("DIATREND", []):
        content += f"- [[{exp.name}]]\n"

    content += """
### REPLACE-BG
"""

    for exp in exp_by_dataset.get("REPLACE-BG", []):
        content += f"- [[{exp.name}]]\n"

    content += """
### T1DiabetesGranada
"""

    for exp in exp_by_dataset.get("T1DiabetesGranada", []):
        content += f"- [[{exp.name}]]\n"

    content += """
---

## Enlaces
- [[Dashboard]]
- [[MOC_Balancing]]
- [[MOC_Datasets]]
"""
    return content


def generate_moc_scripts() -> str:
    """Generate MOC for scripts."""
    content = """
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
"""
    return content


def generate_moc_memoria() -> str:
    """Generate MOC for the LaTeX memory."""
    content = """
# 🗺️ MOC: Memoria (LaTeX)

## Capítulos de la Memoria

### [[Introducción]]
- Archivo: `01_Introduccion.tex`
- Contexto del TFM y motivación

### [[Estado del Arte]]
- Archivo: `02_Estado_del_arte.tex`
- Revisión de literatura y fundamentos

### [[Desarrollo del TFM]]
- Archivo: `03_Desarrollo_del_TFM.tex`
- Metodología e implementación

### [[Resultados y Discusión]]
- Archivo: `04_Resultados_y_Discusion.tex`
- Experimentos y análisis

### [[Conclusiones]]
- Archivo: `05_Conclusiones.tex`
- Conclusiones y trabajo futuro

---

## Estructura de Carpetas

    memoria/
    ├── capitulos/
    │   ├── 01_Introduccion.tex
    │   ├── 02_Estado_del_arte.tex
    │   ├── 03_Desarrollo_del_TFM.tex
    │   ├── 04_Resultados_y_Discusion.tex
    │   └── 05_Conclusiones.tex
    ├── anexos/
    ├── imagenes/
    └── tablas/

---

## Relación con Notebooks de EDA

| Capítulo | Notebooks Relacionados |
|----------|------------------------|
| Introducción | [[00_Quick_Inspection.ipynb]] |
| Estado del Arte | [[01_Inspection_T1DiabetesGranada_v2.ipynb]] |
| Desarrollo | [[Get_patient_info_variables.ipynb]] |

---

## Enlaces
- [[Dashboard]]
- [[MOC_Scripts]]
"""
    return content


def generate_memory_chapter_note(tex_filename: str, title: str) -> str:
    """Generate a note for a LaTeX chapter."""
    tex_path = PATHS["memory"] / tex_filename
    
    # Try to extract abstract/introduction from the tex file
    abstract = ""
    if tex_path.exists():
        try:
            with open(tex_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Try to find the introduction or abstract
                import re
                # Look for content after chapter or section command
                match = re.search(r'\\section\{Introducción?\}.*?(?=\\section|\\subsection|\\chapter|$)', content, re.DOTALL)
                if match:
                    abstract = match.group(0)[:500] + "..."
        except:
            pass
    
    content = f"""
# 📖 {title}

## Archivo Fuente

`{tex_filename}`

---

## Resumen

{abstract if abstract else "Resumen del capítulo."}

---

## Contenido

- [[{title.replace(' ', '_')}]]: Contenido principal del capítulo

---

## Referencias Citadas

[Lista de referencias citadas en este capítulo]

---

## Relación con Notebooks

| Notebook | Relación |
|----------|----------|
| ... | ... |

---

## Enlaces

- [[MOC_Memoria]]
- [[Dashboard]]
"""
    return content


def generate_pipeline_note() -> str:
    """Generate a note about the data pipeline."""
    content = """
# 🏗️ Pipeline de Datos

## Visión General

    ┌─────────────────┐
    │   Raw Data      │
    │  (CGM + Demos)  │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ 1-generate_windows│
    │  (Ventanas HW=8) │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ 2-folds_creation │
    │   (5-fold CV)    │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │2b-trainset_     │
    │  balancing      │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │2c-compact_folds │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │   3-train.py    │
    │   (LSTM Model)  │
    └─────────────────┘

---

## Restricciones Metodológicas

1. **Balanceo solo sobre train:** No se tocan val ni test.
2. **No modificar el modelo:** Solo se transforman los datos de entrada.
3. **Patient-wise CV:** Un paciente no puede aparecer en train y test simultáneamente.
4. **Semilla fija:** Reproducibilidad de todos los procesos aleatorios.

---

## Detalle de Cada Paso

### 1-generate_windows.py
- **Entrada:** Series de glucosa CGM.
- **Salida:** Ventanas de 8 mediciones con objetivo a PH4.
- **Característica:** Se generan ventanas por paciente respetando la secuencia.

### 2-folds_creation.py
- **Entrada:** Ventanas generadas.
- **Salida:** Asignación de fold (train/val/test) para cada ventana.
- **Característica:** 5 folds estratificados por paciente.

### 2b-trainset_balancing.py
- **Entrada:** Archivo con folds, más datos demográficos (Patient_info).
- **Salida:** Archivos por fold, con el train balanceado según técnica.
- **Característica:** Val y test se copian sin modificar.

### 2c-compact_folds.py
- **Entrada:** Archivos por fold (balanceados).
- **Salida:** Un único archivo por combinación (dataset, demographic, technique).
- **Característica:** Se verifica que val y test sean idénticos a los originales.

### 3-train.py
- **Entrada:** Archivo compacto balanceado.
- **Salida:** Modelo LSTM entrenado y métricas.
- **Característica:** El script no sabe que los datos fueron balanceados.

---

## Enlaces
- [[MOC_Scripts]]
- [[Dashboard]]
"""
    return content


def generate_experiment_note(exp: Experiment) -> str:
    """Generate a note for a specific experiment combination."""
    content = f"""
# 🔬 {exp.name}

## Configuración

- **Dataset:** [[{exp.dataset}]]
- **Dimensión Demográfica:** [[{exp.demographic.capitalize()}]]
- **Técnica:** [[{exp.technique.capitalize()}]]
- **Fold:** {exp.fold}
- **Horizonte:** {exp.prediction_horizon}

---

## Archivo Generado

`{exp.path.name}`

Ruta: `{exp.path}`

---

## Metadatos

- **Fecha de generación:** {datetime.fromtimestamp(exp.path.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if exp.path.exists() else "Desconocida"}
- **Tamaño:** {exp.path.stat().st_size / (1024*1024):.2f} MB (si existe)

---

## Contexto

Este experimento forma parte de la batería de pruebas para evaluar el impacto del balanceo demográfico en la predicción de glucosa.

---

## Enlaces

- [[MOC_Experiments]]
- [[{exp.dataset}]]
- [[{exp.technique.capitalize()}]]
- [[{exp.demographic.capitalize()}]]
"""
    return content


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_vault() -> None:
    """Generate the entire Obsidian vault."""
    print(f"Generating Obsidian vault at: {VAULT_ROOT.resolve()}")
    ensure_dir(VAULT_ROOT)
    
    # 1. Home
    home_dir = VAULT_ROOT / "00_Home"
    ensure_dir(home_dir)
    write_md(home_dir / "Dashboard.md", generate_dashboard())
    write_md(home_dir / "Roadmap_TFM.md", generate_roadmap())
    write_md(home_dir / "Pregunta_Investigacion.md", generate_research_question())
    
    # 2. Knowledge
    knowledge_dir = VAULT_ROOT / "01_Knowledge"
    
    # Datasets
    datasets_dir = knowledge_dir / "Datasets"
    ensure_dir(datasets_dir)
    for ds in DATASETS:
        write_md(datasets_dir / f"{ds}.md", generate_dataset_note(ds))
    
    # Balancing techniques
    balancing_dir = knowledge_dir / "Balancing"
    ensure_dir(balancing_dir)
    for tech in BALANCING_TECHNIQUES:
        write_md(balancing_dir / f"{tech.capitalize()}.md", generate_technique_note(tech))
    
    # MOCs
    mocs_dir = VAULT_ROOT / "05_MOCs"
    ensure_dir(mocs_dir)
    write_md(mocs_dir / "MOC_Datasets.md", generate_moc_datasets())
    write_md(mocs_dir / "MOC_Balancing.md", generate_moc_balancing())
    write_md(mocs_dir / "MOC_Experiments.md", generate_moc_experiments())
    write_md(mocs_dir / "MOC_Scripts.md", generate_moc_scripts())
    write_md(mocs_dir / "MOC_Memoria.md", generate_moc_memoria())
    
    # 3. Project
    project_dir = VAULT_ROOT / "02_Project"
    
    # Pipeline
    pipeline_dir = project_dir / "Pipeline"
    ensure_dir(pipeline_dir)
    write_md(pipeline_dir / "Pipeline.md", generate_pipeline_note())
    
    # Scripts
    scripts_dir = project_dir / "Scripts"
    ensure_dir(scripts_dir)
    for script in SCRIPTS:
        write_md(scripts_dir / f"{script}.md", generate_script_note(script))
    
    # Notebooks
    notebooks_dir = project_dir / "Notebooks"
    ensure_dir(notebooks_dir)
    for nb in NOTEBOOKS:
        nb_name = Path(nb).name
        write_md(notebooks_dir / f"{nb_name}.md", f"# {nb_name}\n\nNotebook de EDA: `{nb}`")
    
    # Experiments
    experiments_dir = project_dir / "Experiments"
    ensure_dir(experiments_dir)
    experiments = discover_experiments()
    # Group by (dataset, demographic, technique) to create one note per combination
    from collections import defaultdict
    grouped = defaultdict(list)
    for exp in experiments:
        key = (exp.dataset, exp.demographic, exp.technique)
        grouped[key].append(exp)
    
    for (ds, demo, tech), exps in grouped.items():
        # Create a combined note for this group
        content = f"""
# 🔬 {ds} - {demo.capitalize()} - {tech.capitalize()}

## Configuración

- **Dataset:** [[{ds}]]
- **Dimensión Demográfica:** [[{demo.capitalize()}]]
- **Técnica:** [[{tech.capitalize()}]]
- **Horizonte:** {exps[0].prediction_horizon}

---

## Folds Generados

| Fold | Archivo |
|------|---------|
"""
        for exp in exps:
            content += f"| {exp.fold} | `{exp.path.name}` |\n"
        
        content += f"""
---

## Enlaces

- [[MOC_Experiments]]
- [[{ds}]]
- [[{tech.capitalize()}]]
- [[{demo.capitalize()}]]
"""
        group_name = f"{ds}_{demo}_{tech}"
        write_md(experiments_dir / f"{group_name}.md", content)
    
    # 4. Memory
    memory_dir = VAULT_ROOT / "03_Memory"
    ensure_dir(memory_dir)
    for tex_file, title in LATEX_CHAPTERS.items():
        clean_title = title.replace(" ", "_")
        write_md(memory_dir / f"{clean_title}.md", generate_memory_chapter_note(tex_file, title))
    
    # 5. Artifacts (empty directories)
    artifacts_dir = VAULT_ROOT / "04_Artifacts"
    for sub in ["Parquets", "Figures", "Tables"]:
        ensure_dir(artifacts_dir / sub)
    
    # 6. Templates
    templates_dir = VAULT_ROOT / "Templates"
    ensure_dir(templates_dir)
    template_content = """---
type: 
tags: []
---"""
    write_md(templates_dir / "Default.md", template_content)
    
    print("✅ Vault generation complete.")
    print(f"   - Home: {home_dir}")
    print(f"   - Knowledge: {knowledge_dir}")
    print(f"   - Project: {project_dir}")
    print(f"   - Memory: {memory_dir}")
    print(f"   - MOCs: {mocs_dir}")
    print(f"   - Total experiments discovered: {len(experiments)}")
    print(f"   - Open Obsidian and set the vault folder to: {VAULT_ROOT.resolve()}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Obsidian vault for TFM.")
    parser.add_argument("--vault-root", type=str, default="Obsidian_Vault_TFM",
                        help="Root directory for the vault (default: Obsidian_Vault_TFM)")
    args = parser.parse_args()
    
    # Override VAULT_ROOT if provided
    if args.vault_root:
        VAULT_ROOT = Path(args.vault_root)
    
    generate_vault()