# Desarrollo Metodológico: Balanceo Demográfico de Datasets de Glucosa para Predicción con LSTM

## Índice

1. [Contexto y Motivación](#1-contexto-y-motivación)
2. [Descripción de los Datasets de Entrada](#2-descripción-de-los-datasets-de-entrada)
3. [Transformación a Estructura de Ventanas Deslizantes](#3-transformación-a-estructura-de-ventanas-deslizantes)
4. [Esquema de Validación Cruzada: Group K-Fold](#4-esquema-de-validación-cruzada-group-k-fold)
5. [El Problema del Desbalanceo Demográfico](#5-el-problema-del-desbalanceo-demográfico)
6. [Dimensiones de Balanceo](#6-dimensiones-de-balanceo)
7. [Técnicas de Balanceo Implementadas](#7-técnicas-de-balanceo-implementadas)
8. [Metodología de Balanceo por Fold: Diseño sin Data Leakage](#8-metodología-de-balanceo-por-fold-diseño-sin-data-leakage)
9. [Implementación: `2b-trainset_balancing.py`](#9-implementación-2b-trainset_balancingpy)
10. [Adaptación del Pipeline de Entrenamiento: `3-train-from-balancing.py`](#10-adaptación-del-pipeline-de-entrenamiento-3-train-from-balancingpy)
11. [Estructura de Archivos Resultante](#11-estructura-de-archivos-resultante)
12. [Diseño Experimental Completo](#12-diseño-experimental-completo)
13. [Garantías de Integridad Metodológica](#13-garantías-de-integridad-metodológica)

---

## 1. Contexto y Motivación

El objetivo central de este Trabajo de Fin de Máster (TFM) es investigar si la aplicación de **técnicas de balanceo demográfico** sobre los datos de entrenamiento mejora la capacidad predictiva de un modelo LSTM de glucosa cuando se evalúa sobre subgrupos poblacionales específicos. Esta investigación parte de una premisa clínica relevante: los modelos de predicción de glucosa entrenados sobre datasets con distribuciones demográficas sesgadas pueden mostrar peor rendimiento en grupos minoritarios, lo que tiene implicaciones directas en la equidad clínica del sistema.

El dominio de trabajo es la predicción de glucosa en tiempo continuo mediante datos de Monitores Continuos de Glucosa (CGM). A diferencia de los problemas de clasificación, en los que el balanceo busca equilibrar clases de salida, aquí el objetivo es **equilibrar la representación de los pacientes según sus características demográficas** (edad, sexo) en el conjunto de entrenamiento, con el fin de que el modelo no sea "atraído" por los patrones de grupos sobrerrepresentados.

Esta distinción es fundamental: no se balancea la variable de salida `y` (nivel de glucosa futuro), sino la distribución de pacientes en el conjunto de entrenamiento en función de sus atributos demográficos. La hipótesis es que un dataset de entrenamiento más equilibrado demográficamente producirá un modelo con mejor generalización inter-grupo.

---

## 2. Descripción de los Datasets de Entrada

Se trabaja con tres datasets clínicos de CGM, cada uno con características propias de tamaño, demografía y estructura temporal:

| Dataset | Abreviatura | Pacientes aproximados | Observaciones |
|---|---|---|---|
| T1DiabetesGranada | T1DG | ~643 | Dataset local, alta densidad de mediciones |
| REPLACE-BG | REPLACE | Variable | Ensayo clínico multicéntrico |
| DiaTrend | DIAT | Variable | Dataset longitudinal con seguimiento prolongado |

Cada dataset presenta desequilibrios demográficos distintos:

- **Por sexo:** proporción desigual de pacientes masculinos y femeninos.
- **Por edad:** sobrerepresentación de ciertos rangos etarios (adultos de mediana edad) frente a otros (adolescentes, mayores de 60 años).

Además de la información CGM, cada dataset viene acompañado de un archivo de metadatos de paciente (`patient_info.parquet` o `patient_info.csv`) que contiene variables demográficas: sexo, edad o año de nacimiento, fechas de inicio/fin de medición, entre otras.

---

## 3. Transformación a Estructura de Ventanas Deslizantes

El primer paso del pipeline (previo al balanceo) es la transformación de las series temporales de CGM en una estructura tabular de **ventanas deslizantes**. Esta transformación convierte cada medición de glucosa en una fila que contiene:

- **8 valores históricos de glucosa** (`x0` a `x7`): el valor actual y los 7 anteriores, cada uno separado 15 minutos del siguiente. Representan la ventana de observación de 105 minutos.
- **Variable objetivo** (`y`): el valor de glucosa en el horizonte de predicción `PH` (en el experimento principal, `PH=4`, es decir, 60 minutos hacia el futuro: 4 × 15 min).
- **Metadatos temporales**: fecha (`x_date_7`) y hora (`x_time_7`) del último punto de la ventana.
- **Identificador de paciente** (`patient_id`): indispensable para el Group K-Fold y para el balanceo demográfico.

La estructura resultante de cada fila es:

```
x0    x1    x2    x3    x4    x5    x6    x7    y     x_date_7    x_time_7    patient_id
165.0 161.0 141.0 137.0 131.0 128.0 117.0 122.0 122.0 2019-06-10  17:15:00    42
226.0 220.0 215.0 211.0 203.0 194.0 176.0 164.0 180.0 2019-11-14  03:15:00    30
```

Sobre este archivo de ventanas (denominado `windows_with_5folds_<DATASET>_PH<H>.parquet`) se ejecuta a continuación la asignación de folds y el balanceo.

---

## 4. Esquema de Validación Cruzada: Group K-Fold

### 4.1 Por qué Group K-Fold y no K-Fold estándar

En series temporales médicas con múltiples pacientes, la validación cruzada estándar (K-Fold aleatorio) introduce **data leakage a nivel de paciente**: ventanas del mismo paciente pueden acabar simultáneamente en train y en test, lo que infla artificialmente las métricas al evaluar sobre datos "casi vistos". Para evitar esto, se utiliza **Group K-Fold**, donde el grupo es el `patient_id`: **un paciente solo puede aparecer en uno de los tres subconjuntos** (train, val, test) dentro de cada fold.

### 4.2 Codificación de los folds en el archivo de ventanas

El resultado de la asignación de folds se almacena directamente en el archivo de ventanas como columnas adicionales `fold_0`, `fold_1`, ..., `fold_4`. Cada columna contiene, para cada fila (ventana), el rol que tiene esa ventana en ese fold específico: `"train"`, `"val"` o `"test"`.

```
... patient_id  fold_0  fold_1  fold_2  fold_3  fold_4
... 42          train   train   train   val     test
... 30          train   train   test    train   train
... 17          val     test    train   train   train
```

Esta estructura compacta permite que un único archivo contenga toda la información necesaria para realizar los 5 folds de validación cruzada. Sin embargo, como se detalla en la siguiente sección, esta estructura compacta debe ser **descompuesta** antes del balanceo para garantizar la ausencia de data leakage.

### 4.3 Implicaciones para el balanceo

El hecho de que el balanceo deba realizarse **exclusivamente sobre el conjunto de train de cada fold** impone una restricción crítica: no se puede balancear el dataset completo una sola vez y reutilizar el resultado para todos los folds. Cada fold define una partición diferente de los pacientes en train/val/test, por lo que la distribución demográfica del conjunto de train varía de un fold a otro. Balancear antes de particionar equivaldría a "contaminar" los datos de val y test con información artificial generada a partir de ellos mismos.

---

## 5. El Problema del Desbalanceo Demográfico

### 5.1 Naturaleza del desbalanceo en predicción

En clasificación, el desbalanceo se refiere a la distribución desigual de la variable objetivo (clases). En predicción de series temporales, el desbalanceo que nos ocupa es de naturaleza distinta: es un **desbalanceo en la distribución de los sujetos generadores de datos**. Si el 80% de las ventanas de entrenamiento provienen de pacientes masculinos de entre 30 y 50 años, el modelo tenderá a aprender predominantemente los patrones glucémicos de ese subgrupo, a expensas de los patrones de mujeres, adolescentes o pacientes mayores.

### 5.2 Métricas de evaluación del impacto

El impacto del balanceo se evalúa comparando las métricas de predicción (RMSE, MAE, TIR —Time In Range—, TAR —Time Above Range—, TBR —Time Below Range—) obtenidas por el modelo en cada subgrupo demográfico, comparando la condición original (sin balanceo) frente a cada una de las técnicas aplicadas sobre cada dimensión demográfica.

---

## 6. Dimensiones de Balanceo

El balanceo se aplica de forma independiente sobre dos dimensiones demográficas. Cada dimensión genera su propio conjunto de datasets balanceados, y en ningún caso se combinan ambas dimensiones simultáneamente en el mismo experimento.

### 6.1 Dimensión Sexo (`sex`)

La variable de grupo se deriva del campo `Sex` (o `sex`) del archivo `patient_info`. Los valores se normalizan a `"M"` (masculino) y `"F"` (femenino). Pacientes con información no disponible se categorizan como `"Unknown"` y se excluyen del proceso de balanceo (manteniéndose en el dataset sin modificación).

La columna derivada en el dataset de ventanas se denomina `sex_group`.

### 6.2 Dimensión Edad (`age`)

La edad se obtiene preferentemente del campo `Age` del archivo de metadatos. Si este campo no existe, se infiere a partir del año de nacimiento (`Birth_year`) y la fecha de la última medición disponible. La edad numérica se discretiza en **cinco rangos etarios clínicamente relevantes**:

| Etiqueta | Rango de edad |
|---|---|
| `<=18` | Hasta 18 años (adolescentes) |
| `19-30` | 19 a 30 años |
| `31-45` | 31 a 45 años |
| `46-60` | 46 a 60 años |
| `>60` | Mayores de 60 años |

La columna derivada se denomina `age_group`. Esta discretización permite tratar el problema de desbalanceo por edad como un problema de balanceo multi-clase, aplicando las mismas técnicas que para el sexo.

---

## 7. Técnicas de Balanceo Implementadas

Se implementan seis técnicas de balanceo, seleccionadas para cubrir el espectro de enfoques posibles (reducción, generación y combinación) y adaptadas al dominio temporal de series CGM.

### 7.1 `undersampling` — Submuestreo aleatorio

**Principio:** reducir todos los grupos demográficos al tamaño del grupo minoritario, eliminando filas al azar.

**Procedimiento:** para cada grupo demográfico presente en el conjunto de train del fold, se seleccionan aleatoriamente (sin reemplazo) exactamente `min(count_group_i)` ventanas, siendo este mínimo el tamaño del grupo con menos ventanas.

**Ventajas:** simple, elimina el sesgo de forma directa. **Desventajas:** puede suponer una pérdida sustancial de datos en datasets con grupos muy desequilibrados.

**Fórmula del target por grupo:**
```
target = min(n_g  para todo grupo g)
```

### 7.2 `oversampling` — Sobremuestreo aleatorio con reemplazo

**Principio:** aumentar todos los grupos al tamaño del grupo mayoritario, duplicando filas existentes.

**Procedimiento:** para cada grupo, se muestrean aleatoriamente (con reemplazo) ventanas del propio grupo hasta alcanzar `max(count_group_i)`.

**Ventajas:** no pierde información. **Desventajas:** introduce redundancia exacta; el modelo puede sobreajustarse a las ventanas repetidas.

**Fórmula del target por grupo:**
```
target = max(n_g  para todo grupo g)
```

### 7.3 `patient_aware_undersampling` — Submuestreo consciente del paciente

**Principio:** submuestreo que respeta la distribución de pacientes dentro de cada grupo, evitando que el proceso elimine desproporcionadamente las ventanas de un único paciente.

**Procedimiento:**
1. Para el grupo que se va a reducir, se calcula el número máximo de ventanas por paciente como `ceil(target / n_pacientes_en_grupo)`.
2. Para cada paciente del grupo, se toman como máximo ese número de ventanas, seleccionadas aleatoriamente.
3. Si tras este proceso el total supera el `target`, se realiza un submuestreo adicional aleatorio del conjunto resultante.

**Motivación clínica:** en datasets CGM, un único paciente puede tener miles de ventanas. El submuestreo aleatorio puro tiende a eliminar pacientes enteros (por acumulación de probabilidad), lo que reduce la diversidad de patrones glucémicos más que la reducción del volumen. Esta técnica evita ese efecto.

### 7.4 `smote` — Synthetic Minority Over-sampling Technique (adaptado a series temporales)

**Principio:** generar ventanas sintéticas para los grupos minoritarios mediante interpolación entre ventanas reales vecinas en el espacio de características.

**Procedimiento:**
1. Para cada grupo con menos ventanas que el target, se calculan los `k=5` vecinos más cercanos (o `k = n-1` si el grupo tiene pocos ejemplos) en el espacio de características `x0..x7, y`, usando distancia euclídea sobre los vectores normalizados (media cero, varianza unitaria).
2. Para generar cada ventana sintética, se elige aleatoriamente una ventana base y uno de sus k vecinos, y se interpola linealmente con un factor `λ ∈ [0,1]` uniforme:
   ```
   x_synthetic = x_base + λ · (x_neighbor - x_base)
   ```
3. Los metadatos no numéricos (fecha, hora, `patient_id`) se copian del ejemplo base.
4. Los valores sintéticos se recortan a los límites físicos del sensor correspondiente.

**Límites de sensor por dataset:**
- T1DiabetesGranada: [40, 500] mg/dL
- REPLACE-BG: [39, 401] mg/dL
- DiaTrend: [39, 401] mg/dL

**Adaptación temporal:** dado que cada ventana es un vector de mediciones ya preprocesadas (no una serie temporal raw), SMOTE opera sobre el espacio de representación tabular, evitando los problemas de coherencia temporal que surgirían al interpolar directamente sobre series brutas.

### 7.5 `jittering` — Perturbación gaussiana

**Principio:** generar ventanas sintéticas añadiendo ruido gaussiano calibrado a las ventanas existentes del grupo minoritario.

**Procedimiento:**
1. Para cada valor de glucosa en `x0..x7, y`, se calcula la desviación estándar de esa columna dentro del grupo.
2. Para cada ventana sintética necesaria, se selecciona aleatoriamente una ventana base del grupo y se le añade ruido gaussiano con media 0 y desviación `σ_col × scale`, donde `scale = 0.20` (20% de la variabilidad natural de la columna).
3. Se aplica el mismo recorte a los límites del sensor.

**Motivación:** el jittering es más simple que SMOTE pero produce mayor variedad en la estructura temporal de la ventana al perturbar cada punto de forma independiente, simulando variabilidad fisiológica intra-paciente.

### 7.6 `reference_proportional` — Balanceo a proporciones de referencia externas

**Principio:** en lugar de igualar todos los grupos, se ajusta la distribución del conjunto de train a unas **proporciones de referencia** definidas externamente (por ejemplo, la distribución demográfica de la población diabética española según datos epidemiológicos).

**Procedimiento:**
1. Se lee un archivo de referencia (`JSON` o `CSV`) con la proporción objetivo de cada grupo: `{grupo: proporción}`.
2. Se calcula cuántas ventanas debe tener cada grupo para que el tamaño total del fold de train sea el mismo (se preserva el volumen de datos).
3. Para cada grupo: si el target calculado es mayor que el actual → se aplica oversampling; si es menor → se aplica undersampling.
4. Los grupos no presentes en el archivo de referencia se mantienen en su tamaño original.

**Ventaja:** permite incorporar conocimiento epidemiológico externo al proceso de balanceo, orientando el entrenamiento hacia una distribución clínicamente representativa en lugar de una distribución uniforme.

---

## 8. Metodología de Balanceo por Fold: Diseño sin Data Leakage

Esta sección describe el núcleo metodológico del TFM: cómo se realiza el balanceo de forma que sea **matemáticamente correcto** respecto a la validación cruzada, garantizando que ninguna información de val o test influye en el proceso de generación de datos sintéticos o submuestreo.

### 8.1 El principio fundamental: balancear dentro del fold

La regla central es:

> **El balanceo se aplica única y exclusivamente sobre las ventanas asignadas a `train` en el fold `i`. Las ventanas asignadas a `val` y `test` en ese mismo fold nunca se modifican, nunca se muestrean, y nunca participan en el cálculo de estadísticas usadas para el balanceo.**

Esta regla se deriva directamente del principio de no data leakage: si un algoritmo generativo (SMOTE, jittering) calculara estadísticas sobre la totalidad del dataset antes de la partición, estaría "viendo" los datos de test de forma indirecta. Del mismo modo, si se hiciera oversampling sobre el dataset completo y después se particionara, algunas ventanas duplicadas podrían caer en train y otras en test, introduciendo leakage directo.

### 8.2 Flujo de procesamiento por fold

Para cada combinación de `(dataset, fold_i, dimensión_demográfica, técnica)`:

```
┌─────────────────────────────────────────────────────────────┐
│  Dataset de ventanas completo                               │
│  windows_with_5folds_<DATASET>_PH4.parquet                  │
│  (N filas × columnas: x0..x7, y, metadatos, fold_0..fold_4)│
└───────────────────────────┬─────────────────────────────────┘
                            │
                     Paso 1: Seleccionar fold_i
                            │
            ┌───────────────┴──────────────────┐
            │  fold_i == "train"               │  fold_i == "val" o "test"
            │  (ventanas de entrenamiento)      │  (ventanas de evaluación)
            └───────────────┬──────────────────┘
                            │
                     Paso 2: Unir con patient_info
                     (obtener sex_group / age_group
                      para cada ventana de train)
                            │
                     Paso 3: Calcular distribución demográfica
                     de train → {grupo: n_ventanas}
                            │
                     Paso 4: Aplicar técnica de balanceo
                     solo sobre train_con_demografía
                            │
                     train_balanceado
                            │
            ┌───────────────┴──────────────────┐
            │           CONCATENAR              │
            │  train_balanceado + val + test    │
            │  (val y test intactos, sin        │
            │   modificación alguna)            │
            └───────────────┬──────────────────┘
                            │
                     Paso 5: Eliminar columnas fold_*
                     y columnas demográficas auxiliares
                            │
                     Paso 6: Añadir columna "split"
                     ("train" / "val" / "test")
                            │
                     Paso 7: Guardar parquet
                     <stem>_fold{i}_{grupo}_{técnica}.parquet
```

### 8.3 Por qué se eliminan las columnas `fold_*`

El archivo de ventanas original contiene 5 columnas `fold_0` a `fold_4`. Una vez que hemos extraído la información de fold `i` y construido el archivo balanceado para ese fold específico, las otras 4 columnas de fold son **irrelevantes e incorrectas** para ese archivo:

- **Irrelevantes:** el archivo de salida representa únicamente el fold `i`; no tiene sentido que contenga asignaciones de otros folds.
- **Potencialmente confusas:** si el script de entrenamiento las encontrara, podría intentar filtrar por ellas, obteniendo resultados incorrectos.
- **Redundantes:** la columna `split` que se añade contiene exactamente la información necesaria para que el script de entrenamiento distinga train, val y test.

### 8.4 Independencia entre folds

Dado que el proceso se ejecuta independientemente para cada fold, y dado que la semilla aleatoria incorpora el índice del fold y el hash de la condición experimental, cada archivo de salida es **determinista pero distinto** de los demás folds. Esto garantiza reproducibilidad total del experimento.

La semilla se calcula como:
```python
seed + fold_idx * 1000 + abs(hash((dataset_name, group_name, technique))) % 1000
```

---

## 9. Implementación: `2b-trainset_balancing.py`

### 9.1 Estructura general del script

El script está organizado en capas funcionales bien delimitadas:

```
2b-trainset_balancing.py
│
├── CLI (parse_args)
│   └── --datasets, --groups, --techniques, --horizons, --folds, --seed, --reference-file, ...
│
├── Capa de disco
│   ├── discover_dataset_dirs()   → lista carpetas de dataset
│   ├── resolve_patient_info()    → localiza patient_info.parquet/.csv
│   └── load_table()              → carga parquet o csv
│
├── Capa demográfica
│   ├── infer_age_series()        → extrae edad desde "Age" o "Birth_year + fecha"
│   ├── build_demographic_lookup()→ DataFrame indexado por _patient_key con sex_group, age, age_group
│   └── attach_demographics()     → une ventanas de train con el lookup demográfico
│
├── Técnicas de balanceo
│   ├── undersample_group()
│   ├── oversample_group()
│   ├── patient_aware_undersample_group()
│   ├── smote_group()
│   ├── jitter_group()
│   ├── select_target_counts()    → decide cuántas ventanas debe tener cada grupo
│   └── resample_train_fold()     → orquesta el balanceo completo del fold de train
│
└── Orquestación
    ├── process_dataset()         → itera folds × grupos × técnicas para un dataset
    └── main()                    → itera datasets → llama process_dataset()
```

### 9.2 Flujo detallado de `process_dataset()`

```python
for horizon in horizons:
    # 1. Cargar el archivo de ventanas completo
    df_windows = load_table(windows_file)

    for group_name in groups:          # "age", "sex"
        group_col = f"{group_name}_group"

        for fold_idx in folds:         # 0, 1, 2, 3, 4
            fold_col = f"fold_{fold_idx}"

            # 2. Separar train / val+test usando SOLO la columna de este fold
            train_mask    = df_windows[fold_col] == "train"
            val_test_mask = df_windows[fold_col].isin(["val", "test"])

            raw_train     = df_windows[train_mask].copy()
            raw_untouched = df_windows[val_test_mask].copy()

            # 3. Adjuntar demografía al conjunto de train
            train_with_demo = attach_demographics(raw_train, demo_lookup)

            for technique in techniques:
                # 4. Balancear solo el train
                balanced_train = resample_train_fold(
                    train_df   = train_with_demo,
                    group_col  = group_col,
                    technique  = technique,
                    rng        = np.random.default_rng(seed_derivado),
                    sensor_limits = sensor_limits,
                )

                # 5. Añadir columna "split"
                balanced_train["split"] = "train"
                raw_untouched["split"]  = df_windows[val_test_mask][fold_col]  # "val" o "test"

                # 6. Seleccionar solo columnas de salida (sin fold_*)
                out_df = pd.concat([
                    balanced_train[KEEP_COLUMNS + ["split"]],
                    raw_untouched[KEEP_COLUMNS  + ["split"]],
                ], ignore_index=True)

                # 7. Guardar
                out_df.to_parquet(output_dir / f"{stem}_fold{fold_idx}_{group_name}_{technique}.parquet")
```

### 9.3 La función `resample_train_fold()` en detalle

Esta función recibe el dataframe de train con la columna demográfica ya añadida y realiza:

```python
def resample_train_fold(train_df, group_col, technique, rng, sensor_limits, reference_props):
    # 1. Contar ventanas por grupo demográfico
    counts = train_df[group_col].value_counts(dropna=False)
    # Ejemplo: {"M": 12000, "F": 4500}

    # 2. Calcular el target para cada grupo según la técnica
    target_counts = select_target_counts(counts, technique, len(train_df), reference_props)
    # Ejemplo (undersampling): {"M": 4500, "F": 4500}
    # Ejemplo (oversampling):  {"M": 12000, "F": 12000}

    # 3. Para cada grupo, aplicar el resampling
    parts = []
    for group_value, target in target_counts.items():
        group_df = train_df[train_df[group_col] == group_value]
        resampled = resample_group(group_df, target, technique, rng, feature_columns, sensor_limits)
        parts.append(resampled)

    # 4. Concatenar y mezclar aleatoriamente
    balanced = pd.concat(parts, ignore_index=True)
    return balanced.sample(frac=1.0, random_state=rng_state).reset_index(drop=True)
```

### 9.4 Columnas de salida

Cada archivo de salida contiene exactamente las siguientes columnas, sin ninguna columna `fold_*` ni columna demográfica auxiliar:

| Columna | Tipo | Descripción |
|---|---|---|
| `x0` | float | Glucosa t-105 min (o t-7 pasos) |
| `x1` | float | Glucosa t-90 min |
| `x2` | float | Glucosa t-75 min |
| `x3` | float | Glucosa t-60 min |
| `x4` | float | Glucosa t-45 min |
| `x5` | float | Glucosa t-30 min |
| `x6` | float | Glucosa t-15 min |
| `x7` | float | Glucosa t (presente) |
| `y` | float | Glucosa t+PH (objetivo de predicción) |
| `x_date_7` | date | Fecha del punto presente |
| `x_time_7` | time | Hora del punto presente |
| `patient_id` | int/str | Identificador del paciente |
| `split` | str | `"train"`, `"val"` o `"test"` |

### 9.5 Nomenclatura de los archivos de salida

```
<dataset_dir>/balanced_outputs/
    windows_with_5folds_<DATASET>_PH4_fold0_age_undersampling.parquet
    windows_with_5folds_<DATASET>_PH4_fold0_age_oversampling.parquet
    windows_with_5folds_<DATASET>_PH4_fold0_age_smote.parquet
    windows_with_5folds_<DATASET>_PH4_fold0_age_jittering.parquet
    windows_with_5folds_<DATASET>_PH4_fold0_age_patient_aware_undersampling.parquet
    windows_with_5folds_<DATASET>_PH4_fold0_age_reference_proportional.parquet
    windows_with_5folds_<DATASET>_PH4_fold0_sex_undersampling.parquet
    ...
    windows_with_5folds_<DATASET>_PH4_fold4_sex_reference_proportional.parquet
```

Total por dataset: 5 folds × 2 grupos × 6 técnicas = **60 archivos**.

---

## 10. Adaptación del Pipeline de Entrenamiento: `3-train-from-balancing.py`

### 10.1 El problema con la arquitectura original

El script de entrenamiento original estaba diseñado para recibir **un único archivo parquet** que contenía las columnas `fold_0` a `fold_4`. El loop de folds hacía:

```python
# VERSIÓN ORIGINAL (incorrecta para el nuevo esquema)
df = pd.read_parquet(target.parquet_path)           # Una sola lectura

for fold_i in range(K_FOLDS):
    df_train = df[df[f'fold_{fold_i}'] == 'train']  # Filtro por columna fold
    df_val   = df[df[f'fold_{fold_i}'] == 'val']
    df_test  = df[df[f'fold_{fold_i}'] == 'test']
    # ... entrenamiento ...
```

Este diseño era incompatible con el nuevo esquema de salida del balanceo, donde:
- No existe un único archivo con todos los folds.
- No existen columnas `fold_*` en los archivos balanceados.
- Cada fold tiene su propio archivo, identificado por `*_fold{i}_*.parquet`.
- La asignación de cada ventana a train/val/test viene dada por la columna `split`.

### 10.2 Solución: modo dual con detección automática

Se introduce el concepto de **modo de operación dual** en `ExperimentTarget`:

```python
@dataclass
class ExperimentTarget:
    dataset_name:        str
    balancing_condition: str
    min_sensor:          float
    max_sensor:          float

    # Modo original: un único archivo con columnas fold_0..fold_4
    parquet_path: Optional[Path] = None

    # Modo balanceado: 5 archivos, uno por fold, ordenados [fold0, fold1, ..., fold4]
    fold_files: list = None   # List[Path], len = K_FOLDS
```

El modo activo se detecta automáticamente:
```python
is_balanced_mode = bool(target.fold_files)
```

### 10.3 El nuevo loop de folds

```python
is_balanced_mode = bool(target.fold_files)

for fold_i in range(K_FOLDS):

    if is_balanced_mode:
        # ── MODO BALANCEADO: leer el archivo específico de este fold ──────
        fold_path = target.fold_files[fold_i]
        df = pd.read_parquet(fold_path)

        # Filtrar por columna "split" (generada por 2b-trainset_balancing.py)
        split_values = df["split"].astype(str).str.lower()
        df_train_raw = df[split_values == "train"]
        df_val_raw   = df[split_values == "val"]
        df_test_raw  = df[split_values == "test"]

    else:
        # ── MODO ORIGINAL: leer el archivo único y filtrar por fold_i ─────
        df = pd.read_parquet(target.parquet_path)
        fold_values  = df[f"fold_{fold_i}"].astype(str).str.lower()
        df_train_raw = df[fold_values == "train"]
        df_val_raw   = df[fold_values == "val"]
        df_test_raw  = df[fold_values == "test"]

    # A partir de aquí, el código de entrenamiento es idéntico en ambos modos
    df_train    = df_train_raw.reset_index(drop=True)[used_columns]
    df_val      = df_val_raw.reset_index(drop=True)[used_columns]
    df_test     = df_test_raw.reset_index(drop=True)[used_columns]
    df_test_info = df_test_raw.reset_index(drop=True)[info_columns]

    # ... normalización, construcción de tensores, entrenamiento LSTM ...

    del df, df_train_raw, df_val_raw, df_test_raw, df_train, df_val, df_test
```

### 10.4 Descubrimiento automático de targets (`discover_targets()`)

La función `discover_targets()` se reescribe para manejar ambos modos:

**Modo original:** busca en la raíz de cada dataset archivos con el patrón:
```
windows_with_5folds_<DATASET>_*_PH{HORIZON}.parquet
```
Crea un `ExperimentTarget` con `parquet_path` y `fold_files=[]`.

**Modo balanceado:** busca en `<dataset>/balanced_outputs/` archivos con el patrón:
```
*_fold{i}_{group}_{technique}.parquet    (i = 0..K_FOLDS-1)
```
Agrupa los K_FOLDS archivos de cada `(stem, group, technique)` usando un diccionario. Solo crea el `ExperimentTarget` si están presentes **todos los K_FOLDS archivos** (para evitar experimentos incompletos que producirían métricas no comparables):

```python
for (stem, group, technique), fold_dict in sorted(groups_map.items()):
    missing = [i for i in range(K_FOLDS) if i not in fold_dict]
    if missing:
        print(f"[WARN] Faltan folds {missing}; experimento omitido.")
        continue

    fold_files = [fold_dict[i] for i in range(K_FOLDS)]
    targets.append(ExperimentTarget(
        dataset_name        = dataset_name,
        balancing_condition = f"balanced_{group}_{technique}",
        fold_files          = fold_files,
        ...
    ))
```

### 10.5 Modos de ejecución del script de entrenamiento

El script soporta tres modos de invocación:

**Modo 1: Descubrimiento automático (por defecto)**
```bash
python 3-train-from-balancing.py
python 3-train-from-balancing.py --datasets T1DiabetesGranada DiaTrend
```
Descubre y ejecuta todos los targets disponibles (originales + balanceados).

**Modo 2: Archivo original único**
```bash
python 3-train-from-balancing.py --file data/T1DiabetesGranada/windows_with_5folds_T1DiabetesGranada_PH4.parquet
```
Ejecuta la validación cruzada sobre un único archivo con columnas `fold_*`.

**Modo 3: Carpeta de folds balanceados**
```bash
python 3-train-from-balancing.py --fold-dir data/T1DiabetesGranada/balanced_outputs/
```
Ejecuta la validación cruzada sobre los 5 archivos `*_fold{i}_*.parquet` de esa carpeta.

---

## 11. Estructura de Archivos Resultante

```
data/
├── T1DiabetesGranada/
│   ├── patient_info.parquet
│   ├── windows_with_5folds_T1DiabetesGranada_PH4.parquet   ← archivo original
│   └── balanced_outputs/
│       ├── windows_with_5folds_T1DiabetesGranada_PH4_fold0_age_undersampling.parquet
│       ├── windows_with_5folds_T1DiabetesGranada_PH4_fold0_age_oversampling.parquet
│       ├── windows_with_5folds_T1DiabetesGranada_PH4_fold0_age_smote.parquet
│       ├── windows_with_5folds_T1DiabetesGranada_PH4_fold0_age_jittering.parquet
│       ├── windows_with_5folds_T1DiabetesGranada_PH4_fold0_age_patient_aware_undersampling.parquet
│       ├── windows_with_5folds_T1DiabetesGranada_PH4_fold0_age_reference_proportional.parquet
│       ├── windows_with_5folds_T1DiabetesGranada_PH4_fold0_sex_undersampling.parquet
│       ├── ...  (×6 técnicas × 2 grupos × 5 folds = 60 archivos)
│       └── windows_with_5folds_T1DiabetesGranada_PH4_fold4_sex_reference_proportional.parquet
│
├── DiaTrend/
│   ├── patient_info.parquet
│   ├── windows_with_5folds_DiaTrend_PH4.parquet
│   └── balanced_outputs/
│       └── ...  (60 archivos)
│
└── REPLACE-BG/
    ├── patient_info.csv
    ├── windows_with_5folds_REPLACE-BG_PH4.parquet
    └── balanced_outputs/
        └── ...  (60 archivos)
```

---

## 12. Diseño Experimental Completo

El experimento global se estructura de la siguiente manera:

```
3 datasets
  × 1 condición original (baseline desbalanceado)
  + 2 dimensiones × 6 técnicas  (12 condiciones balanceadas por dataset)
= 3 × 13 = 39 condiciones experimentales totales
```

Cada condición se evalúa sobre los **5 folds de validación cruzada**, produciendo en total:
```
39 condiciones × 5 folds = 195 ejecuciones de entrenamiento LSTM
```

Los resultados de las 5 ejecuciones de cada condición se agregan para obtener métricas robustas (media ± desviación estándar), y posteriormente se comparan mediante tests estadísticos (Friedman + Nemenyi post-hoc) para determinar si las diferencias entre técnicas y la condición original son estadísticamente significativas.

### 12.1 Tabla de condiciones experimentales

| ID | Dataset | Dimensión | Técnica | Condición |
|---|---|---|---|---|
| 1 | T1DG | — | — | original |
| 2 | T1DG | age | undersampling | balanced_age_undersampling |
| 3 | T1DG | age | oversampling | balanced_age_oversampling |
| 4 | T1DG | age | patient_aware_undersampling | balanced_age_patient_aware_undersampling |
| 5 | T1DG | age | smote | balanced_age_smote |
| 6 | T1DG | age | jittering | balanced_age_jittering |
| 7 | T1DG | age | reference_proportional | balanced_age_reference_proportional |
| 8 | T1DG | sex | undersampling | balanced_sex_undersampling |
| ... | ... | ... | ... | ... |
| 13 | T1DG | sex | reference_proportional | balanced_sex_reference_proportional |
| 14 | DiaTrend | — | — | original |
| ... | ... | ... | ... | ... |
| 39 | REPLACE-BG | sex | reference_proportional | balanced_sex_reference_proportional |

---

## 13. Garantías de Integridad Metodológica

### 13.1 Ausencia de data leakage

| Riesgo | Mitigación implementada |
|---|---|
| Estadísticas de normalización calculadas sobre val/test | La normalización (media/std) se calcula exclusivamente sobre el conjunto de train del fold; val y test se normalizan con esos mismos parámetros. |
| Generación de sintéticos (SMOTE/jittering) con información de val/test | Los vecinos más cercanos se buscan solo dentro del grupo demográfico del train. Las ventanas de val/test no participan en ninguna estadística. |
| Sobremuestreo que introduce duplicados entre train y test | El oversampling se aplica solo después de extraer las filas de val/test del fold. No existe posibilidad de duplicar una ventana de test en train. |
| Uso de la misma ventana en train y test de un fold | Group K-Fold garantiza separación a nivel de paciente. El balanceo ocurre posterior a esta partición. |

### 13.2 Reproducibilidad

- Todas las operaciones aleatorias usan `np.random.default_rng()` con semilla derivada deterministamente de `(seed_global, fold_idx, dataset_name, group_name, technique)`.
- Los archivos de salida son bit-a-bit reproducibles entre ejecuciones con los mismos parámetros.
- El flag `--overwrite` permite forzar la regeneración; por defecto, los archivos existentes se saltan, lo que permite reanudar ejecuciones interrumpidas.

### 13.3 Integridad de las ventanas de evaluación

Las filas de `val` y `test` de cada fold se copian literalmente del archivo de ventanas original, sin ninguna transformación. Sus valores de glucosa (`x0..x7, y`) son exactamente los mismos en todas las condiciones experimentales para el mismo fold, lo que garantiza que las comparaciones entre condiciones son justas: todas se evalúan sobre exactamente el mismo conjunto de puntos.

### 13.4 Trazabilidad

Cada archivo de salida codifica en su nombre toda la información necesaria para reproducir el experimento:

```
windows_with_5folds_T1DiabetesGranada_PH4_fold2_age_smote.parquet
│                   │                  │   │    │   └─ técnica
│                   │                  │   │    └───── dimensión demográfica
│                   │                  │   └────────── índice de fold
│                   │                  └────────────── horizonte de predicción
│                   └───────────────────────────────── dataset de origen
└───────────────────────────────────────────────────── prefijo estándar
```

Esta nomenclatura permite al script de entrenamiento inferir automáticamente todos los metadatos del experimento sin necesidad de archivos de configuración adicionales.
