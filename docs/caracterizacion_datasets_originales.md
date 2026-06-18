**ANÁLISIS CIENTÍFICO EXHAUSTIVO**

_Trabajo Fin de Máster — Predicción de Glucosa en DM1_

Desbalanceo en Series CGM: Justificación, Experimentos y Técnicas de Balanceo

Junio 2026

# **PARTE I — BASE DE CONOCIMIENTO: CARACTERIZACIÓN DE LOS DATASETS**

## **1.1 Dataset DIATREND**

### **1.1.1 Descripción general**

DIATREND es un dataset de monitorización continua de glucosa (CGM) que recoge **9.155.089 mediciones de glucosa** procedentes de **51 pacientes con diabetes tipo 1 utilizables** (54 en metadatos, 51 con series de glucosa válidas). El rango temporal abarca desde el **1 de diciembre de 2015 hasta el 28 de junio de 2022**, con una antigüedad máxima de 1.907 días y una mediana de cobertura que sitúa a la mayoría de pacientes por encima de los 1.000 días de seguimiento. La frecuencia de muestreo predominante es de **5 minutos**, conforme a las especificaciones estándar de los dispositivos Dexcom utilizados en la cohorte.

### **1.1.2 Variables disponibles y calidad de datos**

El esquema incluye **Patient\_ID**, **Measurement\_date**, **Measurement\_time**, **Measurement** (glucosa en mg/dL, tipo Int16), y columnas de timestamp resampleadas a 5 y 15 minutos.

Los valores perdidos ascienden a un **16.25%** en las columnas de medición y timestamp, mientras que la columna **15min** presenta una nulidad del 68.14% (irrelevante dado que el dataset opera principalmente a 5 min). El rango medido es de **39 a 401 mg/dL** con 363 valores únicos, consistente con los límites de detección de los sensores CGM modernos.

### **1.1.3 Distribución de clases y desbalanceo**

La distribución global de clases es: **normoglucemia ~51%**, **hiperglucemia ~47%**, **hipoglucemia ~2%**. Este patrón refleja una cohorte de pacientes jóvenes (edades entre 19 y 69 años, media 19-30) con control glucémico subóptimo generalizado pero hipoglucemia extremadamente rara en términos proporcionales.

A nivel individual, el desbalanceo es dramático: **41 de 51 pacientes (80.4%) presentan ratio máximo normo/minoritaria > 1:20**, calificado como 'severo'. Los ratios individuales van desde **5.71:1 hasta 1376.14:1**, con mediana 46.46:1, percentil 75 de 112.46:1 y percentil 90 de 162.34:1. El paciente con ratio más extremo (1376:1) tiene una hipoglucemia tan escasa que cualquier modelo global la ignoraría sistemáticamente.

**Concentración de eventos:** el top 5 de pacientes acumula el **53.94% de todos los episodios de hipoglucemia** y el 29.96% de los de hiperglucemia. Esta distribución de tipo 'cola larga' implica que un modelo entrenado sin ajuste aprendería principalmente los patrones de 5 pacientes, con nula capacidad de generalización.

### **1.1.4 Estructura temporal**

La autocorrelación media es **hypo=0.124 e hyper=0.825**. La baja autocorrelación de hipoglucemia confirma que las crisis son eventos abruptos y poco persistentes — no se acumulan en 'rachas' largas —, mientras que la hiperglucemia es altamente persistente (rafagas claras). Las medianas de longitud de episodio son **hipo=4 pasos, normo=14 pasos, hiper=14 pasos**. Los episodios hipoglucémicos duran de media 6.4 pasos (máx 186), confirmando su brevedad. La ventana de lookback óptima identificada es de **48 pasos**.

### **1.1.5 Hallazgos clave con implicación para modelado**

- Hipoglucemia extremadamente rara (~2% global, <0.1% en algunos pacientes): el modelo sin corrección optimizará únicamente las regiones normo/hiper y producirá predicciones sistemáticamente alejadas del umbral de 70 mg/dL.
- Alta concentración en 5 pacientes: el gradiente de pérdida proviene casi exclusivamente de esos pacientes; el modelo 'aprende' hipoglucemia de muy pocas personas.
- Episodios cortos y poco autocorrelados: el balanceo punto a punto es más destructivo aquí que en datasets con hipoglucemias persistentes, porque destruye la única información temporal que hay.
- Señales precursoras separables desde 48 pasos antes: la pendiente previa (pre\_slope) y la varianza previa (pre\_var) divergen entre clases, justificando ventanas de contexto largas.
- Variable demográfica más explicativa del desbalanceo: Edad (score 0.155), seguida por grupo de edad (0.135) y sexo (0.061). El subgrupo 46-60 años concentra el 25% del decil extremo.

## **1.2 Dataset REPLACE-BG**

### **1.2.1 Descripción general**

REPLACE-BG es el mayor de los tres datasets con **11.579.408 mediciones de glucosa** de **223 pacientes con diabetes tipo 1**. Procede del ensayo clínico aleatorizado REPLACE-BG, que comparó la gestión de insulina con CGM frente a monitorización estándar. El rango temporal es **2000-01-13 a 2000-11-25**, correspondiente a aproximadamente 10 meses de seguimiento. La frecuencia de muestreo es de **5 minutos**, con solo un 0.44% de nulos en la columna 5min.

### **1.2.2 Variables disponibles y calidad de datos**

El esquema es análogo a DIATREND. La nulidad media es del **9.91%** en medición y timestamp, la más baja de los tres datasets. El rango glucémico cubre de 39 a 401 mg/dL con 363 valores únicos. La **cobertura temporal individual** oscila entre 141 y 182 días por paciente, con medias de 176 días (paciente representativo), haciendo de este dataset el más denso en cobertura por paciente.

### **1.2.3 Distribución de clases y desbalanceo**

La distribución global es: **normoglucemia 63.62%, hiperglucemia 32.68%, hipoglucemia 3.70%**. Comparado con DIATREND, la hipoglucemia es casi el doble de frecuente, y la normoglucemia es más dominante. El perfil es más típico de pacientes adultos en seguimiento clínico controlado.

El desbalanceo por paciente muestra **47.1% con ratio > 1:20 (105/223)** y 117 pacientes con desbalanceo 'moderado' (1:5 a 1:20). Los ratios extremos alcanzan **3072:1**, aunque la distribución es menos concentrada que en DIATREND: el top 5 de pacientes solo concentra el **6.84% de hipoglucemias**, lo que indica una distribución mucho más homogénea. Los ratios de simulación son: mediana **18.63:1**, P75 **38.94:1**, P90 **68.01:1**.

### **1.2.4 Estructura temporal**

La autocorrelación media es **hypo=0.750, hyper=0.920** — notablemente más alta que en DIATREND para hipoglucemia. Esto indica que las crisis hipoglucémicas en REPLACE-BG son **persistentes y en rachas**, durando varias mediciones consecutivas. Las medianas de longitud de episodio son **hipo=6, normo=24, hiper=18 pasos**. La ventana de lookback óptima es de **8 pasos**, más corta que en DIATREND, sugiriendo que los precursores son más inmediatos.

### **1.2.5 Hallazgos clave**

- Hipoglucemia moderada (3.7%) con alta autocorrelación (0.750): el modelo puede aprender patrones persistentes en la región hipoglucémica, lo que hace viable el SMOTE temporal.
- Distribución homogénea entre pacientes (top 5 = 6.8% vs 53.9% en DIATREND): menor riesgo de sobreajuste a pacientes individuales; las técnicas globales son más aplicables.
- Episodios más largos (mediana 6 pasos, máx 128): justifica técnicas que preserven episodios completos (T1-ROS y T7-Episode-Aware).
- Ningún paciente sin hipoglucemia (0%): cobertura completa de la clase minoritaria, lo que permite aplicar todas las técnicas de oversampling sin restricciones.
- Mayor predictor demográfico: Edad (ρ\_Spearman=0.281), con el grupo >60 años concentrando el 20% del decil extremo.

## **1.3 Dataset T1DiabetesGranada**

### **1.3.1 Descripción general**

T1DiabetesGranada es un dataset clínico de pacientes del **Hospital Universitario Virgen de las Nieves (Granada)**, con **30.478.234 filas brutas** procedentes de **643–736 pacientes** (la discrepancia refleja pacientes en metadatos vs. series válidas). Opera principalmente a **15 minutos** de frecuencia de muestreo, siendo el único de los tres en este régimen. Cubre el período de **enero 2018 a marzo 2022**.

### **1.3.2 Variables y calidad de datos**

Incluye metadatos ricos en el Patient\_info.csv: **Birth\_year, Sex, Initial/Final\_measurement\_date, Number\_of\_days\_with\_measures, Number\_of\_biochemical\_parameters, Number\_of\_diagnostics**. La nulidad en medición y timestamp es del **27.05%**, la más alta de los tres datasets. Crucialmente, la columna **5min está 100% vacía**, confirmando operación exclusiva a 15 min. El rango glucémico es 40–500 mg/dL.

### **1.3.3 Discontinuidad y fragmentación: el rasgo definitorio**

Este dataset presenta una **fragmentación temporal severa**. Usando ventanas clínicas validadas (1 día, 14 días, 90 días):

| **Ventana** | **Pacientes elegibles** | **Sub-series válidas** | **Cobertura media (días)** |
| --- | --- | --- | --- |
| 1 día | 643 | 50.762 | 3.3 |
| 14 días | 7 | 13 | 35.5 |
| 90 días | 1 | 1 | 93.0 |

Solo **7 pacientes** tienen datos continuos de 14 días, y apenas **1 paciente** alcanza 90 días continuos. Esto significa que el **99.9% de los pacientes no tiene seguimiento continuo suficiente para el estándar AGP**, el informe glucémico de referencia clínica. Los gaps no son aleatorios: coinciden con periodos de hiperglucemia severa o ingreso hospitalario, introduciendo **sesgo de selección no ignorable (missingness not at random, MNAR)**

### **1.3.4 Distribución de clases y desbalanceo**

La distribución global es: **normoglucemia 60.8%, hiperglucemia 34.9%, hipoglucemia 4.3%**. Similar a REPLACE-BG en proporciones globales. Sin embargo, la distribución por paciente muestra mayor heterogeneidad que en REPLACE-BG: la mediana de ratio es **17.79:1**, con P75 37.62:1 y P90 97.51:1. Hay **2 pacientes sin hipoglucemia** y 1 sin hiperglucemia. La concentración de eventos es moderada: top 5 hipo = 6.8%, top 5 hiper = 4.0%.

### **1.3.5 Estructura temporal**

Autocorrelación media: **hipo=0.766, hiper=0.973**. Los episodios hipoglucémicos tienen mediana de **3 mediciones** (15 min/paso → mediana de 45 min de duración real), mientras que los episodios de hiperglucemia tienen mediana de **7 mediciones** (105 minutos). La separabilidad es máxima en la ventana de **1 día (Cohen d = 0.369 para hipoglucemia)**

### **1.3.6 Hallazgos clave**

- Fragmentación extrema: el mayor riesgo metodológico. Las técnicas que construyen ventanas cruzando gaps obtienen patrones ficticios que el modelo aprenderá como reales.
- MNAR (Missing Not At Random): los gaps coinciden con eventos clínicos graves. Ignorarlos introduce sesgo de selección que infravaloraría los episodios más peligrosos.
- Alta autocorrelación de hipoglucemia (0.766): los episodios son más persistentes que en DIATREND (0.124), lo que hace más viable el SMOTE temporal, pero solo dentro de sub-series continuas.
- Frecuencia 15 min: las ventanas de contexto cubren mayor tiempo real por paso, lo que altera el cálculo de Rate of Change (RoC) y la interpretación de las pendientes previas.
- Variable más explicativa del desbalanceo: Edad (score 0.198), con grupos mayores mostrando mayor ratio de desbalanceo.

## **1.4 Tabla comparativa de los tres datasets**

| **Característica** | **DIATREND** | **REPLACE-BG** | **T1DGranada** |
| --- | --- | --- | --- |
| **N pacientes** | 51 | 223 | 643 (EDA) |
| **N mediciones** | 9.15 M | 11.58 M | 30.48 M (bruto) |
| **Frecuencia** | 5 min | 5 min | 15 min |
| **% hipoglucemia** | ~2% | ~3.7% | ~4.3% |
| **% hiperglucemia** | ~47% | ~32.7% | ~34.9% |
| **Ratio mediano (p50)** | 46.46:1 | 18.63:1 | 17.79:1 |
| **Ratio P90** | 162.34:1 | 68.01:1 | 97.51:1 |
| **Autocorr. hipo** | 0.124 (bajo) | 0.750 (alto) | 0.766 (alto) |
| **Mediana episodio hipo** | 4 pasos | 6 pasos | 3 pasos (15min) |
| **Top 5 concentración hipo** | 53.94% | 6.84% | 6.8% |
| **Fragmentación temporal** | Baja | Baja | MUY ALTA |
| **Lookback óptimo** | 48 pasos | 8 pasos | ventana 1d |





# — SÍNTESIS: MAPA DE DECISIÓN EXPERIMENTAL**

**¿Cómo seleccionar la técnica de balanceo óptima para un dataset CGM?** El siguiente árbol de decisión sintetiza los hallazgos del TFM:

1. ¿La autocorrelación de hipoglucemia es < 0.3? → Usar ROS (T1). Los episodios son demasiado cortos para interpolación válida.
2. ¿La autocorrelación de hipoglucemia es ≥ 0.3 Y el dataset tiene sub-series continuas suficientes? → Usar SMOTE\_TEMP (T4) con restricción intra-segmento.
3. ¿El desbalanceo varía drásticamente entre pacientes (ratio max/min > 10x)? → Envolver cualquier técnica base en balance\_by\_patient (T6).
4. ¿La fragmentación temporal es severa (< 20% de pacientes con 14 días continuos)? → Usar técnicas episódicas (T1, T7) con reconstrucción de segmentos explícita.
5. En todos los escenarios: incluir cost-sensitive learning (T3) como baseline y como complemento.

**Conclusión central del TFM:** El desbalanceo en series CGM no es un problema de redistribución estadística de muestras, sino un problema de representación del espacio de aprendizaje en presencia de dependencia temporal. Las técnicas que respetan la estructura episódica y la autocorrelación de las series producen modelos con menor sesgo en las regiones clínicamente críticas (< 70 mg/dL), sin introducir artefactos fisiológicos medibles.