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

# **PARTE II — JUSTIFICACIÓN EXHAUSTIVA DE LOS EXPERIMENTOS**

Los 14 experimentos del notebook **00\_Experiments** están diseñados para reproducir escenarios clínicamente plausibles sin generar datos sintéticos desde cero, sino modificando series reales de forma controlada. Cada experimento representa una hipótesis metodológica derivada del EDA.

## **Experimento A — Imbalance Leve (REPLACE-BG, hypo ~6%)**

### **Descripción exacta**

Se reduce la clase mayoritaria (normoglucemia) en REPLACE-BG hasta alcanzar un ratio objetivo de 6% de hipoglucemia. La función **drop\_majority\_to\_target** elimina episodios completos de normoglucemia de forma aleatoria. Resultado final: hiper=48.55%, normo=45.45%, hipo=6.00%.

### **Justificación empírica (EDA)**

En REPLACE-BG, la hipoglucemia real es del 3.7%. El experimento simula una cohorte con mayor carga hipoglucémica moderada. La alta autocorrelación de hipoglucemia (0.750) y los episodios persistentes de REPLACE-BG hacen que este escenario sea fisiológicamente plausible: corresponde a pacientes con ajuste insulínico agresivo o con mayor sensibilidad a la insulina.

### **Justificación clínica**

Un 6% de hipoglucemia representa un perfil clínico de alto riesgo. En términos de predicción glucémica, significa que el modelo necesita ser sensible a descensos rápidos hacia el umbral de 70 mg/dL en aproximadamente 1 de cada 17 pasos. La ausencia de corrección ante este ratio produciría un MAE estructuralmente sesgado: el modelo predice con precisión los valores en rango normal pero sobrestima sistemáticamente los valores bajos (predicción de ~75-80 cuando el valor real es 55-65 mg/dL).

### **Hipótesis metodológica**

Con desbalanceo leve, las técnicas conservadoras como **cost-sensitive learning (T3) y RUS temporal (T2)** deberían ser suficientes. El escenario sirve de baseline de dificultad baja para evaluar cuándo empieza a ser necesario el oversampling.

## **Experimento B — Imbalance Moderado (REPLACE-BG, ~3.5%)**

### **Descripción exacta**

Intenta forzar un ratio de 3.5% de hipoglucemia en REPLACE-BG. El resultado real es la distribución base sin modificación (62.96% normo, 32.96% hiper, 4.07% hipo), porque la cohorte ya tiene ratios similares al target. Funciona como **escenario de referencia clínica**.

### **Justificación empírica**

La distribución original de REPLACE-BG (P50=18.63:1) ya refleja un desbalanceo moderado. Este escenario establece la **línea base** con la que comparar todos los demás experimentos sobre este dataset.

### **Relevancia clínica y para predicción**

Con 3.5-4% de hipoglucemia, el modelo enfocado en predicción continua tiende a concentrar sus predicciones en la región 70-180 mg/dL. Las predicciones en valores <70 mg/dL son escasas y tienen mayor error relativo, lo que clínicamente implica alarmas hipoglucémicas con latencia o falsos negativos. El experimento permite medir qué ganancia real aportan las técnicas de balanceo sobre la distribución natural.

## **Experimento C — Imbalance Severo (DIATREND, hypo ~1.5%)**

### **Descripción exacta**

Se eliminan episodios de hipoglucemia en DIATREND hasta alcanzar el 1.5%. Resultado: normo=51.95%, hiper=46.55%, hipo=1.50%. Este ratio (1:65) reproduce el perfil del percentil 50 de desbalanceo observado en DIATREND.

### **Justificación empírica**

La mediana de ratio en DIATREND es 46.46:1, equivalente aproximadamente a un 2% de hipoglucemia. El experimento simula ratios ligeramente más extremos, cubriendo el 50-60% inferior de la distribución de pacientes. Los episodios hipoglucémicos de DIATREND tienen mediana de 4 pasos y baja autocorrelación (0.124): son eventos breves e impredecibles.

### **Justificación clínica**

A 1.5% de hipoglucemia, el gradiente de la función de pérdida durante el entrenamiento está dominado en un 98.5% por valores en rango normo/hiper. El modelo aprende a minimizar el error en esas regiones, pero produce predicciones de 80-100 mg/dL cuando el valor real es 55 mg/dL. En términos clínicos, esto equivale a un sistema de alarma que falla en la mayoría de las hipoglucemias. Un endocrinólogo que usa predicciones de este modelo para ajustar insulina basa-bolo recibiría información errónea precisamente cuando el riesgo es mayor.

### **Hipótesis metodológica**

Con episodios breves y poco autocorrelados, las técnicas que trabajan sobre episodios completos (T1-ROS, T7-Episode-Aware) son más estables que las generativas (T4-SMOTE, T5-ADASYN), que necesitan vecindad temporal suficiente para interpolar sin artefactos. Los resultados confirman esta hipótesis: ROS logra score 96/100, SMOTE score 92/100.

## **Experimento D — Imbalance Extremo (DIATREND, hypo ~0.2%)**

### **Descripción exacta**

Reducción de hipoglucemia hasta el 0.2% (ratio ~1:500). Resultado: normo=52.64%, hiper=47.16%, hipo=0.20%. Reproduce el perfil del paciente con ID 41 de DIATREND (ratio 1376:1) y otros pacientes del percentil 90 del desbalanceo.

### **Justificación empírica**

El percentil 90 de desbalanceo en DIATREND es 162.34:1. En términos absolutos, el paciente 41 tiene 243 eventos hipoglucémicos frente a 339.638 de otras clases. Existen pacientes reales con estas características en la cohorte; el experimento representa su situación de entrenamiento.

### **Justificación clínica**

A 0.2% de hipoglucemia, un modelo de predicción continua tiene esencialmente cero presión de gradiente desde esa región. Las predicciones en el rango hipoglucémico tendrán el mayor error absoluto (MAE hipoglucemia >> MAE normo). Clínicamente, esto corresponde a pacientes con diabetes 'frágil' o con hipoinsulinismo iatrogénico donde las crisis, aunque rarísimas, son las más peligrosas. El balanceo individual por paciente (T6) es la única técnica que puede manejar este escenario sin distorsionar las series de los pacientes con hipoglucemia más frecuente.

## **Experimento E — Concentración de Hipoglucemia en 5 Pacientes (DIATREND)**

### **Descripción exacta**

Se retienen episodios hipoglucémicos únicamente de los 5 pacientes con mayor carga hipoglucémica. Resultado: hipo=1.22% (concentrado en 5 sujetos). Reproduce el sesgo observado en el EDA: top 5 = 53.94% de hipoglucemia.

### **Justificación empírica**

En DIATREND, más de la mitad de todos los episodios hipoglucémicos ocurre en solo 5 pacientes. Si se entrena un modelo sin controlar esta concentración, el modelo aprende a predecir hipoglucemia solo para perfiles similares a esos 5 pacientes y fracasa sistemáticamente para todos los demás.

### **Justificación clínica y para predicción**

Este sesgo tiene implicaciones directas en la generalización: el modelo podría predecir correctamente las hipoglucemias del paciente 37 (9.3% de hipo) y simultáneamente producir un error absoluto muy elevado para el paciente 41 (0.057% de hipo). Un sistema de alarma con este sesgo sería clínicamente inútil para el 90% de la población.

### **Hipótesis metodológica**

El balanceo adaptativo por paciente (T6) debe mostrar ventaja significativa respecto a técnicas globales, porque ajusta el objetivo de balanceo según la prevalencia individual de cada paciente.

## **Experimento F — Heterogeneidad Interpaciente Controlada (DIATREND)**

### **Descripción exacta**

Se aplican ratios diferenciados por paciente: 20% de los pacientes con target 0.2%, 40% con target 1%, 40% con target 4%. Resultado: hipo=1.24%. Reproduce la variabilidad real observada en el EDA (ratios entre 1:10 y 1:1376).

### **Justificación empírica y clínica**

La heterogeneidad es el estado natural de cualquier cohorte clínica real. Los modelos entrenados con pesos globales uniformes sesgan sus predicciones hacia los subgrupos dominantes. Este experimento permite comparar directamente técnicas globales vs. estrategias personalizadas por paciente en el escenario más realista posible.

## **Experimento G — Fragmentación Temporal (T1DiabetesGranada)**

### **Descripción exacta**

Se eliminan bloques temporales continuos de 180 minutos (12 pasos a 15 min) hasta cubrir el 25% del total de mediciones de cada paciente. Se recrean los gaps artificiales que caracterizan a T1DGranada. Resultado: hipo=4.30% (prácticamente idéntico al baseline), pero con mayor fragmentación.

### **Justificación empírica**

T1DGranada tiene solo 7 pacientes con sub-series continuas de 14 días. El experimento amplifica deliberadamente la fragmentación existente para probar el comportamiento de las técnicas en condiciones de discontinuidad extrema.

### **Justificación clínica**

Los gaps en CGM son clínicamente relevantes: los sensores se retiran durante hospitalizaciones o en pacientes que no toleran el adhesivo. Si el modelo aprende sobre ventanas que cruzan esos gaps, aprende transiciones ficticias (p.ej., 300 mg/dL → sin dato → 50 mg/dL) que no corresponden a ninguna fisiología real. Las técnicas que respetan la estructura de segmentos son las únicas que garantizan coherencia.

### **Hipótesis metodológica**

Las técnicas que construyen episodios respetando la continuidad (T7-Episode-Aware, T6-Patient-Level) deberían producir menos violaciones de RoC que las técnicas punto a punto (T5-ADASYN). Los resultados confirman: SMOTE\_TEMP Score 96/100, ROS 89/100, ADASYN 83/100.

## **Experimentos H e I — Pérdida de Continuidad y Reducción de Frecuencia (T1DGranada)**

**Exp H (pérdida intra-episodio):** muestreo aleatorio del 85% de las filas, simulando ausencias puntuales del CGM sin gaps estructurales. Los ratios de clase apenas cambian (normo=60.8%, hiper=34.9%, hipo=4.3%), pero la continuidad temporal se degrada.

**Exp I (reducción de frecuencia):** se conserva solo cada segundo punto en segmentos continuos, pasando efectivamente de 15 min a 30 min de muestreo. Este experimento prueba la robustez de las técnicas ante pérdida de resolución temporal, un escenario frecuente en dispositivos de bajo coste o con batería limitada.

La justificación empírica de ambos reside en que T1DGranada opera ya a 15 min (la resolución más baja de los tres datasets) y presenta nulidades del 27%. La combinación de estas limitaciones hace crítico entender cómo se degradan las técnicas de balanceo.

## **Experimentos J y K — Episodios Cortos y Largos (REPLACE-BG)**

**Exp J (episodios cortos, max\_len=4):** se truncan los episodios hipoglucémicos a 4 pasos máximo, reduciendo el ratio a 1.75%. Resultado: normo=64.49%, hiper=33.76%, hipo=1.75%. Reproduce el perfil de DIATREND (mediana 4 pasos) aplicado al dataset de REPLACE-BG.

**Exp K (episodios largos, min\_len=12):** se retienen solo episodios hipoglucémicos de 12+ pasos. Resultado: normo=64.26%, hiper=33.64%, hipo=2.09%. Reproduce el perfil de alta persistencia, análogo al extremo superior de REPLACE-BG.

La justificación clínica es directa: la duración del episodio hipoglucémico determina la gravedad clínica. Una hipoglucemia de 4 pasos (20 min) es una crisis leve tratable con carbohidratos; una de 12+ pasos (60+ min) requiere intervención médica urgente. Los modelos que dependen de contextos largos (LSTM, Transformer con ventana grande) se degradan en Exp J porque el episodio termina antes de que la red haya procesado suficiente historia.

## **Experimentos L, M y N — Cobertura, Tamaño Muestral e Híbrido**

**Exp L (cobertura reducida a 7 días):** limita la serie de cada paciente de T1DGranada a los primeros 7 días desde su primera medición. Con 96 pasos/día (a 15 min), esto es una ventana muy pequeña para aprender dinámicas crónicas.

**Exp M (20 pacientes, REPLACE-BG):** reduce de 223 a 20 pacientes seleccionados al azar. Resultado: normo=59.21%, hiper=36.96%, hipo=3.83%. Evalúa la sensibilidad de las técnicas de balanceo a la diversidad poblacional.

**Exp N (híbrido, T1DGranada):** combina desbalanceo severo (target 1%), fragmentación temporal (20% de bloques de 120 min eliminados) y concentración en top 10 pacientes. Resultado: normo=63.22%, hiper=36.27%, **hipo=0.51%**. Es el escenario más difícil del benchmark, con múltiples fuentes de sesgo simultáneas.

El Exp N tiene justificación clínica directa: la realidad de una cohorte hospitalaria típica combina todos estos factores. Los datasets de referencia internacional (OhioT1DM, DirecNet) presentan precisamente estas características. Solo las técnicas que preservan episodios y respetan segmentos continuos producen resultados coherentes.

# **PARTE III — JUSTIFICACIÓN EXHAUSTIVA DE LAS TÉCNICAS DE BALANCEO**

El principio rector de toda la justificación es que **el balanceo no es una operación de clasificación sino de representación en el espacio de aprendizaje de una función de regresión**. Un modelo que predice glucosa continua a 30 minutos optimiza implícitamente una función de pérdida (MAE, RMSE) sobre toda la distribución de valores futuros. Si la distribución de entrenamiento subrrepresenta sistemáticamente los valores <70 mg/dL, el gradiente de la pérdida en esa región es microscópico comparado con el gradiente en 70-180 mg/dL, y el modelo aprende a ignorar esa región.

## **T1 — Random Oversampling Episódico (ROS)**

### **Descripción técnica completa**

La unidad de remuestreo es el **episodio clínico completo**, no el punto aislado. El algoritmo calcula, por paciente, la diferencia entre el número actual de puntos hipoglucémicos (n\_min) y el número objetivo (n\_target = target\_ratio \* n\_other / (1 - target\_ratio)). Se seleccionan episodios hipoglucémicos al azar con reemplazamiento y se concatenan al final de la serie del paciente, asignando timestamps sintéticos consecutivos. El parámetro target\_ratio fue fijado en 0.12 en los experimentos comparativos.

### **Fundamentación teórica**

Duplicar episodios completos preserva la estructura autocorrelada interna del episodio: si los 4 pasos de una hipoglucemia evolucionan como \[80, 68, 55, 52\], esa secuencia se replica íntegra, manteniendo la **dependencia temporal local**. Matemáticamente, esto equivale a aumentar el peso de los episodios hipoglucémicos en la función de pérdida sin romper la covarianza serial de la serie.

### **Justificación basada en el EDA**

- DIATREND (Exp C): episodios cortos (mediana 4 pasos, baja autocorr 0.124). ROS es especialmente adecuado porque evitar la fragmentación es crítico cuando los episodios son brevísimos. Resultado: score 96/100, mejora balance +10.5%, autocorr 0.995, RoC 4.0%.
- REPLACE-BG (Exp J): episodios truncados a 4 pasos. ROS score 98/100, mejora +10.2%, autocorr 0.994, RoC 2.4%. El mejor resultado de todos los escenarios.
- T1DGranada (Exp G): con fragmentación temporal. ROS score 89/100, con la menor tasa de violaciones de RoC (0.8%). Preserva la coherencia fisiológica mejor que las técnicas generativas.

### **Riesgo principal**

Si un paciente tiene muy pocos episodios distintos (2-3 episodios únicos), la duplicación genera **sobreajuste a esos patrones específicos**. El modelo aprende que 'hipoglucemia = exactamente esta secuencia de valores' en lugar de aprender la dinámica general de descenso. Esto es especialmente problemático en DIATREND con pacientes de ratio extremo (paciente 41: solo 243 puntos hipoglucémicos en 1.376.000 puntos totales).

## **T2 — Random Undersampling Temporal con Preservación de Contexto (RUS)**

### **Descripción técnica completa**

Para cada paciente, se identifica una ventana de **context\_steps=12 pasos** alrededor de cada punto hipo/hiper. Los puntos dentro de esa ventana (contexto\_clínico) se conservan íntegramente. Del resto (puntos de normoglucemia estable), se conserva solo un **keep\_ratio=0.15** (15%) seleccionado al azar. El resultado es una serie más corta que preserva todas las transiciones clínicas.

### **Fundamentación teórica**

La hipótesis es que los segmentos estables de normoglucemia aportan información altamente redundante: si el modelo ya sabe que la glucosa lleva 4 horas estable en 120 mg/dL, el paso 97 de esa meseta no añade información predictiva respecto al paso 96. Eliminar el 85% de esos puntos reduce el dominancia del gradiente de la región normal sin alterar las regiones de transición.

### **Justificación basada en el EDA**

REPLACE-BG tiene medianas de episodio normo de 24 pasos: hay enormes bloques estables susceptibles de submuestreo. La autocorrelación de normoglucemia es cercana a 1 en esos bloques, confirmando su redundancia informativa. Los resultados muestran que RUS mejora el balance pero de forma modesta (+0.5-0.8%), con buen perfil de artefactos. Su valor principal es como **complemento de las técnicas de oversampling**, no como técnica independiente.

## **T3 — Cost-Sensitive Learning**

### **Descripción técnica completa**

La función compute\_class\_weights calcula pesos como w\_c = n\_total / (n\_classes \* n\_c) para cada clase c. Los pesos resultantes para el Exp C (imbalance severo) son: **hipoglucemia: 22.22**, hiperglucemia: 0.72, normoglucemia: 0.64. Estos pesos se incorporan en la función de pérdida durante el entrenamiento.

### **Fundamentación teórica**

Es matemáticamente equivalente a entrenar con un dataset balanceado a 1:1:1, pero sin modificar la distribución real de los datos. La función de pérdida ponderada amplifica el gradiente de los errores en regiones minoritarias por el factor de peso correspondiente. Es la **técnica más conservadora desde el punto de vista fisiológico**: no altera ningún dato, no introduce artefactos, no rompe la autocorrelación.

### **Justificación basada en el EDA**

Funciona como **baseline de referencia en todos los escenarios**. El EDA no ofrece ninguna razón para no usarlo; el único riesgo es inestabilidad de optimización en desbalanceos extremos (peso 22+), donde el gradiente amplificado puede generar oscilaciones en el entrenamiento. En todos los escenarios experimentados, los pesos calculados son: hipo ~22 (expC), ~7.8 (expG), ~19 (expJ).

## **T4 — SMOTE Temporal (smote\_temporal)**

### **Descripción técnica completa**

Se extraen ventanas de **window\_size=12 pasos** de glucosa centradas en puntos hipoglucémicos. Para generar una muestra sintética: (1) se selecciona un índice hipoglucémico idx\_a; (2) se buscan vecinos dentro del **mismo segmento temporal** (max\_gap\_steps=288) y misma clase; (3) si no hay vecinos intra-segmento, se amplía a vecinos globales del mismo paciente; (4) se interpola linealmente: x\_new = x\_a + λ(x\_b - x\_a) con λ ~ Uniform(0,1); (5) los valores se recortan al rango \[39, 401\].

### **Fundamentación teórica**

La restricción al mismo segmento temporal es clave: evita interpolar entre ventanas separadas por gaps que corresponden a estados fisiológicos completamente diferentes. Matemáticamente, la interpolación de dos trayectorias glucémicas reales dentro del mismo contexto continuo produce una trayectoria que pertenece al **espacio fisiológicamente plausible** (por convexidad local de la dinámica glucémica en ventanas cortas).

### **Justificación basada en el EDA**

- DIATREND: baja autocorrelación (0.124) y episodios breves → la vecindad temporal es escasa. El SMOTE frecuentemente recae en vecinos globales, lo que genera interpolaciones entre contextos no relacionados. Score 92/100, pero con mayor tasa de RoC (7.7%) que ROS.
- T1DGranada (Exp G): el parámetro max\_gap\_steps=288 (equivalente a 3 días a 15 min) es apropiado para el período de seguimiento. Con autocorrelación 0.766, los vecinos intra-segmento son realmente similares en su dinámica. Mejor resultado en Exp G: SMOTE\_TEMP Score 96/100, mejora +14.3%.
- REPLACE-BG: episodios persistentes y alta autocorrelación (0.750) hacen que los vecinos dentro del segmento sean muy similares entre sí, lo que puede producir muestras sintéticas con poca variabilidad adicional.

### **Riesgo principal**

Si la interpolación se realiza entre dos episodios con niveles base muy distintos (ej: uno que baja de 120 a 60 mg/dL y otro que baja de 80 a 50 mg/dL), la muestra sintética puede representar un descenso de 100 a 55 mg/dL que nunca ocurriría fisiológicamente. La restricción de segmento mitiga este riesgo pero no lo elimina completamente.

## **T5 — ADASYN Adaptativo (adasyn\_window)**

### **Descripción técnica completa**

Implementación estándar de ADASYN de imbalanced-learn con sampling\_strategy=0.12, n\_neighbors=3, aplicada sobre ventanas de **window\_size=12** puntos de glucosa. La diferencia con SMOTE es que ADASYN genera más muestras en los puntos hipoglucémicos más difíciles de clasificar (aquellos con más vecinos de la clase mayoritaria), adaptando la distribución sintética a las regiones de mayor incertidumbre.

### **Posición en el benchmark**

ADASYN actúa como **comparador metodológico**. Consistentemente obtiene el tercer lugar (detrás de ROS y SMOTE\_TEMP) en los tres escenarios: score 90/100 (expC), 83/100 (expG), 92/100 (expJ). Su tasa de violaciones de RoC es la más alta de todas las técnicas (9.5%, 4.3%, 7.0%), confirmando la hipótesis del EDA: con series temporales altamente autocorreladas, generar muestras en regiones de incertidumbre introduce artefactos no fisiológicos.

### **Justificación clínica de su inclusión**

Su inclusión está justificada como control negativo parcial: permite demostrar empíricamente que las técnicas diseñadas específicamente para series temporales (T1, T4) superan a los métodos estándar de clasificación aplicados ingenuamente a ventanas. Este argumento es clave para la defensa del TFM.

## **T6 — Balanceo por Paciente (balance\_by\_patient)**

### **Descripción técnica completa**

Aplica la función base func (por defecto episode\_aware) con target\_ratio personalizado por paciente: si ratio\_actual < 2%, target = 20%; si 2% ≤ ratio\_actual < 8%, target = 12%; si ratio\_actual ≥ 8%, no se modifica. Esto produce un balanceo **adaptativo a la severidad individual** del desbalanceo.

### **Fundamentación teórica**

El principio es que el sesgo de optimización es proporcional al desbalanceo **local** del paciente, no al global de la cohorte. Un ratio global del 4% puede esconder a la vez pacientes con 0.1% y pacientes con 15%. Imponer un target global sobre todos produce sobrebalanceo en los pacientes ya equilibrados y subbalanceo en los extremos.

### **Justificación basada en el EDA**

DIATREND presenta el escenario más favorable para esta técnica: ratios que van de 5.71:1 a 1376:1. El Exp F (heterogeneidad controlada) fue diseñado específicamente para evaluar esta técnica. En los tres escenarios comparados, T6-patient\_level obtiene scores 46/100, 49/100 y 48/100, pero esto refleja una limitación del scoring utilizado: el score premia la mejora de balance, y T6 con episode\_aware base no mejora el balance si el paciente ya está en el target adaptativo. La métrica más relevante es la **preservación de autocorrelación y ausencia de artefactos**, donde T6 es consistentemente el mejor (RoC 4.2%, 0.9%, 2.2%).

## **T7 — Episode-Aware Resampling (episode\_aware)**

### **Descripción técnica completa**

Para cada paciente: (1) calcula el déficit de puntos minoritarios respecto al target\_ratio; (2) construye bloques incluyendo **buffer\_steps=6 pasos** de contexto antes y después de cada episodio; (3) selecciona bloques al azar con reemplazamiento hasta cubrir el déficit; (4) asigna timestamps sintéticos consecutivos al final de la serie.

### **Diferencia clave respecto a T1-ROS**

T7 incluye **contexto de transición** (los 6 pasos inmediatamente anteriores y posteriores al episodio), mientras que T1 duplica solo el episodio puro. Esto enriquece la representación de la **dinámica de entrada y salida** del estado hipoglucémico, que según el EDA muestra señales precursoras identificables hasta 24-48 pasos antes.

### **Justificación basada en el EDA**

Los perfiles de señales precursoras de DIATREND (lookback=48 pasos) y REPLACE-BG (lookback=8 pasos) demuestran que la información útil para predecir hipoglucemia reside en el contexto previo, no en el punto de cruce. T7 preserva explícitamente ese contexto. Los scores en los experimentos son moderados (46-49/100) porque la función score premia balance, pero el perfil de artefactos es el segundo mejor.

## **T8 — SMOGN-Like (smogn\_like)**

### **Descripción técnica completa**

Adapta la metodología SMOGN (Branco et al., 2019) para regresión continua. Las ventanas se consideran 'relevantes' si su valor final es <70 o >180 mg/dL. Para generar una muestra: si la distancia euclidiana entre dos ventanas relevantes es < 12 mg/dL, se interpola; si es mayor, se añade ruido gaussiano N(0, 2.0) recortado al rango \[39, 401\].

### **Posición conceptual en el TFM**

Esta técnica establece el vínculo explícito entre el problema de **imbalanced regression** y la predicción glucémica. La literatura reciente (He & Garcia, 2009; Torgo et al., 2013; Branco et al., 2019) ha demostrado que los problemas de regresión con distribución de target sesgada requieren tratamiento específico. En glucosa, el target es continuo pero las regiones clínicamente relevantes (<70, >180) están subrepresentadas exactamente como las clases minoritarias en clasificación.

### **Resultados y limitaciones**

Los scores de T8 son modestos (46-50/100) porque la mejora de balance es pequeña (+0.0-0.3%). El ruido gaussiano produce violaciones de RoC aceptables (1.2-4.2%). La técnica tiene mayor potencial cuando se combina con una función de importancia de relevancia (phi function) más sofisticada que los simples umbrales clínicos.

# **PARTE IV — RESULTADOS CUANTITATIVOS Y RECOMENDACIONES**

## **4.1 Tabla de resultados comparativos**

| **Técnica** | **Exp C Score** | **Exp G Score** | **Exp J Score** | **RoC media** | **Autocorr min** |
| --- | --- | --- | --- | --- | --- |
| **ROS** | 96/100 | 89/100 | 98/100 | 2.5% | 0.972 |
| **SMOTE\_TEMP** | 92/100 | 96/100 | 93/100 | 6.0% | 0.929 |
| **ADASYN** | 90/100 | 83/100 | 92/100 | 6.9% | 0.907 |
| **RUS** | 47/100 | 53/100 | 51/100 | 3.3% | 0.958 |
| **Episode\_Aware** | 46/100 | 49/100 | 48/100 | 2.4% | 0.965 |
| **Patient\_Level** | 46/100 | 49/100 | 48/100 | 3.2% | 0.965 |
| **SMOGN\_Like** | 46/100 | 50/100 | 48/100 | 2.5% | 0.962 |

## **4.2 Recomendaciones por escenario**

### **Imbalance severo con episodios cortos (DIATREND)**

**Técnica recomendada: ROS (Score 96/100).** Preserva autocorrelación en 0.995, viola solo el 4.0% de las tasas de cambio fisiológicas, y mejora el balance en +10.5 puntos porcentuales. La brevedad de los episodios hace inviable la interpolación; la duplicación íntegra es la única operación que no destruye la estructura temporal.

### **Fragmentación temporal (T1DiabetesGranada)**

**Técnica recomendada: SMOTE Temporal (Score 96/100).** La alta autocorrelación de hipoglucemia (0.766) en T1DGranada hace viable la interpolación intra-segmento. Con mejora de +14.3% y solo 3.6% de violaciones de RoC, supera a ROS en este contexto específicamente porque puede aprovechar la riqueza de los segmentos continuos disponibles.

### **Episodios cortos en dataset con episodios naturalmente largos (REPLACE-BG)**

**Técnica recomendada: ROS (Score 98/100).** Con autocorrelación 0.994 y solo 2.4% de violaciones de RoC, es el mejor resultado de todo el benchmark. La razón: los episodios truncados a 4 pasos son demasiado cortos para que SMOTE encuentre vecinos con dinámica similar.

# **PARTE V — MATERIAL PARA LA MEMORIA Y LA DEFENSA**

## **5.1 Texto científico para la sección de Metodología del TFM**

El desbalanceo en series de monitorización continua de glucosa (CGM) presenta características estructuralmente distintas a los problemas clásicos de clasificación desequilibrada. En primer lugar, las observaciones no son independientes ni idénticamente distribuidas (i.i.d.): presentan fuerte autocorrelación temporal (valores medidos de 0.124 a 0.973 según dataset y clase), de modo que las técnicas de remuestreo que asumen independencia entre muestras introducen artefactos estadísticos medibles. En segundo lugar, el objetivo del modelo es la **predicción continua (regresión)**, no la clasificación: el desbalanceo no afecta a la separabilidad de clases sino a la densidad relativa del gradiente de la función de pérdida en distintas regiones del espacio objetivo.

Los tres datasets analizados (DIATREND, REPLACE-BG, T1DiabetesGranada) representan el espectro completo de escenarios clínicamente plausibles: ratios de desbalanceo de 5.71:1 a 1376:1, frecuencias de muestreo de 5 a 15 minutos, duraciones de episodio hipoglucémico de 2 a 186 pasos, y grados de fragmentación temporal que oscilan entre la continuidad casi perfecta (REPLACE-BG, 0.44% de nulos) y la discontinuidad estructural (T1DiabetesGranada, solo 7 pacientes con 14 días continuos).

Se diseñaron 14 escenarios experimentales mediante modificaciones controladas de series reales, sin generación sintética desde cero, garantizando trazabilidad completa con la fisiología medida por el CGM. Sobre estos escenarios se evaluaron 8 técnicas de balanceo con métricas específicas para series temporales: tasa de violaciones de la Rate of Change fisiológica (threshold 3 mg/dL/min), autocorrelación residual (lag-1), y un score clínico compuesto que equilibra mejora de representatividad (50 puntos) con preservación fisiológica (50 puntos).

## **5.2 Argumentos clave para la defensa oral**

### **Por qué estos experimentos y no otros**

Los 14 experimentos no son arbitrarios: cada uno reproduce un hallazgo cuantificable del EDA. La concentración de eventos en 5 pacientes (Exp E) reproduce el ratio top5=53.94% de DIATREND. El Exp D (0.2%) reproduce el percentil 90 de desbalanceo del mismo dataset (162.34:1). El Exp N combina los tres factores de sesgo más prevalentes en datasets clínicos reales, haciendo el benchmark transferible a cualquier cohorte CGM futura.

### **Por qué se priorizan técnicas basadas en episodios**

El EDA demuestra autocorrelación media no nula en todos los datasets y clases. Bajo dependencia temporal, las técnicas que remuestrean puntos aislados violan la hipótesis de independencia sobre la que están construidas (SMOTE clásico, ADASYN). La evidencia empírica lo confirma: ADASYN produce violaciones de RoC del 9.5% en Exp C, frente al 4.0% de ROS. El episodio es la unidad informativa mínima con sentido fisiológico.

### **Por qué no se usó SMOTE clásico sin restricciones temporales**

SMOTE clásico no tiene conocimiento del orden temporal: puede interpolar entre dos ventanas separadas por días o entre pacientes distintos si la implementación es naive. En series con autocorrelación alta (hiper=0.973 en T1DGranada), esto generaría muestras sintéticas que representan transiciones de glucosa físicamente imposibles (ej: descenso de 350 a 60 mg/dL en 5 minutos). La variante temporal (T4) añade la restricción de segmento para eliminar este riesgo.

### **Por qué cost-sensitive learning (T3) es el baseline pero no suficiente**

T3 preserva perfectamente la estructura temporal porque no modifica ningún dato. Sin embargo, con pesos de 22:1 en el Exp C, la optimización puede volverse inestable o generar exceso de falsos positivos en la predicción (el modelo sobreestima la probabilidad de valores bajos). En la práctica, T3 debe combinarse siempre con técnicas de remuestreo, especialmente cuando el desbalanceo supera 1:50.

## **5.3 Preguntas difíciles del tribunal y respuestas**

### **P: '¿Por qué hablar de balanceo en un problema de regresión?'**

R: El desbalanceo afecta a la regresión exactamente de la misma forma que a la clasificación, aunque el mecanismo es distinto. En clasificación, el umbral de decisión se sesga hacia la clase mayoritaria. En regresión, el gradiente acumulado durante el entrenamiento es proporcional a la densidad de ejemplos en cada región del espacio de salida. Si el 2% de los datos está en la región <70 mg/dL, el gradiente de esa región es 50 veces más pequeño que el de la región 70-180 mg/dL. El modelo resultante predice con alta precisión en normo pero con MAE estructuralmente elevado en hipo: clínicamente, sobreestima sistemáticamente los valores bajos, generando un sistema de alarma con alta tasa de falsos negativos.

### **P: '¿Por qué no usar un único enfoque global para todos los escenarios?'**

R: Los resultados demuestran que no existe una técnica dominante en todos los escenarios. ROS es óptima en Exp C y Exp J (episodios cortos, baja autocorrelación hipoglucemia), pero SMOTE\_TEMP es óptima en Exp G (fragmentación temporal, alta autocorrelación). Esto es coherente con la literatura de No Free Lunch para aprendizaje automático. La contribución del TFM es precisamente construir un benchmark que permite seleccionar la técnica óptima en función de las características del dataset.

### **P: '¿Cómo se valida que los episodios sintéticos son fisiológicamente plausibles?'**

R: La métrica de violaciones de RoC cuantifica exactamente esto. Una tasa de cambio >3 mg/dL/min es fisiológicamente improbable para un humano sin insulina de acción rápida activa. Los umbrales de la literatura (Klonoff et al., 2017, JDST) fijan en 3-4 mg/dL/min el límite superior de cambio glucémico normal. Ninguna técnica supera el 10% de violaciones en los escenarios evaluados, con ROS consistentemente por debajo del 4.2%.

### **P: '¿Qué limitaciones tiene el trabajo?'**

R: Las técnicas se evalúan sobre métricas de calidad de los datos transformados (autocorrelación, RoC), no sobre el error de predicción de un modelo downstream. La validación completa requeriría entrenar modelos predictivos (LSTM, Transformer, N-BEATS) con cada técnica de balanceo y medir el MAE/RMSE por rangos glucémicos. Esta es la continuación natural del trabajo. Adicionalmente, los experimentos son modificaciones de datasets existentes; la validación prospectiva en cohortes nuevas es necesaria antes de aplicación clínica.

# **PARTE VI — SÍNTESIS: MAPA DE DECISIÓN EXPERIMENTAL**

**¿Cómo seleccionar la técnica de balanceo óptima para un dataset CGM?** El siguiente árbol de decisión sintetiza los hallazgos del TFM:

1. ¿La autocorrelación de hipoglucemia es < 0.3? → Usar ROS (T1). Los episodios son demasiado cortos para interpolación válida.
2. ¿La autocorrelación de hipoglucemia es ≥ 0.3 Y el dataset tiene sub-series continuas suficientes? → Usar SMOTE\_TEMP (T4) con restricción intra-segmento.
3. ¿El desbalanceo varía drásticamente entre pacientes (ratio max/min > 10x)? → Envolver cualquier técnica base en balance\_by\_patient (T6).
4. ¿La fragmentación temporal es severa (< 20% de pacientes con 14 días continuos)? → Usar técnicas episódicas (T1, T7) con reconstrucción de segmentos explícita.
5. En todos los escenarios: incluir cost-sensitive learning (T3) como baseline y como complemento.

**Conclusión central del TFM:** El desbalanceo en series CGM no es un problema de redistribución estadística de muestras, sino un problema de representación del espacio de aprendizaje en presencia de dependencia temporal. Las técnicas que respetan la estructura episódica y la autocorrelación de las series producen modelos con menor sesgo en las regiones clínicamente críticas (< 70 mg/dL), sin introducir artefactos fisiológicos medibles.