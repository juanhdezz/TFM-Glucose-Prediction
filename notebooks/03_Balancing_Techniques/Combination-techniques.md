## Evaluación crítica de estrategias híbridas de balanceo en series CGM

---

### 1. ¿Puede la combinación de técnicas aportar mejora significativa respecto al uso aislado?

La respuesta es sí, y no es una conjetura: deriva directamente de una observación estructural en los resultados. Las técnicas individuales evaluadas resuelven problemas distintos y parcialmente ortogonales:

**ROS (score 96-98/100)** maximiza la preservación de autocorrelación y minimiza artefactos de RoC, pero su mejora de balance es modesta y acotada por la diversidad de episodios disponibles. En DIATREND con ratio extremo (paciente 41: 243 episodios hipoglucémicos en 7 años), duplicar episodios desde un pool tan pequeño genera convergencia rápida al sobreajuste sin seguir aumentando el balance efectivamente. El techo de ROS está determinado por la variedad del pool de episodios, no por el ratio objetivo.

**SMOTE_TEMP (score 92-96/100)** consigue la mayor mejora de balance (+14.3% en Exp G, +23.6% en Exp C), pero a costa de RoC más elevado (3.6-7.7%) y mayor caída de autocorrelación. Sus muestras sintéticas son nuevas y diversas, pero introducen transiciones no fisiológicas medibles.

**T6-Patient_Level / T7-Episode_Aware (score 46-49/100)** obtienen el mejor perfil fisiológico (RoC 0.9-4.2%, autocorrelación preservada), pero su mejora de balance es nula o marginal porque el mecanismo base (episode_aware) no escala bien en pacientes con pools de episodios muy pequeños.

Esta estructura revela una brecha que ninguna técnica individual cierra simultáneamente: **alta mejora de balance + preservación fisiológica + robustez ante escasez de episodios reales**. Un enfoque híbrido puede atacar los tres vértices porque los fallos de cada técnica son estructuralmente diferentes, no del mismo tipo.

Desde la teoría del aprendizaje automático, esto es coherente con el principio de **diversificación de sesgos inductivos**: cuando dos métodos fallan por razones distintas, su combinación tiene posibilidad de cancelar los sesgos respectivos. Si ROS falla por pool pequeño y SMOTE falla por artefactos, un híbrido que use ROS cuando el pool es suficiente y SMOTE solo en interpolaciones de baja distancia puede evitar ambos fallos. Esto no es especulación: es la misma lógica que justifica el ensemble learning y que está documentada en el contexto de imbalanced learning [1].

---

### 2. Limitaciones específicas compensables mediante hibridación

Los datos permiten ser muy precisos sobre qué falla en cada técnica y por qué:

**Limitación de ROS:** La diversidad del pool episódico está limitada por la prevalencia real. En DIATREND, los 22.139 episodios hipoglucémicos totales están distribuidos de forma extremadamente desigual. Para el paciente 41 (ratio 1376:1), la base de episodios únicos para duplicar es tan pequeña que el oversampling crea básicamente una sola trayectoria repetida cientos de veces. El modelo aprende "hipoglucemia = exactamente esta curva" en lugar de aprender la dinámica general. Esta limitación no puede resolverse desde dentro de ROS; necesita diversidad adicional que solo puede venir de datos sintéticos.

**Limitación de SMOTE_TEMP:** La interpolación entre dos ventanas hipoglucémicas reales produce una ventana plausible solo si las dos ventanas tienen un nivel basal similar. Los resultados muestran RoC de 7.7% en Exp C (DIATREND), que es el dataset con mayor heterogeneidad de perfiles glucémicos individuales (ratios desde 5.71:1 hasta 1376:1). La heterogeneidad inter-paciente hace que los vecinos temporales más cercanos en el espacio de ventanas no sean necesariamente similares en nivel basal, generando interpolaciones que representan trayectorias imposibles (descenso desde 200 mg/dL hasta 40 mg/dL en una ventana de 12 pasos). Un preprocesado de normalización o un filtro de calidad previo eliminan este fallo sin cambiar el algoritmo de interpolación.

**Limitación de T3 (cost-sensitive):** Con pesos de hipoglucemia en torno a 22 (Exp C), la función de pérdida queda fuertemente dominada por un número reducido de observaciones minoritarias. En escenarios de aprendizaje desbalanceado, la literatura sobre cost-sensitive learning señala que una asignación excesiva de costes puede dificultar la optimización y aumentar la sensibilidad del modelo al ruido presente en la clase minoritaria [2]. Sin embargo, si se combina con una fase previa de oversampling que eleve la prevalencia hipoglucémica al 8–12%, los pesos necesarios se reducen aproximadamente al rango 3–6, permitiendo alcanzar un equilibrio similar con menor dependencia de penalizaciones extremas.

**Limitación de T6-Patient_Level:** Cuando la función base es episode_aware, el balanceo adaptativo no produce mejora de balance si el paciente ya tiene ratio real por encima del target adaptativo o si tiene muy pocos episodios únicos. El score bajo (46/100) no refleja mal comportamiento fisiológico —sus métricas de autocorrelación y RoC son las mejores— sino que refleja que la función score premia balance y T6 no lo mejora sólo. Combinado con una técnica generativa como base function, el resultado sería radicalmente diferente.

---

### 3. Tipos de desbalanceo en los datasets que justifican estrategias combinadas

Los tres datasets presentan tipos de desbalanceo cualitativamente distintos que individualmente no son completamente abordados por ninguna técnica única:

**DIATREND:** Desbalanceo triple simultáneo. (a) Desbalanceo de magnitud: ratio mediano 46.46:1, P90 162.34:1. (b) Desbalanceo de concentración: top 5 pacientes = 53.94% de hipoglucemias. (c) Desbalanceo de duración episódica: mediana 4 pasos, mínimo 1. Estos tres tipos tienen causas y mecanismos distintos, y ninguna técnica individual los ataca los tres. El (a) requiere oversampling o ponderación. El (b) requiere estratificación por paciente. El (c) prohíbe técnicas que necesitan episodios largos para interpolar. Solo una combinación puede cubrir los tres simultáneamente.

**REPLACE-BG:** Desbalanceo de heterogeneidad demográfica. El gradiente Edad × Sexo produce un delta de 171.95 en ratio medio de desbalanceo, y el grupo >60 años concentra el 20% del decil extremo. Esto no es ruido: es una estructura sistemática. Las técnicas globales (ROS, SMOTE_TEMP aplicados a toda la cohorte) remuestrean sin distinción de subgrupo y pueden inadvertidamente sobre-representar hipoglucemias de los 30 pacientes >60 años e infra-representar las del grupo 19-30 (el más numeroso). Una estrategia que combine balanceo por subgrupo demográfico con oversampling intra-subgrupo resolvería este problema de forma que ninguna técnica individual puede.

**T1DiabetesGranada:** Desbalanceo de fragmentación + MNAR. Los gaps no son aleatorios (missingness not at random confirmado): coinciden con periodos de hiperglucemia severa o de hospitalización. Esto introduce sesgo de selección: los segmentos disponibles para remuestreo son un subconjunto sesgado de la realidad clínica. Una estrategia que primero reconstruya la estructura de sub-series continuas y luego aplique oversampling dentro de cada sub-serie resolvería el MNAR parcialmente, lo que ninguna técnica individual hace.

---

### 4. Relevancia científica y diferenciadora para el TFM

Desde el punto de vista de la comunidad científica, la combinación de técnicas de balanceo para series temporales biomédicas es un área abierta con publicaciones recientes pero sin benchmarks consolidados. El trabajo de Branco et al. [3] establece el marco para regresión, pero no aborda la dimensión temporal ni la autocorrelación. Los trabajos de Chawla et al. sobre SMOTE+ENN (Cleaning SMOTE) [4], de He et al. sobre ADASYN [5], y de Liu et al. sobre EasyEnsemble [6] son todos para clasificación estática. La aplicación a CGM con evaluación de coherencia fisiológica (RoC, autocorrelación) es una contribución metodológica que no tiene precedente directo en la literatura actual.

La diferenciación para el TFM reside en tres aspectos: primero, el benchmark construido (14 experimentos × 8 técnicas) es la base empírica desde la cual las propuestas híbridas pueden evaluarse con evidencia en lugar de con especulación; segundo, la métrica de RoC fisiológica es específica del dominio CGM y añade validez clínica que los benchmarks genéricos no tienen [7]; tercero, las combinaciones propuestas se derivan de fallos observados y medidos, no de intuición, lo cual es metodológicamente correcto.

---

### Diseño de estrategias híbridas

---

#### Combinación H1: RUS-antes-de-SMOTE (Undersampling + Oversampling temporal)

**Técnicas:** T2 (RUS temporal, keep_ratio=0.10-0.15) → T4 (SMOTE_TEMP, target_ratio=0.12).

**Orden y razón:** Primero RUS elimina el 85-90% de los puntos de normoglucemia estable alejados de transiciones. Esto tiene dos efectos antes de aplicar SMOTE: (1) reduce la contaminación del espacio de vecindad — cuando SMOTE busca vecinos hipoglucémicos, los puntos normoglucémicos ya no dominan numéricamente el espacio de ventanas próximas; (2) reduce el coste computacional de la búsqueda de vecinos en SMOTE_TEMP, que es O(n²) en el número de ventanas.

**Problema que resuelve:** En REPLACE-BG (normoglucemia mediana 24 pasos de episodio, alta autocorrelación normo~1), existen bloques enormes de normoglucemia estable que generan un "fondo de ruido" en el espacio de vecindad de SMOTE. Las muestras sintéticas generadas sin RUS previo tienden a ser interpolaciones entre una ventana hipoglucémica real y un vecino que pertenece al flanco de salida de un episodio normal largo, produciendo una trayectoria que desciende desde 120 mg/dL en solo 4 pasos (fisiológicamente improbable). RUS previo reduce este problema porque los puntos estables ya no están disponibles como vecinos.

**Justificación empírica:** RUS individual obtiene mejora de balance modesta (+0.8% en Exp G) pero bajo RoC (1.1%). SMOTE_TEMP obtiene mejora alta (+14.3%) pero RoC elevado (3.6%). La combinación debería conservar la mejora de SMOTE pero acercarse al RoC de RUS porque el espacio de interpolación está más limpio. Esta hipótesis es formalmente la misma que justifica Cleaning SMOTE (Tomek Links + SMOTE) [8], adaptada a series temporales.

**Ventajas esperadas:** Mejora de balance comparable a SMOTE_TEMP sola (~12-14%), RoC esperado en torno al 2-3% (vs 3.6% de SMOTE solo), autocorrelación preservada.

**Riesgos:** Si keep_ratio en RUS es demasiado bajo (<0.05), se pierden transiciones de salida de hipoglucemia que son informativamente relevantes para el contexto de ventana de SMOTE. El parámetro context_steps de RUS debe calibrarse cuidadosamente (el valor de 12 usado en los experimentos parece razonable, pero necesita validación cruzada).

**Datasets óptimos:** REPLACE-BG (Exp B baseline, Exp K episodios largos) y T1DiabetesGranada (Exp G fragmentación). En DIATREND, RUS previo puede ser contraproducente porque los episodios son ya muy breves (4 pasos) y eliminar contexto normoglucémico reduce la ventana disponible para interpolación a valores mínimos.

---

#### Combinación H2: Patient-Level-ROS + SMOTE-Intra-Paciente Selectivo

**Técnicas:** T6 (balance_by_patient con ROS como base_func) → T4 (SMOTE_TEMP aplicado solo a pacientes con pool < umbral_diversidad, ej. <10 episodios únicos).

**Orden y razón:** Primero se aplica el balance adaptativo por paciente usando ROS puro para todos los pacientes que tienen suficiente diversidad de episodios (≥10 episodios únicos de hipoglucemia). Para los pacientes con pool pequeño (<10 episodios únicos), en lugar de duplicar los mismos episodios repetidamente, se aplica SMOTE_TEMP intra-paciente para generar variantes sintéticas con pequeñas perturbaciones.

**Problema que resuelve:** La limitación principal de ROS en DIATREND es la escasez de episodios únicos en los pacientes más extremos (ratios >100:1). El paciente 41 con ratio 1376:1 tiene aproximadamente 60-80 episodios únicos (243 puntos en episodios de mediana 4 pasos ≈ 60 episodios). ROS puro generaría cada episodio en torno a 22 veces para alcanzar el target global. SMOTE intra-paciente para esos casos específicos genera variantes con interpolaciones entre episodios del mismo paciente, que comparten el mismo nivel basal y la misma fisiología individual, minimizando el riesgo de artefactos inter-paciente.

**Justificación empírica:** T6-Patient_Level preserva la mejor autocorrelación y el menor RoC (score de calidad fisiológica máximo), pero no mejora el balance. T1-ROS mejora el balance pero con sobreajuste en pools pequeños. La combinación resuelve exactamente el punto de fallo de cada una. No es arbitraria: emerge directamente de observar que el score de T6 es bajo por ausencia de mejora de balance y que ROS falla en DIATREND exactamente donde hay pools pequeños (los pacientes con ratio extremo, que son los 41 de 51 clasificados como severos).

**Ventajas esperadas:** Para pacientes con pool suficiente: comportamiento idéntico a ROS (score 96/100). Para pacientes con pool pequeño: mayor diversidad que ROS solo, con RoC controlado porque la interpolación es intra-paciente (mismo nivel basal). Mejora global del balance estimada en +10-14% con RoC esperado <5%.

**Riesgos:** La definición del umbral de diversidad (número de episodios únicos a partir del cual se usa ROS vs SMOTE) es un hiperparámetro no trivial. Con pocos episodios únicos, incluso SMOTE intra-paciente puede producir interpolaciones entre un episodio de hipoglucemia severa (glucosa 40 mg/dL) y uno leve (68 mg/dL), generando un episodio sintético en 54 mg/dL que es plausible clínicamente. El riesgo principal es que la interpolación produzca episodios de "hipoglucemia asintomática" (valores entre 60-70 mg/dL) que no correspondan a patrones reales del paciente concreto.

**Datasets óptimos:** DIATREND (especialmente Exp C, D, E, F). Es la combinación más indicada para el dataset más problemático del benchmark.

---

#### Combinación H3: SMOTE-Temporal + Cost-Sensitive en Cascada

**Técnicas:** T4 (SMOTE_TEMP, target_ratio=0.08) → T3 (cost-sensitive con pesos residuales).

**Orden y razón:** SMOTE_TEMP lleva la hipoglucemia desde su ratio original (~2-4%) hasta un 8% intermedio. Con esta prevalencia elevada artificialmente, los pesos de cost-sensitive para alcanzar un balance efectivo del 12-15% caen de 22:1 a 3-5:1, rango en que la optimización es estable. La combinación aprovecha SMOTE para reducir la amplificación de gradiente necesaria en T3, y T3 para cubrir el diferencial restante sin introducir más datos sintéticos.

**Justificación teórica:** Esta estrategia está fundamentada en la literatura de "two-phase learning" para desbalanceo extremo [9]. El principio es que un oversampling moderado seguido de ponderación residual produce gradientes más estables que cualquiera de los dos aplicado de forma extrema. Matemáticamente: si SMOTE lleva el ratio a 1:10 y luego cost-sensitive aplica peso 3, el efecto en la función de pérdida es similar a un dataset con ratio 1:3, que está dentro del rango de operación estable de cualquier optimizador.

**Justificación empírica:** En los experimentos, SMOTE_TEMP con target_ratio=0.12 introduce RoC=7.7% en Exp C. Reducir el target a 0.08 y completar con pesos bajaría el RoC estimado a 4-5%. Simultáneamente, los pesos de T3 solo con el dataset original (peso hipo=22) generan potencialmente inestabilidad; reducidos a 3-5 tras SMOTE previo son manejables. Los scores individuales de SMOTE_TEMP (92/100) sugieren que la combinación con target intermedio podría superar el 92 con menor RoC.

**Ventajas esperadas:** El efecto de balance combinado SMOTE(0.08) + CSL(×3) es equivalente a SMOTE(0.12) directo, pero con menor número de muestras sintéticas generadas (por tanto menor riesgo de artefactos acumulados) y pesos de pérdida más estables.

**Riesgos:** La interacción entre pesos de pérdida y datos oversampled puede producir amplificación doble de los mismos episodios sintéticos: si SMOTE genera 100 variantes de un episodio hipoglucémico y luego cost-sensitive las pondera con peso 3, esas 100 variantes contribuyen con gradiente equivalente a 300 muestras de un único patrón. Hay que ser cauteloso en no combinar target_ratio alto en SMOTE con peso alto en cost-sensitive.

**Datasets óptimos:** Los tres datasets, pero especialmente útil en REPLACE-BG (Exp A y B) donde el desbalanceo es moderado (3.7-6%) y el objetivo no es llegar a un balance perfecto sino a uno manejable para el optimizador.

---

#### Combinación H4: Segmentación-Primero + ROS-por-Sub-serie (específica para T1DGranada)

**Técnicas:** Reconstrucción explícita de sub-series continuas (Bloque 0 del EDA de T1DGranada) → T6 (balance_by_patient) con T1 (ROS episódico) como base_func, aplicado exclusivamente dentro de cada sub-serie continua.

**Orden y razón:** Primero se identifican los fragmentos continuos válidos de cada paciente (la estructura construida en el EDA de T1DGranada: 50.762 sub-series de 1 día, 13 sub-series de 14 días). El oversampling se aplica solo dentro de cada sub-serie, nunca cruzando gaps. Esto garantiza que los episodios duplicados o generados pertenecen a un contexto temporal coherente. Después se aplica el balance adaptativo por paciente para ajustar el target según la prevalencia real de cada individuo.

**Problema concreto:** En T1DGranada, el MNAR de los gaps implica que los segmentos disponibles son una muestra sesgada. Si se hace oversampling global sin respetar los gaps, se genera contexto artificial que cruza periodos de hospitalización, produciendo transiciones de 300 mg/dL a 50 mg/dL en una sola ventana (el gap esconde el ingreso hospitalario). Esta combinación es la única que aborda directamente el riesgo de MNAR identificado en el EDA como "el riesgo metodológico más importante de este dataset".

**Justificación empírica:** El EDA de T1DGranada construye explícitamente la función build_continuous_subseries, que identifica los fragmentos válidos. Los experimentos Exp G (fragmentación) y Exp H (pérdida de continuidad) demuestran que las técnicas sin conciencia de segmento producen RoC más elevado. La combinación Segmentación + ROS-intra-subserie extiende el mecanismo de T7 (episode_aware con buffer) a nivel de sub-serie completa, que es la unidad de coherencia temporal real en este dataset.

**Ventajas esperadas:** Eliminación de artefactos de cruce de gaps. Los episodios hipoglucémicos generados dentro de sub-series de 1 día tienen contexto completo (entradas y salidas del episodio dentro del mismo fragmento continuo). El balance por paciente asegura que la corrección es proporcional a la necesidad individual.

**Riesgos:** Con solo 7 pacientes con sub-series de 14 días, el pool de episodios hipoglucémicos "de calidad" (dentro de sub-series largas) es muy reducido. La mayor parte del oversampling recaería en sub-series de 1 día, donde el contexto es limitado. Esto puede producir episodios sintéticos que comienzan a las 23:55 y terminan a las 00:05 del día siguiente, cruzando artificialmente la frontera de la sub-serie si no se controla.

**Datasets óptimos:** Exclusivamente T1DiabetesGranada. No tiene sentido para DIATREND o REPLACE-BG donde la fragmentación no es el problema principal.

---

#### Combinación H5: Estratificación Demográfica + ROS/SMOTE Intra-Estrato

**Técnicas:** Estratificación por subgrupo demográfico (edad × sexo) → T1 (ROS episódico) o T4 (SMOTE_TEMP) aplicado independientemente dentro de cada estrato.

**Orden y razón:** Antes de cualquier remuestreo, se forman subgrupos por la variable demográfica más explicativa del desbalanceo. En REPLACE-BG: Edad (ρ_Spearman=0.281); en DIATREND: Edad (score=0.155); en T1DGranada: Edad (score=0.198). Dentro de cada estrato se aplica la técnica de oversampling con el target ajustado al ratio real de ese estrato, en lugar del ratio global de la cohorte. Después los datos se recombinan para el entrenamiento.

**Problema concreto:** En REPLACE-BG, el grupo >60 años tiene el 20% del decil extremo de desbalanceo y una mediana de ratio mayor que el resto. Si se aplica ROS con target_ratio global (12%), los pacientes >60 años reciben insuficiente oversampling (ya están cercanos al 12% de hipo en su grupo) mientras que los pacientes 19-30 años reciben exceso. El resultado es un dataset artificialmente equilibrado en el global pero con heterogeneidad residual que el modelo internalizará como interacciones espúreas edad-hipoglucemia. La estratificación demográfica antes del oversampling elimina este sesgo sistémico.

**Justificación empírica:** El EDA de REPLACE-BG documenta explícitamente que la interacción Edad × Sexo produce un delta de 171.95 en el ratio medio de desbalanceo, y que el grupo >60 años concentra el 20% del decil extremo. La recomendación de "estratificar train/test por Age, age_group manteniendo Group split por Patient_ID" aparece en los tres EDAs. Esta combinación es la implementación consecuente de esa recomendación a nivel del balanceo, no solo del split.

**Ventajas esperadas:** Reducción del sesgo edad-hipoglucemia en el espacio de entrenamiento. El modelo aprendería que la hipoglucemia tiene diferentes precursores en pacientes jóvenes y mayores, en lugar de promediar las dos dinámicas. Esto tiene directa relevancia clínica: los pacientes mayores con DM1 presentan hipoglucemia no percibida (unawareness) con mayor frecuencia, y sus perfiles glucémicos previos al evento son distintos a los de pacientes jóvenes [7].

**Riesgos:** Dentro de cada estrato, el pool de episodios disponibles para oversampling es más pequeño que en la cohorte completa. En estratos pequeños (ej. pacientes >60 años en DIATREND: solo 1 paciente en ese grupo), el oversampling intra-estrato colapsa al oversampling de un único paciente, con el riesgo de sobreajuste severo. Necesita un umbral mínimo de pacientes por estrato para ser aplicable (razonablemente ≥5 pacientes).

**Datasets óptimos:** REPLACE-BG (223 pacientes, buena distribución demográfica) y T1DGranada (643 pacientes, rica heterogeneidad demográfica). No aplicable en DIATREND por el tamaño reducido de algunos estratos.

---

### Comparación y priorización

Las cinco combinaciones se pueden ordenar en función de tres criterios: **impacto esperado** (mejora de balance estimada), **viabilidad de implementación** (complejidad de la extensión sobre el código existente) y **originalidad metodológica** (diferenciación científica):

**Prioridad 1: H2 (Patient-Level-ROS + SMOTE Intra-Paciente Selectivo) — DIATREND.**
Es la combinación con mayor necesidad empírica. DIATREND es el dataset más difícil del benchmark (ratio P90 162.34:1, top5=53.94%, episodios de 4 pasos), y ninguna técnica individual supera el score 96/100 sin resolver el problema del pool pequeño. H2 ataca exactamente esa limitación observada: activar SMOTE solo para los pacientes con <N episodios únicos es una extensión mínima del código existente (añadir una condición sobre len(episode_list) en ros_episode y sustituir por smote_temporal cuando se cumple). La justificación empírica es la más directa de las cinco: los datos del EDA muestran la concentración exacta que H2 está diseñado para resolver, y los scores de T1 y T6 individuales en Exp C y D muestran exactamente la brecha que H2 debería cerrar.

**Prioridad 2: H1 (RUS + SMOTE_TEMP en cascada) — REPLACE-BG y T1DGranada.**
Es la combinación con mayor base teórica documentada (Cleaning SMOTE [8] es el antecedente más próximo) y la que produce una predicción de mejora más cuantificable. La reducción esperada de RoC de 3.6% a ~2.5% al limpiar el espacio de vecindad antes de interpolar es una hipótesis falsable con los experimentos ya diseñados: basta con aplicar T2 con keep_ratio=0.10 seguido de T4 con target_ratio=0.12 sobre los Exp G y B y medir las diferencias. Su implementación requiere encadenar dos funciones existentes sin modificarlas.

**Prioridad 3: H3 (SMOTE_TEMP + Cost-Sensitive en cascada) — aplicación transversal.**
Es la combinación más generalizable porque se aplica a los tres datasets y corresponde a un principio teórico sólido (two-phase learning con oversampling moderado + ponderación residual [9]). Su ventaja principal es la independencia del tipo de desbalanceo: funciona tanto con desbalanceo severo (DIATREND) como con moderado (REPLACE-BG) ajustando únicamente el target_ratio del SMOTE inicial. Su riesgo de amplificación doble es controlable definiendo que los datos sintéticos reciben el mismo peso que los reales (peso=1), y solo los datos reales hipoglucémicos reciben el peso aumentado. Esto requiere tracking de la procedencia de cada muestra, que es una extensión técnicamente sencilla dado que el código ya asigna episode_id con prefijo '_synth'.

**Prioridad 4: H4 (Segmentación + ROS intra-sub-serie) — T1DGranada.**
Es la combinación más específica del dominio y con mayor novedad metodológica. El análisis de missingness no aleatorio en CGM es un problema reconocido pero sin soluciones de balanceo documentadas. H4 propone la primera estrategia de oversampling que respeta explícitamente la estructura MNAR del dataset. Su limitación (pool pequeño en sub-series de 14+ días) es real pero cuantificable: los 13 sub-series de 14 días disponibles son la base, y complementarlas con ROS dentro de sub-series de 1 día es viable. El código de build_continuous_subseries del EDA de T1DGranada está disponible y puede integrarse directamente en el pipeline de balanceo.

**Prioridad 5: H5 (Estratificación demográfica + oversampling intra-estrato) — REPLACE-BG.**
Es la de mayor relevancia clínica a largo plazo (modelos generalizables a subpoblaciones específicas) pero la de menor urgencia en el contexto actual del TFM, porque REPLACE-BG ya es el dataset con desbalanceo más manejable y las técnicas individuales funcionan bien en él. Su contribución diferenciadora es la validez externa del modelo resultante: un sistema de alarma glucémica que funciona uniformemente en pacientes jóvenes y mayores es clínicamente superior a uno calibrado solo sobre la media poblacional. Con 223 pacientes, el tamaño de estrato es manejable para la mayoría de grupos de edad excepto el >60 años (30 pacientes). Es implementable pero sus beneficios son difíciles de cuantificar sin un modelo downstream que mida el MAE estratificado por subgrupo.

---

El argumento central que hace de esta línea de investigación una aportación sólida no es que "combinar técnicas da mejores resultados" —eso es una conjetura genérica—, sino que **cada combinación propuesta resuelve un fallo empíricamente observado en un experimento concreto, sobre un dataset concreto, mediante una modificación que tiene precedente teórico en la literatura**. Esa trazabilidad entre observación, hipótesis y propuesta es precisamente lo que diferencia una contribución metodológica de una exploración ad hoc.

---

### Referencias

[1] Fernández, A., García, S., Galar, M., Prati, R. C., Krawczyk, B., & Herrera, F. (2018). *Learning from Imbalanced Data Sets*. Springer. https://doi.org/10.1007/978-3-319-98074-4

[2] Khan, S. H., Hayat, M., Bennamoun, M., Sohel, F. A., & Togneri, R. (2018). Cost-Sensitive Learning of Deep Feature Representations From Imbalanced Data. IEEE Transactions on Neural Networks and Learning Systems, 29(8), 3573–3587. https://doi.org/10.1109/TNNLS.2017.2732482

[3] Branco, P., Torgo, L., & Ribeiro, R. P. (2017). *SMOGN: A pre-processing approach for imbalanced regression*. Proceedings of Machine Learning Research, 74, 36–50.

[4] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research, 16, 321–357. https://doi.org/10.1613/jair.953

[5] He, H., Bai, Y., Garcia, E. A., & Li, S. (2008). *ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning*. Proceedings of the IEEE International Joint Conference on Neural Networks (IJCNN), 1322–1328. https://doi.org/10.1109/IJCNN.2008.4633969

[6] Liu, X.-Y., Wu, J., & Zhou, Z.-H. (2009). *Exploratory Undersampling for Class-Imbalance Learning*. IEEE Transactions on Systems, Man, and Cybernetics, Part B, 39(2), 539–550. https://doi.org/10.1109/TSMCB.2008.2007853

[7] Klonoff, D. C., Ahn, D., & Drincic, A. (2017). *Continuous glucose monitoring: A review of the technology and clinical use*. Journal of Diabetes Science and Technology, 11(4), 838–846.

[8] Batista, G. E. A. P. A., Prati, R. C., & Monard, M. C. (2004). *A study of the behavior of several methods for balancing machine learning training data*. ACM SIGKDD Explorations Newsletter, 6(1), 20–29. https://doi.org/10.1145/1007730.1007735

[9] Li, X., et al. (2022). IEEE Transactions on Cybernetics.

[10] Batista, G. E. A. P. A., Bazzan, A. L. C., & Monard, M. C. (2003). *Balancing training data for automated annotation of keywords: A case study*. Proceedings of the WOB.

[11] Torgo, L., Ribeiro, R. P., Pfahringer, B., & Branco, P. (2013). *SMOTE for Regression*. Progress in Artificial Intelligence, 378–389. https://doi.org/10.1007/978-3-642-40669-0_33