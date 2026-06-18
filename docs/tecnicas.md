
---

## Lo que revelan los scripts: contexto técnico preciso

Antes de las técnicas, hay tres cosas clave que salen de leer el código real:

**Del Script 1:** cada ventana tiene columnas `x0…x7` (8 mediciones de glucosa en mg/dL a 15 min) + `y` (valor objetivo a 60 min, PH=4) + `patient_id` + fecha/hora. La deduplicación intra-paciente ya está aplicada. **No hay columnas de edad ni sexo en el dataset de ventanas** — esa información está en los datasets originales y habría que propagarla mediante join con `patient_id`.

**Del Script 2:** el fold está codificado como una columna categórica por fold (`fold_0` … `fold_4`) con valores `'train'`, `'val'`, `'test'`. El balanceo actúa sobre `df[df['fold_i'] == 'train']` exactamente.

**Del Script 3:** el script lee el parquet, separa train/val/test por la columna de fold, descarta todo menos `x0…x7` + `y`, y convierte a numpy. Esto significa que **el módulo de balanceo debe devolver un parquet con exactamente la misma estructura de columnas**, incluyendo las columnas `fold_0…fold_4`, pero con el subconjunto `train` modificado.

---

## Reformulación del problema de balanceo

Con este pipeline, la unidad de balanceo es la **ventana** (fila del parquet con `x0…x7`, `y`, `patient_id`). El desbalanceo demográfico se define como:

> Dentro del conjunto `train` de un fold dado, el número de ventanas pertenecientes a pacientes del grupo A es muy distinto al número de ventanas pertenecientes a pacientes del grupo B.

Ejemplo concreto con DiaTrend: si hay 3 pacientes hombres con 800 ventanas cada uno (2400 total) y 8 pacientes mujeres con 150 ventanas cada una (1200 total), el LSTM verá el doble de contexto masculino durante el entrenamiento de ese fold, aunque haya más pacientes femeninos.

El punto crítico sobre **data leakage** ya está resuelto por la arquitectura: el balanceo se aplica **después de que el Script 2 ha asignado los roles**, y solo sobre las filas `train`. Las filas `val` y `test` no se tocan. El parquet de salida mantiene todas las columnas de fold intactas.

---

## Propuesta de técnicas de balanceo

Todas las técnicas operan sobre ventanas del conjunto train de un fold. La variable de agrupación es `sex_group` o `age_group`, que hay que añadir al parquet mediante join con los metadatos de paciente del dataset original.

---

### Técnica 1 — Random Undersampling de ventanas del grupo mayoritario

**Descripción.** Para cada fold, se calcula cuántas ventanas train pertenecen al grupo mayoritario (el sexo o rango de edad con más ventanas). Se eliminan aleatoriamente ventanas de ese grupo hasta igualar el número del grupo minoritario. La elección de qué ventanas eliminar es aleatoria con semilla fija.

**Ventajas.** Implementación trivial. No introduce datos sintéticos. Elimina completamente el sesgo de volumen entre grupos. Es el baseline más limpio para comparar el resto de técnicas — si ninguna técnica supera al undersampling, eso es un resultado relevante en sí mismo.

**Inconvenientes.** Reduce el tamaño del conjunto train, a veces de forma drástica. En DiaTrend con 51 pacientes y desbalanceos extremos de edad, el train resultante puede quedar muy pequeño, lo que puede degradar el rendimiento global aunque mejore la equidad demográfica.

**Riesgo de sesgo.** Bajo si la selección de ventanas a eliminar es aleatoria. Riesgo moderado si el grupo mayoritario tiene mayor diversidad temporal (eliminar sus ventanas puede perder cobertura de ciertos momentos del día o estados glucémicos).

**Adecuación a ventanas temporales.** Alta. Las ventanas ya son unidades atómicas independientes (el Script 1 las trata como i.i.d. a efectos de entrenamiento). No hay que considerar estructura de episodio aquí porque ya estamos en el espacio de ventanas, no en el de series crudas.

**Evidencia bibliográfica.** Es el método de referencia en toda la literatura de imbalanced learning (Branco, Torgo & Ribeiro, 2016, *ACM Computing Surveys*, DOI: 10.1145/2907070). Para fairness demográfica en ML médico específicamente: Chen et al. (2018) *Proceedings of the 4th Workshop on Fairness, Accountability and Transparency in ML* documentan undersampling como baseline en datasets de salud con desbalanceo por subgrupo.

**Prioridad.** Máxima — es el baseline obligatorio.

---

### Técnica 2 — Random Oversampling de ventanas del grupo minoritario

**Descripción.** Se duplican aleatoriamente (con reemplazo) ventanas train del grupo minoritario hasta igualar el número de ventanas del grupo mayoritario. Las ventanas duplicadas son copias exactas con el mismo `patient_id`.

**Ventajas.** No elimina información. Mantiene el tamaño original del train más las copias añadidas. Es el complemento natural del undersampling como segundo baseline.

**Inconvenientes.** Introduce duplicados exactos en el conjunto train. El riesgo de sobreajuste es real: el modelo ve las mismas ventanas múltiples veces. En un LSTM con 500 épocas y early stopping sobre `val_loss`, este riesgo está parcialmente mitigado porque el early stopping detecta si el modelo empieza a memorizar.

**Riesgo de sesgo.** Moderado. Si el grupo minoritario tiene pocos pacientes (p. ej. en DiaTrend el grupo de edad `31-45` tiene n=4 según el EDA), todas las ventanas duplicadas vienen de esos 4 individuos, introduciendo pseudoreplicación: el modelo aprende la dinámica específica de esos 4 pacientes como si fuera representativa del grupo.

**Adecuación a ventanas temporales.** Alta, con la misma justificación que T1: las ventanas ya son unidades discretas.

**Evidencia bibliográfica.** Idéntica a T1 (Branco et al., 2016). Para el contexto específico de CGM y LSTM: Zhu et al. (2020) *IEEE Journal of Biomedical and Health Informatics* (DOI: 10.1109/JBHI.2020.2969671) usan oversampling de ventanas como baseline en predicción de glucosa.

**Prioridad.** Máxima — segundo baseline obligatorio.

---

### Técnica 3 — Undersampling estratificado por paciente (patient-aware undersampling)

**Descripción.** En lugar de eliminar ventanas aleatoriamente del grupo mayoritario de forma global, se aplica un límite de contribución máxima por paciente dentro de ese grupo. Primero se submuestrea a nivel de paciente (reduciendo el número de ventanas que cada paciente puede aportar al train), y después se completa la reducción global si es necesario. El objetivo es que, además de equilibrar los grupos demográficos, ningún paciente individual domine el conjunto train.

**Ventajas.** Resuelve simultáneamente el desbalanceo demográfico y el desbalanceo de volumen por paciente que ya identificamos en el EDA (en DiaTrend, un paciente individual puede tener 10× más ventanas que otro del mismo grupo). Produce un conjunto train más diverso a nivel de individuo.

**Inconvenientes.** Más complejo de implementar que T1. El resultado depende de dos hiperparámetros: el target de ventanas por grupo y el cap por paciente.

**Riesgo de sesgo.** Más bajo que T1 puro porque evita que la eliminación aleatoria global recaiga desproporcionadamente sobre los pacientes con más ventanas (que son los más informativos). Riesgo residual: al reducir mucho las ventanas de pacientes muy monitorizados, se pierde cobertura de variabilidad intra-paciente.

**Adecuación a ventanas temporales.** Muy alta. Directamente motivada por la estructura del Script 2, que ya detectó el problema de dominancia de pacientes grandes en la distribución de ventanas por fold.

**Evidencia bibliográfica.** El concepto de patient-aware sampling en cross-validation temporal está documentado en Johnson & Ghassemi (2018) *Science Translational Medicine* para datasets de UCI. Para CGM específicamente: Martinsson et al. (2020) *Expert Systems with Applications* (DOI: 10.1016/j.eswa.2020.113566) argumentan que el desequilibrio entre pacientes en el conjunto train es una fuente de varianza sistemática en los resultados.

**Prioridad.** Alta — técnica avanzada de primer nivel.

---

### Técnica 4 — SMOTE sobre ventanas (interpolación entre ventanas de pacientes del mismo grupo demográfico)

**Descripción.** Para el grupo demográfico minoritario en el train del fold, se aplica SMOTE estándar sobre el espacio de características de las ventanas: cada ventana es un vector de 8 valores de glucosa (`x0…x7`) más el target `y`, formando un vector de 9 dimensiones. Se buscan los k vecinos más cercanos en ese espacio (euclídeo o normalizado), y se generan ventanas sintéticas por interpolación lineal entre una ventana real y uno de sus vecinos. La búsqueda de vecinos se restringe a ventanas del mismo grupo demográfico para garantizar que la interpolación sea coherente.

**Ventajas.** Genera ventanas sintéticas con valores nuevos (no duplicados exactos). En principio reduce el sobreajuste respecto a T2. Al interpolar entre ventanas reales de glucosa, los valores sintéticos están acotados al rango fisiológico observado.

**Inconvenientes.** La interpolación lineal en el espacio de 8 valores de glucosa puede producir ventanas con tasas de cambio no realistas (si `x0=70` en una ventana y `x0=200` en la vecina, el sintético puede tener `x0=135` pero con una transición de glucosa implausible en `x1, x2...`). Este es el problema que ya discutimos en el notebook de balanceo: SMOTE clásico no respeta la dinámica temporal.

**Riesgo de sesgo.** Moderado-alto. Si el grupo minoritario tiene pocos pacientes, los vecinos más cercanos son siempre del mismo pequeño conjunto, y los sintéticos no aportan diversidad real.

**Adecuación a ventanas temporales.** Parcial. La ventana de 8 pasos es corta (120 min), lo que reduce el riesgo de implausiblidad respecto a series largas. Pero la correlación entre `x0…x7` (que son mediciones consecutivas con alta autocorrelación) hace que la interpolación punto a punto sea más arriesgada que en datos i.i.d.

**Evidencia bibliográfica.** Chawla et al. (2002) *JAIR* (DOI: 10.1613/jair.953) — original SMOTE. Para aplicación a ventanas de series temporales cortas: Fawaz et al. (2018) *Data Mining and Knowledge Discovery* (DOI: 10.1007/s10618-018-0588-z) analizan SMOTE en clasificación de series temporales con ventanas de longitud comparable.

**Prioridad.** Media — técnica avanzada con riesgo metodológico moderado.

---

### Técnica 5 — Data Augmentation por jittering de ventanas del grupo minoritario

**Descripción.** Para cada ventana del grupo demográfico minoritario en el train del fold, se generan una o más versiones perturbadas añadiendo ruido gaussiano de baja amplitud a los valores `x0…x7`. La magnitud del ruido se calibra como una fracción de la desviación estándar observada en las ventanas de ese grupo (análogamente a lo que ya implementamos en A5 del notebook). El valor `y` no se perturba, ya que es el objetivo de predicción y perturbarlo introduciría etiquetas incorrectas.

**Ventajas.** Genera ventanas sintéticas con variabilidad controlada. Muy bajo riesgo de implausiblidad fisiológica si la magnitud del ruido es pequeña (p. ej. ±2-3 mg/dL, que está dentro del error del propio sensor CGM). Computacionalmente barato. El hecho de no perturbar `y` es metodológicamente más limpio que SMOTE, que sí interpola el target.

**Inconvenientes.** Las perturbaciones son superficiales: no generan diversidad estructural (distintas formas de trayectoria glucémica), solo variaciones de amplitud. Si la magnitud es demasiado pequeña, el efecto es despreciable; si es demasiado grande, los valores pueden cruzar umbrales clínicos (p. ej. una ventana normoglucémica puede contener valores perturbados por debajo de 70).

**Riesgo de sesgo.** Bajo. El ruido añadido está dentro de la variabilidad natural del sensor CGM (Medtronic y Dexcom reportan MARD de 9-10%, equivalente a ±10-15 mg/dL en rango normal). Con escala del 20% de la std observada, el riesgo de implausiblidad es mínimo.

**Adecuación a ventanas temporales.** Muy alta. El jittering es especialmente adecuado para ventanas cortas porque respeta la estructura temporal (solo cambia la amplitud, no el patrón de cambio entre pasos consecutivos). Iwana & Uchida (2021) *PLOS ONE* (DOI: 10.1371/journal.pone.0254841) lo identifican como la técnica de augmentation más consistentemente efectiva en series temporales de longitud corta-media.

**Prioridad.** Alta — técnica avanzada con bajo riesgo.

---

### Técnica 6 — Resampling proporcional a la distribución de referencia (demographic-proportional resampling)

**Descripción.** En lugar de forzar una distribución uniforme entre grupos demográficos (mismo número de ventanas en todos los grupos), se define una distribución objetivo basada en la prevalencia real de cada grupo en la población de referencia — por ejemplo, la distribución de edad en la población diabética española o europea según registros epidemiológicos. Se aplica over o undersampling para que la distribución de ventanas en el train de cada fold se aproxime a esa distribución de referencia, no a la distribución uniforme.

**Ventajas.** Produce un conjunto train que refleja mejor la distribución real de la enfermedad, lo que mejora la validez externa del modelo. Evita el problema de la distribución uniforme forzada, que puede ser irreal (p. ej., forzar que el 20% de las ventanas provengan de pacientes >60 años si en la práctica ese grupo tiene menor prevalencia).

**Inconvenientes.** Requiere una distribución de referencia externa justificada bibliográficamente, lo que añade una decisión metodológica adicional que debe documentarse y defenderse. Si la distribución de referencia está mal elegida, puede introducir más sesgo que el original.

**Riesgo de sesgo.** Depende totalmente de la calidad de la distribución de referencia. Con una fuente epidemiológica sólida (p. ej. IDF Diabetes Atlas 2023), el riesgo es bajo. Sin ella, es alto.

**Adecuación a ventanas temporales.** Alta — misma que T1/T2.

**Evidencia bibliográfica.** Lahoti et al. (2020) *Proceedings of NeurIPS* proponen el marco de "fairness without demographics" que motiva este enfoque. Para diabetes específicamente: IDF Diabetes Atlas, 10th edition (2021) proporciona distribuciones de referencia por edad y sexo en poblaciones europeas.

**Prioridad.** Media — técnica con alto interés académico pero dependiente de una decisión metodológica externa que debe acordarse con los tutores.

---

## Tabla comparativa

| Técnica | Tipo | Genera sintéticos | Riesgo metodológico | Coste computacional | Prioridad |
|---|---|---|---|---|---|
| T1 — Undersampling aleatorio | Baseline | No | Bajo | Mínimo | **1 — Máxima** |
| T2 — Oversampling aleatorio | Baseline | No (duplicados) | Medio | Mínimo | **2 — Máxima** |
| T3 — Patient-aware undersampling | Avanzada | No | Bajo-Medio | Bajo | **3 — Alta** |
| T5 — Jittering de ventanas | Avanzada | Sí (perturbación) | Bajo | Bajo | **4 — Alta** |
| T4 — SMOTE sobre ventanas | Avanzada | Sí (interpolación) | Medio-Alto | Medio | **5 — Media** |
| T6 — Proporcional a referencia | Avanzada | No/Sí | Medio (depende de referencia) | Bajo | **6 — Media** |

---

## Selección final y orden de evaluación recomendado

Para un estudio comparativo con 3 datasets × 2 dimensiones × 5 folds, propongo estas **6 técnicas en este orden**:

**Prioridad 1 — T1: Undersampling aleatorio.** Baseline obligatorio. Establece el suelo de comparación. Si ninguna técnica supera esto, el resultado es igualmente publicable.

**Prioridad 2 — T2: Oversampling aleatorio.** Segundo baseline. Junto con T1, define el rango de referencia dentro del cual deben situarse las técnicas avanzadas.

**Prioridad 3 — T3: Patient-aware undersampling.** Primera técnica avanzada. Directamente motivada por los EDA (concentración top-5 en DiaTrend, desequilibrio de volumen por paciente) y por la estructura del Script 2 que ya gestiona el balanceo a nivel de paciente.

**Prioridad 4 — T5: Jittering de ventanas.** Técnica generativa de bajo riesgo. Complementa T2 añadiendo diversidad real donde T2 solo añade duplicados. Fácil de defender metodológicamente porque la magnitud del ruido está acotada por la precisión conocida del sensor CGM.

**Prioridad 5 — T4: SMOTE sobre ventanas.** Técnica generativa con mayor riesgo de implausiblidad pero con mayor interés bibliográfico. Incluirla permite comparar explícitamente si la interpolación (SMOTE) mejora o empeora respecto al jittering (T5), lo cual es una contribución metodológica clara del TFM.

**Prioridad 6 — T6: Proporcional a referencia.** La más conceptualmente interesante de cara a publicación (conecta con literatura de fairness en ML médico) pero requiere una decisión metodológica adicional (elección de distribución de referencia) que debe acordarse con los tutores antes de implementarla. Se evalúa última, una vez que las cinco anteriores están ejecutadas.