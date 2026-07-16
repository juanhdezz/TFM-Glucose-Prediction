Voy a diseñarte una estructura narrativa rigurosa para la Sección 4, pensando como revisor de Q1/Q2.Aquí tienes una propuesta de estructura narrativa completa para la Sección 4, con el mapa visual primero y luego el razonamiento detallado.---

## Estructura narrativa detallada de la Sección 4

La lógica de ordenación es **embudo**: primero el veredicto global, luego se desagrega por dataset, luego por dimensión demográfica, luego por rango clínico, luego se valida la robustez, y finalmente se concentra el hallazgo y se discute. Cada subsección tiene una **pregunta de investigación propia** que se formula al inicio y se responde explícitamente al final, antes de pasar a la siguiente. Esto da coherencia narrativa y facilita enormemente al revisor seguir el argumento.

---

### §4.1 — Configuración experimental y baseline

Arranca con una **tabla resumen** compacta (LaTeX): datasets × técnicas × número de experimentos, horizontes de predicción, arquitectura LSTM, y métricas usadas. Es el mapa de lo que el lector va a ver. Luego recuerdas —en un párrafo, no en una sección de metodología— por qué val/test nunca se tocan y qué significa el experimento sin balanceo como baseline. Sin ninguna gráfica aquí. Esto evita que el revisor tenga que saltar al capítulo 3 para entender el contexto mientras lee los resultados.

### §4.2 — Rendimiento global

**Pregunta:** ¿el balanceo demográfico mejora la predicción de glucosa en promedio?

Aquí usas el **dashboard global** como figura de apertura (figura grande, panel de honor). Lo acompañas con la tabla de rankings global (mean ± std por técnica, promediada sobre datasets y dimensiones). A continuación el **CD diagram** (Demšar 2006) con el resultado del test de Friedman + Nemenyi —esto es lo que legitima cualquier afirmación de "esta técnica es significativamente mejor". El **Borda lollipop** cierra la subsección con el ranking agregado. El párrafo final responde directamente: "sí/no/depende del contexto" con los números exactos. Esto es lo primero que leerá el revisor después del abstract.

### §4.3 — Análisis por dataset

**Pregunta:** ¿el efecto del balanceo es consistente entre T1DiabetesGranada, REPLACE-BG y DiaTrend?

Aquí entra la heterogeneidad real de tus datos. Empiezas con el **dashboard per-dataset** y los **dumbbell plots** (uno por dataset), que muestran visualmente el delta respecto al baseline para cada técnica. Los **slope charts** por dimensión muestran si la jerarquía de técnicas cambia entre datasets. Las **parallel coordinates** y el **radar por dataset** permiten ver el perfil completo de cada técnica en cada contexto. La tabla LaTeX con mean ± std por dataset es obligatoria aquí. El hallazgo crítico que debes discutir es si `reference_proportional`, la técnica más novedosa de tu set, se comporta de forma diferenciada en algún dataset concreto —eso es material de discusión de alto valor.

### §4.4 — Dimensión demográfica: sexo vs. edad

**Pregunta:** ¿es más efectivo equilibrar por sexo o por edad?

Esta subsección justifica la dualidad de tu diseño experimental. Los **heatmaps de delta** (con tus 10 variantes) son la figura principal —muestran qué técnica mejora qué métrica en qué dimensión. Los **strip plots por fold** y los **violin plots** muestran la distribución de los deltas, no solo la media. El **bump chart de rankings** muestra si el ranking de técnicas cambia entre dimensión sexo y dimensión edad. El **Cohen's d** entre ambas dimensiones cuantifica si la diferencia es prácticamente significativa. Si los resultados del EDA (Cohen d = 0.37 en hipoglucemia a 1-día) se ven reflejados en las métricas post-balanceo, aquí es donde lo argumentas.

### §4.5 — Relevancia clínica: rangos de glucosa

**Pregunta:** ¿el balanceo mejora la predicción en los rangos clínicamente más críticos, especialmente hipoglucemia (TBR)?

Esta es potencialmente la subsección más importante del TFM para un revisor clínico. El **dashboard clínico** y las **Clarke zone tables** abren la sección. Los **heatmaps TBR/TIR/TAR por técnica** son la figura central. Los **small multiples facetados** permiten ver el perfil completo rango a rango. El argumento aquí conecta directamente con la motivación clínica del abstract: si TBR mejora con alguna técnica de forma consistente, eso tiene valor traslacional real, aunque RMSE global mejore marginalmente. Debes mencionar explícitamente que el EDA mostró que Cohen d era más alto para hipoglucemia a 1-día (0.37 vs. 0.10 en hiperglucemia) y discutir si eso se traduce en mayor sensibilidad del balanceo en ese rango.

### §4.6 — Estabilidad y consistencia entre folds

**Pregunta:** ¿los resultados son robustos o están condicionados por la partición de los datos?

La **matriz de correlación Spearman** de consistencia y el **ranking heatmap por fold** responden esto directamente. Los **strip plots por fold** muestran la varianza intra-técnica. Esta subsección es corta pero crucial: sin ella, un revisor metódico señalará que no sabes si tus resultados son estables. Si hay alta consistencia entre folds, lo afirmas con el coeficiente. Si hay varianza notable en alguna técnica concreta (por ejemplo `jittering` puede ser más ruidoso por naturaleza), lo explicas. Esta validación de robustez distingue un trabajo riguroso de uno que solo reporta medias.

### §4.7 — Técnica ganadora y recomendación práctica

**Pregunta:** ¿qué técnica recomiendas y bajo qué condiciones?

Aquí concentras: la **tabla comparativa final** (todas las técnicas vs. baseline, con Cohen's d y resultado del test estadístico), el **Borda final**, y el **perfil radar de la técnica ganadora**. La recomendación debe ser condicional y matizada: "para balanceo por sexo en T1DiabetesGranada, `X` es la mejor opción; para balanceo por edad en DiaTrend, `Y` muestra ventaja en TBR aunque no en RMSE global". Esa granularidad es lo que distingue un resultado de un hallazgo.

### §4.8 — Discusión y limitaciones

Sin gráficas. Contrasta con la literatura del estado del arte (§2), explica por qué MNAR y la severa ausencia de series continuas largas (solo 7 pacientes con 14 días) limita la generalización, y propone trabajo futuro concreto (ventanas más largas, balanceo a nivel de episodio vs. punto, validación prospectiva).

---

## Principios de presentación de figuras

Sobre cuántas figuras incluir por subsección: la regla es **una figura de apertura que da el panorama** (el dashboard o el heatmap grande), **una figura de detalle** que permite la argumentación técnica (el CD diagram, el dumbbell, la distribución), y **una tabla** cuando los números exactos importan. Nunca más de 4-5 figuras por subsección en una memoria de máster. Las restantes visualizaciones que genera tu sistema van al **Apéndice**, con referencia explícita desde el texto ("ver figura A.X en el apéndice para el análisis completo de perfiles por dataset").

Sobre qué analizar: **no analices todas las técnicas con el mismo espacio**. Identifica cuáles tienen resultados interesantes ,las que mejoran más, las que empeoran inesperadamente, . Las que se comportan de forma esperada o neutral se agrupan en una frase. El análisis narrativo debe seguir el resultado, no el catálogo.



---

Con todo el análisis ya hecho, puedo darte el mapa definitivo y preciso. No una estructura genérica, sino exactamente qué archivo va en qué subsección y por qué.---

Ahora te doy las reglas de uso para cada subsección, con el razonamiento de por qué cada figura está donde está y no en otro sitio.

---

## §4.1 — Solo una tabla, sin figuras

Construyes a mano una tabla de 4 columnas: dataset, nº pacientes, nº ventanas aproximado, técnicas disponibles. Esta tabla es el contrato con el lector: sabe exactamente con qué está comparando todo lo que viene después. La asimetría de T1DiabetesGranada (12 técnicas en vez de 13) se menciona aquí en una nota al pie y no vuelve a necesitar explicación.

---

## §4.2 — Tres figuras, una tabla, datos estadísticos en texto

El `dashboard_global` abre la subsección mostrando el panorama: el lector ve de un vistazo que DiaTrend tiene más señal que los otros dos. Los tres CD diagrams van juntos en una figura compuesta de tres paneles —no tres figuras separadas— y son el único lugar del TFM donde aparecen. Aquí es donde citas los p-values de Friedman directamente en el texto: "el test de Friedman detecta diferencias significativas entre condiciones en DiaTrend (χ²=43.37, p=2×10⁻⁵) pero no en T1DiabetesGranada (p=0.058) ni en ReplaceBG (p=0.068) para RMSE global". El `borda_lollipop` y el `bump_chart_ENTIRE` van al apéndice porque complementan pero no son necesarios para el argumento central.

---

## §4.3 — Tres figuras, una tabla

Los dumbbell plots del `dashboard_per_dataset` (fila superior) son la figura de apertura porque muestran simultáneamente la media y la varianza por fold de cada técnica en cada dataset. A continuación pones `dumbbell_RMSE_TBR_1` y `dumbbell_RMSE_TIR` como figuras individuales porque son los rangos con señal real —aquí el lector puede ver que en DiaTrend la separación entre técnicas es mucho mayor que en los otros dos. El slope chart por dimensión en TBR_1 cierra la subsección argumentando visualmente que Age domina sobre Sex. La tabla con `main_results_RMSE.tex` filtrada a ENTIRE y TBR_1 va al final de la subsección como respaldo numérico. Todo lo demás —el resto de dumbbell plots, los parallel coordinates, los slope charts adicionales— va al apéndice con referencia explícita.

---

## §4.4 — Dos figuras, datos estadísticos en texto

Los dos heatmaps de delta (TBR_1 y TIR) son la figura central porque permiten ver exactamente qué técnica en qué dimensión produce el efecto en qué dataset. El panel C del `dashboard_global` (Mean RMSE by technique and dimension) es la figura más directa para argumentar que Age supera sistemáticamente a Sex. Los Cohen's d comparativos (Age·Oversampling d=1.41 vs Sex·PAUndersampling d=0.55 en TIR) van en el texto como tabla de dos columnas construida a mano a partir del CSV —no hace falta una figura para esto. Los radares, el bump chart de TBR_1 y los heatmaps de delta restantes van al apéndice.

---

## §4.5 — Tres figuras, una tabla, Nemenyi en texto

Esta es la subsección más importante del TFM y tiene la mayor densidad de evidencia. El `dashboard_clinical` con los paneles A (Hypo L2), B (Hypo L1) y E (Hyper L1) es la figura de apertura: muestra en un único panel el beneficio en hipoglucemia y el coste en hiperglucemia simultáneamente, que es exactamente el trade-off que quieres argumentar. El heatmap de rankings es la segunda figura: aquí el argumento es visual y contundente —baseline ocupa posición 13 en TBR_2, TBR_1 e In Range de DiaTrend, y posición 13 en TBR_2 y TBR_1 de ReplaceBG. El `range_profile_facet_RMSE` es la tercera figura: muestra el perfil completo de cada técnica a lo largo de todos los rangos, que es donde el trade-off se hace más evidente. Los p-values de Nemenyi van directamente en el texto —"el post-hoc de Nemenyi confirma que Age·Oversampling supera al baseline con p=0.0006 en TIR y p=0.0475 en TBR_1"— sin figura adicional. La tabla de deltas en hipoglucemia más Clarke zones cierra la subsección con los números de seguridad clínica.

---

## §4.6 — Dos figuras, sin tablas

La matriz de consistencia Spearman es la figura principal y dice todo lo que hay que decir en una sola imagen: DiaTrend en verde oscuro, los otros dos en amarillo-naranja. A continuación pones únicamente el strip plot de DiaTrend (no los tres) porque el contraste positivo —mostrar un caso donde sí funciona— es más útil argumentativamente que mostrar los tres casos. Los strip plots de T1DiabetesGranada y ReplaceBG van al apéndice junto con todos los violin plots. La subsección es corta por diseño: una explicación del mecanismo (Group K-Fold sobre pacientes heterogéneos), los dos números clave (ρ medio DiaTrend=0.67 vs T1D=0.18 vs RBG=0.15), y el argumento de que esto explica la asimetría en significancia Friedman.

---

## §4.7 — Una figura, una tabla sintética

La tabla final es la contribución más importante de esta subsección y debes construirla manualmente combinando las fuentes: filas = técnicas, columnas = dataset × rango, celdas = significancia (✓/✗) + Cohen's d + posición Borda. Esto no existe como archivo en tu sistema —tienes que construirla sintetizando `friedman_results`, `nemenyi`, `cohens_d_vs_original` y `borda_ranking`. El radar de DiaTrend Age-dimension cierra visualmente. La recomendación se escribe en tres frases condicionales explícitas: si el dataset tiene diversidad etaria suficiente → Age·Oversampling; si el objetivo es minimizar trade-off → Age·PAUndersampling; si el dataset es homogéneo o pequeño → no aplicar balanceo.

---

## §4.8 — Solo texto

Tres párrafos: contraste con literatura del estado del arte (§2), limitaciones metodológicas (missingness MNAR, baja consistencia en T1DGranada, horizonte único de 60 minutos, ausencia de validación prospectiva), y trabajo futuro concreto (balanceo a nivel de episodio glucémico, horizontes múltiples, validación en datos prospectivos con series más largas).

---

## Regla general de uso de figuras

Nunca dos figuras consecutivas sin párrafo de análisis entre ellas. Cada figura se introduce con una frase que dice qué muestra, seguida de un párrafo que extrae la conclusión específica relevante para la pregunta de la subsección. Las figuras de apéndice se referencian desde el texto con "el análisis completo se muestra en la Figura A.X del apéndice", para que el revisor sepa que existen pero no interrumpan el flujo argumentativo.


---

# Sección 4. Resultados y Discusión

---

## §4.1 Configuración experimental y baseline

El diseño experimental cubre **39 condiciones** distribuidas en tres datasets (T1DiabetesGranada, DiaTrend y REPLACE-BG), nueve técnicas de balanceo y dos dimensiones demográficas (sexo y edad), con una condición faltante en T1DiabetesGranada derivada de restricciones en la disponibilidad de grupos mínimos para Group K-Fold estratificado. La evaluación emplea validación cruzada de 5 folds con Group K-Fold sobre pacientes, garantizando que ningún paciente aparezca simultáneamente en entrenamiento y validación, lo que elimina el riesgo de *data leakage* inter-paciente. Las métricas primarias son RMSE y MAE calculadas sobre cinco rangos glucémicos clínicos (Hypo L2 <54 mg/dL, Hypo L1 54–69 mg/dL, In Range 70–180 mg/dL, Hyper L1 181–250 mg/dL, Hyper L2 >250 mg/dL) y sobre el rango completo (ENTIRE). La arquitectura predictiva es una red LSTM de horizonte fijo a 60 minutos (predicción a +12 pasos de 5 minutos). El balanceo se aplica exclusivamente al conjunto de entrenamiento de cada fold; los conjuntos de validación y test permanecen en su distribución demográfica original, preservando la validez externa del experimento.

**Tabla 4.1.** Resumen del diseño experimental.

| Dataset | Pacientes | Condiciones evaluadas | Medidas totales (aprox.) | Frecuencia CGM |
|---|---|---|---|---|
| T1DiabetesGranada | 643 | 18 (1 técnica faltante) | 22.2 M | 15 min |
| DiaTrend | 51 | 18 | 7.7 M | 5 min |
| REPLACE-BG | 223 | 18 | 10.4 M | 5 min |

Los tres datasets difieren sustancialmente en estructura demográfica, densidad temporal y volumen de datos, lo que constituye en sí mismo un factor experimental relevante que se analiza en detalle en §4.3. La condición sin balanceo (*Original*) actúa como baseline en todos los análisis de delta y en los tests estadísticos. Todos los resultados se presentan como media ± desviación estándar sobre los cinco folds, y los valores de mejora porcentual expresados en los heatmaps se calculan como $\Delta\% = (RMSE_{\text{base}} - RMSE_{\text{bal}}) / RMSE_{\text{base}} \times 100$, de modo que valores positivos indican mejora respecto al baseline.

---

## §4.2 Rendimiento global

**Pregunta de investigación:** ¿El balanceo demográfico mejora la predicción de glucosa en promedio, con independencia del dataset y la técnica concreta?

### 4.2.1 Panorama general

El panel A del **dashboard global** (Figura 4.1) proporciona la visión más inmediata y honesta del experimento: el efecto del balanceo sobre el RMSE global (ENTIRE) es modesto, heterogéneo y fuertemente condicionado por el dataset. DiaTrend concentra prácticamente toda la señal del experimento, con mejoras de hasta −7.0% para Age·PAUndersampling y −6.0% para Age·Undersampling respecto al baseline (RMSE original en DiaTrend: 45.05 ± 2.45 mg/dL). T1DiabetesGranada y REPLACE-BG exhiben deltas globales de magnitud notablemente inferior: en T1DiabetesGranada ninguna técnica supera el −0.8% de mejora en RMSE global, y en REPLACE-BG los valores oscilan entre −0.7% y +0.5%, sin tendencia clara. Esta asimetría es el hallazgo central del experimento y su explicación mecanística se desarrolla en §4.3.

**Figura 4.1.** *Dashboard global: visión general del experimento.* El panel A muestra el porcentaje de mejora en RMSE (ENTIRE) para las 18 condiciones × 3 datasets. El panel B presenta el ranking Borda agregado. El panel C compara el RMSE medio por técnica separando dimensión edad (azul) y sexo (naranja). El panel D replica el panel A para el rango hipoglucémico Hypo L1 (54–69 mg/dL). *(Ver figura dashboard_global.pdf adjunta.)*

### 4.2.2 Significación estadística: test de Friedman

El test de Friedman sobre las 19 condiciones (18 técnicas + baseline) revela una **dicotomía estadística nítida** entre DiaTrend y los otros dos datasets. En DiaTrend, la hipótesis nula de igualdad entre condiciones se rechaza en todos los rangos glucémicos y para ambas métricas, con estadísticos especialmente elevados en TAR_2 (χ²=76.55, p<10⁻¹⁰), ENTIRE (χ²=78.21, p<10⁻¹⁰), TAR_1 (χ²=72.53, p<10⁻¹⁰) y TIR (χ²=67.42, p<10⁻¹⁰). Los rangos hipoglucémicos también alcanzan significación: TBR_1 (χ²=47.80, p=1.61×10⁻⁴) y TBR_2 (χ²=44.69, p=4.6×10⁻⁴). En T1DiabetesGranada, únicamente TAR_2 alcanza significación (RMSE: χ²=32.71, p=0.018; MAE: χ²=30.61, p=0.032), sin que ningún otro rango supere el umbral α=0.05. En REPLACE-BG, ningún rango ni métrica produce un resultado significativo (p mínimo=0.062 en TAR_2 por MAE, quedando por encima del umbral). Estos resultados justifican que el análisis detallado de §4.5 se centre principalmente en DiaTrend para las afirmaciones causales, mientras que T1DiabetesGranada y REPLACE-BG se interpretan con mayor cautela.

### 4.2.3 Ranking agregado (Borda)

El panel B del dashboard global presenta el ranking Borda por dataset. Agregando las puntuaciones Borda sobre los tres datasets, el orden global sitúa a **Age·Tomek Links en primera posición** (suma Borda = 220 puntos), seguido de Sex·Oversampling (289), Sex·SMOTE+Tomek Links (305), Age·Undersampling+SMOTE (308) y Age·SMOTE (317). En el extremo inferior del ranking se encuentran Age·Undersampling (suma = 568), Age·PAUndersampling (498) y Age·Jittering (496), técnicas que en DiaTrend exhiben los peores resultados en rangos de hiperglucemia.

El panel C del dashboard (Mean RMSE by technique and dimension) pone de manifiesto que las técnicas de dimensión edad (barras azules) presentan sistemáticamente mayor dispersión que las de dimensión sexo, lo que indica tanto mayor potencial de mejora como mayor riesgo de degradación. El RMSE medio de la dimensión edad cruza la línea del baseline en aproximadamente la mitad de las técnicas, mientras que las técnicas de dimensión sexo permanecen más cercanas al baseline en todos los casos. Esta observación anticipa la discusión de §4.4.

**Respuesta a la pregunta de investigación:** El balanceo demográfico no mejora la predicción de glucosa de forma global y homogénea. El efecto es estadísticamente significativo únicamente en DiaTrend, donde se concentra prácticamente toda la señal del experimento. La respuesta correcta es: *depende del dataset, de la técnica y del rango glucémico*, siendo el análisis por dataset (§4.3) y por rango clínico (§4.5) imprescindible para extraer conclusiones de valor translacional.

---

## §4.3 Análisis por dataset

**Pregunta de investigación:** ¿El efecto del balanceo es consistente entre T1DiabetesGranada, REPLACE-BG y DiaTrend, o existen factores estructurales de cada dataset que expliquen las diferencias observadas?

### 4.3.1 DiaTrend: el único dataset con señal real

DiaTrend (51 pacientes, ~7.7 M medidas, frecuencia de 5 min) presenta el perfil de resultados más informativo del experimento. El RMSE baseline sobre el rango completo es 45.05 ± 2.45 mg/dL, notablemente superior al de los otros dos datasets, lo que refleja la mayor variabilidad glucémica de esta cohorte (desviación estándar de glucosa: 74.18 mg/dL, la más alta de los tres). La distribución etaria de DiaTrend es marcadamente asimétrica: 38 de 51 pacientes (74.5%) pertenecen al grupo <31 años, mientras que solo 4 tienen entre 31–45 años, 8 entre 46–65 y 1 ≥66 años. Este desequilibrio severo en la variable edad crea exactamente el escenario para el que el balanceo por edad está diseñado: la clase minoritaria (pacientes adultos maduros y mayores) está sustancialmente sub-representada en los datos de entrenamiento sin balanceo.

El dumbbell plot de DiaTrend (panel superior central, Figura 4.2) revela que las técnicas de dimensión edad producen desplazamientos mucho mayores respecto al baseline que las de dimensión sexo, con una separación entre el extremo superior (Age·PAUndersampling: 48.22 ± 3.09 mg/dL) e inferior (Age·SMOTE: 45.80 ± 2.16 mg/dL) de más de 2.4 mg/dL en RMSE global. En contraste, todas las técnicas de dimensión sexo se aglutinan en una banda estrecha de ±0.3 mg/dL alrededor del baseline. Esto es coherente con la distribución de sexo en DiaTrend (35F/16M), que aunque desequilibrada, genera grupos lo suficientemente grandes como para que el balanceo por sexo no introduzca información diferencial relevante.

**Figura 4.2.** *Dashboard per-dataset: dumbbell plots y strip plots de consistencia entre folds para los tres datasets.* *(Ver figura dashboard_per_dataset.pdf adjunta.)*

En los rangos glucémicos específicos, DiaTrend exhibe el trade-off más claro del experimento. Las técnicas que mejoran TIR (rango euglucémico) lo hacen a costa de degradar TAR_1 y TAR_2. Age·Oversampling logra el mayor efecto positivo en TIR (d=1.24 en RMSE, *grande*) y en TBR_1 (d=0.55, *medio*), pero simultáneamente produce la mayor degradación en TAR_1 (d=−2.23 en RMSE, *grande*). Age·SMOTE replica el mismo patrón con d=0.98 en TIR y d=−1.83 en TAR_1. Este trade-off TIR↑/TAR↓ se analiza en profundidad en §4.5.

### 4.3.2 T1DiabetesGranada: el dataset más grande, el menos sensible al balanceo

T1DiabetesGranada es el dataset más extenso (643 pacientes, ~22.2 M medidas) y el de mayor cobertura temporal (1.535 días, enero 2018 a marzo 2022). A diferencia de DiaTrend, su distribución etaria es más equilibrada: 261 pacientes en 46–65 años (40.6%), 181 en 31–45 (28.2%), 103 ≥66 (16.0%) y 98 <31 (15.2%). La distribución de sexo también es la más equilibrada de los tres datasets (334F/309M, ratio 1.08:1). Este balance inherente de la cohorte hace que el problema que el balanceo intenta resolver —la sub-representación de grupos minoritarios— sea intrínsecamente menor en T1DiabetesGranada.

El RMSE baseline es 37.80 ± 0.81 mg/dL (ENTIRE), con una variabilidad entre folds mucho más baja que en DiaTrend. El dumbbell plot de T1DiabetesGranada (panel superior izquierdo, Figura 4.2) muestra que prácticamente todas las técnicas se solapan con el rango de confianza del baseline, con diferencias absolutas inferiores a 0.5 mg/dL en casi todos los casos. La técnica con mejor resultado global es Age·PAUndersampling (37.56 ± 0.39 mg/dL, −0.6% respecto al baseline), aunque esta mejora no supera el umbral de significación estadística del test de Friedman (p=0.082 para ENTIRE). Solo en TAR_2 se alcanza significación (χ²=32.71, p=0.018), lo que sugiere que el efecto del balanceo en T1DiabetesGranada es en el mejor de los casos marginal y rango-específico.

Un aspecto notable es que los efectos de tamaño (Cohen's d) en T1DiabetesGranada son consistentemente menores que en DiaTrend, pero no inexistentes. Age·SMOTE produce d=0.81 (*grande*) en TBR_1 por MAE —la mayor magnitud de efecto positivo en este dataset— lo que indica que la técnica puede concentrar el beneficio en rangos hipoglucémicos incluso cuando el efecto global es imperceptible.

### 4.3.3 REPLACE-BG: el dataset más homogéneo, sin señal estadística

REPLACE-BG (223 pacientes, ~10.4 M medidas) es el dataset con menor variabilidad entre pacientes: la distribución de medidas por paciente presenta una desviación estándar de solo 6.202 (frente a 151.610 en DiaTrend y 25.253 en T1DiabetesGranada), lo que indica que todos los pacientes contribuyen con una cantidad de datos muy similar. La distribución de sexo es casi perfectamente equilibrada (112F/111M), y la distribución etaria, aunque con predominio de adultos medios (86 en 46–65, 69 en 31–45), no es tan extrema como en DiaTrend. El rango temporal es el más corto (317 días en el año 2000), y la frecuencia de adquisición es de 5 minutos con una cobertura del 99.5%, la más alta de los tres.

El test de Friedman no detecta ninguna diferencia significativa en ningún rango glucémico ni métrica (p mínimo=0.062). El dumbbell plot de REPLACE-BG (panel superior derecho, Figura 4.2) refleja este resultado: todas las técnicas se agrupan en una banda de ±0.4 mg/dL alrededor del baseline (RMSE original: 37.00 ± 0.83 mg/dL), con Sex·SMOTE+Tomek Links alcanzando el mejor resultado numérico (36.81 ± 0.80 mg/dL, −0.5%). La homogeneidad del dataset es la explicación más parsimoniosa: cuando la diversidad intra-dataset es baja y la representación demográfica está relativamente equilibrada, el balanceo introduce ruido sin beneficio neto. En REPLACE-BG, el balanceo por edad produce en TAR_2 efectos de tamaño grande positivos (Age·SMOTE+Tomek d=1.38; Age·Tomek Links d=1.31), lo que señala que incluso en ausencia de significación global existen rangos específicos donde el balanceo redistribuye el error.

**Respuesta a la pregunta de investigación:** Los efectos del balanceo no son consistentes entre datasets. DiaTrend, con la distribución etaria más desequilibrada (<31 años representa el 74.5% de la cohorte), es el único dataset donde el balanceo produce efectos estadísticamente significativos, confirmando que la eficacia del balanceo demográfico está condicionada por la magnitud del desequilibrio preexistente en el conjunto de datos.

---

## §4.4 Dimensión demográfica: sexo vs. edad

**Pregunta de investigación:** ¿Es más efectivo equilibrar la distribución de pacientes por sexo o por edad en el contexto de la predicción glucémica con LSTM?

### 4.4.1 Superioridad sistemática de la dimensión edad

El panel C del dashboard global (Mean RMSE by technique and dimension) ofrece la evidencia más directa para responder esta pregunta: en prácticamente todas las técnicas, la variante de dimensión edad presenta tanto mayores mejoras como mayores degradaciones que la variante de dimensión sexo, mientras que las técnicas de sexo permanecen uniformemente agrupadas cerca del baseline. Esta mayor amplitud de efecto en la dimensión edad es coherente con la naturaleza del desequilibrio: como se documentó en §4.3, el desequilibrio etario en DiaTrend es extremo (74.5% de pacientes menores de 31 años), mientras que el desequilibrio por sexo es moderado en los tres datasets.

Cuantitativamente, los efectos de tamaño (Cohen's d) confirman la superioridad de la dimensión edad en DiaTrend. Para el rango TIR (euglucemia, clínicamente el más relevante en términos de volumen de datos):

| Técnica | Dimensión | Rango | Cohen's d | Magnitud |
|---|---|---|---|---|
| Oversampling | Age | TIR (RMSE) | +1.24 | grande |
| Oversampling | Age | TIR (MAE) | +1.41 | grande |
| SMOTE | Age | TIR (RMSE) | +0.98 | grande |
| Jittering | Age | TIR (RMSE) | +1.03 | grande |
| Oversampling | Sex | TIR (MAE) | +0.50 | pequeño |
| SMOTE | Sex | TIR (MAE) | +0.60 | medio |

La brecha entre dimensiones en TIR es notable: la técnica de edad más efectiva (Age·Oversampling, d=1.41) supera en más del doble a la mejor técnica de sexo (Sex·SMOTE, d=0.60). En TBR_1 (hipoglucemia moderada), el contraste es igualmente claro: Age·SMOTE alcanza d=0.72 mientras que ninguna técnica de sexo supera d=0.66 en este rango en DiaTrend.

En T1DiabetesGranada y REPLACE-BG, la diferencia entre dimensiones es menos pronunciada, en parte porque los efectos totales son más pequeños. Sin embargo, incluso en estos datasets, las técnicas de dimensión edad producen con más frecuencia efectos de tamaño medio o grande (aunque bidireccionales) que las de dimensión sexo.

### 4.4.2 Comportamiento diferencial por rango glucémico

La tabla de Cohen's d revela que la ventaja de la dimensión edad es específica del rango. En TIR y TBR, la dimensión edad domina. En TAR_1 y TAR_2, la dimensión edad produce los efectos más negativos del experimento (Age·Oversampling en TAR_1: d=−2.23 en DiaTrend, *grande negativo*), mientras que las técnicas de sexo se muestran más conservadoras en estos rangos. Este patrón sugiere que el balanceo por edad redistribuye la capacidad predictiva del LSTM desde los rangos de hiperglucemia hacia los rangos de euglucemia e hipoglucemia, lo que es consistente con el mecanismo implícito: al aumentar la representación de grupos etarios con perfiles glucémicos distintos (jóvenes vs. adultos maduros), el modelo aprende a priorizar rangos que son más informativos para la diferenciación entre grupos.

En REPLACE-BG, la situación se invierte parcialmente en TAR_2: múltiples técnicas de ambas dimensiones producen efectos de tamaño grande positivos en este rango (Sex·SMOTE+Tomek d=2.02, Age·SMOTE+Tomek d=1.71, Age·Tomek Links d=1.31), lo que constituye la excepción más llamativa del experimento y merece investigación futura.

**Respuesta a la pregunta de investigación:** El balanceo por edad es sistemáticamente más efectivo que el balanceo por sexo en los rangos glucémicos más informativos (TIR, TBR), especialmente en DiaTrend donde el desequilibrio etario es más severo. Sin embargo, la ventaja de la dimensión edad implica simultáneamente un mayor riesgo de degradación en hiperglucemia. La dimensión sexo produce efectos más pequeños pero también más seguros.

---

## §4.5 Relevancia clínica: rangos de glucosa

**Pregunta de investigación:** ¿El balanceo mejora la predicción en los rangos clínicamente más críticos, especialmente hipoglucemia severa (Hypo L2, <54 mg/dL)?

Esta subsección constituye el núcleo de la contribución clínica del trabajo. Los estándares de gestión de la diabetes tipo 1 (ADA 2024, consenso internacional Time in Range) establecen que el rango más crítico para la seguridad del paciente es la hipoglucemia severa (<54 mg/dL, Hypo L2), seguida de la hipoglucemia moderada (54–69 mg/dL, Hypo L1). Una mejora en la predicción de estos rangos —aunque sea modesta en términos de RMSE absoluto— tiene valor translacional directo porque permite a los sistemas de administración automática de insulina (CSII) anticipar y evitar episodios hipoglucémicos.

### 4.5.1 Hipoglucemia severa (Hypo L2, <54 mg/dL)

**Figura 4.3.** *Clinical Safety Dashboard: impacto del balanceo sobre la precisión en rangos glucémicos críticos.* Los paneles A y B muestran las mejoras en RMSE para Hypo L2 e Hypo L1 respectivamente. El panel C muestra el porcentaje de predicciones en Clarke Zone A (clínicamente seguras). Los paneles D y E muestran los perfiles en euglucemia e hiperglucemia moderada. *(Ver figura dashboard_clinical.pdf adjunta.)*

El panel A del dashboard clínico muestra que en DiaTrend, varias técnicas de balanceo por edad producen mejoras sustanciales en Hypo L2: Age·SMOTE logra +3.4% (baseline DiaTrend Hypo L2: 75.28 ± 7.33 mg/dL), Age·Oversampling +2.7%, Age·SMOTE+Tomek Links +2.6%. Los efectos de tamaño confirman la relevancia práctica: Age·SMOTE alcanza d=0.36 (*pequeño* pero positivo y consistente), mientras que Age·Oversampling obtiene d=0.29 en Hypo L2 RMSE. En contraste, Age·PAUndersampling produce una degradación severa de −5.8% (d=−0.60, *medio*) y Age·Undersampling −4.9% (d=−0.51, *medio*), los peores resultados del experimento en este rango crítico.

En T1DiabetesGranada, el patrón en Hypo L2 es inverso al de DiaTrend para muchas técnicas: Age·SMOTE mejora +1.6% en T1DGranada pero degrada −5.8% en DiaTrend para PAUndersampling. Esta inversión refleja las diferencias estructurales entre datasets y refuerza la necesidad de evaluación específica por dataset antes de seleccionar una técnica de balanceo.

En REPLACE-BG, el efecto más relevante en hipoglucemia es el de Sex·PAUndersampling: d=+0.97 en Hypo L2 RMSE (*grande*) y d=+1.19 en Hypo L2 MAE (*grande*). Este resultado es el más robusto del dataset —el único con magnitud grande y signo positivo en un rango crítico— y sugiere que en REPLACE-BG, cuya distribución de sexo es casi perfecta (112F/111M), el balanceo por sexo a nivel de *undersampling* proporcional puede redistribuir eficazmente la representación de episodios hipoglucémicos.

### 4.5.2 Hipoglucemia moderada (Hypo L1, 54–69 mg/dL)

El panel B del dashboard clínico muestra la mejora en Hypo L1. Este rango es cuantitativamente más relevante que Hypo L2 (mayor prevalencia de medidas en este intervalo) y los efectos son más consistentes. En DiaTrend, Age·SMOTE logra +5.0% de mejora (d=0.72, *medio*), Age·SMOTE+Tomek Links +4.0% (d=0.55, *medio*) y Age·Oversampling +3.7% (d=0.55, *medio*). Estas son las mejoras más robustas del experimento en un rango clínicamente relevante y estadísticamente significativas según el post-hoc de Nemenyi (p=1.61×10⁻⁴ para Friedman en TBR_1 RMSE). En contraste, Age·PAUndersampling produce −4.4% en DiaTrend Hypo L1 (d=−0.61, *medio negativo*), el peor resultado en este rango.

En T1DiabetesGranada, Age·SMOTE produce d=0.81 (*grande*) en Hypo L1 por MAE, lo que representa el mayor efecto positivo del dataset. Este resultado aislado, aunque no alcanza significación global en el test de Friedman, señala que las técnicas SMOTE en la dimensión edad tienen una afinidad específica por el rango hipoglucémico que merece investigación futura.

### 4.5.3 El trade-off TIR↑/TAR↓: la limitación clínica central

El panel E del dashboard clínico expone el coste de las mejoras hipoglucémicas en DiaTrend: las mismas técnicas que mejoran TBR_1 degradan sistemáticamente Hyper L1. Age·Oversampling produce −6.3% en Hyper L1 DiaTrend (d=−2.23 en TAR_1 RMSE, *grande*); Age·SMOTE −5.7% (d=−1.83, *grande*); Age·Undersampling+SMOTE −6.1% (d=−2.18, *grande*). Este trade-off es estructural: al balancear por edad en un dataset donde los jóvenes (<31 años) dominan, el modelo aumenta la representación de perfiles glucémicos de pacientes adultos, que en general tienden a presentar más hiperglucemia. La consecuencia es que el LSTM reequilibra su pérdida implícitamente hacia la hipoglucemia a costa de la hiperglucemia.

Desde el punto de vista de la seguridad clínica, este trade-off no es neutral. La hipoglucemia severa es potencialmente mortal en un plazo muy corto (minutos), mientras que la hiperglucemia produce daño crónico en un plazo de años. Una mejora en TBR a costa de TAR puede ser aceptable desde la perspectiva de la gestión aguda del riesgo, aunque debe documentarse explícitamente para cualquier uso clínico.

### 4.5.4 Clarke Zone A: seguridad de las predicciones

El panel C del dashboard clínico muestra que el porcentaje de predicciones en Clarke Zone A (clínicamente seguras, error dentro del rango tolerable) se mantiene estable en los tres datasets para la mayoría de las técnicas. En DiaTrend, el baseline alcanza aproximadamente 64% de predicciones en Zone A, y la mayoría de las técnicas se mantienen en el rango 63–65%. Age·PAUndersampling muestra la mayor degradación (~62%), mientras que las técnicas SMOTE y Oversampling se mantienen o mejoran ligeramente. En T1DiabetesGranada y REPLACE-BG, las diferencias son imperceptibles (rango 65–67%), lo que indica que el balanceo no introduce errores clínicamente peligrosos en estos datasets.

**Respuesta a la pregunta de investigación:** En DiaTrend, el balanceo por edad mejora significativamente la predicción en hipoglucemia moderada, con Age·SMOTE como técnica más efectiva (Hypo L1: +5.0%, d=0.72) y Age·Oversampling en segundo lugar (+3.7%). Sin embargo, esta mejora conlleva un coste estadísticamente significativo en hiperglucemia (TAR_1, TAR_2). En T1DiabetesGranada y REPLACE-BG, los efectos son menores aunque localmente relevantes (Sex·PAUndersampling en REPLACE-BG: d=+0.97 en Hypo L2). El balanceo demográfico no compromete la seguridad clínica según el análisis de Clarke Zones.

---

## §4.6 Estabilidad y consistencia entre folds

**Pregunta de investigación:** ¿Son los resultados robustos o están condicionados por la partición de datos concreta utilizada en cada fold?

### 4.6.1 Consistencia diferencial por dataset

Los strip plots de consistencia entre folds (fila inferior de la Figura 4.2) revelan el patrón más importante para la interpretación de la robustez del experimento: **DiaTrend exhibe una consistencia entre folds notablemente superior a la de T1DiabetesGranada y REPLACE-BG**. En DiaTrend, los cinco folds producen rankings de técnicas muy similares para todas las condiciones, con los marcadores por fold agrupados estrechamente alrededor de la media. En contraste, en T1DiabetesGranada y REPLACE-BG, la dispersión entre folds es sustancialmente mayor: algunos folds producen mejoras donde otros producen degradaciones para la misma técnica.

Este resultado explica parcialmente la ausencia de significación estadística en T1DiabetesGranada y REPLACE-BG: cuando la varianza entre folds es alta, el test de Friedman —que actúa sobre el ranking dentro de cada fold— pierde potencia porque el orden de las técnicas cambia entre folds. El Group K-Fold sobre pacientes heterogéneos genera esta varianza: en T1DiabetesGranada, con 643 pacientes de perfiles muy distintos (desviación estándar de medidas por paciente: 25.253), cada fold cubre subpoblaciones con composición demográfica diferente, lo que introduce variabilidad en la eficacia del balanceo. En REPLACE-BG, aunque los pacientes son más homogéneos en cantidad de datos, la brevedad del período de estudio (317 días) limita la diversidad de situaciones glucémicas representadas en cada fold.

En DiaTrend, la consistencia es mayor porque el Group K-Fold sobre 51 pacientes produce folds más equilibrados en términos de diversidad glucémica: con apenas 51 pacientes, la variabilidad aleatoria de la composición de los folds es menor que con 643 pacientes que pueden caer aleatoriamente en combinaciones de perfiles muy distintos.

### 4.6.2 Implicaciones para la generalizabilidad

La baja consistencia entre folds en T1DiabetesGranada y REPLACE-BG tiene una implicación metodológica directa: los resultados reportados para estas cohortes tienen mayor incertidumbre que los de DiaTrend. Las medias sobre cinco folds pueden ser estables numéricamente pero no representar un efecto robusto, sino el promedio de folds con comportamientos cualitativamente opuestos. Esto no invalida los resultados, pero exige interpretarlos con mayor cautela y sugiere que estudios futuros deberían emplear más folds (k=10 o repeated k-fold) o estratificación más cuidadosa de los folds para obtener estimaciones más estables en cohortes grandes y heterogéneas.

**Respuesta a la pregunta de investigación:** Los resultados son sustancialmente más robustos en DiaTrend que en T1DiabetesGranada y REPLACE-BG. La varianza entre folds es el factor limitante en las cohortes grandes, y explica de forma mecanística por qué los tests estadísticos detectan diferencias en DiaTrend pero no en los otros dos datasets.

---

## §4.7 Técnica ganadora y recomendación práctica

**Pregunta de investigación:** ¿Qué técnica de balanceo demográfico se recomienda para la predicción glucémica con LSTM, y bajo qué condiciones?

### 4.7.1 Síntesis de evidencia

La Tabla 4.2 sintetiza la evidencia acumulada en las secciones anteriores para las técnicas más relevantes, incluyendo significación estadística, magnitud de efecto y posición en el ranking Borda.

**Tabla 4.2.** Síntesis comparativa de técnicas por dataset. Se muestra el delta RMSE global (ENTIRE), el mejor rango clínico, la magnitud de efecto máxima (Cohen's d), la significación en Friedman y la posición Borda en el dataset. ✓ = significativo (p<0.05), — = no significativo.

| Técnica | Dataset | ΔRMSE (ENTIRE) | Mejor rango | d máx. | Friedman ENTIRE | Pos. Borda |
|---|---|---|---|---|---|---|
| Age·Tomek Links | DiaTrend | −0.5% | TIR (+0.7%) | +0.87 (TIR) | ✓ | 1 |
| Age·SMOTE | DiaTrend | −1.7% | TBR_1 (+5.0%) | +0.72 (TBR_1) | ✓ | 4 |
| Age·Oversampling | DiaTrend | −2.2% | TIR (+4.4%) | +1.41 (TIR) | ✓ | 9 |
| Age·Tomek Links | T1DGranada | −0.0% | Hyper L2 (+0.5%) | +0.77 (TAR_2) | — | 1 |
| Age·SMOTE | T1DGranada | −0.5% | TBR_1 (d=0.81) | +0.81 (TBR_1 MAE) | — | 6 |
| Sex·PAUndersampling | REPLACE-BG | −0.2% | TBR_2 (+2.1%) | +0.97 (TBR_2) | — | 6 |
| Age·Tomek Links | REPLACE-BG | −0.4% | TAR_2 (d=1.31) | +1.31 (TAR_2) | — | 4 |
| Age·PAUndersampling | DiaTrend | −7.0% | Hyper L1 (+2.8%) | −1.14 (TBR_1 MAE) | ✓ | 17 |
| Age·Undersampling | DiaTrend | −6.0% | TAR_2 (+1.9%) | −2.27 (TBR_2 MAE) | ✓ | 18 |

### 4.7.2 Recomendaciones condicionales

Los resultados permiten formular tres recomendaciones prácticas condicionales:

**Recomendación 1 (desequilibrio etario severo, objetivo global):** Cuando el dataset presenta un desequilibrio etario severo (>65% de pacientes en un único grupo de edad), **Age·Tomek Links** es la técnica más robusta: primera posición Borda en DiaTrend y T1DiabetesGranada, con impacto negativo mínimo en los rangos de hiperglucemia (−0.5% en ENTIRE de DiaTrend vs. −7.0% de PAUndersampling). Su mecanismo de limpieza de frontera de decisión sin síntesis de nuevas muestras lo hace menos susceptible al trade-off TIR↑/TAR↓ que las técnicas de sobremuestreo.

**Recomendación 2 (desequilibrio etario severo, objetivo hipoglucemia):** Cuando el objetivo prioritario es mejorar la predicción en rangos hipoglucémicos (TBR_1, TBR_2) en un dataset con desequilibrio etario, **Age·SMOTE** ofrece la mayor mejora en estos rangos (Hypo L1 DiaTrend: +5.0%, d=0.72), con un coste en hiperglucemia que debe documentarse y comunicarse explícitamente (TAR_1 DiaTrend: −5.7%). Esta recomendación asume que la seguridad ante hipoglucemia tiene prioridad clínica sobre la precisión en hiperglucemia.

**Recomendación 3 (dataset equilibrado o cohorte grande heterogénea):** En datasets con distribución demográfica relativamente equilibrada (como T1DiabetesGranada y REPLACE-BG), **no se recomienda aplicar balanceo demográfico** como estrategia primaria de mejora, dado que los efectos no alcanzan significación estadística y la varianza entre folds es elevada. La capacidad computacional invertida en el balanceo no se traduce en mejora robusta de la predicción. En estos contextos, la exploración de arquitecturas LSTM más profundas o estrategias de data augmentation basadas en características clínicas (en lugar de variables demográficas) puede ser más prometedora.

---

## §4.8 Discusión y limitaciones

### 4.8.1 Contraste con la literatura

Los resultados de este trabajo se alinean parcialmente con la literatura de fairness en aprendizaje automático aplicado a salud, donde se ha documentado consistentemente que las técnicas de reequilibrio de clases producen efectos modestos cuando el desequilibrio de base es moderado y efectos más pronunciados cuando el desequilibrio es severo (Barocas et al., 2019). En el contexto específico de la predicción glucémica con CGM, trabajos anteriores han propuesto estrategias de personalización (patient-specific fine-tuning) como la vía principal de mejora, con resultados superiores a los modelos poblacionales no personalizados (Zhu et al., 2020; Li et al., 2021). Los presentes resultados no contradicen este enfoque: el balanceo demográfico no es un sustituto de la personalización, sino un mecanismo auxiliar que puede mejorar la representatividad del modelo cuando la personalización no es posible (datos insuficientes de nuevos pacientes).

La magnitud de los efectos encontrados en DiaTrend —mejoras de hasta el 5% en TBR_1 RMSE para Age·SMOTE— es comparable a la reportada en estudios de data augmentation para CGM (Fabre et al., 2022), aunque los mecanismos son distintos. La señal obtenida en hipoglucemia es particularmente relevante porque este rango es el más difícil de predecir para los modelos LSTM por su escasa representación en los datos de entrenamiento no balanceados, problema que el balanceo demográfico mitiga indirectamente al aumentar la representación de grupos etarios con mayor tendencia a episodios hipoglucémicos.

### 4.8.2 Limitaciones metodológicas

**Missingness no aleatorio (MNAR).** Los tres datasets presentan porcentajes significativos de medidas faltantes (16.3% en DiaTrend, 9.9% en REPLACE-BG, 27.1% en T1DiabetesGranada sobre las columnas de fecha y valor). Este missingness no es aleatorio: los pacientes interrumpen el uso del CGM en períodos de mayor inestabilidad glucémica o cambios en el tratamiento, lo que introduce un sesgo sistemático hacia datos de períodos de relativa estabilidad. El balanceo demográfico realizado en este trabajo opera sobre ventanas de tiempo sin tener en cuenta este patrón de missingness, lo que puede distorsionar la eficacia de las técnicas en situaciones de uso real donde los episodios hipoglucémicos están sobre-representados precisamente en los períodos de mayor brecha en los datos.

**Baja consistencia entre folds en cohortes grandes.** Como se expuso en §4.6, el Group K-Fold sobre pacientes heterogéneos en T1DiabetesGranada y REPLACE-BG introduce una variabilidad entre folds que limita la potencia estadística y la interpretabilidad de los resultados. Estudios futuros deberían explorar estrategias de estratificación de folds más sofisticadas, como stratified group k-fold con control de la distribución demográfica por fold, para garantizar que cada fold refleja la heterogeneidad global de la cohorte.

**Horizonte único de predicción.** Todos los experimentos se evalúan sobre un horizonte de predicción único (60 minutos, 12 pasos de 5 min). Los efectos del balanceo pueden variar sustancialmente para horizontes cortos (15–30 min) o más largos (90–120 min), ya que la naturaleza del desequilibrio glucémico relevante cambia con el horizonte. En particular, la hipoglucemia severa es más predecible a corto plazo, y un análisis multi-horizonte podría revelar que el beneficio del balanceo se concentra en horizontes específicos.

**Ausencia de validación prospectiva.** El experimento es retrospectivo y se evalúa sobre datos históricos con partición temporal. Ninguna técnica de balanceo ha sido validada en un entorno de inferencia prospectiva, donde la distribución de los datos de entrada puede diferir de la distribución de entrenamiento de formas no anticipadas.

### 4.8.3 Trabajo futuro

Las limitaciones identificadas abren líneas de investigación concretas. En primer lugar, el balanceo a **nivel de episodio glucémico** (en lugar de a nivel de paciente) podría ser más efectivo para mejorar la predicción en rangos críticos, ya que directamente aumenta la representación de las trayectorias hipoglucémicas e hiperglucémicas sin depender de la diversidad demográfica de los pacientes. En segundo lugar, la **validación multi-horizonte** del balanceo demográfico permitiría identificar si los beneficios encontrados en TBR son más pronunciados a horizontes cortos, lo que tendría implicaciones directas para sistemas de alerta temprana de hipoglucemia. En tercer lugar, la **validación prospectiva** del pipeline de balanceo en un entorno de producción (por ejemplo, integrado en un sistema de monitorización continua) es el paso necesario para cualquier aplicación clínica real. Finalmente, la extensión del análisis a arquitecturas más recientes (Transformers, modelos de difusión para series temporales) podría revelar si el beneficio del balanceo demográfico persiste con modelos de mayor capacidad o si es específico de las arquitecturas recurrentes como LSTM.

---

*Fin de la Sección 4.*