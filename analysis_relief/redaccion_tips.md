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

Sobre qué analizar: **no analices todas las técnicas con el mismo espacio**. Identifica cuáles tienen resultados interesantes (las que mejoran más, las que empeoran inesperadamente, y `reference_proportional` como técnica no anticipada) y dedícales párrafo argumentativo. Las que se comportan de forma esperada o neutral se agrupan en una frase. El análisis narrativo debe seguir el resultado, no el catálogo.



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