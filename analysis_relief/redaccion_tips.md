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


---

# Datos reales 

§4.1 — Configuración experimental y baseline
El conjunto de experimentos cubre tres datasets de monitorización continua de glucosa (T1DiabetesGranada, DiaTrend y REPLACE-BG), seis técnicas de balanceo demográfico (jittering, oversampling, patient-aware undersampling, reference proportional, SMOTE y undersampling) aplicadas sobre dos dimensiones (edad y sexo), con validación cruzada de 5 folds por condición. En total se evaluaron 39 condiciones experimentales —13 por dataset, incluyendo el baseline sin balanceo—, cada una entrenando un modelo LSTM con ventana de historia H=8H=8
H=8 mediciones y horizonte de predicción PH=4PH=4
PH=4 pasos (60 minutos). Las métricas de rendimiento se calcularon de forma independiente sobre seis rangos clínicos: hipoglucemia severa (TBR-2, $<$54 mg/dL), hipoglucemia leve (TBR-1, 54–69 mg/dL), rango en objetivo (TIR, 70–180 mg/dL), hiperglucemia leve (TAR-1, 181–250 mg/dL) e hiperglucemia severa (TAR-2, $>$250 mg/dL), además del rango global (ENTIRE). La condición sin balanceo (original) actúa como baseline de referencia a lo largo de todo el análisis; val y test permanecen inalterados en todos los experimentos, garantizando la comparabilidad entre condiciones.
Los valores de RMSE del baseline oscilan entre 37.80 ± 0.81 mg/dL (T1DiabetesGranada, global) y 45.05 ± 2.45 mg/dL (DiaTrend, global), con REPLACE-BG en una posición intermedia (37.00 ± 0.83 mg/dL). La mayor incertidumbre entre folds se concentra en DiaTrend, especialmente en los rangos de hipoglucemia (TBR-2: 75.28 ± 7.33 mg/dL), lo que anticipa una mayor sensibilidad a las técnicas de balanceo en ese dataset.
[Tabla §4.1: resumen de datasets, nº condiciones, RMSE baseline por rango — construir a mano a partir de main_results_RMSE.csv, fila "Original"]



### §4.2 — Rendimiento global: ¿mejora el balanceo en promedio?

**[Figura §4.2-A: `dashboard_global` — panel de honor]**

El test de Friedman detecta diferencias estadísticamente significativas entre las 13 condiciones experimentales en DiaTrend para todos los rangos clínicos ($\chi^2=50.79$, $p=1\times10^{-6}$ en ENTIRE; $\chi^2=45.12$, $p=1\times10^{-5}$ en TIR; $\chi^2=39.24$, $p=9.6\times10^{-5}$ en TBR-1; $\chi^2=47.57$, $p=4\times10^{-6}$ en TAR-1). Por el contrario, ningún rango alcanza significancia en REPLACE-BG ($p \geq 0.24$ en todos los casos). T1DiabetesGranada presenta un patrón intermedio: el test resulta significativo en TBR-2 ($\chi^2=26.45$, $p=0.009$), TBR-1 ($\chi^2=21.62$, $p=0.042$), TIR ($\chi^2=24.35$, $p=0.018$) y TAR-1 ($\chi^2=25.57$, $p=0.012$), pero no en TAR-2 ni en el RMSE global ($p=0.457$).

**[Figura §4.2-B: CD diagrams — tres paneles compuestos]**

El post-hoc de Nemenyi sobre DiaTrend·TIR confirma que cuatro técnicas de balanceo por edad superan al baseline con significancia estadística: Age·Oversampling ($p=0.0055$), Age·SMOTE ($p=0.0088$), Age·Ref. Proportional ($p=0.0102$) y Age·Jittering ($p=0.0321$). Ninguna técnica de sexo alcanza diferencia significativa respecto al baseline en TIR ($p \geq 0.66$ en todos los casos). En TAR-1, el patrón se invierte: Age·Oversampling se separa del baseline con $p=0.0012$ en sentido negativo —es decir, el balanceo por edad degrada significativamente la predicción en hiperglucemia leve—, y Age·Ref. Proportional alcanza $p=0.0034$. Este trade-off se analiza en detalle en §4.5.

En T1DiabetesGranada, el post-hoc de Nemenyi no identifica ningún par de condiciones con diferencia significativa en TBR-1, TBR-2 ni TIR a pesar de la significancia global de Friedman, lo que indica que las diferencias entre condiciones son difusas y no localizables en una técnica concreta —patrón consistente con la baja consistencia entre folds documentada en §4.6.

En cuanto al ranking Borda por dataset, Age·SMOTE lidera en T1DiabetesGranada (puntuación 45, posición 1) y ocupa el tercer puesto en DiaTrend (63, posición 3); Age·PAUndersampling encabeza REPLACE-BG (43, posición 1). En el dashboard global (panel B), Age·Undersampling y Age·PAUndersampling acumulan las peores posiciones agregadas, impulsadas por su comportamiento en TAR-1 de DiaTrend.

**[Figura §4.2-C: Borda lollipop — cierre de subsección]**

La respuesta a la pregunta de investigación de esta subsección es contextual: el balanceo demográfico produce efectos estadísticamente distinguibles únicamente cuando el dataset presenta suficiente heterogeneidad demográfica entre folds. DiaTrend es el único caso donde esta condición se satisface de forma sistemática.

---

### §4.3 — Análisis por dataset

**[Figura §4.3-A: panel A del `dashboard_global` — RMSE improvement heatmap]**

La heterogeneidad entre datasets constituye el hallazgo estructural de mayor alcance del estudio. En DiaTrend, las técnicas de balanceo por edad producen mejoras de hasta −7.3\% en RMSE global (Age·PAUndersampling) y −6.4\% (Age·Undersampling), concentradas en TIR. Las técnicas de sexo permanecen dentro de ±0.5\% en todos los rangos globales de DiaTrend. En T1DiabetesGranada y REPLACE-BG las diferencias respecto al baseline son marginales en RMSE global (entre −0.6\% y +0.3\% en T1DG; entre −0.0\% y +0.3\% en REPLACE-BG), con efectos de tamaño negligible en ENTIRE en ambos datasets (Cohen $d < 0.28$ en todos los casos).

**[Figura §4.3-B: dumbbell plots para TBR-1 y TIR por dataset]**

La asimetría entre DiaTrend y los otros dos datasets se explica por tres factores convergentes: mayor variabilidad basal entre folds en DiaTrend (std de 2.45 mg/dL en ENTIRE frente a 0.81 y 0.83 en los otros dos), mayor diversidad etaria de sus pacientes —que amplifica el impacto del balanceo por edad—, y la alta consistencia del ranking de técnicas entre folds (Spearman $\rho$ entre 0.69 y 0.92), que permite al test estadístico detectar efectos sistemáticos. Estos tres factores se detallan en §4.6.

---

### §4.4 — Dimensión demográfica: edad vs. sexo

**[Figura §4.4-A: panel C del `dashboard_global` — Mean RMSE by technique and dimension]**

El balanceo por edad produce efectos sistemáticamente superiores a los del balanceo por sexo en DiaTrend, tanto en magnitud como en significancia estadística. En TIR, cuatro técnicas de edad alcanzan efecto large en Cohen $d$ (jittering $d=0.86$, oversampling $d=1.07$, reference proportional $d=0.95$, SMOTE $d=0.99$), mientras que las seis técnicas de sexo en el mismo rango se quedan en medium o small ($d$ entre 0.26 y 0.52). El post-hoc de Nemenyi confirma que ninguna técnica de sexo alcanza diferencia significativa respecto al baseline en TIR de DiaTrend ($p \geq 0.66$).

En TBR-1 de DiaTrend, Age·Oversampling ($d=0.58$ medium) y Age·SMOTE ($d=0.63$ medium) producen mejoras moderadas, mientras que todas las técnicas de sexo presentan efectos negligibles ($d < 0.19$). El post-hoc de Nemenyi en TBR-1 no identifica diferencias significativas entre ninguna técnica y el baseline, aunque Age·Oversampling y Age·SMOTE tienen los p-values más bajos frente al original (0.54 y 0.48 respectivamente), coherentes con su dirección de efecto.

En T1DiabetesGranada, ninguna de las dos dimensiones produce efectos clínicamente relevantes en ENTIRE o TIR (todos los $d$ negligibles). Los efectos más pronunciados de T1DG se localizan en TAR-1, donde Age·PAUndersampling ($d=0.79$ RMSE, $d=1.33$ MAE) y Age·Undersampling ($d=0.90$ RMSE, $d=1.23$ MAE) son las técnicas con mayor señal, aunque en sentido de mejora en hiperglucemia —a diferencia de lo observado en DiaTrend.

**[Tabla §4.4: efectos large y medium seleccionados de cohens\_d\_vs\_original — construir a mano con las filas más informativas]**

---

### §4.5 — Relevancia clínica: rangos glucémicos

**[Figura §4.5-A: panel D del `dashboard_global` — RMSE improvement en Hypo L1]**

El análisis por rangos revela un trade-off sistemático que es el hallazgo clínico más importante del TFM: las técnicas de balanceo por edad que mejoran TIR degradan simultáneamente TAR-1 en DiaTrend, y este efecto es estadísticamente significativo.

En TIR, Age·Oversampling reduce el RMSE de 30.72 a 29.45 mg/dL ($d=1.07$, large; $p_\text{Nemenyi}=0.0055$ vs. baseline). Age·SMOTE produce una reducción comparable (30.72 → 29.57 mg/dL, $d=0.99$, large; $p=0.0088$). Age·Ref. Proportional y Age·Jittering obtienen efectos large similares ($d=0.95$ y $d=0.86$; $p=0.0102$ y $p=0.0321$). Ninguna de estas técnicas, sin embargo, supera al baseline en TBR-1 con significancia estadística en el post-hoc de Nemenyi.

En TAR-1, el patrón se invierte con igual contundencia: Age·Oversampling incrementa el RMSE de 42.08 a 44.93 mg/dL ($d=-2.14$, large; $p_\text{Nemenyi}=0.0012$ vs. baseline), Age·Ref. Proportional ($d=-1.95$, large; $p=0.0034$) y Age·Jittering ($d=-1.70$, large; $p=0.0212$) siguen el mismo patrón. Age·SMOTE, a pesar de su efecto large negativo en TAR-1 ($d=-2.14$), alcanza $p=0.0088$ —es decir, la mejora en TIR y la degradación en TAR-1 son ambas estadísticamente significativas para la misma técnica.

La única técnica de edad que atenúa este trade-off es Age·PAUndersampling, cuyo efecto en TAR-1 es medium ($d=-0.88$; $p=0.63$ frente al baseline, no significativo), a costa de no mejorar significativamente TIR ($d=0.49$, small) ni TBR-1 ($d=-0.70$, medium, en sentido negativo).

**[Figura §4.5-B: heatmap de rankings por rango y técnica en DiaTrend]**

En T1DiabetesGranada, el test de Friedman detecta diferencias en TBR-1 ($p=0.042$) y TBR-2 ($p=0.009$), pero el post-hoc de Nemenyi no identifica ningún par significativo en esos rangos, lo que indica que el efecto del balanceo en hipoglucemia en T1DG es estadísticamente presente pero difuso —no atribuible a ninguna técnica concreta con la potencia estadística disponible (5 folds, 13 condiciones).

En términos de seguridad clínica, estos resultados implican que la selección de técnica de balanceo no puede basarse únicamente en RMSE global: una técnica que mejora TIR a costa de TAR-1 puede ser aceptable para perfiles predominantemente normoglucémicos, pero contraproducente en pacientes con tendencia a hiperglucemia.

---

### §4.6 — Estabilidad entre folds

**[Figura §4.6: `consistency_folds_RMSE_ENTIRE` — tres paneles, figura principal]**

La matriz de correlación Spearman del ranking de técnicas entre pares de folds revela una brecha estructural entre datasets. DiaTrend presenta consistencia alta y uniforme ($\rho$ entre 0.69 y 0.92, con media $\bar{\rho}=0.81$), lo que indica que el ranking de condiciones es estable independientemente de la partición de pacientes. Por el contrario, T1DiabetesGranada muestra correlaciones que oscilan entre $-0.44$ y $0.40$ —con valores negativos frecuentes—, y REPLACE-BG exhibe un patrón igualmente inconsistente ($\rho$ entre $-0.48$ y $0.31$).

Esta asimetría explica directamente la de Friedman: cuando el ranking de técnicas se invierte entre folds, el test no detecta diferencias sistemáticas entre condiciones, no por ausencia de efectos sino porque la varianza inducida por la heterogeneidad de pacientes entre particiones supera la señal del balanceo. En T1DiabetesGranada y REPLACE-BG el Group K-Fold opera sobre conjuntos de pacientes cuyas dinámicas glucémicas no son intercambiables, introduciendo un nivel de varianza estructural que el balanceo de ventanas no puede compensar. Este resultado es metodológicamente esperable y refleja la dificultad real de generalizar modelos de predicción glucémica a través de poblaciones heterogéneas.

---

### §4.7 — Técnica ganadora y recomendación práctica

Del análisis acumulado se extraen tres recomendaciones condicionales.

Si el dataset presenta diversidad etaria suficiente y el objetivo clínico prioritario es minimizar el error en rango normoglucémico (TIR), **Age·Oversampling** es la técnica con mayor efecto confirmado ($d=1.07$ large en TIR de DiaTrend; $p_\text{Nemenyi}=0.0055$ frente al baseline). Su coste en TAR-1 es estadísticamente significativo ($d=-2.14$, $p=0.0012$) y debe evaluarse explícitamente frente al perfil glucémico de la población.

Si el objetivo es maximizar el rendimiento agregado minimizando el trade-off entre rangos, **Age·SMOTE** ofrece el mejor equilibrio según el ranking Borda intra-dataset (posición 1 en T1DG con puntuación 45; posición 3 en DiaTrend con 63), con efecto large en TIR ($d=0.99$) y la misma advertencia en TAR-1.

Si el dataset es homogéneo o presenta baja consistencia entre folds (patrón T1DiabetesGranada/REPLACE-BG), **no aplicar balanceo** es la opción más conservadora: ninguna técnica produce mejoras globales estadísticamente significativas, el post-hoc de Nemenyi no localiza ningún par significativo en ningún rango, y el riesgo de introducir varianza incontrolada supera el beneficio esperado.

**[Tabla §4.7: síntesis — construir a mano: filas = técnicas, columnas = Friedman sig. (DiaTrend), $p$ Nemenyi vs. baseline (TIR), Cohen $d$ TIR, Cohen $d$ TAR-1, posición Borda por dataset]**

**[Figura §4.7: radar DiaTrend dimensión Age — perfil de técnica ganadora]**

