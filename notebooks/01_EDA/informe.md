# Informe de Análisis y Selección de Técnicas de Balanceo para TFM — Predicción de Glucosa

# Parte I. Caracterización de los Datasets

## 1.1 DIATREND

### Tamaño y estructura

DIATREND contiene 9.155.089 mediciones de glucosa procedentes de 54 pacientes, de los cuales 51 disponen de series temporales utilizables. El rango temporal abarca desde el 1 de diciembre de 2015 hasta el 28 de junio de 2022, convirtiéndolo en el dataset con mayor profundidad longitudinal. Existe una gran variabilidad entre pacientes: mientras que el más denso alcanza 649.908 registros, algunos apenas superan las 4.000 observaciones. En varios casos se dispone de más de cinco años de monitorización prácticamente continua.

### Calidad de los datos

El porcentaje global de valores nulos es del 17,2%, afectando principalmente a variables temporales derivadas. La resolución de 15 minutos presenta un 68,14% de valores ausentes, mientras que la resolución de 5 minutos únicamente registra un 4,42%, lo que indica que esta última constituye la frecuencia de muestreo principal. El timestamp compuesto presenta un 16,25% de nulos, coherente con la ausencia de información temporal en determinados registros. Los valores de glucosa se encuentran dentro del rango 39–401 mg/dL y existen 363 valores únicos en la escala discreta del CGM. No se observan valores nulos en la información de pacientes.

### Distribución del target y desbalanceo

La distribución global muestra aproximadamente un 51% de normoglucemia, un 47% de hiperglucemia y únicamente un 2% de hipoglucemia. Sin embargo, esta distribución agregada oculta una fuerte heterogeneidad entre pacientes: la prevalencia de hipoglucemia oscila entre el 0,05% y el 9,31% según el individuo. La concentración de eventos es especialmente llamativa, ya que los cinco pacientes con mayor número de hipoglucemias acumulan el 53,94% de todos los episodios hipoglucémicos del dataset, mientras que los cinco pacientes con más hiperglucemias concentran el 29,96% de dichos eventos.

El nivel de desbalanceo es muy elevado. Un total de 41 de los 51 pacientes (80,4%) presentan ratios normoglucemia/minoritaria superiores a 1:20, considerados casos de desbalanceo severo. Los 10 pacientes restantes se sitúan en la categoría moderada, con ratios entre 1:5 y 1:20. El caso más extremo alcanza una proporción de 1.376:1.

### Particularidades temporales

Desde el punto de vista temporal, la hipoglucemia presenta una autocorrelación muy baja (0,124), indicando que los eventos aparecen de forma relativamente dispersa y poco predecible. Por el contrario, la hiperglucemia alcanza una autocorrelación de 0,825 y la normoglucemia mantiene valores elevados y estables. La duración mediana de los episodios es de 4 pasos para la hipoglucemia y de 14 pasos para normoglucemia e hiperglucemia. Esta combinación de baja autocorrelación y corta duración convierte a los episodios hipoglucémicos en fenómenos aislados que difícilmente forman ráfagas largas, aspecto especialmente relevante para la selección de técnicas de oversampling.

## 1.2 REPLACE-BG

### Tamaño y estructura

REPLACE-BG está compuesto por 11.579.408 mediciones correspondientes a 223 pacientes. El periodo cubierto se extiende entre enero y noviembre del año 2000, aproximadamente diez meses de seguimiento. Aunque la duración temporal es mucho menor que en DIATREND, se trata del dataset con mayor número de pacientes.

### Calidad de los datos

La calidad de los datos es comparable a la observada en DIATREND. El porcentaje de valores nulos en medición, fecha y hora es del 9,91%. La resolución de 15 minutos presenta un 66,81% de valores ausentes, mientras que la resolución de 5 minutos apenas alcanza el 0,44%. Los valores de glucosa vuelven a encontrarse dentro del rango fisiológico esperado de 39–401 mg/dL.

### Distribución del target y desbalanceo

La distribución global difiere notablemente de la observada en DIATREND. La normoglucemia representa el 63,62% de las observaciones, la hiperglucemia el 32,68% y la hipoglucemia el 3,70%. Aunque sigue existiendo desbalanceo, la hipoglucemia tiene una presencia relativa considerablemente mayor.

A nivel de pacientes, el 47,1% de la cohorte presenta ratios superiores a 1:20. Sin embargo, la concentración de eventos es mucho menor que en DIATREND. Los cinco pacientes con más episodios hipoglucémicos acumulan únicamente el 6,84% de todas las hipoglucemias registradas, mientras que los cinco con más hiperglucemias concentran el 4,78%. Esto sugiere una distribución clínica mucho más homogénea dentro de la cohorte.

### Particularidades temporales

El hallazgo más relevante es la elevada autocorrelación de todas las clases. La hipoglucemia presenta valores cercanos a 0,8, mientras que normoglucemia e hiperglucemia rondan 0,95. Los episodios tienen además una duración superior a la observada en DIATREND, con medianas de 6 pasos para hipoglucemia, 24 para normoglucemia y 18 para hiperglucemia. Esto indica que la hipoglucemia aparece frecuentemente en forma de ráfagas persistentes y no como eventos aislados.

### Posibles sesgos

El hecho de que todos los datos procedan del año 2000 sugiere que el dataset proviene de un entorno clínico controlado. Esta característica puede introducir sesgos derivados de la monitorización intensiva y del propio diseño experimental, haciendo que algunos patrones difieran de los observados en contextos de práctica clínica real.

## 1.3 T1DiabetesGranada

### Tamaño y estructura

T1DiabetesGranada es el dataset de mayor volumen, con 30.477.434 mediciones brutas correspondientes a 736 pacientes y 643 pacientes únicos con registros glucémicos válidos. El rango temporal cubre desde enero de 2018 hasta marzo de 2022.

### Calidad de los datos

La principal diferencia respecto a los otros datasets es la frecuencia de muestreo. La columna correspondiente a resolución de 5 minutos está completamente vacía, mientras que la resolución de 15 minutos presenta únicamente un 1,19% de nulos. En conjunto, el 27,05% de las mediciones y timestamps son valores ausentes.

### Problema estructural: fragmentación temporal

La característica más importante de este dataset es la fuerte discontinuidad temporal de las series. El análisis de gaps revela que la mayoría de los pacientes no disponen de registros continuos prolongados. Aunque 643 pacientes son elegibles para ventanas de 1 día, únicamente 7 pacientes mantienen series continuas de 14 días y tan solo uno alcanza una ventana de 90 días. La cobertura media efectiva es de 3,3 días para ventanas de un día y de 35,5 días para ventanas de 14 días. En consecuencia, lo que aparentemente es una cohorte con varios años de seguimiento se comporta realmente como una colección de múltiples fragmentos temporales cortos.

### Distribución del target y desbalanceo

La distribución global es la más equilibrada de los tres datasets. Aproximadamente el 55% de las observaciones corresponden a normoglucemia, el 35% a hiperglucemia y el 8% a hipoglucemia. Aun así, persiste una gran heterogeneidad entre pacientes. Existen tres pacientes sin representación de la clase minoritaria, 296 pacientes con desbalanceo severo, 312 con desbalanceo moderado y únicamente 32 con ratios considerados leves.

### Particularidades temporales

La hipoglucemia presenta autocorrelaciones entre 0,7 y 0,8, similares a las observadas en REPLACE-BG, mientras que normoglucemia e hiperglucemia superan habitualmente 0,95. La concentración de eventos es reducida: los cinco pacientes con más hipoglucemias acumulan únicamente el 6,8% de los eventos y los cinco con más hiperglucemias el 4,0%, indicando una distribución relativamente homogénea a nivel poblacional.

# Parte II. Insights Críticos

## 2.1 Naturaleza diferente del desbalanceo

Aunque los tres datasets presentan hipoglucemia infrarrepresentada, la naturaleza del problema es distinta en cada caso. DIATREND combina una fuerte concentración de eventos en pocos pacientes con episodios cortos y poco autocorrelados. REPLACE-BG presenta episodios más largos y persistentes, distribuidos de forma más homogénea. T1DiabetesGranada comparte la estructura temporal de REPLACE-BG, pero introduce además un problema severo de fragmentación temporal. Esta heterogeneidad implica que una misma técnica de balanceo no puede asumirse igualmente válida para todos los datasets.

## 2.2 Limitaciones de SMOTE clásico

Las técnicas basadas en interpolación, como SMOTE, asumen independencia relativa entre muestras. Cuando la hipoglucemia aparece en ráfagas altamente autocorreladas, los vecinos más cercanos pertenecen normalmente al mismo episodio. La interpolación entre ellos no genera nueva información clínica relevante, sino variaciones artificiales de una misma secuencia temporal. Por este motivo, la aplicación directa de SMOTE resulta cuestionable, especialmente en REPLACE-BG y T1DiabetesGranada.

## 2.3 Necesidad de estrategias adaptadas al paciente

Los ratios de desbalanceo observados varían desde aproximadamente 2,6:1 hasta 1.376:1. Esta enorme variabilidad hace difícil justificar un único ratio objetivo para toda la cohorte. Una estrategia uniforme produciría inevitablemente sobrebalanceo en algunos pacientes y subbalanceo en otros. Por ello, resulta más razonable plantear enfoques adaptativos basados en paciente o en estratos de severidad.

## 2.4 Particularidades de T1DiabetesGranada

A diferencia de los otros datasets, T1DiabetesGranada requiere una decisión previa sobre la unidad temporal de análisis. La presencia de gaps impide aplicar directamente técnicas de resampling sobre la serie completa, ya que cualquier generación de muestras sintéticas que atraviese discontinuidades produciría secuencias fisiológicamente inconsistentes.

## 2.5 Naturaleza de la hiperglucemia

La hiperglucemia constituye una clase abundante y temporalmente estable, especialmente en DIATREND, donde representa aproximadamente el 47% de las observaciones. Aunque no requiere oversampling, su distribución desigual entre pacientes puede introducir sesgos que deben controlarse durante la evaluación.

## 2.6 Restricciones fisiológicas

Los límites 39 y 401 mg/dL representan saturaciones reales del sensor CGM. Cualquier técnica generativa debe respetar estrictamente este rango para evitar la creación de observaciones fisiológicamente imposibles.

## 2.7 El problema real es de regresión desbalanceada

Aunque gran parte del análisis se apoya en etiquetas discretas de hipoglucemia, normoglucemia e hiperglucemia, el objetivo final es la predicción de valores continuos de glucosa. Desde esta perspectiva, el problema debe entenderse como un caso de imbalanced regression, donde las regiones de interés son principalmente los valores inferiores a 70 mg/dL y, en menor medida, los superiores a 180 mg/dL.

# Parte III. Selección de Técnicas Candidatas

## Técnicas estándar

### T1. Random Oversampling (ROS) a nivel episódico

Consiste en replicar ventanas temporales completas pertenecientes a la clase minoritaria en lugar de observaciones individuales. Esta aproximación preserva la estructura temporal de los episodios y evita la creación de muestras sintéticas. Su principal riesgo es el sobreajuste, aunque representa la alternativa más segura para series temporales.

### T2. Random Undersampling (RUS) con preservación temporal

Busca reducir la dominancia de la normoglucemia eliminando ventanas poco informativas, manteniendo siempre las regiones próximas a transiciones clínicas relevantes. Resulta especialmente atractivo en datasets de gran tamaño por sus beneficios computacionales.

### T3. Cost-Sensitive Learning

Introduce pesos inversamente proporcionales a la frecuencia de las clases dentro de la función de pérdida. Al no modificar los datos originales, evita cualquier artefacto asociado a técnicas de generación o eliminación de muestras.

## Técnicas avanzadas

### T4. SMOTE adaptado temporalmente

Propone restringir la interpolación a ventanas del mismo paciente y contextos temporales similares. Aunque puede resultar útil en datasets con episodios persistentes, requiere definir cuidadosamente qué se entiende por vecindad temporal coherente.

### T5. ADASYN

Genera muestras sintéticas de forma adaptativa en las regiones más difíciles de aprender. Su principal atractivo radica en reforzar las zonas de transición entre normoglucemia e hipoglucemia, donde suelen concentrarse los errores predictivos.

### T6. Balanceo adaptativo por paciente

Consiste en aplicar estrategias distintas según el nivel de desbalanceo de cada paciente. Este enfoque está directamente respaldado por los hallazgos del EDA y aborda de forma explícita la gran heterogeneidad observada.

### T7. Episode-Aware Oversampling

Utiliza episodios completos como unidad de resampling, respetando así la estructura temporal natural de la enfermedad. Conceptualmente resulta más coherente que los métodos basados en puntos individuales.

## Técnicas de investigación

### T8. SMOGN y WERCS

Estas técnicas trasladan el concepto de oversampling al contexto de regresión desbalanceada. Dado que el problema real consiste en predecir glucosa continua, representan una de las alternativas metodológicamente más alineadas con el objetivo final del TFM.

### T9. VAE condicional

Los autoencoders variacionales permiten generar nuevas secuencias glucémicas aprendiendo una representación latente de los episodios reales. Aunque su interés científico es elevado, la complejidad de implementación y evaluación dificulta su adopción en un TFM con recursos limitados.

### T10. TimeGAN

Representa una de las aproximaciones más avanzadas para la generación de series temporales sintéticas. Sin embargo, el elevado coste computacional y la dificultad de validación limitan su viabilidad práctica.

### T11. Focal Loss para regresión

Adapta el concepto de focalización de errores a escenarios de regresión, incrementando el peso de las regiones infrarrepresentadas sin necesidad de modificar los datos originales.

# Parte IV. Priorización Final

## Prioridad alta

Las técnicas recomendadas como núcleo experimental son T1 (ROS episódico), T2 (RUS con preservación temporal), T3 (Cost-Sensitive Learning) y T6 (Balanceo adaptativo por paciente). Estas alternativas presentan una combinación favorable de viabilidad, fundamento metodológico y alineación con los hallazgos del análisis exploratorio.

## Prioridad media

En un segundo nivel se sitúan T4 (SMOTE adaptado temporalmente), T5 (ADASYN) y T7 (Episode-Aware Oversampling). Su aplicación resulta especialmente interesante en REPLACE-BG y T1DiabetesGranada, donde la estructura temporal de los episodios favorece este tipo de enfoques.

## Prioridad baja

Finalmente, T8 (SMOGN/WERCS), T9 (VAE), T10 (TimeGAN) y T11 (Focal Loss para regresión) constituyen propuestas con elevado interés académico y potencial para enriquecer la discusión metodológica, aunque no necesariamente requieren una implementación completa dentro del alcance del TFM.

# Síntesis Final

El análisis realizado demuestra que los tres datasets comparten la presencia de hipoglucemia infrarrepresentada, pero difieren profundamente en la forma en que dicho desbalanceo se manifiesta. DIATREND se caracteriza por una concentración extrema de eventos en pocos pacientes y episodios breves; REPLACE-BG presenta una distribución más homogénea y episodios claramente agrupados en ráfagas; y T1DiabetesGranada combina una representación relativamente equilibrada de la hipoglucemia con un importante problema de fragmentación temporal. Como consecuencia, no existe una única técnica de balanceo universalmente válida. El valor científico del TFM residirá precisamente en analizar cómo las características estructurales de cada dataset condicionan la eficacia de las distintas estrategias de balanceo y cómo estas afectan al rendimiento predictivo frente al baseline de referencia.
