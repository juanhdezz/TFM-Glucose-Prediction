La metodología es la siguiente:

---

**Punto de partida: lo que ya teníamos**

Antes de los nuevos scripts, el proceso era sencillo: se generaban ventanas deslizantes a partir de las series de glucosa, se asignaban los pacientes a folds (train/val/test), y el resultado era un único archivo por dataset que el modelo usaba para entrenarse. Sin balanceo.

El objetivo ahora es introducir el balanceo demográfico (por edad y por sexo) sin tocar ni el proceso de generación de ventanas ni el de entrenamiento.

---

**El problema que resolvemos**

Imaginemos que en DiaTrend los pacientes hombres generan 1.800.000 ventanas de entrenamiento y las mujeres generan 500.000. El modelo LSTM aprende principalmente de los hombres, no porque sean más importantes clínicamente, sino simplemente porque hay más datos suyos. Si después evaluamos el modelo por separado en cada grupo, es probable que prediga peor en mujeres. Lo mismo ocurre con la edad: si los pacientes de 45-60 años dominan los datos, el modelo aprende su dinámica glucémica y generaliza peor a adolescentes o mayores de 60.

El balanceo corrige esto ajustando cuántas ventanas de cada grupo llegan al entrenamiento.

---

**La restricción más importante: no contaminar val ni test**

El balanceo solo puede aplicarse sobre los datos de entrenamiento. Si modificamos también las ventanas de validación o test, los resultados del modelo no serían comparables entre experimentos ni defendibles metodológicamente, porque estaríamos evaluando sobre datos que también han sido manipulados.

Por eso toda la intervención ocurre exclusivamente dentro del conjunto train de cada fold, y val y test se mantienen exactamente igual que en el experimento sin balanceo.

---

**Cómo funciona el nuevo proceso paso a paso**

El pipeline completo tiene ahora cuatro pasos en lugar de tres.

*Paso 1 — Generación de ventanas* (sin cambios). A partir de las series temporales de glucosa se generan ventanas de 8 mediciones con su valor objetivo a 60 minutos. Resultado: un parquet por dataset.

*Paso 2 — Creación de folds* (sin cambios). Se asigna cada ventana a un fold, indicando si es train, val o test en cada uno de los 5 folds. Resultado: un único parquet por dataset donde cada fila tiene cinco etiquetas, una por fold.

*Paso 2b — Balanceo del train* (nuevo). Para cada fold por separado, se toman únicamente las ventanas de entrenamiento y se aplica la técnica de balanceo. Por ejemplo, para fold 0 con técnica de sobremuestreo por sexo: se duplican ventanas de mujeres hasta igualar el número de ventanas de hombres. Las ventanas de validación y test de ese fold no se tocan en absoluto. El resultado son cinco archivos separados, uno por fold.

Un ejemplo concreto: para DiaTrend, fold 0, técnica jittering, agrupación por sexo, el archivo de salida se llamaría `windows_with_5folds_DiaTrend_PH4_fold0_sex_jittering.parquet` y contiene el train balanceado de ese fold más el val y test originales intactos.

*Paso 2c — Compactación de folds* (nuevo). El script de entrenamiento espera recibir un único archivo que contenga los cinco folds juntos, igual que el archivo original del paso 2. Este paso lee los cinco archivos generados en 2b para una misma combinación de técnica y agrupación, y los fusiona en un único archivo con el formato esperado. Para hacer la fusión correctamente, el val y el test se toman siempre del archivo original del paso 2, nunca de los archivos balanceados, garantizando que esos conjuntos son idénticos en todos los experimentos.

El resultado final para el ejemplo anterior sería un único archivo llamado `DiaTrend_balanced_sex_jittering_PH4.parquet` que el script de entrenamiento puede leer exactamente igual que leía el archivo original, sin ningún cambio.

*Paso 3 — Entrenamiento* (sin cambios). El script de entrenamiento recibe el archivo compactado, separa train/val/test exactamente como siempre, y entrena el modelo. No sabe ni le importa que el train ha sido balanceado previamente.

---

**Qué obtenemos al final**

Siguiendo este proceso con 3 datasets, 5 técnicas de balanceo y 2 dimensiones demográficas (edad y sexo) obtenemos 30 archivos balanceados, cada uno directamente usable por el script de entrenamiento. Añadiendo el experimento sin balanceo como baseline, tenemos 33 condiciones experimentales en total, cada una con sus 5 folds de validación cruzada.

Comparando las métricas del modelo entre estas condiciones podemos responder directamente la pregunta del TFM: ¿mejora la predicción de glucosa cuando equilibramos la representación demográfica en los datos de entrenamiento?

---

**Por qué esta aproximación es metodológicamente válida**

Hay tres decisiones de diseño que hacen que el resultado sea defendible:

La primera es que el balanceo ocurre después de que los folds están definidos. Los folds se crean sobre los datos originales sin balancear, y solo después se interviene sobre el train. Esto evita que el proceso de balanceo influya en qué pacientes van a test, que es la partición sobre la que se mide el rendimiento real del modelo.

La segunda es que val y test siempre son los datos originales. El script de compactación los toma directamente del archivo del paso 2, ignorando lo que contienen los archivos balanceados. Hay una verificación automática que comprueba que los conteos de val y test son exactamente iguales a los originales antes de escribir el archivo final, y si no lo son el proceso se detiene con un error.

La tercera es que cada experimento es reproducible. Todos los procesos aleatorios usan una semilla fija, y los archivos de salida tienen nombres que identifican exactamente qué dataset, qué dimensión demográfica, qué técnica y qué horizonte de predicción se usaron.