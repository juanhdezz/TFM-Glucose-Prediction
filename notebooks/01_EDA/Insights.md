# Análisis de Desbalanceo y Diseño Experimental: T1DiabetesGranada

## 1. Caracterización y Severidad del Desbalanceo
La caracterización detallada del dataset revela que el **46.3% de los pacientes (341 de 736)** presentan un desbalanceo severo (ratio **> 1:20**) o carecen totalmente de alguna de las clases minoritarias. Esta realidad clínica exige abandonar el enfoque de "un solo ratio global" y adoptar una estrategia de **simulación por estratos de severidad**.

### 1.1. Escenarios de Simulación (Basados en Percentiles)
En lugar de un balanceo 1:1 irreal, los experimentos deben replicar los desafíos observados en la distribución real:
* **Escenario Base (Mediana - 18.01:1):** Representa la dificultad promedio en la práctica clínica.
* **Escenario Desafiante (P75 - 39.84:1):** Evalúa si el modelo ignora la clase minoritaria bajo presión de desbalanceo alta.
* **Escenario Extremo (P90 - 105.05:1):** Crucial para probar la estabilidad de técnicas como *SMOTE temporal* cuando la señal es ínfima.

### 1.2. Estrategia de Evaluación por Estratos (Bucketing)
El rendimiento no debe reportarse como una media global, sino segmentado por categorías de `imbalance_severity` (Leve, Moderado, Severo, Extremo).
* **Objetivo:** Analizar si el modelo mantiene una **sensibilidad (recall)** aceptable en pacientes "Severos", evitando que una media global alta oculte un fracaso total en grupos críticos.

### 1.3. Mitigación del Sesgo de "Pacientes Dominantes"
El **Top 5** de pacientes concentra el **6.73%** de los episodios de hipoglucemia, creando el riesgo de que el modelo aprenda solo patrones de estos individuos.
* **Acción:** Implementar un **Resampling consciente del paciente (Patient-aware)** para limitar la contribución de individuos dominantes y asegurar diversidad sintética.

### 1.4. Estructura de Partición (Group-Stratified Split)
Dado que la **Edad (Age)** es la variable más predictora del nivel de desbalanceo, una partición aleatoria simple invalidaría los resultados.
* **Implementación:** Realizar un **Group Split por Patient_ID** (evita fuga temporal) estratificado por **Edad y Sexo**. Esto garantiza una representación proporcional de la dificultad de balanceo en entrenamiento y test.

### 1.5. Selección de Métricas Clínicas vs. Estadísticas
Con ratios de **105:1**, el *Accuracy* es inútil.
* **Prioridad:** La métrica reina es la **Sensibilidad (Recall)** en hipoglucemia (el coste de no detectar un evento es mayor que una falsa alarma).
* **Complementos:** Uso de **PR-AUC** (Área bajo la curva Precisión-Recall) por ser más informativo que ROC-AUC en clases escasas.

### 1.6. Validación de "Ausencia Real" vs. "Infradetección"
Existen 5 pacientes sin registros de eventos minoritarios (3 sin hipo, 2 sin hiper).
* **Acción:** Contrastar estos perfiles con grupos de riesgo (ej. >60 años) para asegurar que no se confunda la ausencia de eventos con problemas técnicos o falta de uso del sensor.

---

## 2. Estructura Temporal (Efecto "Ráfaga")
Los datos demuestran que las clases minoritarias tienen una fuerte dependencia temporal.

* **Autocorrelación:** Muy alta en minoritarias (**0.768** para hipo y **0.973** para hiper), indicando que los eventos aparecen en ráfagas.
* **Duración de Episodios:** Mediana de **3.0** mediciones para hipoglucemia y **7.0** para hiperglucemia.
* **Implicación:** Priorizar técnicas como *window-level balancing* para no romper la dinámica de los episodios.

---

## 3. Propiedades de las Ventanas Temporales (Lookback)
* **Ventana Óptima:** **24 pasos** (mediciones) es la longitud ideal para capturar el contexto previo y la máxima separabilidad de la señal (varianza y pendiente).
* **Separabilidad:** Las señales previas al inicio de un episodio son distinguibles de la normoglucemia, justificando el uso de *Data Augmentation*.

---

## 4. Síntesis de Parámetros Experimentales
| Parámetro | Configuración Sugerida |
| :--- | :--- |
| **Ratios de Simulación** | 18.01:1 (Base), 39.84:1 (Difícil), 105.05:1 (Extremo) |
| **Técnicas de Balanceo** | SMOTE temporal, Window-level balancing |
| **Validación** | Group-Stratified Split (ID, Age, Sex) |
| **Métricas Clave** | Recall (Hipo), PR-AUC, Balanced Accuracy |
| **Control de Leakage** | Respeto estricto al orden cronológico |