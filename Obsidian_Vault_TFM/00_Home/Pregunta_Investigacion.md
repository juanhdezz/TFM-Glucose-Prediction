
# ❓ Pregunta de Investigación

## Pregunta Principal

**¿Cómo afecta la composición demográfica —sexo y edad— del conjunto de entrenamiento al rendimiento predictivo de un modelo LSTM para predicción de glucosa, y puede corregirse mediante técnicas de balanceo aplicadas a las ventanas de entrenamiento?**

---

## Hipótesis

1. **Hipótesis nula (H₀):** El balanceo demográfico del conjunto de entrenamiento no tiene un efecto significativo en el rendimiento predictivo del modelo LSTM en subgrupos demográficos infrarrepresentados.

2. **Hipótesis alternativa (H₁):** El balanceo demográfico del conjunto de entrenamiento mejora significativamente el rendimiento predictivo del modelo LSTM en subgrupos demográficos infrarrepresentados, reduciendo la disparidad de rendimiento entre grupos.

---

## Sub-preguntas

1. ¿Existe disparidad de rendimiento por sexo en modelos LSTM entrenados sin balanceo demográfico?

2. ¿Existe disparidad de rendimiento por edad en modelos LSTM entrenados sin balanceo demográfico?

3. ¿Qué técnica de balanceo ([[Oversampling]], [[Undersampling]], [[Patient-aware Undersampling]], [[SMOTE]], [[Jittering]]) es más efectiva para mitigar la disparidad de rendimiento?

4. ¿El efecto del balanceo es consistente a través de los tres datasets ([[DIATREND]], [[REPLACE-BG]], [[T1DiabetesGranada]])?

5. ¿La paridad demográfica en la composición del dataset garantiza la paridad en el rendimiento del modelo?

---

## Justificación Fisiológica

- [[Sexo]]: Las hormonas sexuales y la distribución de masa corporal modulan la homeostasis de la glucosa.
- [[Edad]]: La dinámica glucémica difiere sustancialmente entre grupos etarios.

## Justificación Metodológica

- Los modelos LSTM minimizan una pérdida agregada sobre el conjunto de entrenamiento.
- Si el dataset está dominado por un subgrupo demográfico, el modelo optimiza para ese subgrupo.
- El balanceo corrige esta distorsión sin modificar la arquitectura del modelo.

---

## Restricciones Metodológicas

1. El balanceo se aplica **exclusivamente** al conjunto de entrenamiento.
2. Validation y Test permanecen intactos (no se balancean).
3. El modelo LSTM no se modifica (solo se transforman los datos de entrada).
4. Validación cruzada patient-wise (un paciente no aparece en train y test simultáneamente).

---

## Referencias Clave

- [[Wang2024Disparate]]: Disparidad por sexo y edad en enfermedades crónicas
- [[Ovalle2024Racial]]: Disparidad racial en predicción de glucosa con LSTM
- [[Straw2022SexBias]]: Sesgo por sexo en modelos clínicos
- [[Ricci2024Sociodemographic]]: Revisión de sesgo sociodemográfico en ML clínico
