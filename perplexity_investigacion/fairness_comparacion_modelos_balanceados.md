Sí: **es un problema metodológico reconocido**, pero con un matiz importante. En la literatura sobre *imbalanced learning* no se considera que el “problema” sea comparar contra el dataset original per se; el problema reconocido es que el efecto observado puede mezclar al menos tres cosas distintas: cambio en la distribución de clases, cambio en el tamaño efectivo del entrenamiento y cambio en la geometría/localidad inducida por la generación sintética. SMOTE, por ejemplo, fue concebido para generar muestras sintéticas en el entrenamiento, y la literatura insiste en aplicarlo **solo al train** para evitar fuga de información; además, revisiones modernas muestran que el efecto de resampling depende fuertemente del clasificador, la métrica y el ratio de desbalance, por lo que no hay una “ventaja” universal atribuible solo al balanceo. [openreview](https://openreview.net/pdf?id=CF7FzuUkUck)

## Qué dice la literatura

La evidencia empírica y las revisiones coinciden en varios puntos. Primero, técnicas como SMOTE no son equivalentes a “simplemente duplicar datos”: generan puntos sintéticos por interpolación, lo que puede alterar la distribución real del minority class y su frontera de decisión. Segundo, la mejora al usar oversampling no puede atribuirse únicamente a “más datos”, porque en SMOTE los ejemplos nuevos no aportan información independiente en el sentido estadístico clásico; sí aportan densidad local alrededor de la minoría, pero también pueden introducir artefactos o sobreajuste en ciertas condiciones. Tercero, hay consenso en que el resampling debe aplicarse solo al conjunto de entrenamiento dentro de cada partición de validación cruzada, precisamente para no contaminar evaluación y preservar independencia entre train y test. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0020025519306838)

Sobre tu preocupación concreta, la literatura apoya que comparar un modelo original frente a un modelo entrenado con oversampling/SMOTE **es habitual y metodológicamente aceptado**, siempre que la comparación se formule como “efecto del método de reequilibrado en el pipeline completo”. Muchas comparaciones publicadas hacen exactamente eso: baseline sin resampling frente a variantes con undersampling, oversampling y SMOTE, evaluadas sobre el mismo test intacto. Lo que no es correcto es interpretar automáticamente cualquier mejora como “solo efecto del balanceo” sin reconocer que el método cambia simultáneamente el tamaño y la estructura del conjunto de entrenamiento. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC8641296/)

## Consenso y debate

Hay **consenso** en que el resampling es un paso de entrenamiento y debe estar anidado dentro de la validación cruzada; también hay consenso en que el efecto depende del clasificador y del problema, y que no existe una técnica universalmente superior. El **debate** está en la interpretación causal de la mejora: si proviene de la corrección del sesgo por desbalance, de la mayor exposición a la minoría, de la regularización implícita o de la simple duplicación/interpolación de observaciones. La literatura no ofrece una respuesta única porque esos mecanismos no se separan automáticamente en un único experimento estándar. [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/10.1002/sam.11538)

En otras palabras: tu revisor tendría razón si dijera que “más muestras de entrenamiento” es un confusor potencial. Pero también sería correcto responder que, en estudios de resampling, ese aumento del tamaño **forma parte del método** y no un artefacto accidental. Por eso, la comparación principal suele ser pragmática: “¿qué pipeline funciona mejor bajo las mismas condiciones de evaluación?” y no “¿cuál es el efecto puramente causal del balanceo aislado de todo lo demás?”. [bmj](https://www.bmj.com/content/372/bmj.n71)

## Estrategias usadas en artículos

La literatura usa varias estrategias para reducir este sesgo interpretativo. Una es comparar resampling con **cost-sensitive learning** o con loss reweighting, porque estos enfoques corrigen el desbalance sin aumentar necesariamente el tamaño del train, lo que ayuda a separar el efecto del reequilibrado del efecto del tamaño. Otra estrategia es hacer análisis por familias: comparar solo métodos de oversampling entre sí, solo undersampling entre sí, o familias híbridas por separado, para no mezclar mecanismos distintos. También se usan *ablation studies* o experimentos adicionales con distintos ratios de oversampling para ver si la mejora se satura o si aparece sobreajuste, lo que ayuda a interpretar si el resultado depende del número de muestras generadas. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0020025519306838)

Además, en la práctica es frecuente reportar la evaluación con métricas robustas al desbalance, como balanced accuracy, F1, G-mean o AUC, en lugar de accuracy simple, porque la accuracy puede ocultar el comportamiento en la minoría. En estudios biomédicos y de aprendizaje con datos desbalanceados, las revisiones recomiendan describir claramente el esquema de particionado, la semilla, el balanceo por fold y las métricas, porque la reproducibilidad y la comparabilidad dependen de ello. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/34867255/)

## Recomendaciones oficiales

No existe una “norma oficial” específica que diga que debas igualar el tamaño del train tras oversampling. Lo más cercano a una recomendación ampliamente aceptada es PRISMA para revisiones sistemáticas, pero PRISMA regula **cómo reportar** una revisión, no cómo diseñar exactamente un experimento de resampling. Para tu caso, la guía metodológica más sólida viene de la propia literatura de *imbalanced learning*: aplicar el balanceo solo en train, evaluar con test intacto, usar validación cruzada correctamente anidada y describir de forma transparente las limitaciones del método. [prisma-statement](https://www.prisma-statement.org/prisma-2020-statement)

## Qué añadiría en tu TFM

Sí, sería recomendable añadir al menos algunos experimentos o análisis adicionales, pero no necesariamente todos. Lo más defendible sería:  
- separar resultados por **familias**: undersampling, oversampling, sintéticos e híbridos;  
- añadir un análisis con **cost-sensitive baseline** o sample weighting, si tu modelo lo admite;  
- reportar el número final de muestras de train por técnica, para que el lector vea cuánto cambia el tamaño;  
- si es viable, hacer un subanálisis donde se **fuerce un tamaño de train comparable** entre técnicas, por ejemplo mediante submuestreo de la estrategia oversampled o mediante curvas de aprendizaje;  
- si no puedes repetir todo, al menos hacer un experimento de sensibilidad con una o dos técnicas representativas, como Random Oversampling vs. SMOTE vs. una técnica de undersampling. [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/10.1002/sam.11538)

Una idea metodológicamente útil es distinguir entre dos preguntas:  
1. “¿Qué técnica produce mejor rendimiento global?”  
2. “¿Cuánto de esa mejora puede atribuirse al aumento del tamaño del train?”  
La literatura responde bien a la primera, pero la segunda requiere diseños adicionales de control o de sensibilidad. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0020025519306838)

## Cómo justificar la limitación

En la sección de Metodología puedes escribir que tu objetivo no es aislar un efecto causal puro del tamaño muestral, sino comparar el rendimiento de pipelines de balanceo en condiciones realistas de entrenamiento, manteniendo intactos validation y test. También puedes indicar explícitamente que algunas técnicas modifican el tamaño efectivo del train y que eso constituye una propiedad intrínseca del método, no un error de implementación. [imbalanced-learn](https://imbalanced-learn.org/stable/user_guide.html)

En la Discusión, la formulación más sólida es reconocer que el efecto observado de oversampling y SMOTE refleja la combinación de: redistribución de clases, incremento del número de instancias de entrenamiento y, en el caso de SMOTE, interpolación sintética. Añade que, por limitaciones de tiempo, no se realizó un diseño factorial completo para descomponer esos mecanismos, por lo que las conclusiones deben interpretarse como comparaciones de **pipeline** y no como atribución causal aislada. Esa redacción es científicamente honesta y está alineada con cómo se presentan habitualmente estos estudios. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S1743919121000406)

## Amenazas a la validez interna

Las amenazas más importantes que deberías mencionar son:  
- **Confusión entre balanceo y tamaño del train** en oversampling y SMOTE, porque cambian dos factores a la vez. [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/10.1002/sam.11538)
- **Dependencia del clasificador**, ya que el efecto del resampling varía con el modelo usado. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/34867255/)
- **Varianza por fold y semilla**, especialmente con datasets médicos pequeños o desbalanceados. [bmj](https://www.bmj.com/content/372/bmj.n71)
- **Fuga de información** si cualquier forma de oversampling se aplicara antes de separar train/validation/test; tu protocolo correcto evita esto. [openreview](https://openreview.net/pdf?id=CF7FzuUkUck)
- **Sesgo por métrica**, si solo se reporta accuracy en lugar de métricas sensibles a la clase minoritaria. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/34867255/)
- **No independencia entre pacientes o ventanas** en CGM si hubiera múltiples observaciones por paciente, lo que exige particionado patient-wise. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/35500032/)

## Qué diría un revisor

Un revisor razonable probablemente diría algo como esto: “El estudio compara de forma válida varios métodos de resampling, pero la interpretación causal de las mejoras en oversampling/SMOTE debe moderarse porque el procedimiento aumenta el número de muestras de entrenamiento; por tanto, parte del efecto puede deberse al mayor tamaño y no solo al reequilibrio de clases.” También podría pedir un análisis de sensibilidad con sample weighting, una comparación por familias o al menos una discusión explícita de esta limitación. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0020025519306838)

## Recomendaciones prácticas

- Mantén la comparación principal como pipeline completo, pero no la presentes como aislamiento causal puro del balanceo. [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/10.1002/sam.11538)
- Reporta siempre el tamaño final del train por técnica y fold. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0020025519306838)
- Separa resultados por familias de métodos.
- Añade, si es posible, un baseline cost-sensitive o con class weights. [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/10.1002/sam.11538)
- Usa métricas robustas al desbalance.
- Asegura particionado patient-wise si hay múltiples ventanas por paciente. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/35500032/)
- Aplica resampling solo dentro del train de cada fold. [imbalanced-learn](https://imbalanced-learn.org/stable/user_guide.html)
- Declara explícitamente que no realizas un diseño factorial para separar “más datos” de “balanceo”.

## Referencias clave

- Chawla, N. V., et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*.  
- He, H., & Garcia, E. A. (2009). Learning from imbalanced data.  
- Branco, P., Torgo, L., & Ribeiro, R. P. (2016/2017). Survey sobre *imbalanced learning*.  
- Buda, M., et al. (2018). Systematic study of class imbalance in CNNs.  
- Fernández, A., et al. (2018). Learning from imbalanced data sets.  
- Lemaître, G., et al. (2017). *Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning*.  
- Page, M. J., et al. (2021). The PRISMA 2020 statement. [prisma-statement](https://www.prisma-statement.org/prisma-2020-statement)

## Conclusión

La metodología más defendible científicamente para tu TFM es comparar cada técnica como parte de un **pipeline completo**, con resampling solo en train, validation/test intactos y métricas sensibles al desbalance, pero dejando claro que en oversampling y SMOTE la mejora observada no puede atribuirse exclusivamente al balanceo porque el tamaño efectivo del entrenamiento también cambia. Si puedes añadir un análisis complementario por familias o un baseline con ponderación de clases, tu discusión será mucho más robusta; si no, la limitación sigue siendo aceptable siempre que la formules como una amenaza a la validez interna y no como un fallo del estudio. [bmj](https://www.bmj.com/content/372/bmj.n71)