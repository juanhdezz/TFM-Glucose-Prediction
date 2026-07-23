## Justificación de la clasificación de Oversampling + Tomek Links en la familia `oversampling`

### 1. Descripción del algoritmo implementado

La técnica **Oversampling + Tomek Links** implementada en este trabajo sigue el siguiente procedimiento:

1. **Tomek Links inicial**: se aplica el algoritmo de Tomek Links sobre el conjunto de entrenamiento completo (todos los grupos demográficos). Este paso elimina pares de puntos que son vecinos más cercanos mutuos y pertenecen a clases distintas, actuando como una limpieza de la frontera de decisión.

2. **Cálculo de targets equitativos**: se determina cuántas muestras debería tener cada grupo demográfico para alcanzar una proporción equitativa (50% para dos grupos, 25% para cuatro grupos), usando como referencia el tamaño total del conjunto de entrenamiento original.

3. **ROS compensatorio**: para cada grupo que queda por debajo de su target tras la limpieza Tomek, se aplica Random Over-Sampling (ROS) con reemplazo, muestreando del conjunto original, hasta alcanzar el tamaño objetivo.

4. **Reincorporación de Unknown**: las filas con grupo demográfico "Unknown" se mantienen intactas y se reincorporan al final.

### 2. Por qué no garantiza `same_size`

Aunque el paso 2 calcula los targets sobre el tamaño original, el paso 3 **expande activamente los grupos minoritarios** hasta alcanzar dichos targets. Esto implica que el tamaño final del conjunto balanceado puede ser **significativamente mayor** que el original cuando:

- Existen grupos muy minoritarios que requieren una gran cantidad de muestras sintéticas/duplicadas para alcanzar el target equitativo.
- El grupo mayoritario pierde pocas muestras en la limpieza Tomek (o ninguna), por lo que no se reduce.

En consecuencia, el algoritmo **no garantiza** que el tamaño final sea igual al original. El tamaño resultante depende enteramente de la distribución inicial de los grupos demográficos en cada fold.

### 3. Evidencia experimental

Los siguientes resultados, extraídos del dataset **DIATREND**, ilustran el comportamiento de la técnica en dos dimensiones demográficas distintas.

#### Caso A: Dimensión EDAD (4 grupos, muy desbalanceada)

| Fold | Original | Balanceado | Diferencia | Grupos |
|------|----------|------------|------------|--------|
| 0 | 1,598,896 | 2,681,611 | +67.72% | <31 (94.7%), 31-45 (1.3%), 46-65 (3.0%), ≥66 (1.0%) |
| 1 | 1,613,241 | 2,703,199 | +67.56% | <31 (94.6%), 31-45 (1.7%), 46-65 (2.6%), ≥66 (1.0%) |
| 2 | 1,582,201 | 2,667,623 | +68.60% | <31 (95.4%), 31-45 (0.8%), 46-65 (2.8%), ≥66 (1.0%) |
| 3 | 1,617,106 | 2,736,011 | +69.19% | <31 (95.8%), 31-45 (1.5%), 46-65 (1.8%), ≥66 (1.0%) |
| 4 | 1,604,177 | 2,600,731 | +62.12% | <31 (96.7%), 31-45 (1.8%), 46-65 (1.6%) |

**Análisis**: El grupo `<31` concentra ~95% de las muestras. Los tres grupos minoritarios (31-45, 46-65, ≥66) suman apenas ~5%. Al aplicar Tomek Links, el grupo mayoritario pierde muy pocas muestras (los pares Tomek se concentran en la frontera, que es pequeña en proporción al tamaño del grupo). A continuación, el ROS compensatorio expande cada grupo minoritario hasta alcanzar ~400,000 muestras (25% del total cada uno), lo que **triplica** el tamaño de esos grupos y produce un aumento global de ~65%.

**Conclusión**: En dimensiones con grupos muy desbalanceados, la técnica se comporta como **oversampling puro**, con aumentos de tamaño superiores al 60%.

#### Caso B: Dimensión SEXO (2 grupos, moderadamente desbalanceada)

| Fold | Original | Balanceado | Diferencia | Grupos |
|------|----------|------------|------------|--------|
| 0 | 1,598,896 | 2,077,988 | +29.96% | F (85.0%), M (15.0%) |
| 1 | 1,613,241 | 1,613,241 | +0.00% | F (85.0%), M (15.0%) |
| 2 | 1,582,201 | 1,928,638 | +21.90% | F (78.5%), M (21.5%) |
| 3 | 1,617,106 | 1,999,400 | +23.64% | F (79.8%), M (20.2%) |
| 4 | 1,604,177 | 1,604,177 | +0.00% | F (67.7%), M (32.3%) |

**Análisis**: Con dos grupos, el desbalance es menos extremo. En los folds 1 y 4, Tomek Links elimina suficientes muestras del grupo mayoritario (F) como para que, tras el ROS compensatorio del grupo minoritario (M), el tamaño total coincida exactamente con el original. En los folds 0, 2 y 3, Tomek elimina menos muestras del grupo mayoritario, por lo que el ROS del grupo minoritario produce un aumento neto de ~20-30%.

**Conclusión**: En dimensiones con dos grupos, el comportamiento es **variable**: a veces same_size (0%), a veces oversampling moderado (+20-30%).

#### Caso C: Datasets con grupos naturalmente equilibrados

| Dataset | Dimensión | Proporción original | Aumento típico | Comportamiento |
|---------|-----------|--------------------|----------------|----------------|
| REPLACE-BG | SEXO | 56% / 44% | 0% | **Same size** |
| REPLACE-BG | EDAD | 40/30/23/8% | +3% a +9% | **Same size** aproximado |
| T1DiabetesGranada | SEXO | 53% / 47% | 0% | **Same size** |
| T1DiabetesGranada | EDAD | 38/26/23/13% | +3% a +4% | **Same size** aproximado |

**Análisis**: Cuando los grupos están naturalmente más equilibrados (proporciones cercanas al 50% o al 25%), Tomek Links elimina suficientes muestras de todos los grupos como para que el ROS compensatorio apenas aumente el tamaño total. En el límite, si los grupos ya son perfectamente equitativos, Tomek elimina pares en todas las fronteras por igual y el ROS compensatorio solo repone lo eliminado, resultando en 0% de aumento.

### 4. Clasificación adoptada

Dado que:

1. La técnica **no garantiza** mantener el tamaño original en todos los escenarios.
2. En los casos más desbalanceados (que son los más relevantes para el balanceo), el aumento de tamaño es **significativo** (≥60%).
3. El mecanismo principal que modifica el tamaño es el **ROS compensatorio**, que expande los grupos minoritarios.

Se clasifica **Oversampling + Tomek Links** dentro de la familia **`oversampling`**:
FAMILY_SIZE = {
...
'oversampling_tomek': 'oversampling',
...
}


Esta clasificación refleja su comportamiento predominante y permite compararla justamente con otras técnicas de oversampling como ROS, SMOTE y Jittering.

### 5. Implicaciones para el análisis intra-familia

Al agrupar `oversampling_tomek` con el resto de técnicas de oversampling:

- Se compara con técnicas que también **aumentan** el tamaño del conjunto de entrenamiento.
- La comparación es justa porque todas las técnicas de esta familia comparten la característica de expandir los grupos minoritarios.
- La principal diferencia con ROS y SMOTE es que `oversampling_tomek` incluye un paso previo de limpieza de frontera (Tomek Links), lo que puede conferirle una ventaja cualitativa al eliminar muestras ruidosas antes de expandir.

### 6. Comparación con otras técnicas `same_size`

| Técnica | ¿Garantiza mismo tamaño? | ¿Fuerza proporción equitativa? | Mecanismo |
|---------|--------------------------|--------------------------------|-----------|
| `undersampling_oversampling` | ✅ Sí | ✅ Sí (50/50 o 25/25/25/25) | RUS + ROS |
| `undersampling_smote` | ✅ Sí | ✅ Sí (50/50 o 25/25/25/25) | RUS + SMOTE |
| `smote_tomek` | ✅ Sí | ✅ Sí (50/50 o 25/25/25/25) | SMOTE + Tomek + RUS |
| `oversampling_tomek` | ❌ No | ✅ Sí (50/50 o 25/25/25/25) | Tomek + ROS compensatorio |

La diferencia fundamental es que las tres primeras utilizan **RUS explícito** para reducir los grupos mayoritarios al target exacto, garantizando el tamaño original. `oversampling_tomek` delega la reducción en Tomek Links, que es un algoritmo **no paramétrico** y no controlable en cuanto a número de eliminaciones.

### 7. Texto resumen para la memoria

> *"La técnica Oversampling + Tomek Links se clasifica en la familia oversampling debido a que su mecanismo principal de modificación del tamaño es la expansión de los grupos minoritarios mediante ROS. Aunque incorpora un paso previo de limpieza de frontera con Tomek Links, este actúa como un filtro de ruido y no como un mecanismo de undersampling controlado. En dimensiones con grupos muy desbalanceados (como edad en DIATREND, donde el grupo mayoritario concentra el 95% de las muestras), el aumento de tamaño puede superar el 60%, equiparándose a técnicas clásicas de oversampling como ROS o SMOTE. En dimensiones más equilibradas, el aumento es menor o nulo. Esta variabilidad es inherente al algoritmo y debe considerarse al interpretar los resultados comparativos."*