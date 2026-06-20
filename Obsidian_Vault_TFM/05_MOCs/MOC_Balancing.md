
# 🗺️ MOC: Técnicas de Balanceo

## Índice de Técnicas

### Remuestreo
- [[Oversampling]]
- [[Undersampling]]
- [[Patient-aware Undersampling]]

### Generación Sintética
- [[SMOTE]]
- [[Jittering]]

### Basado en Referencia
- [[Reference Proportional]]

---

## Dimensiones Demográficas

### [[Sexo]]
- Balanceo entre grupos `M` y `F`

### [[Edad]]
- Balanceo entre 5 grupos etarios:
  - `<=18`
  - `19-30`
  - `31-45`
  - `46-60`
  - `>60`

---

## Matriz de Experimentos

| Dataset | Técnica | Sexo | Edad |
|---------|---------|------|------|
| DIATREND | Oversampling | ✅ | ✅ |
| DIATREND | Undersampling | ✅ | ✅ |
| DIATREND | Patient-aware Undersampling | ✅ | ✅ |
| DIATREND | SMOTE | ✅ | ✅ |
| DIATREND | Jittering | ✅ | ✅ |
| ... | ... | ... | ... |

---

## Decisiones de Diseño

### ¿Por qué balancear por ventana y no por paciente?
- Un paciente con una serie larga genera más ventanas.
- El modelo se entrena sobre ventanas, no sobre pacientes.
- Balancear por paciente no garantiza balance por ventana.

### ¿Por qué no modificar el modelo?
- Restricción metodológica: el balanceo debe ser un preprocesado.
- Permite comparar directamente el efecto del balanceo.

---

## Enlaces
- [[Dashboard]]
- [[Pregunta_Investigacion]]
