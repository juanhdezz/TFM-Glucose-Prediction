
# 🗺️ Roadmap del TFM

## Fase 1: Preparación de Datos
- [x] Curación de datos de glucemia
- [x] Curación de datos demográficos (sexo, edad)
- [x] Generación de ventanas deslizantes (HW=8, PH=4)
- [x] Creación de 5-fold cross-validation

## Fase 2: Balanceo Demográfico
- [x] Implementación de [[Undersampling]]
- [x] Implementación de [[Oversampling]]
- [x] Implementación de [[Patient-aware Undersampling]]
- [x] Implementación de [[SMOTE]]
- [x] Implementación de [[Jittering]]
- [x] Generación de archivos balanceados por fold

## Fase 3: Entrenamiento y Evaluación
- [ ] Entrenamiento de modelo LSTM baseline
- [ ] Entrenamiento con balanceo por sexo
- [ ] Entrenamiento con balanceo por edad
- [ ] Evaluación de disparidad por subgrupo

## Fase 4: Análisis y Memoria
- [ ] Análisis de resultados
- [ ] Redacción de capítulo de resultados
- [ ] Redacción de conclusiones

---

## 📅 Cronograma Estimado

| Fase | Duración | Estado |
|------|----------|--------|
| Preparación de Datos | 2 semanas | ✅ Completado |
| Balanceo Demográfico | 2 semanas | ✅ Completado |
| Entrenamiento | 3 semanas | ⏳ En progreso |
| Análisis y Memoria | 3 semanas | ⏳ Pendiente |

---

## 🎯 Próximos Pasos

1. Ejecutar entrenamiento para todas las combinaciones
2. Comparar métricas entre experimentos
3. Analizar disparidad por subgrupo demográfico
4. Documentar resultados en la memoria
