
# 🏗️ Pipeline de Datos

## Visión General

    ┌─────────────────┐
    │   Raw Data      │
    │  (CGM + Demos)  │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ 1-generate_windows│
    │  (Ventanas HW=8) │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ 2-folds_creation │
    │   (5-fold CV)    │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │2b-trainset_     │
    │  balancing      │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │2c-compact_folds │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │   3-train.py    │
    │   (LSTM Model)  │
    └─────────────────┘

---

## Restricciones Metodológicas

1. **Balanceo solo sobre train:** No se tocan val ni test.
2. **No modificar el modelo:** Solo se transforman los datos de entrada.
3. **Patient-wise CV:** Un paciente no puede aparecer en train y test simultáneamente.
4. **Semilla fija:** Reproducibilidad de todos los procesos aleatorios.

---

## Detalle de Cada Paso

### 1-generate_windows.py
- **Entrada:** Series de glucosa CGM.
- **Salida:** Ventanas de 8 mediciones con objetivo a PH4.
- **Característica:** Se generan ventanas por paciente respetando la secuencia.

### 2-folds_creation.py
- **Entrada:** Ventanas generadas.
- **Salida:** Asignación de fold (train/val/test) para cada ventana.
- **Característica:** 5 folds estratificados por paciente.

### 2b-trainset_balancing.py
- **Entrada:** Archivo con folds, más datos demográficos (Patient_info).
- **Salida:** Archivos por fold, con el train balanceado según técnica.
- **Característica:** Val y test se copian sin modificar.

### 2c-compact_folds.py
- **Entrada:** Archivos por fold (balanceados).
- **Salida:** Un único archivo por combinación (dataset, demographic, technique).
- **Característica:** Se verifica que val y test sean idénticos a los originales.

### 3-train.py
- **Entrada:** Archivo compacto balanceado.
- **Salida:** Modelo LSTM entrenado y métricas.
- **Característica:** El script no sabe que los datos fueron balanceados.

---

## Enlaces
- [[MOC_Scripts]]
- [[Dashboard]]
