# RELIEF-T1D — Sistema de Análisis Experimental

Análisis completo de los resultados de 39 experimentos LSTM
sobre 3 datasets de Diabetes Tipo 1, con 6 técnicas de balanceo
aplicadas sobre 2 dimensiones demográficas (Age / Sex).

## Estructura

```
analysis/           Código fuente del sistema de análisis
outputs/            Figuras, tablas y estadísticos generados
main.py             Punto de entrada principal
setup_analysis_project.py   Este script de inicialización
```

## Uso rápido

```bash
# 1. Configurar rutas en analysis/config.py
# 2. Ejecutar pipeline completo
python main.py

# Ejecutar solo carga y preprocesado
python main.py --only load

# Ejecutar solo visualizaciones
python main.py --only viz

# Ejecutar solo estadísticos
python main.py --only stats

# Ejecutar solo tablas
python main.py --only tables
```

## Requisitos

```bash
pip install pandas numpy scipy scikit-posthocs matplotlib seaborn
pip install ptitprince  # raincloud plots
```
