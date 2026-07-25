# -*- coding: utf-8 -*-
"""EDA para el dataset DIATREND (archivo original filtrado)."""
from pathlib import Path
from eda_common import run_eda

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]   # raíz del proyecto
    DATA_ROOT = PROJECT_ROOT / "data"                    # donde están los datasets
    OUTPUT_ROOT = PROJECT_ROOT / "results"               # donde se guardarán los resultados EDA

    run_eda("DIATREND", DATA_ROOT, OUTPUT_ROOT)