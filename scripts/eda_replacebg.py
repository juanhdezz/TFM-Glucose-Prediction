# -*- coding: utf-8 -*-
"""EDA para el dataset REPLACE-BG (archivo original filtrado)."""
from pathlib import Path
from eda_common import run_eda

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_ROOT = PROJECT_ROOT / "data"
    OUTPUT_ROOT = PROJECT_ROOT / "results"

    run_eda("REPLACE-BG", DATA_ROOT, OUTPUT_ROOT)