import pandas as pd
import numpy as np

files = {
    "DIATREND": "data/DIATREND/Glucose_measurements_DiaTrend_FILTERED_2026-03-27.parquet",
    "REPLACE-BG": "data/REPLACE-BG/Glucose_measurements_REPLACE-BG_FILTERED_2026-03-27.parquet",
    "T1DiabetesGranada": "data/T1DiabetesGranada/Glucose_measurements_T1DiabetesGranada_FILTERED_2026-03-27.parquet"
}

for name, path in files.items():
    print(f"\n{'='*20} {name} {'='*20}")
    try:
        df = pd.read_parquet(path)
        print(f"Shape: {df.shape}")
        print("Columns & Types:")
        print(df.dtypes)
        print("\nNull counts:")
        print(df.isnull().sum())
        print("\nHead:")
        print(df.head(3))
        print("\nUnique patients:", df['Patient_ID'].nunique())
        
        # Check target stats
        print("\nMeasurement description:")
        print(df['Measurement'].describe())
        
    except Exception as e:
        print(f"Error loading {name}: {e}")
