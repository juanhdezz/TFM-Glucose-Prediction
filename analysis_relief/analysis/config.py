# -*- coding: utf-8 -*-
"""config.py — Configuración central RELIEF-T1D."""
from pathlib import Path

# ---------------------------------------------------------------------------
# RUTAS — AJUSTA OUTPUT_ROOT
# ---------------------------------------------------------------------------
OUTPUT_ROOT = Path(r"C:\Users\User\Desktop\TFM-Glucose-Prediction\results")

ANALYSIS_OUT   = Path(__file__).parent.parent / "outputs"
FIGURES_DIR    = ANALYSIS_OUT / "figures"
TABLES_DIR     = ANALYSIS_OUT / "tables"
STATS_DIR      = ANALYSIS_OUT / "stats"
DASHBOARDS_DIR = ANALYSIS_OUT / "dashboards"

for _d in [FIGURES_DIR, TABLES_DIR, STATS_DIR, DASHBOARDS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# DATASETS
# ---------------------------------------------------------------------------
DATASETS      = ["DIATREND", "REPLACE-BG", "T1DiabetesGranada"]
DATASET_ORDER = ["T1DiabetesGranada", "DIATREND", "REPLACE-BG"]
DATASET_LABELS = {
    "DIATREND":          "DiaTrend",
    "REPLACE-BG":        "ReplaceBG",
    "T1DiabetesGranada": "T1DiabetesGranada",
}

# ---------------------------------------------------------------------------
# DIMENSIONES
# ---------------------------------------------------------------------------
DIMENSIONS       = ["age", "sex"]
DIMENSION_LABELS = {"age": "Age", "sex": "Sex", "original": "—"}

# ---------------------------------------------------------------------------
# TÉCNICAS — nombres exactos de los directorios reales
# ---------------------------------------------------------------------------
TECHNIQUES = [
    "jittering",
    "oversampling",
    "patient_aware_undersampling",
    "tomek_links",
    "undersampling_oversampling",
    "undersampling_smote",
    "smote_tomek",
    "smote",
    "undersampling",
    "oversampling_tomek",

]

TECHNIQUE_LABELS = {
    "original":                    "Original",
    "jittering":                   "Jittering",
    "oversampling":                "Oversampling",
    "patient_aware_undersampling": "PAUndersampling",
    "smote":                       "SMOTE",
    "undersampling":               "Undersampling",
    "tomek_links":                 "Tomek Links",
    "undersampling_oversampling":  "Undersampling + Oversampling",
    "undersampling_smote":         "Undersampling + SMOTE",
    "oversampling_tomek":          "Oversampling + Tomek Links",

}

# Orden canónico para todos los plots (original siempre primero)
TECHNIQUE_ORDER = [
    "original",
    "jittering",
    "oversampling",
    "patient_aware_undersampling",
    "smote",
    "undersampling",
    "tomek_links",
    "undersampling_oversampling",
    "undersampling_smote",
    "smote_tomek",
    "oversampling_tomek",
]

# Clasificación 1: efecto sobre el tamaño del conjunto de entrenamiento
FAMILY_SIZE = {
    'original':                    'baseline',
    'oversampling':                'oversampling',
    'smote':                       'oversampling',
    'jittering':                   'oversampling',
    'undersampling':               'undersampling',
    'tomek_links':                 'undersampling',
    'patient_aware_undersampling': 'undersampling',
    'undersampling_oversampling':  'same_size',
    'undersampling_smote':         'same_size',
    'smote_tomek':                 'same_size',
    'oversampling_tomek':          'oversampling',
}
 
# Clasificación 2: mecanismo de selección/generación de muestras
FAMILY_MECHANISM = {
    'original':                    'baseline',
    'oversampling':                'random',
    'undersampling':               'random',
    'undersampling_oversampling':  'random',
    'smote':                       'guided',
    'undersampling_smote':         'random+guided',
    'smote_tomek':                 'guided',
    'jittering':                   'guided',
    'tomek_links':                 'guided',
    'oversampling_tomek':          'random+guided',
    'patient_aware_undersampling': 'guided',
}

# Etiquetas legibles para familias
FAMILY_SIZE_LABELS = {
    'baseline':     'Baseline',
    'oversampling': 'Oversampling',
    'hybrid':       'Híbrido',
    'undersampling':'Undersampling',
}
FAMILY_MECHANISM_LABELS = {
    'baseline':      'Baseline',
    'random':        'Aleatorio',
    'guided':        'Guiado',
    'random+guided': 'Aleatorio + Guiado',}

# ---------------------------------------------------------------------------
# RANGOS GLUCÉMICOS — nomenclatura real de los CSVs
# ---------------------------------------------------------------------------
RANGE_INFO = {
    "ENTIRE": {"label": "All Ranges",    "mgdl": "full range",   "priority": 3},
    "TBR_2":  {"label": "Hypo L2",       "mgdl": "<54 mg/dL",    "priority": 1},
    "TBR_1":  {"label": "Hypo L1",       "mgdl": "54–69 mg/dL",  "priority": 2},
    "TIR":    {"label": "In Range",      "mgdl": "70–180 mg/dL", "priority": 4},
    "TAR_1":  {"label": "Hyper L1",      "mgdl": "181–250 mg/dL","priority": 5},
    "TAR_2":  {"label": "Hyper L2",      "mgdl": ">250 mg/dL",   "priority": 6},
}
RANGE_ORDER  = ["TBR_2", "TBR_1", "TIR", "TAR_1", "TAR_2", "ENTIRE"]
RANGE_LABELS = {k: v["label"] for k, v in RANGE_INFO.items()}

# ---------------------------------------------------------------------------
# MÉTRICAS — columnas reales de los CSVs
# ---------------------------------------------------------------------------
ALL_METRIC_COLS  = ["A", "B", "C", "D", "E", "A + B", "RMSE", "MSE", "MAE", "MAPE"]
ERROR_METRICS    = ["RMSE", "MAE", "MAPE", "MSE"]
CEG_METRICS      = ["A", "B", "C", "D", "E", "A + B"]
ANALYSIS_METRICS = ["RMSE", "MAE"]          # principales para análisis comparativo
PRIMARY_METRIC   = "RMSE"
LOWER_IS_BETTER  = ["RMSE", "MAE", "MAPE", "MSE", "C", "D", "E"]
HIGHER_IS_BETTER_CEG = ["A", "A + B"]

N_FOLDS = 5
ALPHA   = 0.05

# ---------------------------------------------------------------------------
# PALETA VISUAL — Wong (2011), colorblind-safe, 7 colores + negro
# ---------------------------------------------------------------------------
PALETTE = {
    "original":                    "#000000",   # negro
    "jittering":                   "#009E73",   # verde
    "oversampling":                "#E69F00",   # naranja
    "patient_aware_undersampling": "#0072B2",   # azul
    "reference_proportional":      "#CC79A7",   # rosa/morado
    "smote":                       "#D55E00",   # rojo-naranja
    "undersampling":               "#56B4E9",   # azul cielo
    "tomek_links":                 "#F0E442",   # amarillo
    "undersampling_oversampling":  "#BC8F8F",   # caqui
    "undersampling_smote":         "#984EA3",   # morado
    "smote_tomek":                 "#FF7F00",   # naranja oscuro
    "oversampling_tomek":          "#999999",   # gris
}

DIMENSION_PALETTE = {
    "age":      "#0072B2",
    "sex":      "#D55E00",
    "original": "#000000",
}

DATASET_PALETTE = {
    "DIATREND":          "#009E73",
    "REPLACE-BG":        "#E69F00",
    "T1DiabetesGranada": "#CC79A7",
}

# ---------------------------------------------------------------------------
# ESTILO MATPLOTLIB
# ---------------------------------------------------------------------------
MPL_RC = {
    "font.family":           "DejaVu Sans",
    "font.size":             11,
    "axes.titlesize":        13,
    "axes.titleweight":      "bold",
    "axes.labelsize":        11,
    "xtick.labelsize":       10,
    "ytick.labelsize":       10,
    "legend.fontsize":        9,
    "legend.framealpha":     0.9,
    "legend.edgecolor":      "0.8",
    "figure.dpi":            150,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.15,
    "axes.grid":             True,
    "grid.alpha":            0.3,
    "grid.linestyle":        "--",
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.axisbelow":        True,
}