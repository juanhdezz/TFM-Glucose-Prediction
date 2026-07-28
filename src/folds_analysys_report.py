"""
folds_analysis_report.py
========================
Genera informes de distribución demográfica del balanceo para la memoria del TFM.

Para cada combinación (dataset × técnica × dimensión) genera un fichero .txt
con el análisis de todos los folds disponibles:
  1. Tamaño del conjunto de entrenamiento (original vs. balanceado)
  2. Distribución de la variable balanceada (por grupo, con n y %)

Uso:
    python folds_analysis_report.py

Salida:
    ROOT_DIR/folds_analysis/<dataset>_<dim>_<tecnica>.txt   (60 ficheros)
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "input"
OUT_DIR  = ROOT_DIR / "folds_analysis"

DATASETS = {
    "DIATREND":          DATA_DIR / "DIATREND",
    "REPLACE-BG":        DATA_DIR / "REPLACE-BG",
    "T1DiabetesGranada": DATA_DIR / "T1DiabetesGranada",
}

# ── Constantes ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [f"x{i}" for i in range(8)] + ["y"]
KEY_COLS     = FEATURE_COLS + ["patient_id"]

AGE_BINS   = [-np.inf, 30, 45, 65, np.inf]
AGE_LABELS = ["<31", "31-45", "46-65", ">=66"]

TECHNIQUES = [
    "undersampling",
    "oversampling",
    "smote",
    "tomek_links",
    "patient_aware_undersampling",
    "jittering",
    "undersampling_oversampling",
    "undersampling_smote",
    "smote_tomek",
    "oversampling_tomek",
]

DIMENSIONS = ["sex", "age"]

# Nombre legible para técnicas en el encabezado del fichero
TECHNIQUE_LABELS = {
    "undersampling":              "RUS (Random Undersampling)",
    "oversampling":               "ROS (Random Oversampling)",
    "smote":                      "SMOTE",
    "tomek_links":                "Tomek Links",
    "patient_aware_undersampling":"Patient-Aware Undersampling",
    "jittering":                  "Jittering",
    "undersampling_oversampling": "Undersampling + Oversampling (RUS+ROS)",
    "undersampling_smote":        "Undersampling + SMOTE",
    "smote_tomek":                "SMOTE + Tomek",
    "oversampling_tomek":         "Oversampling + Tomek",
    "original":                   "Sin balanceo (original)",
}

# Patrón del nombre de los ficheros balanceados
BAL_PATTERN = re.compile(
    r".*_fold(?P<fold>\d+)_(?P<group>age|sex)_(?P<technique>.+)\.parquet$"
)

N_FOLDS = 5


# ══════════════════════════════════════════════════════════════════════════════
# Helpers de carga  (reutilizados del notebook de validación)
# ══════════════════════════════════════════════════════════════════════════════

def load_original(dataset_dir: Path) -> tuple[pd.DataFrame, str]:
    """Carga el fichero parquet original del dataset."""
    candidates = sorted(dataset_dir.glob("windows_with_5folds_*.parquet"))
    # Excluir ficheros que pertenecen a balanced_outputs
    candidates = [c for c in candidates if "balanced_outputs" not in str(c)]
    if not candidates:
        raise FileNotFoundError(f"No se encontró archivo original en {dataset_dir}")
    df = pd.read_parquet(candidates[0])
    return df, candidates[0].name


def get_patient_info(dataset_dir: Path) -> pd.DataFrame:
    """Carga el fichero de información de pacientes."""
    for cand in ["Patient_info.parquet", "patient_info.parquet",
                 "Patient_info.csv",     "patient_info.csv"]:
        p = dataset_dir / cand
        if p.exists():
            return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"No se encontró patient_info en {dataset_dir}")


def get_original_split(df_orig: pd.DataFrame, fold_idx: int, split: str) -> pd.DataFrame:
    """Extrae el subconjunto (train/val/test) de un fold concreto."""
    col = f"fold_{fold_idx}"
    return df_orig[df_orig[col].str.lower() == split].reset_index(drop=True)


def find_balanced_file(dataset_dir: Path, fold_idx: int,
                       group: str, technique: str) -> Path | None:
    """Busca el fichero balanceado correspondiente a (fold, grupo, técnica)."""
    bal_dir = dataset_dir / "balanced_outputs"
    if not bal_dir.exists():
        return None
    pattern = f"*_fold{fold_idx}_{group}_{technique}.parquet"
    matches = sorted(bal_dir.glob(pattern))
    return matches[0] if matches else None


# ══════════════════════════════════════════════════════════════════════════════
# Helpers demográficos  (reutilizados del notebook de validación)
# ══════════════════════════════════════════════════════════════════════════════

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Busca una columna por nombre, insensible a mayúsculas y espacios."""
    norm = {c.strip().lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower().replace(" ", "_")
        if key in norm:
            return norm[key]
    raise KeyError(f"Ninguna de estas columnas encontrada: {candidates}")


def add_age_group(df: pd.DataFrame, pid_col: str,
                  pat_info: pd.DataFrame) -> pd.DataFrame:
    """Añade columna age_group al dataframe."""
    pi = pat_info.copy()
    pid_c = _find_col(pi, ["patient_id", "Patient_ID", "patientid"])
    pi["_pid"] = pi[pid_c].astype(str).str.strip()

    try:
        age_c = _find_col(pi, ["Age", "age"])
        pi["_age"] = pd.to_numeric(pi[age_c], errors="coerce")
    except KeyError:
        birth_c = _find_col(pi, ["Birth_year", "birth_year"])
        pi["_age"] = 2026 - pd.to_numeric(pi[birth_c], errors="coerce")

    pi["_age_group"] = pd.cut(
        pi["_age"], bins=AGE_BINS, labels=AGE_LABELS,
        include_lowest=True, right=True
    ).astype(str)

    lookup = pi.set_index("_pid")["_age_group"].to_dict()
    out = df.copy()
    out["age_group"] = df[pid_col].astype(str).str.strip().map(lookup).fillna("Unknown")
    return out


def add_sex_group(df: pd.DataFrame, pid_col: str,
                  pat_info: pd.DataFrame) -> pd.DataFrame:
    """Añade columna sex_group al dataframe."""
    pi = pat_info.copy()
    pid_c = _find_col(pi, ["patient_id", "Patient_ID", "patientid"])
    pi["_pid"] = pi[pid_c].astype(str).str.strip()
    sex_c = _find_col(pi, ["Sex", "sex"])
    lookup = pi.set_index("_pid")[sex_c].str.upper().to_dict()
    out = df.copy()
    out["sex_group"] = df[pid_col].astype(str).str.strip().map(lookup).fillna("Unknown")
    return out


def add_demo_group(df: pd.DataFrame, pid_col: str,
                   pat_info: pd.DataFrame, group: str) -> pd.DataFrame:
    """Añade la columna de grupo demográfico según la dimensión solicitada."""
    return (add_age_group(df, pid_col, pat_info) if group == "age"
            else add_sex_group(df, pid_col, pat_info))


def group_col_name(group: str) -> str:
    return f"{group}_group"


# ══════════════════════════════════════════════════════════════════════════════
# Estadísticos  (reutilizados del notebook de validación)
# ══════════════════════════════════════════════════════════════════════════════

def describe_distribution(df: pd.DataFrame, gcol: str) -> pd.DataFrame:
    """
    Devuelve DataFrame con columnas [n, %] indexado por grupo.
    """
    counts = df[gcol].value_counts(dropna=False).sort_index()
    pct    = (100 * counts / counts.sum()).round(1)
    return pd.DataFrame({"n": counts, "%": pct})


# ══════════════════════════════════════════════════════════════════════════════
# Formateo de texto
# ══════════════════════════════════════════════════════════════════════════════

def fmt_size_section(orig_n: int, bal_n: int) -> str:
    """Formatea el apartado 1: Tamaño del conjunto de entrenamiento."""
    delta     = bal_n - orig_n
    delta_str = f"{delta:+,}"
    if orig_n > 0:
        delta_pct = 100 * delta / orig_n
        pct_str   = f"({delta_pct:+.2f}%)"
    else:
        pct_str   = ""

    lines = [
        "  1. TAMAÑO DEL CONJUNTO DE ENTRENAMIENTO",
        f"     Original  : {orig_n:>12,}",
        f"     Balanceado: {bal_n:>12,}",
        f"     Diferencia: {delta_str:>12}  {pct_str}",
    ]
    return "\n".join(lines)


def fmt_distribution_section(orig_df: pd.DataFrame, bal_df: pd.DataFrame,
                              gcol: str) -> str:
    """
    Formatea el apartado 2: Distribución de la variable balanceada.
    Genera una tabla alineada con columnas Original y Balanceado.
    """
    dist_orig = describe_distribution(orig_df, gcol)
    dist_bal  = describe_distribution(bal_df,  gcol)

    # Unión de grupos presentes en cualquiera de los dos
    all_groups = sorted(
        set(dist_orig.index.tolist()) | set(dist_bal.index.tolist())
    )

    col_group = gcol.upper()
    header = (
        f"  2. DISTRIBUCIÓN POR {col_group} EN TRAIN\n"
        f"     {'Grupo':<12}  {'Original n':>12}  {'Original %':>10}  "
        f"{'Balanceado n':>14}  {'Balanceado %':>13}"
    )
    sep = "     " + "-" * 68

    rows = [header, sep]
    for g in all_groups:
        o_n   = int(dist_orig["n"].get(g, 0))
        o_pct = float(dist_orig["%"].get(g, 0.0))
        b_n   = int(dist_bal["n"].get(g, 0))
        b_pct = float(dist_bal["%"].get(g, 0.0))
        rows.append(
            f"     {g:<12}  {o_n:>12,}  {o_pct:>9.1f}%  "
            f"{b_n:>14,}  {b_pct:>12.1f}%"
        )

    return "\n".join(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Análisis por fold
# ══════════════════════════════════════════════════════════════════════════════

def analyse_fold_balanced(ds_name: str, dataset_dir: Path,
                          df_orig: pd.DataFrame, pat_info: pd.DataFrame,
                          fold_idx: int, group: str, technique: str) -> str | None:
    """
    Analiza un fold balanceado concreto.
    Devuelve el texto formateado o None si el fichero no existe.
    """
    bal_file = find_balanced_file(dataset_dir, fold_idx, group, technique)
    if bal_file is None:
        return None

    df_bal    = pd.read_parquet(bal_file)
    orig_train = get_original_split(df_orig, fold_idx, "train")
    bal_train  = df_bal[df_bal["split"] == "train"].reset_index(drop=True)

    orig_train_g = add_demo_group(orig_train, "patient_id", pat_info, group)
    bal_train_g  = add_demo_group(bal_train,  "patient_id", pat_info, group)

    gcol = group_col_name(group)

    blocks = [
        fmt_size_section(len(orig_train), len(bal_train)),
        "",
        fmt_distribution_section(orig_train_g, bal_train_g, gcol),
    ]
    return "\n".join(blocks)


def analyse_fold_original(df_orig: pd.DataFrame, pat_info: pd.DataFrame,
                          fold_idx: int, group: str) -> str:
    """
    Genera el bloque de análisis para el dataset original (sin balanceo).
    Muestra únicamente distribución del train de ese fold.
    """
    orig_train   = get_original_split(df_orig, fold_idx, "train")
    orig_train_g = add_demo_group(orig_train, "patient_id", pat_info, group)
    gcol         = group_col_name(group)

    n = len(orig_train)
    lines = [
        "  1. TAMAÑO DEL CONJUNTO DE ENTRENAMIENTO",
        f"     Original  : {n:>12,}",
        "",
        fmt_distribution_section(orig_train_g, orig_train_g, gcol),
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Generación de ficheros
# ══════════════════════════════════════════════════════════════════════════════

def file_stem(ds_name: str, group: str, technique: str) -> str:
    """Construye el nombre del fichero de salida sin extensión."""
    ds_slug  = ds_name.lower().replace("-", "_")
    tec_slug = technique.replace("-", "_")
    return f"{ds_slug}_{group}_{tec_slug}"


def build_header(ds_name: str, technique: str, group: str) -> str:
    """Construye el encabezado del fichero."""
    width = 60
    sep   = "=" * width
    tec_label = TECHNIQUE_LABELS.get(technique, technique)
    dim_label = "EDAD" if group == "age" else "SEXO"
    lines = [
        sep,
        f"  Dataset  : {ds_name}",
        f"  Técnica  : {tec_label}",
        f"  Dimensión: {dim_label}",
        sep,
    ]
    return "\n".join(lines)


def generate_file_balanced(ds_name: str, dataset_dir: Path,
                            df_orig: pd.DataFrame, pat_info: pd.DataFrame,
                            group: str, technique: str) -> int:
    """
    Genera el fichero .txt para una combinación (dataset, técnica, dimensión).
    Itera sobre todos los folds disponibles.
    Devuelve el número de folds encontrados.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem     = file_stem(ds_name, group, technique)
    out_path = OUT_DIR / f"{stem}.txt"

    header   = build_header(ds_name, technique, group)
    fold_sep = "-" * 60
    sections = [header]

    folds_found = 0
    for fold_idx in range(N_FOLDS):
        fold_text = analyse_fold_balanced(
            ds_name, dataset_dir, df_orig, pat_info, fold_idx, group, technique
        )
        if fold_text is None:
            continue
        folds_found += 1
        sections.append(f"\nFOLD {fold_idx}")
        sections.append(fold_text)
        sections.append(fold_sep)

    if folds_found == 0:
        # No hay ficheros para esta combinación — no se crea el .txt
        return 0

    out_path.write_text("\n".join(sections), encoding="utf-8")
    return folds_found


def generate_file_original(ds_name: str,
                            df_orig: pd.DataFrame, pat_info: pd.DataFrame,
                            group: str) -> None:
    """
    Genera el fichero .txt para el dataset original (sin balanceo),
    para ambas dimensiones (sex y age), con todos los folds.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem     = file_stem(ds_name, group, "original")
    out_path = OUT_DIR / f"{stem}.txt"

    header   = build_header(ds_name, "original", group)
    fold_sep = "-" * 60
    sections = [header]

    for fold_idx in range(N_FOLDS):
        fold_text = analyse_fold_original(df_orig, pat_info, fold_idx, group)
        sections.append(f"\nFOLD {fold_idx}")
        sections.append(fold_text)
        sections.append(fold_sep)

    out_path.write_text("\n".join(sections), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# Punto de entrada
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Carpeta de salida: {OUT_DIR}\n")

    total_files    = 0
    skipped_files  = 0

    for ds_name, dataset_dir in DATASETS.items():
        print(f"── {ds_name}")

        # Carga única de original y patient_info por dataset
        try:
            df_orig, orig_fname = load_original(dataset_dir)
        except FileNotFoundError as e:
            print(f"   ⚠️  {e}")
            continue

        try:
            pat_info = get_patient_info(dataset_dir)
        except FileNotFoundError as e:
            print(f"   ⚠️  {e}")
            continue

        print(f"   Original: {orig_fname}  ({len(df_orig):,} filas)")

        # ── Ficheros "original" (sin balanceo, uno por dimensión) ─────────────
        for group in DIMENSIONS:
            generate_file_original(ds_name, df_orig, pat_info, group)
            stem = file_stem(ds_name, group, "original")
            print(f"   ✅ {stem}.txt  (original, {N_FOLDS} folds)")
            total_files += 1

        # ── Ficheros balanceados (técnica × dimensión) ────────────────────────
        for technique in TECHNIQUES:
            for group in DIMENSIONS:
                n_folds = generate_file_balanced(
                    ds_name, dataset_dir, df_orig, pat_info, group, technique
                )
                stem = file_stem(ds_name, group, technique)
                if n_folds > 0:
                    print(f"   ✅ {stem}.txt  ({n_folds} folds)")
                    total_files += 1
                else:
                    print(f"   ·  {stem}  (sin ficheros balanceados — omitido)")
                    skipped_files += 1

        print()

    print(f"{'─'*60}")
    print(f"Ficheros generados : {total_files}")
    print(f"Combinaciones omitidas (sin datos): {skipped_files}")
    print(f"Salida en: {OUT_DIR}")


if __name__ == "__main__":
    main()