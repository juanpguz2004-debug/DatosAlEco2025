# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="Evaluación Calidad - Guía MinTIC 2025 (0-10 internals)", layout="wide")
GUIDE_PDF_PATH = "/mnt/data/Guía de Calidad e Interoperabilidad 2025 (1).pdf"

# -----------------------
# HELPERS
# -----------------------
def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0

def to_percent(score_0_10: float) -> float:
    return float(np.clip(score_0_10 * 10.0, 0.0, 100.0))

def parse_iso(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        # Accept "2025-11-20T19:13:18Z" and other ISO forms
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        try:
            return pd.to_datetime(dt_str, utc=True).to_pydatetime()
        except Exception:
            return None

def freq_to_days(freq_str: Optional[str]) -> int:
    """
    Flexible interpretation of frequency (option A).
    """
    if not freq_str:
        return 180  # default if unknown
    s = freq_str.strip().lower()
    if any(x in s for x in ["diaria", "daily"]):
        return 1
    if any(x in s for x in ["semanal", "weekly"]):
        return 7
    if any(x in s for x in ["mensual", "monthly"]):
        return 30
    if any(x in s for x in ["trimestral", "quarter", "quarterly"]):
        return 90
    if any(x in s for x in ["semestral", "half", "semi"]):
        return 180
    if any(x in s for x in ["anual", "annual", "yearly"]):
        return 365
    # fallback
    return 180

# -----------------------
# DATA & METADATA INGESTION
# -----------------------
@st.cache_data(show_spinner="Cargando datos desde API...")
def fetch_data_resource(dataset_id: str, limit: int = 50000, app_token: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch rows from Socrata resource endpoint.
    endpoint example: https://www.datos.gov.co/resource/{dataset_id}.json
    """
    base = f"https://www.datos.gov.co/resource/{dataset_id}.json"
    params = {"$limit": limit}
    headers = {}
    if app_token:
        headers["X-App-Token"] = app_token
    try:
        r = requests.get(base, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data)
        # replace empty strings with NaN for proper counts
        df.replace("", np.nan, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error cargando filas desde recurso: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner="Cargando metadatos del recurso...")
def fetch_resource_metadata_view(dataset_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch resource metadata from SODA v3 view endpoint:
    https://www.datos.gov.co/api/v3/views/{dataset_id}
    """
    url = f"https://www.datos.gov.co/api/v3/views/{dataset_id}"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        meta = r.json()
        # normalize common fields (safe extraction)
        categoria = meta.get("category") or meta.get("domain") or ""
        tags = meta.get("tags") or []
        descripcion = meta.get("description") or ""
        fecha_meta = meta.get("metadataUpdatedAt") or meta.get("last_metadata_updated_date")
        fecha_datos = meta.get("dataUpdatedAt") or meta.get("last_data_updated_date")
        # custom fields might live under metadata.custom_fields
        custom = meta.get("metadata", {}).get("custom_fields", {}) if isinstance(meta.get("metadata"), dict) else {}
        frecuencia = ""
        licencia = ""
        publisher = ""
        # try to pull from 'Información de Datos' and 'Common Core' if present
        if isinstance(custom, dict):
            info = custom.get("Información de Datos") or {}
            cc = custom.get("Common Core") or {}
            frecuencia = info.get("Frecuencia de Actualización") or info.get("frecuencia") or ""
            licencia = cc.get("License") or cc.get("license") or meta.get("license") or ""
            publisher = cc.get("Publisher") or cc.get("publisher") or meta.get("attribution") or meta.get("attributionLink") or ""
        # fallback fields
        licencia = licencia or meta.get("license") or meta.get("attribution")
        publisher = publisher or meta.get("attribution") or meta.get("owner") or ""
        return {
            "categoria": categoria,
            "tags": tags,
            "descripcion": descripcion,
            "fecha_metadatos": fecha_meta,
            "fecha_datos": fecha_datos,
            "frecuencia": frecuencia,
            "licencia": licencia,
            "publisher": publisher,
            "raw": meta
        }
    except Exception:
        return None

# -----------------------
# BASIC DF STATS
# -----------------------
def df_basic_stats(df: pd.DataFrame):
    cols = len(df.columns)
    rows = len(df)
    total_cells = df.size
    total_nulls = int(df.isnull().sum().sum())
    col_more_1pct_null = int(((df.isnull().sum() / max(rows, 1)) > 0.01).sum())
    return {
        "rows": rows,
        "cols": cols,
        "total_cells": total_cells,
        "total_nulls": total_nulls,
        "col_more_1pct_null": col_more_1pct_null
    }

# -----------------------
# 17 CRITERIA (INTERNAL 0..10)
# -----------------------
# 1 Accesibilidad (0..10)
def criterio_accesibilidad_10(df: pd.DataFrame, meta: Optional[Dict[str,Any]]) -> float:
    if df is None or df.empty:
        return 0.0
    if meta is None:
        # data present but no metadata: partial
        return 5.0
    # require title/description/license/publisher presence -> full
    has_title = bool(meta.get("raw", {}).get("name") or meta.get("descripcion"))
    has_description = bool(meta.get("descripcion"))
    has_license = bool(meta.get("licencia"))
    has_publisher = bool(meta.get("publisher"))
    has_format = df.shape[1] > 0
    score = 0.0
    if all([has_title, has_description, has_license, has_publisher, has_format]):
        score = 10.0
    elif any([has_title, has_description, has_license, has_publisher, has_format]):
        score = 6.0
    else:
        score = 2.0
    return float(score)

# 2 Actualidad (0..10) - uses fecha_datos and frecuencia
def criterio_actualidad_10(meta: Optional[Dict[str,Any]]) -> float:
    if meta is None:
        return 0.0
    fecha_datos = parse_iso(meta.get("fecha_datos"))
    if fecha_datos is None:
        return 0.0
    frecuencia_str = meta.get("frecuencia", "")
    freq_days = freq_to_days(frecuencia_str)
    now = datetime.now(timezone.utc)
    days_diff = (now - fecha_datos).days
    # heuristic: if updated within frequency window -> full (10), else scale down
    if days_diff <= max(1, freq_days):
        return 10.0
    # if within 3 * freq -> 6
    if days_diff <= 3 * freq_days:
        return 6.0
    # within year -> 3
    if days_diff <= 365:
        return 3.0
    return 0.0

# 3 Completitud (0..10) - official formula pieces
def criterio_completitud_10(df: pd.DataFrame, stats: Dict[str,int]) -> float:
    if stats["total_cells"] == 0 or stats["cols"] == 0:
        return 0.0
    # a) medidaCompletitudDatos = 10 * (1 - (totalNulos/totalCeldas)^1.5)
    medida_datos = 10.0 * (1.0 - (safe_div(stats["total_nulls"], stats["total_cells"]) ** 1.5))
    # b) medidaCompletitudCol = 10 * (1 - (numColPorcNulos/dfColumnas)^2)
    medida_cols = 10.0 * (1.0 - (safe_div(stats["col_more_1pct_null"], stats["cols"]) ** 2))
    # c) medidaColNoVacias = 10 * (cols_without_null / dfColumnas)
    cols_no_null = int((df.isnull().sum() == 0).sum())
    medida_col_no_vacias = 10.0 * safe_div(cols_no_null, stats["cols"])
    score_10 = (medida_datos + medida_cols + medida_col_no_vacias) / 3.0
    # clip 0..10
    return float(np.clip(score_10, 0.0, 10.0))

# 4 Comprensibilidad (0..10)
def criterio_comprensibilidad_10(df: pd.DataFrame, meta: Optional[Dict[str,Any]]) -> float:
    # If metadata description present, assume high comprehensibility
    if meta and meta.get("descripcion"):
        return 10.0
    # else proxy: proportion of readable column names (snake_case, short)
    if df is None or df.shape[1] == 0:
        return 0.0
    col_names = df.columns.tolist()
    understandable = sum(1 for c in col_names if len(str(c)) <= 30 and re.match(r'^[a-z0-9_]+$', str(c).lower()))
    return float(np.clip(10.0 * safe_div(understandable, len(col_names)), 0.0, 10.0))

# 5 Conformidad (0..10)
def criterio_conformidad_10(meta: Optional[Dict[str,Any]]) -> float:
    if meta is None:
        return 0.0
    lic = (meta.get("licencia") or "").lower()
    if lic and ("cc" in lic or "open" in lic or "creative" in lic):
        return 10.0
    if lic:
        return 6.0
    return 2.0

# 6 Confidencialidad (0..10)
PII_HIGH = ["identificacion", "cedula", "documento", "nro_documento", "numero_identificacion"]
PII_MED = ["telefono", "celular", "email", "correo", "direccion"]
PII_LOW = ["fecha_nacimiento", "edad", "sexo", "genero", "nombre"]

def criterio_confidencialidad_10(df: pd.DataFrame) -> float:
    if df is None or df.shape[1] == 0:
        return 0.0
    per_col_risk = {}
    for c in df.columns:
        cl = str(c).lower()
        risk = 0
        if any(k in cl for k in PII_HIGH):
            risk = 3
        elif any(k in cl for k in PII_MED):
            risk = 2
        elif any(k in cl for k in PII_LOW):
            risk = 1
        if risk:
            per_col_risk[c] = risk
    if not per_col_risk:
        return 10.0
    sum_risk = sum(per_col_risk.values())
    # penalty: sum_risk scaled by number of columns and max risk
    denom = max(1, 3 * df.shape[1])
    penalty = (sum_risk / denom) * 10.0
    score = 10.0 - penalty
    return float(np.clip(score, 0.0, 10.0))

# 7 Consistencia (0..10)
def criterio_consistencia_10(df: pd.DataFrame) -> float:
    if df is None or df.shape[1] == 0:
        return 0.0
    cols = df.columns
    cumple = 0
    for c in cols:
        try:
            unique = df[c].nunique(dropna=True)
            # For numeric columns, require variance threshold; else unique >1
            if pd.api.types.is_numeric_dtype(df[c]):
                var = float(df[c].dropna().astype(float).var()) if df[c].dropna().shape[0] > 1 else 0.0
                if var >= 0.1 and unique >= 2:
                    cumple += 1
            else:
                if unique >= 2:
                    cumple += 1
        except Exception:
            continue
    return float(np.clip(10.0 * safe_div(cumple, max(1, len(cols))), 0.0, 10.0))

# 8 Credibilidad (0..10)
def criterio_credibilidad_10(meta: Optional[Dict[str,Any]], df: pd.DataFrame) -> float:
    if meta is None:
        return 4.0  # conservative
    mc = 0
    for k in ["descripcion", "categoria", "licencia", "publisher"]:
        if meta.get(k):
            mc += 1
    medidaMetadatosCompletos = 10.0 * safe_div(mc, 4.0)
    medidaPublicadorValido = 10.0 if meta.get("publisher") else 0.0
    medidaColDescValida = criterio_comprensibilidad_10(df, meta)
    # weights: 70%, 5%, 25% as guide
    score = (0.70 * medidaMetadatosCompletos) + (0.05 * medidaPublicadorValido) + (0.25 * medidaColDescValida)
    return float(np.clip(score, 0.0, 10.0))

# 9 Disponibilidad (0..10)
def criterio_disponibilidad_10(access_10: float, actual_10: float) -> float:
    return float(np.clip((access_10 + actual_10) / 2.0, 0.0, 10.0))

# 10 Eficiencia (0..10)
def criterio_eficiencia_10(df: pd.DataFrame, completitud_10: float) -> float:
    if df is None or df.shape[0] == 0:
        return 0.0
    # filas unicas ratio
    unique_rows = len(df.drop_duplicates())
    filas_ratio = safe_div(unique_rows, len(df))
    # columnas duplicadas: fraction of identical column pairs
    cols = df.columns.tolist()
    equal_pairs = 0
    total_pairs = 0
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            total_pairs += 1
            try:
                if df[cols[i]].equals(df[cols[j]]):
                    equal_pairs += 1
            except Exception:
                continue
    col_dup_factor = safe_div(equal_pairs, max(1, total_pairs))
    # combine: weights 40% completitud, 30% filas no duplicadas, 30% columnas no duplicadas
    score = (0.4 * completitud_10) + (0.3 * (filas_ratio * 10.0)) + (0.3 * ((1.0 - col_dup_factor) * 10.0))
    return float(np.clip(score, 0.0, 10.0))

# 11 Exactitud (0..10) = avg sintáctica + semántica
def criterio_exactitud_10(df: pd.DataFrame) -> float:
    # syntactic: penalize when text columns have very few unique normalized values
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if len(text_cols) == 0:
        synt = 10.0
    else:
        num_similar = 0
        for c in text_cols:
            vals = df[c].dropna().astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()
            unique = vals.nunique()
            if unique <= 2:
                num_similar += 1
        synt = 10.0 * (1.0 - (safe_div(num_similar, len(text_cols)) ** 2))
    # semantic: numeric columns with non-numeric entries penalize
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) == 0:
        sem = 10.0
    else:
        bad = 0
        for c in num_cols:
            non_numeric = df[c].apply(lambda x: False if (pd.isna(x) or isinstance(x, (int, float, np.integer, np.floating))) else True).sum()
            if safe_div(non_numeric, max(1, len(df))) > 0.05:
                bad += 1
        frac_bad = safe_div(bad, len(num_cols))
        sem = 10.0 if bad == 0 else max(0.0, 10.0 - (1.0 - (frac_bad ** 2)))
    return float(np.clip((synt + sem) / 2.0, 0.0, 10.0))

# 12 Portabilidad (0..10)
def criterio_portabilidad_10(completitud_10: float, conformidad_10: float) -> float:
    portabilidad_base = 10.0  # assume standard formats available via API
    score = 0.50 * portabilidad_base + 0.25 * conformidad_10 + 0.25 * completitud_10
    return float(np.clip(score, 0.0, 10.0))

# 13 Precisión (0..10) - proxy using consistency
def criterio_precision_10(consistency_10: float) -> float:
    return float(np.clip(consistency_10, 0.0, 10.0))

# 14 Recuperabilidad (0..10)
def criterio_recuperabilidad_10(access_10: float, meta: Optional[Dict[str,Any]]) -> float:
    if meta is None:
        return float(np.clip(access_10 * 0.5, 0.0, 10.0))
    metadatos_completos = 10.0 if all([meta.get("descripcion"), meta.get("categoria"), meta.get("licencia"), meta.get("publisher")]) else (5.0 if any([meta.get("descripcion"), meta.get("categoria"), meta.get("licencia"), meta.get("publisher")]) else 0.0)
    metadatos_auditados = 10.0 if meta.get("fecha_metadatos") else 0.0
    return float(np.clip((access_10 + metadatos_completos + metadatos_auditados) / 3.0, 0.0, 10.0))

# 15 Relevancia (0..10)
def criterio_relevancia_10(meta: Optional[Dict[str,Any]], df: pd.DataFrame) -> float:
    medida_filas = 10.0 if df is not None and len(df) >= 50 and safe_div(df.isnull().sum().sum(), max(1, df.size)) < 0.2 else 2.0
    medida_categoria = 10.0 if meta and meta.get("categoria") else (5.0 if meta and meta.get("tags") else 2.0)
    return float(np.clip((medida_categoria + medida_filas) / 2.0, 0.0, 10.0))

# 16 Trazabilidad (0..10)
def criterio_trazabilidad_10(actual_10: float, cred_10: float) -> float:
    return float(np.clip((actual_10 + cred_10) / 2.0, 0.0, 10.0))

# 17 Unicidad (0..10)
def criterio_unicidad_10(df: pd.DataFrame) -> float:
    if df is None or df.shape[0] == 0:
        return 0.0
    unique_rows = len(df.drop_duplicates())
    ratio = safe_div(unique_rows, len(df))
    return float(np.clip(ratio * 10.0, 0.0, 10.0))

# -----------------------
# MAIN EVALUATION
# -----------------------
def evaluate_all_10(df: pd.DataFrame, meta: Optional[Dict[str,Any]]) -> Dict[str, float]:
    stats = df_basic_stats(df)
    # compute each criterion (0..10)
    a10 = criterio_accesibilidad_10(df, meta)
    act10 = criterio_actualidad_10(meta)
    comp10 = criterio_completitud_10(df, stats)
    compr10 = criterio_comprensibilidad_10(df, meta)
    conf10 = criterio_conformidad_10(meta)
    confid10 = criterio_confidencialidad_10(df)
    cons10 = criterio_consistencia_10(df)
    cred10 = criterio_credibilidad_10(meta, df)
    disp10 = criterio_disponibilidad_10(a10, act10)
    uniq10 = criterio_unicidad_10(df)
    eff10 = criterio_eficiencia_10(df, comp10)
    exact10 = criterio_exactitud_10(df)
    port10 = criterio_portabilidad_10(comp10, conf10)
    prec10 = criterio_precision_10(cons10)
    rec10 = criterio_recuperabilidad_10(a10, meta)
    rel10 = criterio_relevancia_10(meta, df)
    traz10 = criterio_trazabilidad_10(act10, cred10)

    results_10 = {
        "1. Accesibilidad": a10,
        "2. Actualidad": act10,
        "3. Completitud": comp10,
        "4. Comprensibilidad": compr10,
        "5. Conformidad": conf10,
        "6. Confidencialidad": confid10,
        "7. Consistencia": cons10,
        "8. Credibilidad": cred10,
        "9. Disponibilidad": disp10,
        "10. Eficiencia": eff10,
        "11. Exactitud": exact10,
        "12. Portabilidad": port10,
        "13. Precisión": prec10,
        "14. Recuperabilidad": rec10,
        "15. Relevancia": rel10,
        "16. Trazabilidad": traz10,
        "17. Unicidad": uniq10
    }
    return results_10

# -----------------------
# UI
# -----------------------
def main():
    st.title("Evaluación de Calidad de Conjuntos de Datos — Guía MinTIC 2025")
    st.write("Cálculos internos en escala 0–10 (Guía). Visualización en % (0–100%).")

    st.sidebar.header("Ingesta")
    mode = st.sidebar.radio("Origen de datos", ("Asset Inventory (API)", "Cargar CSV local"))

    df = pd.DataFrame()
    meta = None

    if mode == "Asset Inventory (API)":
        dataset_id = st.sidebar.text_input("Dataset ID (ej. uzcf-b9dh)", value="uzcf-b9dh")
        app_token = st.sidebar.text_input("App Token (opcional)", value="", type="password")
        if st.sidebar.button("Cargar desde API"):
            with st.spinner("Cargando filas..."):
                df = fetch_data_resource(dataset_id.strip(), app_token=app_token.strip() or None)
            with st.spinner("Cargando metadatos..."):
                meta = fetch_resource_metadata_view(dataset_id.strip())
            if meta is None:
                st.warning("No se recuperaron metadatos del recurso; algunos criterios se evaluarán parcialmente.")
    else:
        uploaded = st.sidebar.file_uploader("Subir CSV", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                df.replace("", np.nan, inplace=True)
                st.info("CSV cargado. Nota: en modo CSV no hay metadatos del portal.")
                # create a minimal meta placeholder
                meta = {
                    "categoria": "",
                    "tags": [],
                    "descripcion": "",
                    "fecha_metadatos": None,
                    "fecha_datos": None,
                    "frecuencia": "",
                    "licencia": "",
                    "publisher": ""
                }
            except Exception as e:
                st.error(f"Error leyendo CSV: {e}")
                return

    if df is None or df.empty:
        st.info("Cargue datos para que se ejecute la evaluación.")
        return

    # run evaluation
    results_10 = evaluate_all_10(df, meta)

    # Score global (mean of 17 criteria, on 0..10)
    overall_10 = float(np.mean(list(results_10.values())))
    overall_pct = to_percent(overall_10)

    # Display KPIs
    st.subheader(f"🏆 Score global: {overall_pct:.2f}%  (equiv. {overall_10:.2f}/10)")
    st.markdown("---")
    st.subheader("Criterios (visualizados en % — internals 0..10)")

    # show metrics in a grid, convert to %
    items = list(results_10.items())
    n_cols = 4
    cols = st.columns(n_cols)
    for i, (name, val10) in enumerate(items):
        with cols[i % n_cols]:
            st.metric(label=name, value=f"{to_percent(val10):.2f}%")

    st.markdown("---")
    st.subheader("Detalle: Completitud por Atributo")
    completeness_detail = pd.DataFrame({
        "Atributo": df.columns.astype(str),
        "Valores no nulos": df.count().values,
        "Total filas": len(df),
        "Completitud (%)": (df.count().values / max(1, len(df))) * 100.0
    }).sort_values(by="Completitud (%)", ascending=False)
    st.dataframe(completeness_detail, width="stretch")

    st.markdown("---")
    st.subheader("Vista previa de datos")
    st.dataframe(df.head(10), width="stretch")

    # show metadata summary if present
    st.markdown("---")
    st.subheader("Metadatos del Recurso (resumen)")
    if meta:
        with st.expander("Ver metadatos básicos"):
            st.write({
                "Categoría": meta.get("categoria"),
                "Tags": meta.get("tags"),
                "Descripción (snippet)": (meta.get("descripcion") or "")[:400],
                "Fecha datos": meta.get("fecha_datos"),
                "Frecuencia": meta.get("frecuencia"),
                "Licencia": meta.get("licencia"),
                "Publisher": meta.get("publisher")
            })
    else:
        st.info("No se detectaron metadatos del recurso (usa la API de views/{id}).")

    # warning about missing meta fields
    if meta:
        missing = [k for k in ["categoria","descripcion","fecha_datos","licencia","publisher"] if not meta.get(k)]
        if missing:
            st.warning("Faltan algunos metadatos que la Guía usa para cálculos: " + ", ".join(missing))

    # download guide
    try:
        st.download_button("📄 Descargar Guía 2025 (referencia)", GUIDE_PDF_PATH)
    except Exception:
        st.info(f"Ruta local de referencia a la Guía: {GUIDE_PDF_PATH}")

if __name__ == "__main__":
    main()
