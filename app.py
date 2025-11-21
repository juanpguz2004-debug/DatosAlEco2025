# streamlit_app_corrected.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from typing import Dict, Any

# --- RUTA DEL PDF DE REFERENCIA SUBIDO (SE DEJÓ PARA DESCARGA/REFERENCIA) ---
GUIDE_PDF_PATH = "/mnt/data/Guía de Calidad e Interoperabilidad 2025 (1).pdf"

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Dashboard Calidad Datos Abiertos (Guía 2025 - Corregido)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTES / PARÁMETROS ---
MIN_FILAS_RELEVANCIA = 50
PENAL_EXP_COMPLETITUD = 1.5  # según la guía para completitud de datos
PENAL_QUAD_COMPLETITUD_COL = 2  # exponente 2 para completitud por columnas (según guía)
# Umbrales para consistencia/varianza (ajustables)
VARIANCE_THRESHOLD = 0.1
MIN_UNIQUE_VALUES = 2

# PII keywords y riesgo por keyword (alto=3, medio=2, bajo=1) -- ajustar según política
PII_KEYWORDS_HIGH = ['identificacion', 'documento', 'nro_documento', 'numero_identificacion', 'numero_documento', 'tarjeta', 'pasaporte']
PII_KEYWORDS_MEDIUM = ['telefono', 'celular', 'direccion', 'email', 'correo']
PII_KEYWORDS_LOW = ['fecha_nacimiento', 'edad', 'sexo', 'genero']

# --- UTILIDADES ---
def safe_div(a, b):
    return a / b if b else 0.0

def to_percent(score_0_10: float) -> float:
    """Convierte escala 0-10 a porcentaje 0-100 redondeado."""
    return float(np.clip(score_0_10 * 10, 0, 100))

# --- INGESTA DE DATOS ---
@st.cache_data(show_spinner="Cargando datos...")
def fetch_api_data(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data)
        # normalizar columnas vacías a NaN
        df.replace("", np.nan, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error al obtener datos desde API: {e}")
        return pd.DataFrame()

def handle_csv_upload(uploaded_file) -> pd.DataFrame:
    try:
        df = pd.read_csv(uploaded_file)
        df.replace("", np.nan, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error leyendo CSV: {e}")
        return pd.DataFrame()

# --- CÁLCULOS BASE (variables auxiliares) ---
def df_basic_stats(df: pd.DataFrame):
    df_cols = len(df.columns)
    total_rows = len(df)
    total_cells = df.size
    total_nulls = int(df.isnull().sum().sum())
    # columnas con más de 1% nulos
    num_col_porcent_null = int(((df.isnull().sum() / total_rows) > 0.01).sum()) if total_rows>0 else 0
    return {
        'df_columns': df_cols,
        'total_rows': total_rows,
        'total_cells': total_cells,
        'total_nulls': total_nulls,
        'num_col_porcent_null': num_col_porcent_null
    }

def detect_pii_columns(df: pd.DataFrame) -> Dict[str,int]:
    """
    Detecta columnas con posible PII usando keywords y devuelve:
    { 'num_confidencial': int, 'sum_risk': int, 'per_column_risk': {col: risk} }
    """
    per_column_risk = {}
    sum_risk = 0
    for col in df.columns:
        col_l = col.lower()
        risk = 0
        if any(k in col_l for k in PII_KEYWORDS_HIGH):
            risk = 3
        elif any(k in col_l for k in PII_KEYWORDS_MEDIUM):
            risk = 2
        elif any(k in col_l for k in PII_KEYWORDS_LOW):
            risk = 1
        if risk > 0:
            per_column_risk[col] = risk
            sum_risk += risk
    return {
        'num_confidencial': len(per_column_risk),
        'sum_risk': sum_risk,
        'per_column_risk': per_column_risk
    }

# --- MEDIDAS QUE REQUIEREN METADATOS ---
def read_metadata_from_df(df: pd.DataFrame) -> Dict[str,Any]:
    """
    Intenta inferir metadatos comunes del DataFrame cargado (cuando la ingestión
    trae datos tipo Socrata donde algunas columnas son metadata).
    Si no encuentra, deja None para indicar que la medida debe pedirse al Asset Inventory.
    """
    # Campos que podrían venir en un export plano de la API (intentar leerlos)
    meta = {}
    # ejemplos comunes: 'title', 'description', 'updated_at', 'frequency', 'publisher', 'license'
    for k in ['title', 'description', 'updated_at', 'frequency', 'publisher', 'license', 'tags']:
        meta[k] = None
        if k in df.columns:
            # tomar el primer no-nulo
            val = df[k].dropna().astype(str).iloc[0] if df[k].dropna().shape[0] > 0 else None
            meta[k] = val
    return meta

# --- CÁLCULOS DE CRITERIOS (escala 0..10 según guía) ---

# 1. Accesibilidad (restringido: necesita metadatos de publicación). Aquí: proxy razonable
def calc_accessibility(df: pd.DataFrame, metadata: Dict[str,Any]) -> float:
    """
    Devuelve 0..10. La guía evalúa: metadatos claros, formatos abiertos, enlace operativo.
    Implementación:
    - 10 si detecta al menos: title, description, license y formato en {csv,json}
    - 5 si tiene parte de metadatos
    - 0 si nada
    """
    score = 0.0
    has_title = bool(metadata.get('title'))
    has_description = bool(metadata.get('description'))
    has_license = bool(metadata.get('license'))
    # formato inference: si df came from JSON or CSV can't be 100% but infer from presence of keys
    # as proxy: if len(columns)>0 assume formato abierto
    has_format = df.shape[1] > 0
    if has_title and has_description and has_license and has_format:
        score = 10.0
    elif any([has_title, has_description, has_license, has_format]):
        score = 5.0
    else:
        score = 0.0
    return float(score)

# 2. Actualidad (0..10) - requiere metadata.updated_at y frequency
def calc_actuality(df: pd.DataFrame, metadata: Dict[str,Any]) -> float:
    """
    Implementación simplificada fiel a la guía:
    - Si no hay metadata['updated_at'] -> 0
    - Si updated_at dentro de 3 meses -> 10
    - Si actualizado entre 3 y 12 meses -> 5
    - Si > 12 meses -> 0
    """
    updated = metadata.get('updated_at')
    if not updated:
        return 0.0
    try:
        updated_ts = pd.to_datetime(updated, errors='coerce', utc=True)
        if pd.isna(updated_ts):
            return 0.0
        now = pd.Timestamp.now(tz='UTC')
        months_diff = (now.year - updated_ts.year) * 12 + (now.month - updated_ts.month)
        if months_diff <= 3:
            return 10.0
        elif months_diff <= 12:
            return 5.0
        else:
            return 0.0
    except Exception:
        return 0.0

# 3. Completitud (0..10) - fórmula oficial
def calc_completeness(df: pd.DataFrame, stats: Dict[str,int]) -> float:
    df_columns = stats['df_columns']
    total_nulls = stats['total_nulls']
    total_cells = stats['total_cells']
    num_col_porcent_null = stats['num_col_porcent_null']

    if total_cells == 0 or df_columns == 0:
        return 0.0

    # a) medidaCompletitudDatos = 10 * (1 - (totalNulos/totalCeldas)^{penalizacionFactor})
    medidaCompletitudDatos = 10.0 * (1.0 - (safe_div(total_nulls, total_cells) ** PENAL_EXP_COMPLETITUD))

    # b) medidaCompletitudCol = 10 * (1 - (numColPorcNulos/dfColumnas)^{penalizacionFactor})  (penalizacionFactor=2 en la guía)
    medidaCompletitudCol = 10.0 * (1.0 - (safe_div(num_col_porcent_null, df_columns) ** PENAL_QUAD_COMPLETITUD_COL))

    # c) medidaColNoVacias = 10 * (num_cols_without_null / dfColumnas)
    num_cols_no_null = int((df.isnull().sum() == 0).sum())
    medidaColNoVacias = 10.0 * safe_div(num_cols_no_null, df_columns)

    # promedio (0..10)
    score_10 = float((medidaCompletitudDatos + medidaCompletitudCol + medidaColNoVacias) / 3.0)
    return np.clip(score_10, 0.0, 10.0)

# 4. Comprensibilidad (0..10) - requiere metadatos/diccionario; proxy con nombres y descripción
def calc_comprehensibility(df: pd.DataFrame, metadata: Dict[str,Any]) -> float:
    # si la metadata tiene description y title: 10
    if metadata.get('title') and metadata.get('description'):
        return 10.0
    # si hay algunos nombres claros -> escalar parcialmente
    col_names = list(df.columns)
    if len(col_names) == 0:
        return 0.0
    understandable_count = sum(1 for c in col_names if re.match(r'^[a-z0-9_]+$', c.lower()) and len(c) <= 30)
    score_10 = 10.0 * safe_div(understandable_count, len(col_names))
    return float(score_10)

# 5. Conformidad (0..10) - evaluación de estándares básicos (formato, licencias)
def calc_conformity(df: pd.DataFrame, metadata: Dict[str,Any]) -> float:
    # Si hay license y publisher -> 10
    if metadata.get('license') and metadata.get('publisher'):
        return 10.0
    # si al menos la licencia está -> 5
    if metadata.get('license'):
        return 5.0
    # si no, revisar nombres de columnas y formatos básicos -> pequeño puntaje
    return 2.0 if df.shape[1] > 0 else 0.0

# 6. Confidencialidad (0..10) - implementada con keywords y riesgo
def calc_confidentiality(df: pd.DataFrame) -> float:
    df_columns = df.shape[1]
    pii = detect_pii_columns(df)
    num_conf = pii['num_confidencial']
    sum_risk = pii['sum_risk']
    if num_conf == 0:
        return 10.0
    # Según la guía: se incorpora factor de riesgo. Implementación razonable:
    # confid = 10 - ( (sum_risk / (3 * df_columns)) * 10 )
    denom = max(1, 3 * max(1, df_columns))
    penalty = (sum_risk / denom) * 10.0
    score = 10.0 - penalty
    return float(np.clip(score, 0.0, 10.0))

# 7. Consistencia (0..10) - proporción columnas que cumplen varianza y unicidad (interpretación guía)
def calc_consistency(df: pd.DataFrame) -> float:
    df_columns = df.shape[1]
    if df_columns == 0:
        return 0.0
    columns_cumplen = 0
    for col in df.columns:
        unique_vals = df[col].nunique(dropna=True)
        cumple_unique = unique_vals >= MIN_UNIQUE_VALUES
        cumple_var = True
        if pd.api.types.is_numeric_dtype(df[col]):
            # varianza only for numeric
            try:
                cumple_var = (df[col].dropna().astype(float).var() >= VARIANCE_THRESHOLD)
            except Exception:
                cumple_var = False
        else:
            # For text, we assume variability if number of unique > MIN_UNIQUE_VALUES
            cumple_var = cumple_unique
        if cumple_var and cumple_unique:
            columns_cumplen += 1
    score = safe_div(columns_cumplen, df_columns) * 10.0
    return float(np.clip(score, 0.0, 10.0))

# 8. Credibilidad (0..10)
def calc_credibility(df: pd.DataFrame, metadata: Dict[str,Any]) -> float:
    # medidaMetadatosCompletos: si title, description, publisher, license presentes -> 10
    md_complete_count = sum(bool(metadata.get(k)) for k in ['title','description','publisher','license'])
    medidaMetadatosCompletos = 10.0 * safe_div(md_complete_count, 4.0)
    medidaPublicadorValido = 10.0 if metadata.get('publisher') else 0.0
    # medidaColDescValida: si hay descripciones por columna en metadata (difícil), proxy usando comprehensibility
    medidaColDescValida = calc_comprehensibility(df, metadata)
    score_10 = (0.70 * medidaMetadatosCompletos) + (0.05 * medidaPublicadorValido) + (0.25 * medidaColDescValida)
    return float(np.clip(score_10, 0.0, 10.0))

# 9. Disponibilidad (0..10) = (accesibilidad + actualidad)/2
def calc_availability(accessibility_score_10: float, actuality_score_10: float) -> float:
    return float(np.clip((accessibility_score_10 + actuality_score_10) / 2.0, 0.0, 10.0))

# 10. Eficiencia (0..10) - se compone de completitud, columnas no duplicadas y filas no duplicadas
def calc_efficiency(df: pd.DataFrame, completeness_10: float) -> float:
    # columnas no duplicadas: proportion of columns with all unique names (names always unique) -> measure duplicados en columnas clave: detectar columnas que son idénticas
    df_cols = df.shape[1]
    if df.shape[0] == 0 or df_cols == 0:
        return 0.0
    # filas no duplicadas
    unique_rows = len(df.drop_duplicates())
    filas_no_duplicadas_ratio = safe_div(unique_rows, len(df))
    # columnas duplicadas (contenido idéntico)
    # identificar pares de columnas con exactamente la misma serie (costoso para many cols)
    equal_col_pairs = 0
    checked = set()
    cols = list(df.columns)
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if cols[j] in checked:
                continue
            try:
                if df[cols[i]].equals(df[cols[j]]):
                    equal_col_pairs += 1
            except Exception:
                pass
    # penalización por columnas idénticas
    max_pairs = max(1, df_cols*(df_cols-1)/2)
    col_dup_penalty = safe_div(equal_col_pairs, max_pairs)
    # combine: 40% completitud, 30% filas no duplicadas, 30% no duplicados columnas
    score_10 = (0.4 * completeness_10) + (0.3 * (filas_no_duplicadas_ratio * 10.0)) + (0.3 * ((1.0 - col_dup_penalty) * 10.0))
    return float(np.clip(score_10, 0.0, 10.0))

# 11. Exactitud sintáctica (0..10)
def calc_exactitud_sintactica(df: pd.DataFrame) -> float:
    # Identificar columnas textuales y contar cuántas tienen "valores únicos muy similares".
    # Implementación: para cada texto, normalizar y contar valores únicos relativos.
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if len(text_cols) == 0:
        return 10.0
    num_similar = 0
    for c in text_cols:
        vals = df[c].dropna().astype(str).str.lower().str.normalize('NFKD')
        # quick normalization: strip punctuation / accents (approx)
        vals = vals.str.replace(r'[^\w\s]', '', regex=True).str.strip()
        unique_vals = vals.nunique()
        # si menos del 3 valores únicos en una columna de texto, consideramos alto riesgo de similitud
        if unique_vals <= 2:
            num_similar += 1
    df_text_cols = len(text_cols)
    score_10 = 10.0 * (1.0 - (safe_div(num_similar, df_text_cols) ** 2))
    return float(np.clip(score_10, 0.0, 10.0))

# 11b. Exactitud semántica (0..10) - aproximación
def calc_exactitud_semantica(df: pd.DataFrame) -> float:
    # Requiere comparar título/description vs valores -> si no hay metadata, usamos proxy:
    # Proxy: para columnas numéricas, porcentaje de celdas no numéricas reduce el score.
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) == 0:
        return 10.0
    bad_cols = 0
    for c in num_cols:
        non_numeric = df[c].apply(lambda x: False if (pd.isna(x) or isinstance(x,(int,float,np.integer,np.floating))) else True).sum()
        # si >5% de valores no numéricos en columna numérica -> marca
        if safe_div(non_numeric, max(1, len(df))) > 0.05:
            bad_cols += 1
    df_num_cols = len(num_cols)
    # fórmula guía (exponencial para penalizar): sem = 10 - (1 - (bad/df)^2)
    # ajustamos para que quede en 0..10:
    frac_bad = safe_div(bad_cols, df_num_cols)
    score_10 = 10.0 - (1.0 - (frac_bad ** 2))
    # normalizar a 0..10
    # if frac_bad==0 => 10 - (1 - 0 ) = 9 -> adjust: we prefer 10 when no bad columns
    if bad_cols == 0:
        score_10 = 10.0
    return float(np.clip(score_10, 0.0, 10.0))

def calc_accuracy(df: pd.DataFrame) -> float:
    s_sint = calc_exactitud_sintactica(df)
    s_sem = calc_exactitud_semantica(df)
    return float(np.clip((s_sint + s_sem) / 2.0, 0.0, 10.0))

# 12. Portabilidad (0..10) = 0.50*portabilidad_base + 0.25*conformidad + 0.25*completitud
def calc_portability(df: pd.DataFrame, completeness_10: float, conformity_10: float) -> float:
    # portabilidad_base: puntaje por formato y tamaño. Proxy:
    # formatos abiertos: si df has <= 1GB (we don't access size) -> assume ok: base=10
    portabilidad_base = 10.0
    score_10 = 0.50 * portabilidad_base + 0.25 * conformity_10 + 0.25 * completeness_10
    return float(np.clip(score_10, 0.0, 10.0))

# 13. Precisión (0..10) - proxy usando consistencia/variabilidad
def calc_precision(df: pd.DataFrame, consistency_10: float) -> float:
    # Usar consistency como proxy razonable
    return float(np.clip(consistency_10, 0.0, 10.0))

# 14. Recuperabilidad (0..10) = (accesibilidad + metadatosCompletos + metadatosAuditados)/3
def calc_recoverability(access_10: float, metadata: Dict[str,Any]) -> float:
    metadatos_completos = 10.0 if all(metadata.get(k) for k in ['title','description','publisher','license']) else 5.0 if any(metadata.get(k) for k in ['title','description','publisher','license']) else 0.0
    metadatos_auditados = 10.0 if metadata.get('updated_at') else 0.0
    return float(np.clip((access_10 + metadatos_completos + metadatos_auditados)/3.0, 0.0, 10.0))

# 15. Relevancia (0..10) = (medidaCategoria + medidaFilas)/2
def calc_relevance(df: pd.DataFrame, metadata: Dict[str,Any]) -> float:
    # medidaFilas:
    medidaFilas = 10.0 if len(df) >= MIN_FILAS_RELEVANCIA and safe_div(df.isnull().sum().sum(), df.size) < 0.2 else 0.0
    # medidaCategoria: requiere comparar metadata['description'] y 'title' con taxonomía; proxy:
    medidaCategoria = 10.0 if metadata.get('description') and metadata.get('title') else 5.0 if metadata.get('title') else 0.0
    return float(np.clip((medidaCategoria + medidaFilas) / 2.0, 0.0, 10.0))

# 16. Trazabilidad (0..10) - proxy: (actualidad + credibilidad)/2
def calc_traceability(actuality_10: float, credibility_10: float) -> float:
    return float(np.clip((actuality_10 + credibility_10)/2.0, 0.0, 10.0))

# 17. Unicidad (0..10) - proporción filas únicas y penalización por duplicados en columnas clave
def calc_uniqueness(df: pd.DataFrame) -> float:
    if df.shape[0] == 0:
        return 0.0
    unique_rows = len(df.drop_duplicates())
    ratio = safe_div(unique_rows, len(df))
    # penalización adicional: si hay columnas con muchos repeats exactos
    return float(np.clip(ratio * 10.0, 0.0, 10.0))

# --- FUNCIÓN PRINCIPAL: calcula todos los criterios y los muestra ---
def calculate_and_display_metrics(df: pd.DataFrame):
    st.subheader("Evaluación (alineada a Guía 2025)")
    if df.empty:
        st.info("No hay datos cargados.")
        return

    stats = df_basic_stats(df)
    metadata = read_metadata_from_df(df)  # intenta inferir metadata (si viene embebida)

    # calcular criterios (todos en escala 0..10)
    accessibility_10 = calc_accessibility(df, metadata)
    actuality_10 = calc_actuality(df, metadata)
    completeness_10 = calc_completeness(df, stats)
    comprehensibility_10 = calc_comprehensibility(df, metadata)
    conformity_10 = calc_conformity(df, metadata)
    confidentiality_10 = calc_confidentiality(df)
    consistency_10 = calc_consistency(df)
    credibility_10 = calc_credibility(df, metadata)
    availability_10 = calc_availability(accessibility_10, actuality_10)
    efficiency_10 = calc_efficiency(df, completeness_10)
    accuracy_10 = calc_accuracy(df)
    portability_10 = calc_portability(df, completeness_10, conformity_10)
    precision_10 = calc_precision(df, consistency_10)
    recoverability_10 = calc_recoverability(accessibility_10, metadata)
    relevance_10 = calc_relevance(df, metadata)
    traceability_10 = calc_traceability(actuality_10, credibility_10)
    uniqueness_10 = calc_uniqueness(df)

    # agrupar y convertir a % para display (como tenías)
    metrics_10 = {
        "1. Accesibilidad": accessibility_10,
        "2. Actualidad": actuality_10,
        "3. Completitud": completeness_10,
        "4. Comprensibilidad": comprehensibility_10,
        "5. Conformidad": conformity_10,
        "6. Confidencialidad": confidentiality_10,
        "7. Consistencia": consistency_10,
        "8. Credibilidad": credibility_10,
        "9. Disponibilidad": availability_10,
        "10. Eficiencia": efficiency_10,
        "11. Exactitud": accuracy_10,
        "12. Portabilidad": portability_10,
        "13. Precisión": precision_10,
        "14. Recuperabilidad": recoverability_10,
        "15. Relevancia": relevance_10,
        "16. Trazabilidad": traceability_10,
        "17. Unicidad": uniqueness_10,
    }

    # Score global (en 0..10) y en %
    overall_10 = float(np.mean(list(metrics_10.values())))
    overall_pct = to_percent(overall_10)

    st.markdown(f"## 🏆 Score Global de Calidad: **{overall_pct:.2f}%**  (equiv. {overall_10:.2f}/10)")

    # Mostrar KPI's
    st.subheader("Criterios individuales")
    cols = st.columns(4)
    i = 0
    for name, val_10 in metrics_10.items():
        with cols[i % 4]:
            st.metric(label=name, value=f"{to_percent(val_10):.2f}%")
        i += 1

    st.markdown("---")
    st.subheader("Detalle de Completitud por Atributo")
    completeness_detail = pd.DataFrame({
        'Atributo': df.columns,
        'Valores No Nulos': df.count().values,
        'Total Filas': len(df),
        'Completitud (%)': (df.count().values / max(1, len(df))) * 100
    }).sort_values(by='Completitud (%)', ascending=False)
    st.dataframe(completeness_detail, use_container_width=True)

    st.markdown("---")
    st.subheader("Vista Previa del Dataset (primeras 10 filas)")
    st.dataframe(df.head(10), use_container_width=True)

    # Mostrar advertencias sobre metadatos faltantes
    missing_meta = [k for k,v in metadata.items() if not v]
    if missing_meta:
        st.warning(
            "Algunas medidas requieren metadatos del recurso (ej. title, description, publisher, license, updated_at). "
            "Los siguientes campos no se detectaron en la tabla: " + ", ".join(missing_meta) +
            ". Para resultados 100% fieles a la guía, recupere los metadatos del Asset Inventory/API y cárguelos."
        )

    # Enlace de referencia al PDF (ruta local)
    try:
        st.download_button("📄 Descargar Guía 2025 (referencia)", GUIDE_PDF_PATH)
    except Exception:
        st.info(f"Ruta de referencia a la guía: {GUIDE_PDF_PATH}")

# --- UI y flujo principal ---
def main():
    st.title("Sistema de Monitoreo de Calidad de Datos Abiertos — Versión Alineada a Guía 2025")
    st.caption("Esta versión corrige los cálculos y normaliza a escala 0..10 (mostrado en %).")

    st.sidebar.header("Opciones de Ingesta")
    ingestion_mode = st.sidebar.radio("Origen:", ('API (Asset Inventory)', 'Cargar CSV Local'))

    df = pd.DataFrame()
    if ingestion_mode == 'API (Asset Inventory)':
        api_url = st.sidebar.text_input("API URL (SocRata / datos.gov.co u otro):", value="")
        if st.sidebar.button("Cargar desde API") and api_url.strip():
            df = fetch_api_data(api_url.strip())
    else:
        uploaded = st.sidebar.file_uploader("Subir CSV", type=['csv'])
        if uploaded:
            df = handle_csv_upload(uploaded)

    if not df.empty:
        calculate_and_display_metrics(df)
    else:
        st.info("Cargue un dataset desde la API o suba un CSV para evaluar.")

if __name__ == "__main__":
    main()
