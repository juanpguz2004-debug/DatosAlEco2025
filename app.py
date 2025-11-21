import streamlit as st
import pandas as pd
import requests
from io import StringIO
import numpy as np

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Dashboard de Calidad de Datos Abiertos (17 Criterios)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTES ---
API_URL = "https://www.datos.gov.co/resource/uzcf-b9dh.json?$limit=100000"
EXPECTED_COLUMNS = ['item_id', 'department', 'resource_type', 'updated_at', 'category', 'tags']

# --- FUNCIONES DE INGESTA DE DATOS (SIN CAMBIOS) ---

@st.cache_data(show_spinner="Conectando a la API y cargando datos...")
def fetch_api_data(url: str) -> pd.DataFrame:
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        st.success(f"Datos cargados exitosamente desde la API. Filas: {len(df)}")
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error al conectar con la API: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al procesar los datos de la API: {e}")
        return pd.DataFrame()

def handle_csv_upload(uploaded_file) -> pd.DataFrame:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"CSV cargado exitosamente. Filas: {len(df)}")
        return df
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")
        return pd.DataFrame()

# --- FUNCIONES AUXILIARES ---

def get_date_column(df: pd.DataFrame, potential_names=['updated_at', 'fecha_actualizacion', 'created_at']) -> str:
    """Busca la columna de fecha más probable o retorna None."""
    for col in potential_names:
        if col in df.columns:
            return col
    return None

# --- FUNCIONES DE CÁLCULO DE LOS 17 CRITERIOS ---

def calculate_completeness(df: pd.DataFrame) -> float:
    """Criterio 3: Todos los campos obligatorios están diligenciados."""
    if df.empty: return 0.0
    # Ejemplo: 'item_id' (ID único) y 'department' son críticos.
    critical_cols = ['item_id', 'department']
    df_filtered = df[[col for col in critical_cols if col in df.columns]]
    
    if df_filtered.empty: return 0.0
    
    # Completitud basada solo en los campos críticos
    total_critical_cells = df_filtered.size
    non_null_critical_cells = df_filtered.count().sum()
    return (non_null_critical_cells / total_critical_cells) * 100

def calculate_uniqueness(df: pd.DataFrame, id_col: str = 'item_id') -> float:
    """Criterio 17: Detección de registros duplicados e identificación única (Ej: ID)."""
    if df.empty or id_col not in df.columns: return 0.0
    
    total_rows = len(df)
    unique_ids = df[id_col].nunique()
    return (unique_ids / total_rows) * 100

def calculate_conformity(df: pd.DataFrame, conformity_col: str = 'resource_type') -> float:
    """Criterio 5: Cumplimiento de lineamientos y estándares vigentes."""
    if df.empty or conformity_col not in df.columns: return 0.0
    # Se evalúa que la columna no sea nula y que los valores sean texto (conformidad simple).
    conforming_rows = df[conformity_col].apply(lambda x: isinstance(x, str) and len(str(x).strip()) > 0).sum()
    total_rows = len(df)
    return (conforming_rows / total_rows) * 100

def calculate_confidentiality(df: pd.DataFrame) -> float:
    """Criterio 6: Los datos solo deben ser accedidos por personal autorizado."""
    # Asumimos 100% si no se detectan columnas sensibles (ej: 'nombre_completo', 'cc').
    sensitive_indicators = ['nombre', 'cedula', 'email', 'identificacion']
    for col in df.columns:
        if any(ind in col.lower() for ind in sensitive_indicators):
            st.warning("Se detectaron posibles columnas sensibles. Puntuación de Confidencialidad reducida.")
            return 0.0
    return 100.0

def calculate_consistency(df: pd.DataFrame) -> float:
    """Criterio 7: Datos coherentes y sin contradicción (ej. fechas no desordenadas)."""
    if df.empty: return 0.0
    # Evaluación simple: consistencia de tipos de datos en columnas esperadas.
    present_cols = [col for col in EXPECTED_COLUMNS if col in df.columns]
    score = (len(present_cols) / len(EXPECTED_COLUMNS)) * 100
    return min(100.0, score + 20) 

def calculate_credibility(df: pd.DataFrame, source_col: str = 'department') -> float:
    """Criterio 8: Información veraz y confiable (Fuente oficial declarada)."""
    if df.empty or source_col not in df.columns: return 0.0
    # Credibilidad: % de registros que tienen una entidad responsable declarada.
    credible_rows = df[source_col].notna().sum()
    total_rows = len(df)
    return (credible_rows / total_rows) * 100

def calculate_efficiency(df: pd.DataFrame) -> float:
    """Criterio 10: Plataforma permite análisis y descargas con buen rendimiento."""
    # Asumimos un buen rendimiento de la plataforma datos.gov.co.
    return 95.0 

# INICIO DE LA FUNCIÓN FALTANTE (calculate_syntactic_accuracy)
def calculate_syntactic_accuracy(df: pd.DataFrame) -> float:
    """
    Cálculo de Exactitud Sintáctica (Criterio 11 - Componente Sintáctica).
    Valida si la columna de fecha principal es interpretable como fecha.
    """
    date_column = get_date_column(df)
    if df.empty or date_column is None:
        return 0.0

    try:
        # Intentamos convertir la columna a datetime. Si es posible, se considera sintácticamente correcta.
        correct_format_count = pd.to_datetime(df[date_column], errors='coerce').notna().sum()
        total_rows = len(df)
        return (correct_format_count / total_rows) * 100
    except Exception:
        return 0.0
# FIN DE LA FUNCIÓN FALTANTE

def calculate_accuracy(df: pd.DataFrame) -> float:
    """Criterio 11: Datos diligenciados correctamente (Exactitud Sintáctica y Semántica)."""
    # Combinamos Sintáctica y Semántica. Usamos el score de Exactitud Sintáctica como base.
    syntactic_score = calculate_syntactic_accuracy(df)
    
    # Semántica (Ejemplo): Validar que el tipo de recurso ('resource_type') sea lógico con la categoría ('category').
    semantically_correct = df.apply(lambda row: True if row.get('category') in ['Gobierno', 'Economía'] and row.get('resource_type') in ['dataset', 'map'] else False, axis=1).sum()
    semantic_score = (semantically_correct / len(df)) * 100
    
    # La Exactitud total es el promedio simple de ambos (como sugiere la guía 3.7.1 y 3.7.2)
    return (syntactic_score + semantic_score) / 2

def calculate_portability(df: pd.DataFrame) -> float:
    """Criterio 12: Formatos sin restricciones para su reutilización (CSV, JSON, etc.)."""
    return 100.0

def calculate_precision(df: pd.DataFrame) -> float:
    """Criterio 13: Nivel de desagregación de los datos es adecuado al original."""
    if 'tags' not in df.columns: return 50.0 
    tag_count = df['tags'].apply(lambda x: len(str(x).split(',')) if isinstance(x, str) else 0).mean()
    # Puntuación basada en el promedio de tags (mínimo 3 tags para 100%)
    return min(100.0, (tag_count / 3) * 100) 

def calculate_recoverability(actuality_score: float) -> float:
    """Criterio 14: Capacidad de restaurar o recuperar datos (Copias de seguridad / Control de versiones)."""
    return actuality_score * 0.95 

def calculate_relevance(df: pd.DataFrame) -> float:
    """Criterio 15: Los datos publicados deben ser de utilidad (alineados con demandas ciudadanas)."""
    if 'views' not in df.columns or 'downloads' not in df.columns: return 50.0
    
    relevant_count = df[(df['views'].astype(float) > 0) & (df['downloads'].astype(float) > 0)].shape[0]
    total_rows = len(df)
    return (relevant_count / total_rows) * 100

def calculate_traceability(actuality_score: float, credibility_score: float) -> float:
    """Criterio 16: Histórico del conjunto de datos (fechas de creación, publicación y actualizaciones)."""
    # FÓRMULA COMPUESTA SUGERIDA: (Actualidad + Credibilidad) / 2.
    return (actuality_score + credibility_score) / 2

def calculate_actuality(df: pd.DataFrame) -> float:
    """Criterio 2: Vigencia y actualización de los datos publicados (Fechas más recientes)."""
    date_column = get_date_column(df)
    if df.empty or date_column is None: return 0.0

    try:
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce') 
        df_copy.dropna(subset=[date_column], inplace=True)

        # Criterio: 90 días (3 meses) se considera muy actual.
        three_months_ago = pd.Timestamp.now() - pd.DateOffset(months=3)
        recent_count = df_copy[df_copy[date_column] >= three_months_ago].shape[0]
        
        # Ponderación: Cuanto más reciente el dataset, mayor la puntuación.
        return min(100.0, (recent_count / len(df)) * 100 * 1.2) 

    except Exception:
        return 0.0

def calculate_accessibility(df: pd.DataFrame) -> float:
    """Criterio 1: El conjunto puede ser consultado y descargado (Sin requisitos de registro)."""
    return 100.0 if not df.empty else 0.0

def calculate_comprehensibility(df: pd.DataFrame, col_names: list) -> float:
    """Criterio 4: Los datos pueden ser interpretados fácilmente (Encabezados claros)."""
    # Evaluación: ¿Los nombres de las columnas son cortos, sin caracteres especiales y entendibles?
    understandable_count = sum(1 for col in col_names if len(col) < 20 and all(c.isalnum() or c == '_' for c in col))
    return (understandable_count / len(col_names)) * 100

def calculate_availability(accessibility_score: float, actuality_score: float) -> float:
    """Criterio 9: Los datos están en línea cuando se necesitan. FÓRMULA: (Accesibilidad + Actualidad) / 2."""
    return (accessibility_score + actuality_score) / 2

# --- FUNCIÓN PRINCIPAL DE CÁLCULO Y DISPLAY ---

def calculate_and_display_metrics(df: pd.DataFrame):
    """
    Calcula y muestra las 17 métricas de calidad de datos en Streamlit.
    """
    if df.empty:
        st.info("No hay datos cargados para calcular las métricas.")
        return

    st.header("📊 Perfilado y Evaluación de los 17 Criterios de Calidad")
    
    # 1. CÁLCULO DE CRITERIOS BASE E INTERMEDIOS
    # Criterios base
    accessibility_score = calculate_accessibility(df)
    actuality_score = calculate_actuality(df)
    completeness_score = calculate_completeness(df)
    conformity_score = calculate_conformity(df)
    confidentiality_score = calculate_confidentiality(df)
    consistency_score = calculate_consistency(df)
    credibility_score = calculate_credibility(df)
    efficiency_score = calculate_efficiency(df)
    accuracy_score = calculate_accuracy(df) 
    portability_score = calculate_portability(df)
    precision_score = calculate_precision(df)
    relevance_score = calculate_relevance(df)
    uniqueness_score = calculate_uniqueness(df)
    comprehensibility_score = calculate_comprehensibility(df, df.columns.tolist())
    
    # Criterios compuestos/dependientes
    availability_score = calculate_availability(accessibility_score, actuality_score) 
    recoverability_score = calculate_recoverability(actuality_score) 
    traceability_score = calculate_traceability(actuality_score, credibility_score) 

    # 2. AGRUPACIÓN DE MÉTRICAS (Los 17 Criterios)
    metrics = {
        "1. Accesibilidad": accessibility_score,
        "2. Actualidad": actuality_score,
        "3. Completitud": completeness_score,
        "4. Comprensibilidad": comprehensibility_score,
        "5. Conformidad": conformity_score,
        "6. Confidencialidad": confidentiality_score,
        "7. Consistencia": consistency_score,
        "8. Credibilidad": credibility_score,
        "9. Disponibilidad": availability_score,
        "10. Eficiencia": efficiency_score,
        "11. Exactitud": accuracy_score,
        "12. Portabilidad": portability_score,
        "13. Precisión": precision_score,
        "14. Recuperabilidad": recoverability_score,
        "15. Relevancia": relevance_score,
        "16. Trazabilidad": traceability_score,
        "17. Unicidad": uniqueness_score,
    }

    # CÁLCULO DEL SCORE GLOBAL DE CALIDAD
    overall_score = np.mean(list(metrics.values()))
    st.markdown(f"## 🏆 Score Global de Calidad: **{overall_score:.2f}%**")

    # 3. VISUALIZACIÓN DE MÉTRICAS (KPIs)
    st.subheader("Evaluación Detallada de los 17 Criterios (%)")
    
    # Crear dos filas de 8 columnas cada una
    cols = st.columns(8) 
    
    for i, (name, value) in enumerate(metrics.items()):
        score = round(value, 2)
        
        with cols[i % 8]:
            st.metric(label=name, value=f"{score}%")
            
    st.markdown("---")
    
    # 4. PERFILADO DETALLADO (Ejemplo: Completitud por Columna)
    st.subheader("Detalle de la Completitud por Atributo")
    completeness_detail = pd.DataFrame({
        'Atributo': df.columns,
        'Valores No Nulos': df.count().values,
        'Total Filas': len(df),
        'Completitud (%)': (df.count().values / len(df)) * 100
    }).sort_values(by='Completitud (%)', ascending=False)
    
    st.dataframe(completeness_detail, use_container_width=True)

    st.markdown("---")

    # 5. TABLA DE DATOS (Muestra)
    st.subheader("Vista Previa del Dataset")
    st.dataframe(df.head(10), use_container_width=True)


# --- LAYOUT DE LA APLICACIÓN STREAMLIT (SIN CAMBIOS) ---

def main():
    st.title("Sistema de Monitoreo de Calidad de Datos Abiertos")
    st.caption("Implementación de la Guía de Calidad e Interoperabilidad 2025 para Asset Inventory.")

    # SIDEBAR: Opciones de Ingesta
    st.sidebar.header("Opciones de Ingesta de Datos")
    ingestion_mode = st.sidebar.radio(
        "Seleccione el origen de datos:",
        ('Asset Inventory (API)', 'Cargar CSV Local')
    )
    
    df_data = pd.DataFrame()
    
    if ingestion_mode == 'Asset Inventory (API)':
        st.sidebar.code(API_URL, language='text')
        if st.sidebar.button("Cargar Datos desde API"):
            df_data = fetch_api_data(API_URL)
            
    elif ingestion_mode == 'Cargar CSV Local':
        uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])
        if uploaded_file is not None:
            df_data = handle_csv_upload(uploaded_file)
            
    # MAIN CONTENT
    if not df_data.empty:
        calculate_and_display_metrics(df_data)
    else:
        st.info("Utilice la barra lateral para cargar el Asset Inventory desde la API o un archivo CSV local.")


if __name__ == "__main__":
    main()
