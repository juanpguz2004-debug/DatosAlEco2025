import streamlit as st
import pandas as pd
import requests
from io import StringIO
import numpy as np

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Dashboard de Calidad de Datos Abiertos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTES ---
# URL para el Asset Inventory de datos.gov.co
API_URL = "https://www.datos.gov.co/resource/uzcf-b9dh.json?$limit=100000"

# --- FUNCIONES DE INGESTA DE DATOS (SIN CAMBIOS) ---

@st.cache_data(show_spinner="Conectando a la API y cargando datos...")
def fetch_api_data(url: str) -> pd.DataFrame:
    """
    Conecta a la API de Socrata y descarga los datos.
    """
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
    """
    Maneja la carga de archivos CSV por parte del usuario.
    """
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"CSV cargado exitosamente. Filas: {len(df)}")
        return df
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")
        return pd.DataFrame()

# --- FUNCIONES DE CÁLCULO DE MÉTRICAS DE CALIDAD (ACTUALIZADAS) ---

# Función auxiliar para manejar la columna de fecha (común en Socrata)
def get_date_column(df: pd.DataFrame, potential_names=['updated_at', 'fecha_actualizacion', 'created_at']) -> str:
    """Busca la columna de fecha más probable o retorna None."""
    for col in potential_names:
        if col in df.columns:
            return col
    return None

def calculate_completeness(df: pd.DataFrame) -> float:
    """
    Cálculo de Completitud (Criterio 3.8). 
    FÓRMULA ESTÁNDAR: (Número de celdas no nulas) / (Número total de celdas)
    """
    if df.empty:
        return 0.0
    total_cells = df.size
    non_null_cells = df.count().sum()
    return (non_null_cells / total_cells) * 100

def calculate_uniqueness(df: pd.DataFrame) -> float:
    """
    Cálculo de Unicidad (Criterio 3.15).
    FÓRMULA ESTÁNDAR: (Número de filas únicas) / (Número total de filas)
    """
    if df.empty:
        return 0.0
    total_rows = len(df)
    unique_rows = len(df.drop_duplicates())
    return (unique_rows / total_rows) * 100

def calculate_conformity(df: pd.DataFrame) -> float:
    """
    Cálculo de Conformidad (Criterio 3.6).
    **¡ATENCIÓN!** Se debe implementar la función de penalización exponencial de la guía.
    
    Placeholder mejorado: Busca la columna 'resource_type' y valida si es no nula.
    """
    column = 'resource_type'
    if df.empty or column not in df.columns:
        return 0.0
    # Ejemplo: Si el tipo de recurso es un valor esperado (no nulo)
    conforming_rows = df[column].notna().sum()
    total_rows = len(df)
    # Dejo un 50% de score base si existe el dataset. La lógica de penalización de la guía debe ir aquí.
    return 50.0 * (conforming_rows / total_rows)

def calculate_syntactic_accuracy(df: pd.DataFrame) -> float:
    """
    Cálculo de Exactitud Sintáctica (Criterio 3.7.1).
    **¡ATENCIÓN!** Reemplazar con la validación de formatos específicos de la guía.
    
    Placeholder mejorado: Valida si la columna de fecha principal es interpretable como fecha.
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
        return 0.0 # Si falla la conversión general

def calculate_actuality(df: pd.DataFrame) -> float:
    """
    Cálculo de Actualidad (Criterio 3.4).
    **¡ATENCIÓN!** Reemplazar con el criterio de antigüedad máximo aceptable de la guía.
    
    Placeholder mejorado: Evalúa qué porcentaje de fechas fueron actualizadas en el último año.
    """
    date_column = get_date_column(df)
    if df.empty or date_column is None:
        return 0.0

    try:
        df_copy = df.copy()
        # Forzar el formato, colocando NaT si hay error
        df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce') 
        df_copy.dropna(subset=[date_column], inplace=True) # Solo filas con formato correcto

        one_year_ago = pd.Timestamp.now() - pd.DateOffset(years=1)

        # Contar cuántos registros fueron actualizados en el último año
        recent_count = df_copy[df_copy[date_column] >= one_year_ago].shape[0]
        total_rows = len(df_copy)
        
        if total_rows == 0:
            return 0.0

        return (recent_count / total_rows) * 100

    except Exception:
        return 0.0

# --- NUEVAS FUNCIONES SEGÚN LA GUÍA ---

def calculate_accessibility(df: pd.DataFrame) -> float:
    """
    Cálculo del Criterio de Accesibilidad (Criterio 3.14).
    En el contexto de un Asset Inventory cargado exitosamente, se asume el 100%.
    """
    # Para una implementación más completa, aquí se verificarían formatos abiertos, etc.
    return 100.0 if not df.empty else 0.0

def calculate_availability(accessibility_score: float, actuality_score: float) -> float:
    """
    Cálculo de Disponibilidad (Criterio 3.18).
    FÓRMULA DE LA GUÍA: (accesibilidad + actualidad) / 2 
    """
    return (accessibility_score + actuality_score) / 2

# --- FUNCIÓN PRINCIPAL DE CÁLCULO Y DISPLAY ---

def calculate_and_display_metrics(df: pd.DataFrame):
    """
    Calcula y muestra las métricas de calidad de datos en Streamlit.
    """
    if df.empty:
        st.info("No hay datos cargados para calcular las métricas.")
        return

    st.header("📊 Perfilado y Métricas de Calidad de Datos")

    # 1. CÁLCULO DE MÉTRICAS BASE
    completeness_score = calculate_completeness(df)
    uniqueness_score = calculate_uniqueness(df)
    conformity_score = calculate_conformity(df)
    syntactic_accuracy_score = calculate_syntactic_accuracy(df)
    
    # 2. CÁLCULO DE MÉTRICAS COMPUESTAS / NUEVAS
    actuality_score = calculate_actuality(df) # Criterio 3.4
    accessibility_score = calculate_accessibility(df) # Criterio 3.14
    availability_score = calculate_availability(accessibility_score, actuality_score) # Criterio 3.18 (Corregido)

    metrics = {
        # 6 Métricas originales (algunas con lógica mejorada)
        "Completitud": completeness_score,
        "Unicidad": uniqueness_score,
        "Conformidad": conformity_score,
        "Exactitud Sintáctica": syntactic_accuracy_score,
        "Actualidad": actuality_score,
        "Accesibilidad": accessibility_score,
        "Disponibilidad": availability_score,
        
        # FALTAN DE IMPLEMENTAR:
        # Trazabilidad, Exactitud Semántica, Confidencialidad, Consistencia,
        # Precisión, Portabilidad, Credibilidad, Comprensibilidad, Eficiencia, Recuperabilidad, Relevancia.
        # **AÑADIR AQUÍ LOS 10 CRITERIOS RESTANTES**
    }

    # 3. VISUALIZACIÓN DE MÉTRICAS (KPIs)
    st.subheader("Métricas Clave de Calidad (%)")
    
    # Mostrar todas las métricas implementadas
    cols = st.columns(len(metrics))
    i = 0
    for name, value in metrics.items():
        score = round(value, 2)
        
        with cols[i % len(cols)]:
            st.metric(label=name, value=f"{score}%")
        i += 1
        
    st.info("🚨 **AVISO:** Faltan por implementar 10 criterios (Confidencialidad, Relevancia, Trazabilidad, Exactitud Semántica, Consistencia, Precisión, Portabilidad, Credibilidad, Comprensibilidad, Eficiencia y Recuperabilidad).")

    st.markdown("---")
    
    # 4. PERFILADO DETALLADO (Ejemplo: Completitud por Columna)
    st.subheader("Detalle: Completitud por Atributo")
    # ... (código sin cambios)
    completeness_detail = pd.DataFrame({
        'Atributo': df.columns,
        'Valores No Nulos': df.count().values,
        'Total Filas': len(df),
        'Completitud (%)': (df.count().values / len(df)) * 100
    }).sort_values(by='Completitud (%)', ascending=True)
    
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
