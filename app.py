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

# --- FUNCIONES DE INGESTA DE DATOS ---

@st.cache_data(show_spinner="Conectando a la API y cargando datos...")
def fetch_api_data(url: str) -> pd.DataFrame:
    """
    Conecta a la API de Socrata y descarga los datos.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lanza excepción para códigos de error HTTP
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

# --- FUNCIONES DE CÁLCULO DE MÉTRICAS DE CALIDAD (PLACEHOLDERS) ---

def calculate_completeness(df: pd.DataFrame) -> float:
    """
    Cálculo de Completitud.
    FÓRMULA ESTÁNDAR: (Número de celdas no nulas) / (Número total de celdas)

    **¡ATENCIÓN!** Reemplazar con la fórmula de la guía (e.g., completitud por atributo
    o un promedio ponderado si la guía lo especifica).
    """
    if df.empty:
        return 0.0
    total_cells = df.size
    non_null_cells = df.count().sum()
    return (non_null_cells / total_cells) * 100

def calculate_uniqueness(df: pd.DataFrame) -> float:
    """
    Cálculo de Unicidad.
    FÓRMULA ESTÁNDAR: (Número de filas únicas) / (Número total de filas)
    Se calcula sobre todas las filas, asumiendo unicidad de registro.

    **¡ATENCIÓN!** Reemplazar con la fórmula de la guía. Podría ser unicidad
    de una columna clave específica (ej: 'id') si la guía lo requiere.
    """
    if df.empty:
        return 0.0
    total_rows = len(df)
    unique_rows = len(df.drop_duplicates())
    return (unique_rows / total_rows) * 100

def calculate_conformity(df: pd.DataFrame, column: str = 'entity_type') -> float:
    """
    Cálculo de Conformidad (Ejemplo basado en una columna).
    FÓRMULA ESTÁNDAR: % de valores que cumplen un patrón o un conjunto de valores esperados.
    Aquí se usa un ejemplo simple de si hay valores nulos en el 'entity_type'.

    **¡ATENCIÓN!** Reemplazar con la fórmula de la guía. La conformidad requiere
    reglas de negocio específicas (e.g., formato de fechas, rangos de valores).
    """
    if df.empty or column not in df.columns:
        return 0.0
    # Ejemplo: Si el tipo de entidad es un valor esperado (no nulo)
    conforming_rows = df[column].notna().sum()
    total_rows = len(df)
    return (conforming_rows / total_rows) * 100

def calculate_syntactic_accuracy(df: pd.DataFrame, column: str = 'updated_at') -> float:
    """
    Cálculo de Exactitud Sintáctica (Ejemplo de formato de fecha/hora).
    FÓRMULA ESTÁNDAR: % de valores que cumplen un formato sintáctico esperado.

    **¡ATENCIÓN!** Reemplazar con la fórmula de la guía. Requiere validación de formatos.
    """
    if df.empty or column not in df.columns:
        return 0.0
    # Intentamos convertir la columna a datetime. Si es posible, se considera sintácticamente correcta.
    try:
        correct_format_count = pd.to_datetime(df[column], errors='coerce').notna().sum()
        total_rows = len(df)
        return (correct_format_count / total_rows) * 100
    except Exception:
        return 0.0 # Si falla la conversión general

def calculate_availability(df: pd.DataFrame) -> float:
    """
    Cálculo de Disponibilidad.
    En el contexto de un dataset, si se cargó exitosamente, se asume 100%.
    Para una métrica real, esto mediría el tiempo de actividad del servicio (API).

    **¡ATENCIÓN!** Reemplazar con la fórmula de la guía.
    """
    return 100.0 if not df.empty else 0.0

def calculate_actuality(df: pd.DataFrame, date_column: str = 'updated_at') -> float:
    """
    Cálculo de Actualidad (Timeliness).
    FÓRMULA ESTÁNDAR: Se basa en la antigüedad del último registro.
    Aquí se usa una métrica simple: si el 90% de los registros se actualizaron
    en los últimos 365 días (1 año).

    **¡ATENCIÓN!** Reemplazar con la fórmula de la guía. Esto es una conjetura.
    """
    if df.empty or date_column not in df.columns:
        return 0.0

    try:
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
        one_year_ago = pd.Timestamp.now() - pd.DateOffset(years=1)

        # Contar cuántos registros fueron actualizados en el último año
        recent_count = df_copy[df_copy[date_column] >= one_year_ago].shape[0]
        total_rows = len(df_copy)
        return (recent_count / total_rows) * 100

    except Exception:
        return 0.0


# --- FUNCIÓN PRINCIPAL DE CÁLCULO Y DISPLAY ---

def calculate_and_display_metrics(df: pd.DataFrame):
    """
    Calcula y muestra las métricas de calidad de datos en Streamlit.
    """
    if df.empty:
        st.info("No hay datos cargados para calcular las métricas.")
        return

    st.header("📊 Perfilado y Métricas de Calidad de Datos")

    # 1. CÁLCULO DE MÉTRICAS
    metrics = {
        "Completitud": calculate_completeness(df),
        "Unicidad": calculate_uniqueness(df),
        "Conformidad": calculate_conformity(df),
        "Exactitud Sintáctica": calculate_syntactic_accuracy(df),
        "Actualidad": calculate_actuality(df),
        "Disponibilidad": calculate_availability(df),
        # **AÑADIR AQUÍ EL RESTO DE LAS 17 MÉTRICAS**
        # 'Confidencialidad': formula_confidencialidad(df),
        # 'Trazabilidad': formula_trazabilidad(df),
        # 'Exactitud Semántica': formula_exactitud_semantica(df),
        # 'Portabilidad': formula_portabilidad(df),
        # etc.
    }

    # 2. VISUALIZACIÓN DE MÉTRICAS (KPIs)
    st.subheader("Métricas Clave de Calidad (%)")
    cols = st.columns(len(metrics))
    
    i = 0
    for name, value in metrics.items():
        score = round(value, 2)
        # Mostrar el valor en una caja (método más visual que el metric)
        if score >= 90:
            color = "green"
        elif score >= 70:
            color = "orange"
        else:
            color = "red"
            
        with cols[i % len(cols)]:
            st.metric(label=name, value=f"{score}%")
        i += 1

    st.markdown("---")
    
    # 3. PERFILADO DETALLADO (Ejemplo: Completitud por Columna)
    st.subheader("Detalle: Completitud por Atributo")
    completeness_detail = pd.DataFrame({
        'Atributo': df.columns,
        'Valores No Nulos': df.count().values,
        'Total Filas': len(df),
        'Completitud (%)': (df.count().values / len(df)) * 100
    }).sort_values(by='Completitud (%)', ascending=True)
    
    st.dataframe(completeness_detail, use_container_width=True)

    st.markdown("---")

    # 4. TABLA DE DATOS (Muestra)
    st.subheader("Vista Previa del Dataset")
    st.dataframe(df.head(10), use_container_width=True)


# --- LAYOUT DE LA APLICACIÓN STREAMLIT ---

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
