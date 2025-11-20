import streamlit as st
import pandas as pd
import numpy as np
import requests # Necesario para la conexión a la API
from datetime import datetime
import warnings

# Ocultar advertencias de Pandas/Streamlit
warnings.filterwarnings('ignore') 

# =================================================================
# 0. VARIABLES GLOBALES Y CONFIGURACIÓN
# =================================================================

# --- NUEVA FUENTE DE DATOS: API Socrata ---
API_URL = "https://www.datos.gov.co/resource/uzcf-b9dh.json"
API_LIMIT = 100000 # Límite para obtener un conjunto grande de datos

# Mapeo de columnas: Nombre de la API (con acento/guion) -> Nombre Final Estándar
COL_RENAME_MAP = {
    't_tulo': 'titulo', 
    'descripci_n': 'descripcion', 
    'fecha_actualizaci_n': 'fecha_actualizacion' 
}

# Columnas finales esperadas para todo el sistema de evaluación (sin acentos)
COLUMNAS_CLAVE = [
    'uid', 'dueño', 'tema', 'titulo', 'descripcion', 
    'fecha_actualizacion', 'formato', 'vistas', 'descargas'
] 

# =================================================================
# 1. Funciones de Carga de Datos (Desde API)
# =================================================================

@st.cache_data(ttl=3600) # Caching por 1 hora para evitar recargar la API
def load_data_from_api(url, limit):
    """
    Carga los datos directamente desde la API de Socrata.
    Asegura la robustez de las columnas para evitar fallos.
    """
    st.info(f"Cargando datos de la API: {url}?$limit={limit}. Esto puede tardar unos segundos...")
    try:
        response = requests.get(f"{url}?$limit={limit}")
        response.raise_for_status() # Lanza una excepción para errores HTTP
        
        data = response.json()
        df = pd.DataFrame(data)
        
        if df.empty:
            st.error("La API devolvió un conjunto de datos vacío.")
            return pd.DataFrame()

        # 1. Renombrar columnas
        rename_map = {old: new for old, new in COL_RENAME_MAP.items() if old in df.columns}
        df.rename(columns=rename_map, inplace=True)

        # 2. GARANTIZAR LA EXISTENCIA de las columnas clave (Robustez)
        for col in COLUMNAS_CLAVE:
            if col not in df.columns:
                if col == 'fecha_actualizacion':
                    df[col] = pd.NaT 
                elif col in ['vistas', 'descargas']:
                    df[col] = 0.0 
                else:
                    df[col] = np.nan 
        
        # 3. Filtrar y ordenar las columnas clave
        df = df[COLUMNAS_CLAVE] 
        
        st.success(f"Carga exitosa: {len(df)} activos procesados.")
        return df
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error al conectar o recibir datos de la API: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error inesperado durante la carga o procesamiento: {e}")
        return pd.DataFrame()

# =================================================================
# 2. Funciones de Evaluación de Métricas Estrictas (Guía)
# =================================================================

@st.cache_data
def calculate_quality_metrics(df_base):
    """
    Calcula las métricas de calidad fundamentales basadas en la Guía.
    Devuelve scores de 0 a 100 para Completitud, Unicidad y Actualidad.
    Devuelve un score de 0 a 10 para Relevancia.
    """
    df = df_base.copy()
    n_cols = df.shape[1]
    today = datetime.now()

    # 1. CRITERIO DE COMPLETITUD (3.8 - Densidad de datos)
    # Se evalúa el porcentaje de columnas no nulas por fila.
    df['completitud_score'] = (df.notna().sum(axis=1) / n_cols) * 100
    
    # 2. CRITERIO DE UNICIDAD (3.15 - Duplicados semánticos)
    # Se detectan duplicados en las columnas 'titulo' y 'dueño' (identificador)
    df['es_duplicado_semantico'] = df.duplicated(subset=['titulo', 'dueño'], keep='first')
    df['unicidad_score'] = np.where(df['es_duplicado_semantico'], 0.0, 100.0) # 0 si es duplicado, 100 si es único

    # 3. CRITERIO DE ACTUALIDAD (3.4 - Días desde la última actualización)
    # Puntuación: 100 si es reciente, 0 si supera 365 días de antigüedad.
    df['fecha_actualizacion'] = pd.to_datetime(df['fecha_actualizacion'], errors='coerce', utc=True)
    
    # Calcular días de antigüedad (maneja NaT/NaN)
    antiguedad_dias = (today.replace(tzinfo=None) - df['fecha_actualizacion'].dt.tz_localize(None)).dt.days
    antiguedad_dias.fillna(365, inplace=True) # Si la fecha es NaT/NaN, se asume la antigüedad máxima penalizable (o más)
    df['dias_antiguedad'] = antiguedad_dias
    
    # Score de Actualidad: 100 - (Antigüedad / 365) * 100, con mínimo 0
    df['actualidad_score'] = np.clip(100 - (df['dias_antiguedad'] / 365 * 100), 0, 100)
    
    # 4. CRITERIO DE RELEVANCIA/POTENCIAL DE USO (3.3 / 5.1.3 - CTR)
    # CTR = Descargas / Vistas (ajustado para la guía)
    df['vistas'] = pd.to_numeric(df['vistas'], errors='coerce').fillna(0)
    df['descargas'] = pd.to_numeric(df['descargas'], errors='coerce').fillna(0)
    
    # CTR (tasa de conversión: descargas / vistas)
    df['ctr'] = np.where(df['vistas'] > 0, df['descargas'] / df['vistas'], 0)
    df['ctr'] = np.clip(df['ctr'], 0, 1) # Asegura que el CTR esté entre 0 y 1
    
    # El score de relevancia se calcula de 0 a 10 (multiplicando CTR por 10)
    df['relevancia_score'] = df['ctr'] * 10 
    
    return df

# =================================================================
# 3. Lógica Principal de la Aplicación (Streamlit)
# =================================================================

def main_app_logic():
    st.set_page_config(layout="wide", page_title="Evaluador Básico de Calidad de Datos (API)")

    st.title("Paso 1: Carga desde API y Evaluación Estricta de Métricas 📊")
    st.markdown("---")

    # 1. Carga de Datos desde la API
    df_loaded = load_data_from_api(API_URL, API_LIMIT)

    if df_loaded.empty:
        st.error("No se pudo cargar la data de la API o la API está vacía. No se puede realizar la evaluación.")
        return

    # 2. Evaluación de Métricas de Calidad
    st.header("Resultados de la Evaluación Estricta de Métricas")
    st.markdown("Se han calculado los scores de: **Completitud**, **Unicidad**, **Actualidad** y **Relevancia** (CTR) de 0 a 100 (excepto Relevancia de 0 a 10).")
    
    df_metrics = calculate_quality_metrics(df_loaded)

    # 3. Mostrar las métricas calculadas
    st.subheader("DataFrame con Scores de Calidad Agregados")
    
    # Seleccionamos las columnas clave y los scores para una vista limpia
    cols_to_display = [
        'titulo', 'dueño', 'completitud_score', 'unicidad_score', 
        'actualidad_score', 'relevancia_score', 'dias_antiguedad', 'es_duplicado_semantico'
    ]
    
    st.dataframe(df_metrics[cols_to_display].head(50), 
                 use_container_width=True,
                 column_config={
                    "completitud_score": st.column_config.ProgressColumn("Completitud (%)", format="%f", min_value=0, max_value=100),
                    "unicidad_score": st.column_config.ProgressColumn("Unicidad (%)", format="%f", min_value=0, max_value=100),
                    "actualidad_score": st.column_config.ProgressColumn("Actualidad (%)", format="%f", min_value=0, max_value=100),
                    "relevancia_score": st.column_config.ProgressColumn("Relevancia (0-10)", format="%f", min_value=0, max_value=10),
                 })

    st.subheader("Estadísticas Resumen de Calidad")
    
    # Calculamos y mostramos los promedios de las métricas
    resumen = df_metrics[['completitud_score', 'unicidad_score', 'actualidad_score', 'relevancia_score']].mean().to_frame().T
    resumen.columns = ['Completitud Promedio (%)', 'Unicidad Promedio (%)', 'Actualidad Promedio (%)', 'Relevancia Promedio (0-10)']

    st.dataframe(resumen, use_container_width=True, hide_index=True)


# Ejecución
if __name__ == "__main__":
    main_app_logic()
