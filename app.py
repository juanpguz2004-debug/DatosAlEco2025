import streamlit as st
# La configuración de página DEBE ser lo primero
st.set_page_config(page_title="DataSentinel", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib.ticker import PercentFormatter
import io
from datetime import datetime
import re
import warnings
import os
import base64
import json 
import requests # Necesario para la API
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Ocultar advertencias de Pandas/Streamlit
warnings.filterwarnings('ignore')

# =================================================================
# 0. VARIABLES GLOBALES Y CONFIGURACIÓN
# =================================================================

# URL DEL API DE SOCRATA (Datos.gov.co)
API_URL = "https://www.datos.gov.co/resource/uzcf-b9dh.json"
LIMIT_ROWS = 100000 # Límite para traer todo el dataset

# CRITERIO DE RIESGO
# Umbral de Riesgo Alto (Crítico) - SE MANTIENE EN 3.5 COMO PEDISTE
UMBRAL_RIESGO_ALTO = 3.5

# --- CONFIGURACIÓN DE RIESGOS UNIVERSALES ---
PENALIZACION_DATOS_INCOMPLETOS = 2.0
PENALIZACION_INCONSISTENCIA_TIPO = 0.5
PENALIZACION_DUPLICADO = 1.0
# RIESGO MÁXIMO TEÓRICO UNIVERSAL BASE: 3.5 (Variable según columnas afectadas)

# --- CONFIGURACIÓN DE RIESGOS AVANZADOS (EXPANDIDOS) ---
# **Riesgos Universal/Existentes**
PENALIZACION_INCONSISTENCIA_METADATOS = 1.5 # Inconsistencia de metadatos (ej. frecuencia vs. antigüedad)
PENALIZACION_ANOMALIA_SILENCIOSA = 1.0      # Duplicidad semántica/Cambios abruptos (Anomalía + Baja Popularidad)
PENALIZACION_ACTIVO_VACIO = 2.0           # Activos vacíos en categorías populares

# **Nuevas Penalizaciones Basadas en Criterios Extendidos**
PENALIZACION_CONFIDENCIALIDAD = 1.0       # Público + Falla de Descripción (Confidencialidad Herramientas Datos)
PENALIZACION_TRAZABILIDAD = 1.5           # Dueño desconocido (Trazabilidad Herramientas Datos)
PENALIZACION_CONFORMIDAD_ACTUALIDAD = 2.0 # Incumplimiento O Antigüedad > 1 año (Conformidad Herramientas Datos / Actualidad Herramientas Datos)
PENALIZACION_RELEVANCIA = 1.0            # Baja Popularidad + Alto Riesgo (Relevancia Herramientas Datos)
PENALIZACION_DISPONIBILIDAD = 1.5         # Riesgo Crítico O Incumplimiento (Disponibilidad Herramientas Datos / Recuperabilidad Herramientas Datos / Accesibilidad Herramientas Datos)
PENALIZACION_COMPRENSIBILIDAD = 1.0       # Alto Riesgo + Baja Completitud (Credibilidad/Comprensibilidad/Eficiencia/Portabilidad Herramientas Datos)

# RIESGO MÁXIMO TEÓRICO AVANZADO 
# Ajustado a 15.0 para tener margen con todas las penalizaciones acumulativas
RIESGO_MAXIMO_TEORICO_AVANZADO = 15.0

# CLAVE SECRETA DE GEMINI
try:
    GEMINI_API_SECRET_VALUE = st.secrets["APIKEY"]
except Exception:
    GEMINI_API_SECRET_VALUE = None

# =================================================================
# 1. Funciones de Carga y Procesamiento (API INTEGRATION)
# =================================================================

@st.cache_data(ttl=3600) # Cachear por 1 hora para no saturar el API
def fetch_and_process_api_data():
    """
    Descarga los datos directamente del API de Socrata, normaliza las columnas
    y calcula los scores base necesarios para que el resto de la app funcione.
    """
    try:
        # 1. Petición al API
        params = {'$limit': LIMIT_ROWS}
        response = requests.get(API_URL, params=params)
        response.raise_for_status() # Lanzar error si falla
        
        data = response.json()
        df = pd.DataFrame(data)
        
        if df.empty:
            return pd.DataFrame()

        # 2. Mapeo de columnas (API -> Nombres esperados por la App)
        # Basado en la documentación Socrata proporcionada
        column_mapping = {
            'uid': 'uid',
            'name': 'titulo',
            'description': 'descripcion',
            'owner': 'dueño',
            'category': 'categoria',
            'audience': 'publico', # O 'commoncore_publicaccesslevel' si audience falla
            'commoncore_theme': 'common_core_theme',
            'last_data_updated_date': 'fecha_actualizacion',
            'informacindedatos_frecuenciadeactualizacin': 'frecuencia_actualizacion',
            'visits': 'visitas',
            'downloads': 'descargas'
        }
        
        # Renombrar solo las que existen
        df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
        
        # 3. Procesamiento de Fechas y Antigüedad
        if 'fecha_actualizacion' in df.columns:
            df['fecha_actualizacion'] = pd.to_datetime(df['fecha_actualizacion'], errors='coerce')
            now = pd.Timestamp.now()
            # Calcular antigüedad en días
            df['antiguedad_datos_dias'] = (now - df['fecha_actualizacion']).dt.days
            df['antiguedad_datos_dias'] = df['antiguedad_datos_dias'].fillna(9999) # Penalización por falta de fecha
        else:
            df['antiguedad_datos_dias'] = 0

        # 4. Cálculo de Score de Completitud (Densidad de datos)
        # Porcentaje de columnas no nulas por fila
        n_cols = df.shape[1]
        df['datos_por_fila_score'] = (df.notna().sum(axis=1) / n_cols) * 100
        df['completitud_score'] = df['datos_por_fila_score'] # Mapeo directo para compatibilidad

        # 5. Cálculo de Score de Popularidad (Normalizado)
        # Convertir a numérico
        for col in ['visitas', 'descargas']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0
                
        df['total_interacciones'] = df['visitas'] + df['descargas']
        max_inter = df['total_interacciones'].max()
        if max_inter > 0:
            df['popularidad_score'] = df['total_interacciones'] / max_inter
        else:
            df['popularidad_score'] = 0.0

        # 6. Cálculo de Riesgos Base (Universal)
        # Riesgo Incompletitud
        df['riesgo_datos_incompletos'] = np.where(
            df['completitud_score'] < 70, PENALIZACION_DATOS_INCOMPLETOS, 0.0
        )
        
        # Riesgo Duplicidad
        df['es_duplicado'] = df.duplicated(subset=['titulo', 'dueño'], keep=False) # Heurística de duplicado semántico
        df['riesgo_duplicado'] = np.where(
            df['es_duplicado'], PENALIZACION_DUPLICADO, 0.0
        )
        
        # Riesgo Consistencia (Simplificado para API: ver si hay mezcla de tipos es costoso, asumimos 0 base)
        df['riesgo_consistencia_tipo'] = 0.0

        # SCORE INICIAL DE RIESGO
        df['prioridad_riesgo_score'] = (
            df['riesgo_datos_incompletos'] + 
            df['riesgo_duplicado'] +
            df['riesgo_consistencia_tipo']
        )

        # 7. Estado de Actualización (Heurística simple si no viene del API)
        # Si antigüedad > 365 días -> Incumplimiento (Esto se refina luego en advanced checks)
        df['estado_actualizacion'] = np.where(
            df['antiguedad_datos_dias'] > 365,
            '🔴 INCUMPLIMIENTO',
            '🟢 OK'
        )

        # 8. Limpieza final de Nulos críticos
        df['titulo'] = df['titulo'].fillna("Sin Título")
        df['dueño'] = df['dueño'].fillna("Desconocido")
        df['categoria'] = df['categoria'].fillna("Sin Categoría")
        
        return df

    except Exception as e:
        st.error(f"Error conectando con el API: {e}")
        return pd.DataFrame()

def clean_and_convert_types_external(df):
    """Fuerza a las columnas a ser tipo string para asegurar la detección de inconsistencias."""
    
    # Columnas que suelen ser de tipo 'object' (string)
    object_cols = ['titulo', 'descripcion', 'dueño'] 
    
    data_cols = [col for col in df.columns if col not in object_cols]
    
    for col in data_cols:
        if df[col].dtype != 'object':
            try:
                df[col] = df[col].astype(object) 
            except:
                pass 

    return df

def check_universals_external(df):
    """
    Calcula métricas de calidad universal para archivos externos.
    """
    df_copy = df.copy() 
    n_cols = df_copy.shape[1]
    
    df_copy['datos_por_fila_score'] = (df_copy.notna().sum(axis=1) / n_cols) * 100
    df_copy['riesgo_datos_incompletos'] = np.where(
        df_copy['datos_por_fila_score'] < 70, PENALIZACION_DATOS_INCOMPLETOS, 0.0
    )

    df_copy['riesgo_consistencia_tipo'] = 0.0
    
    object_cols_for_check = [col for col in df_copy.select_dtypes(include='object').columns if col not in ['titulo', 'descripcion', 'dueño']]
    
    for col in object_cols_for_check:
        inconsistencies = df_copy[col].apply(lambda x: not isinstance(x, str) and pd.notna(x))
        df_copy.loc[inconsistencies, 'riesgo_consistencia_tipo'] += PENALIZACION_INCONSISTENCIA_TIPO
        
    df_copy['es_duplicado'] = df_copy.duplicated(keep=False) 
    df_copy['riesgo_duplicado'] = np.where(
        df_copy['es_duplicado'], PENALIZACION_DUPLICADO, 0.0
    )
    
    return df_copy

def process_external_data(df):
    """
    Lógica de riesgo universal para el archivo externo subido.
    """
    
    df = clean_and_convert_types_external(df)
    df = check_universals_external(df)
    
    df['prioridad_riesgo_score'] = (
        df['riesgo_datos_incompletos'] + 
        df['riesgo_consistencia_tipo'] +
        df['riesgo_duplicado']
    )
    
    avg_file_risk = df['prioridad_riesgo_score'].mean()
    quality_score = 100 - (avg_file_risk / RIESGO_MAXIMO_TEORICO_AVANZADO * 100)
    
    df['calidad_total_score'] = np.clip(quality_score, 0, 100)

    return df

# FUNCIÓN PARA DETECCIÓN DE ANOMALÍAS CON ISOLATION FOREST
@st.cache_data
def apply_anomaly_detection(df):
    """
    Detecta anomalías en los activos de datos utilizando Isolation Forest
    basado en métricas clave. Asigna -1 para anomalía y 1 para normal.
    """
    df_copy = df.copy()
    
    # 1. Definir features
    features = ['prioridad_riesgo_score', 'completitud_score', 'antiguedad_datos_dias', 'popularidad_score']
    
    # Verificar existencia de columnas (popularidad_score fue creada en fetch)
    features = [col for col in features if col in df_copy.columns]

    # 2. Preparar los datos
    df_model = df_copy[features].dropna().astype(float)
    
    if len(df_model) < 10: 
        st.sidebar.warning("Advertencia: Menos de 10 filas de datos completos. ML Anomaly Detection se omitirá.")
        df_copy['anomalia_score'] = 1 
        return df_copy
    
    # 3. Inicializar y entrenar Isolation Forest
    iso_forest = IsolationForest(
        random_state=42, 
        contamination='auto',
        n_estimators=100
    )
    
    # 4. Ajustar y predecir
    predictions = iso_forest.fit_predict(df_model)
    
    # 5. Mapear las predicciones
    df_copy['anomalia_score'] = 1 
    df_copy.loc[df_model.index, 'anomalia_score'] = predictions
    
    num_anomalies = (df_copy['anomalia_score'] == -1).sum()
    st.sidebar.markdown(f"**Detección ML:** {num_anomalies} anomalías detectadas.")
    
    return df_copy

# FUNCIÓN PARA CHEQUEOS AVANZADOS
@st.cache_data
def apply_advanced_risk_checks(df):
    """
    Calcula nuevos scores de riesgo avanzados y los añade al score de riesgo existente,
    incorporando los nuevos criterios de la Guía 2025.
    """
    df_copy = df.copy()
    
    # 1. Chequeos Existentes/Universal
    
    # Detección de Inconsistencia de Metadatos
    col_antiguedad = df_copy['antiguedad_datos_dias'] if 'antiguedad_datos_dias' in df_copy.columns else 0
    
    df_copy['riesgo_inconsistencia_metadatos'] = np.where(
        (df_copy['prioridad_riesgo_score'] > UMBRAL_RIESGO_ALTO) & (col_antiguedad < 365), 
        PENALIZACION_INCONSISTENCIA_METADATOS, 
        0.0
    )

    # Duplicidad Semántica/Cambios Abruptos
    if 'anomalia_score' in df_copy.columns:
        df_copy['riesgo_semantico_actualizacion'] = np.where(
            (df_copy['anomalia_score'] == -1) & (df_copy.get('popularidad_score', 0.0) < 0.1),
            PENALIZACION_ANOMALIA_SILENCIOSA,
            0.0
        )
    else:
        df_copy['riesgo_semantico_actualizacion'] = 0.0

    # Activos Vacíos en Categorías Populares
    if 'categoria' in df_copy.columns:
        top_categories = df_copy['categoria'].value_counts().nlargest(5).index.tolist()
        df_copy['riesgo_activos_vacios'] = np.where(
            (df_copy['completitud_score'] < 20.0) & (df_copy['categoria'].isin(top_categories)),
            PENALIZACION_ACTIVO_VACIO,
            0.0
        )
    else:
        df_copy['riesgo_activos_vacios'] = 0.0
    
    # 2. Nuevos Criterios de Riesgo (Heurísticas Ajustadas a la Guía 2025)
    
    # --- A. Confidencialidad ---
    df_copy['riesgo_confidencialidad'] = np.where(
        (df_copy.get('publico') == 'Público') & (df_copy['descripcion'].isna()),
        PENALIZACION_CONFIDENCIALIDAD,
        0.0
    )
    
    # --- B. Trazabilidad ---
    df_copy['riesgo_trazabilidad'] = np.where(
        df_copy['dueño'].isna(),
        PENALIZACION_TRAZABILIDAD,
        0.0
    )

    # --- C. Conformidad ---
    df_copy['riesgo_conformidad'] = np.where(
        df_copy.get('estado_actualizacion') == '🔴 INCUMPLIMIENTO',
        PENALIZACION_CONFORMIDAD_ACTUALIDAD, 
        0.0
    )
    
    # --- D. Actualidad ---
    # Mapeo de frecuencias a días
    freq_map = {
        'Tiempo real': 1, 'Diario': 1, 'Semanal': 7, 'Mensual': 30, 
        'Trimestral': 90, 'Semestral': 180, 'Anual': 365
    }
    
    if 'frecuencia_actualizacion' in df_copy.columns:
        df_copy['freq_days'] = df_copy['frecuencia_actualizacion'].map(freq_map).fillna(365)
        
        df_copy['riesgo_actualidad'] = np.where(
            df_copy.get('antiguedad_datos_dias', 0) > df_copy['freq_days'],
            PENALIZACION_CONFORMIDAD_ACTUALIDAD, 
            0.0
        )
    else:
        df_copy['riesgo_actualidad'] = np.where(
            df_copy.get('antiguedad_datos_dias', 0) > 365,
            PENALIZACION_CONFORMIDAD_ACTUALIDAD,
            0.0
        )
    
    # --- E. Relevancia ---
    df_copy['riesgo_relevancia'] = np.where(
        (df_copy.get('popularidad_score', 0.0) < 0.1) & (df_copy['prioridad_riesgo_score'] > UMBRAL_RIESGO_ALTO),
        PENALIZACION_RELEVANCIA,
        0.0
    )
    
    # --- F. Disponibilidad ---
    df_copy['riesgo_disponibilidad'] = np.where(
        (df_copy['prioridad_riesgo_score'] > RIESGO_MAXIMO_TEORICO_AVANZADO * 0.5) | (df_copy.get('estado_actualizacion') == '🔴 INCUMPLIMIENTO'),
        PENALIZACION_DISPONIBILIDAD,
        0.0
    )
    
    # --- G. Comprensibilidad ---
    df_copy['riesgo_comprensibilidad'] = np.where(
        (df_copy['prioridad_riesgo_score'] > UMBRAL_RIESGO_ALTO) & (df_copy['completitud_score'] < 50.0),
        PENALIZACION_COMPRENSIBILIDAD,
        0.0
    )
    
    # 3. Actualizar el score de riesgo principal
    df_copy['prioridad_riesgo_score_v2'] = (
        df_copy['prioridad_riesgo_score'] + # Riesgos Universales Base
        df_copy['riesgo_inconsistencia_metadatos'] +
        df_copy['riesgo_semantico_actualizacion'] +
        df_copy['riesgo_activos_vacios'] +
        
        # Nuevos criterios
        df_copy['riesgo_confidencialidad'] +
        df_copy['riesgo_trazabilidad'] +
        df_copy['riesgo_conformidad'] + 
        df_copy['riesgo_actualidad'] +
        df_copy['riesgo_relevancia'] +
        df_copy['riesgo_disponibilidad'] +
        df_copy['riesgo_comprensibilidad']
    )
    
    # Sustituir el score principal
    df_copy['prioridad_riesgo_score'] = df_copy['prioridad_riesgo_score_v2']
    df_copy.drop(columns=['prioridad_riesgo_score_v2'], inplace=True, errors='ignore')
    
    # Asegurar que el score no exceda el máximo teórico
    df_copy['prioridad_riesgo_score'] = np.clip(df_copy['prioridad_riesgo_score'], 0, RIESGO_MAXIMO_TEORICO_AVANZADO)

    return df_copy

# Función de Generación de Reporte HTML
def generate_report_html(df_filtrado, umbral_riesgo):
    # 1. Preparación de Datos
    total_activos = len(df_filtrado)
    riesgo_promedio_general = df_filtrado['prioridad_riesgo_score'].mean()
    completitud_promedio_general = df_filtrado['completitud_score'].mean()
    
    df_top_riesgo = df_filtrado.sort_values(by='prioridad_riesgo_score', ascending=False).head(10).copy()
    df_top_riesgo = df_top_riesgo[['titulo', 'prioridad_riesgo_score', 'completitud_score', 'dueño']].rename(columns={'prioridad_riesgo_score': 'Riesgo Score', 'completitud_score': 'Completitud Score', 'dueño': 'Entidad'}).reset_index(drop=True)
    df_top_riesgo['Nivel Riesgo'] = df_top_riesgo['Riesgo Score'].apply(lambda x: 'Alto' if x > umbral_riesgo else 'Bajo/Medio')
    
    df_riesgo_entidad = df_filtrado.groupby('dueño').agg(
        Activos_Totales=('uid', 'count'),
        Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
        Completitud_Promedio=('completitud_score', 'mean')
    ).reset_index().sort_values(by='Riesgo_Promedio', ascending=False).head(5)
    
    # HTML Simplificado
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Reporte Data Sentinel</title></head>
    <body style="font-family: Arial;">
    <h1>Reporte Final - API Data Sentinel</h1>
    <p>Fecha: {datetime.now().strftime("%Y-%m-%d")}</p>
    <h2>Resumen</h2>
    <ul>
        <li>Activos: {total_activos}</li>
        <li>Riesgo Promedio: {riesgo_promedio_general:.2f}</li>
        <li>Completitud Promedio: {completitud_promedio_general:.2f}%</li>
    </ul>
    <h2>Top 10 Riesgos</h2>
    {df_top_riesgo.to_html(index=False)}
    </body>
    </html>
    """
    return html_content

def get_table_download_link(html_content, filename, text):
    b64 = base64.b64encode(html_content.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 5px; font-family: Arial, sans-serif;">{text}</a>'
    return href

def generate_specific_recommendation(risk_dimension):
    if 'Datos Incompletos' in risk_dimension:
        return "Revisar procesos de ingesta. Llenar campos obligatorios."
    elif 'Duplicados' in risk_dimension:
        return "Eliminar copias exactas y verificar claves únicas."
    else:
        return "Normalizar tipos de datos."

# =================================================================
# 2. FUNCIÓN DE GENERACIÓN DE CONTEXTO DINÁMICO
# =================================================================

def generate_dynamic_context(df_in):
    """
    Genera un string de texto con resúmenes estadísticos exhaustivos
    basados en el DataFrame actual (filtrado) para el Agente de IA.
    """
    if df_in.empty:
        return "No hay datos disponibles en la vista actual."

    df_temp = df_in.copy()
    
    # --- B. ESTADÍSTICAS GLOBALES CLAVE (KPIs) ---
    total_activos = len(df_temp)
    riesgo_promedio_total = df_temp['prioridad_riesgo_score'].mean()
    completitud_promedio_total = df_temp['completitud_score'].mean()
    
    activos_publicos = 0
    if 'publico' in df_temp.columns:
        # En API socrata 'audience' suele ser 'Público' o 'Public'
        activos_publicos = int(df_temp['publico'].astype(str).str.contains('Público|Public', case=False, na=False).sum())

    anomalias_detectadas = int((df_temp['anomalia_score'] == -1).sum()) if 'anomalia_score' in df_temp.columns else 0

    global_kpis = {
        "Total de Activos (Vista Actual)": int(total_activos),
        "Activos Públicos": f"{activos_publicos}",
        "Riesgo Promedio": f"{riesgo_promedio_total:.2f} (Escala 0-15)",
        "Completitud Promedio": f"{completitud_promedio_total:.2f}%",
        "Anomalías Detectadas": anomalias_detectadas
    }
    
    context_str = f"## RESUMEN GLOBAL (KPIs)\n{json.dumps(global_kpis, indent=4)}\n\n"

    # --- C. TOP ENTIDADES ---
    if 'dueño' in df_temp.columns:
        resumen_entidades = df_temp.groupby('dueño').agg(
            Riesgo=('prioridad_riesgo_score', 'mean'),
            Count=('uid', 'count')
        ).reset_index().sort_values(by='Riesgo', ascending=False).head(5)
        context_str += f"## TOP 5 ENTIDADES CON MAYOR RIESGO PROMEDIO:\n{resumen_entidades.to_markdown(index=False, floatfmt='.2f')}\n\n"

    # --- E. TOP RIESGOS INDIVIDUALES ---
    cols_select = ['titulo', 'dueño', 'prioridad_riesgo_score']
    cols_select = [c for c in cols_select if c in df_temp.columns]
    
    if cols_select:
        top_riesgo = df_temp[cols_select].sort_values(by='prioridad_riesgo_score', ascending=False).head(5)
        context_str += f"## TOP 5 ACTIVOS INDIVIDUALES MÁS RIESGOSOS:\n{top_riesgo.to_markdown(index=False, floatfmt='.2f')}\n"

    return context_str

# =================================================================
# 3. FUNCIÓN DE AGENTE DE IA
# =================================================================

def generate_ai_response(user_query, df_current, model_placeholder):
    dynamic_context = generate_dynamic_context(df_current)

    if not GENAI_AVAILABLE:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with model_placeholder.chat_message("assistant"):
            st.error("Librería google-genai no está instalada.")
        return

    try:
        client = genai.Client(api_key=GEMINI_API_SECRET_VALUE)
        system_prompt = (
            "Eres un experto en Datos Abiertos. Analiza el siguiente contexto de datos EN VIVO "
            "del dashboard y responde la pregunta del usuario. "
            f"\n\nCONTEXTO:\n{dynamic_context}"
        )

        with model_placeholder.chat_message("assistant"):
            with st.spinner("Consultando IA..."):
                response = client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=[{"role": "user", "parts": [{"text": user_query}]}],
                    config=genai.types.GenerateContentConfig(system_instruction=system_prompt)
                )
                ai_response = response.text
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})

    except Exception as e:
        with model_placeholder.chat_message("assistant"):
            st.error(f"Error IA: {e}")

# =================================================================
# 4. EJECUCIÓN PRINCIPAL
# =================================================================

st.title("Data Sentinel : Priorización de Activos de Datos (Live API)")

try:
    with st.spinner('Conectando con Datos.gov.co (Socrata API)...'):
        # ⚠️ CAMBIO PRINCIPAL: Usamos la función de API en lugar de cargar CSV
        df_analisis_completo = fetch_and_process_api_data()

    if df_analisis_completo.empty:
        st.error("No se pudieron cargar datos del API. Verifique la conexión o el endpoint.")
    else:
        # APLICAR MODELOS Y RIESGOS (Pipeline normal)
        df_analisis_completo = apply_anomaly_detection(df_analisis_completo)
        df_analisis_completo = apply_advanced_risk_checks(df_analisis_completo) 
        
        st.success(f'Datos actualizados desde API. Total activos: {len(df_analisis_completo)}')
        
        # --- Inicialización de variables de estado ---
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # ----------------------------------------------------------------------
        # --- FILTROS (Mantenidos igual) ---
        # ----------------------------------------------------------------------
        with st.sidebar:
            st.header("Filtros para Visualizaciones")
            
            filtro_acceso_publico = False 
            if 'publico' in df_analisis_completo.columns:
                filtro_acceso_publico = st.checkbox("Mostrar Solo Activos públicos", value=False)
            
            filtro_categoria = "Mostrar Todos"
            if 'categoria' in df_analisis_completo.columns:
                categories = sorted(df_analisis_completo['categoria'].astype(str).unique().tolist())
                categories.insert(0, "Mostrar Todos")
                filtro_categoria = st.selectbox("Filtrar por Categoría:", categories)
                
            filtro_tema = "Mostrar Todos" 
            if 'common_core_theme' in df_analisis_completo.columns:
                themes = sorted(df_analisis_completo['common_core_theme'].astype(str).unique().tolist())
                themes.insert(0, "Mostrar Todos")
                filtro_tema = st.selectbox("Tema:", themes)
                
            st.markdown("---")
            if st.button("Generar Reporte HTML"):
                report_html = generate_report_html(df_analisis_completo, UMBRAL_RIESGO_ALTO) 
                filename = f"Reporte_API_{datetime.now().strftime('%Y%m%d')}.html"
                st.markdown(get_table_download_link(report_html, filename, "Descargar Reporte"), unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # --- CONTENIDO PRINCIPAL ---
        # ----------------------------------------------------------------------
        
        owners = sorted(df_analisis_completo['dueño'].astype(str).unique().tolist())
        owners.insert(0, "Mostrar Análisis General")
        
        filtro_dueño = st.selectbox("Selecciona Entidad:", owners)
        
        # APLICAR FILTROS
        df_filtrado = df_analisis_completo.copy()
        
        if filtro_dueño != "Mostrar Análisis General":
             df_filtrado = df_filtrado[df_filtrado['dueño'] == filtro_dueño]
        if filtro_acceso_publico and 'publico' in df_filtrado.columns:
             # Filtro laxo para 'Publico', 'Público', etc.
             df_filtrado = df_filtrado[df_filtrado['publico'].astype(str).str.contains('Public|Público', case=False, na=False)]
        if filtro_categoria != "Mostrar Todos":
            df_filtrado = df_filtrado[df_filtrado['categoria'] == filtro_categoria]
        if filtro_tema != "Mostrar Todos":
            df_filtrado = df_filtrado[df_filtrado['common_core_theme'] == filtro_tema]
            
        st.header("Visualizaciones en Tiempo Real")
        st.info(f"Vista actual: **{len(df_filtrado)} activos**")

        if not df_filtrado.empty:
            
            # KPIs
            c1, c2, c3 = st.columns(3)
            c1.metric("Completitud Promedio", f"{df_filtrado['completitud_score'].mean():.2f}%")
            inc = (df_filtrado['estado_actualizacion'] == '🔴 INCUMPLIMIENTO').sum() if 'estado_actualizacion' in df_filtrado.columns else 0
            c2.metric("Incumplimiento Actualización", f"{inc}")
            anom = (df_filtrado['anomalia_score'] == -1).sum() if 'anomalia_score' in df_filtrado.columns else 0
            c3.metric("Anomalías ML", f"{anom}")
            
            st.markdown("---")

            # TABLA DETALLE
            st.subheader("Top Activos Riesgosos (Vista Filtrada)")
            cols_show = ['titulo', 'prioridad_riesgo_score', 'completitud_score', 'antiguedad_datos_dias']
            cols_show = [c for c in cols_show if c in df_filtrado.columns]
            
            st.dataframe(
                df_filtrado[cols_show].sort_values(by='prioridad_riesgo_score', ascending=False).head(50).style.format({
                    'prioridad_riesgo_score': '{:.2f}', 
                    'completitud_score': '{:.2f}%',
                    'antiguedad_datos_dias': '{:.0f}'
                }),
                use_container_width=True
            )
            
            # PESTAÑAS GRÁFICOS
            tab1, tab2 = st.tabs(["Distribución de Riesgo", "Mapa de Calor (Treemap)"])
            
            with tab1:
                if 'prioridad_riesgo_score' in df_filtrado.columns:
                    fig = px.histogram(df_filtrado, x="prioridad_riesgo_score", nbins=20, title="Distribución del Score de Riesgo")
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                if 'categoria' in df_filtrado.columns:
                    df_tree = df_filtrado.groupby('categoria').agg(Count=('uid','count'), Riesgo=('prioridad_riesgo_score','mean')).reset_index()
                    fig_tree = px.treemap(df_tree, path=['categoria'], values='Count', color='Riesgo', color_continuous_scale='RdYlGn_r')
                    st.plotly_chart(fig_tree, use_container_width=True)

            # ----------------------------------------------------------------------
            # ASISTENTE DE DATOS
            # ----------------------------------------------------------------------
            st.markdown("<hr>", unsafe_allow_html=True)
            st.header("Asistente IA (Datos en Vivo)")
            
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            if prompt := st.chat_input("Pregunta sobre estos datos..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    model_response_placeholder = st.empty() 
                    generate_ai_response(prompt, df_filtrado, model_response_placeholder)

        else:
            st.warning("No hay datos con los filtros actuales.")

except Exception as e:
    st.error(f"ERROR CRÍTICO: {e}")
