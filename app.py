import streamlit as st
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
# --- ADICIÓN: Importación para la descarga de archivos ---
import base64
# --- FIN ADICIÓN ---

# --- Importaciones para el Agente de IA (Usando API nativa de Gemini) ---
from google import genai 
# --- FIN DE IMPORTACIÓN DE GEMINI ---

# --- NUEVAS IMPORTACIONES PARA CLUSTERING NO SUPERVISADO (K-MEANS) ---
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# 🚀 ADICIÓN: Importación para Detección de Anomalías con ML
from sklearn.ensemble import IsolationForest
# --- FIN DE NUEVAS IMPORTACIONES ---

warnings.filterwarnings('ignore') # Ocultar advertencias de Pandas/Streamlit

# =================================================================
# 0. VARIABLES GLOBALES Y CONFIGURACIÓN
# =================================================================

ARCHIVO_PROCESADO = "Asset_Inventory_PROCESSED.csv" 
KNOWLEDGE_FILE = "knowledge_base.txt" 
# CRITERIO DE RIESGO
# Umbral de Riesgo Alto (Crítico) según tu última solicitud
UMBRAL_RIESGO_ALTO = 3.5 

# --- CONFIGURACIÓN DE RIESGOS UNIVERSALES ---
PENALIZACION_DATOS_INCOMPLETOS = 2.0  
PENALIZACION_INCONSISTENCIA_TIPO = 0.5    
PENALIZACION_DUPLICADO = 1.0          
# RIESGO MÁXIMO TEÓRICO UNIVERSAL: 3.5
RIESGO_MAXIMO_TEORICO_UNIVERSAL = 3.5 

# --- NUEVAS PENALIZACIONES AVANZADAS (Agregadas por la modificación) ---
PENALIZACION_INCONSISTENCIA_METADATOS = 1.5 # Inconsistencia de metadatos (ej. frecuencia vs. antigüedad)
PENALIZACION_ANOMALIA_SILENCIOSA = 1.0     # Duplicidad semántica/Cambios abruptos (Anomalía + Baja Popularidad)
PENALIZACION_ACTIVO_VACIO = 2.0          # Activos vacíos en categorías populares
# RIESGO MÁXIMO TEÓRICO AVANZADO (3.5 + 1.5 + 1.0 + 2.0 = 8.0)
RIESGO_MAXIMO_TEORICO_AVANZADO = RIESGO_MAXIMO_TEORICO_UNIVERSAL + PENALIZACION_INCONSISTENCIA_METADATOS + PENALIZACION_ANOMALIA_SILENCIOSA + PENALIZACION_ACTIVO_VACIO

# ⚠️ CLAVE SECRETA DE GEMINI
# REEMPLAZA ESTE VALOR con tu clave secreta real de Gemini (comienza con AIza...).
GEMINI_API_SECRET_VALUE = "AIzaSyDvuJPAAK8AVIS-VQIe39pPgVNb8xlJw3g"

# --- COLUMNAS CLAVE PARA FILTROS DINÁMICOS (NUEVAS Y EXISTENTES) ---
COL_TEMA = 'commoncore_theme' 
# El usuario ha solicitado la columna 'Público\naudience'.
COL_AUDIENCIA = 'Público\naudience' 
# -------------------------------------------------------------------

# =================================================================
# 1. Funciones de Carga y Procesamiento
# =================================================================

@st.cache_data
def load_processed_data(file_path):
    """Carga el archivo CSV YA PROCESADO y lo cachea."""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

def clean_and_convert_types_external(df):
    """Fuerza a las columnas a ser tipo string para asegurar la detección de inconsistencias."""
    
    # Columnas que suelen ser de tipo 'object' (string)
    object_cols = ['titulo', 'descripcion', 'dueño'] 
    
    # Columnas que contienen los datos que queremos chequear por tipo mixto
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
    Calcula métricas de calidad universal: Completitud (Datos), Consistencia, Unicidad 
    para el diagnóstico rápido.
    """
    n_cols = df.shape[1]
    
    # --- 1. COMPLETITUD: Datos por Fila (Densidad) ---
    df['datos_por_fila_score'] = (df.notna().sum(axis=1) / n_cols) * 100
    df['riesgo_datos_incompletos'] = np.where(
        df['datos_por_fila_score'] < 70, PENALIZACION_DATOS_INCOMPLETOS, 0.0
    )

    # --- 2. CONSISTENCIA: Mezcla de Tipos ---
    df['riesgo_consistencia_tipo'] = 0.0
    for col in df.select_dtypes(include='object').columns:
        inconsistencies = df[col].apply(lambda x: not isinstance(x, str) and pd.notna(x))
        df.loc[inconsistencies, 'riesgo_consistencia_tipo'] = PENALIZACION_INCONSISTENCIA_TIPO
        
    # --- 3. UNICIDAD: Duplicados Exactos ---
    df['es_duplicado'] = df.duplicated(keep=False) 
    df['riesgo_duplicado'] = np.where(
        df['es_duplicado'], PENALIZACION_DUPLICADO, 0.0
    )
    
    return df

def process_external_data(df):
    """
    Lógica de riesgo universal para el archivo externo subido.
    """
    
    df = clean_and_convert_types_external(df)
    df = check_universals_external(df)
    
    campos_clave_universal = ['titulo', 'descripcion', 'dueño'] 
    campos_existentes_y_llenos = 0
    num_campos_totales_base = len(campos_clave_universal)

    for campo in campos_clave_universal:
        if campo in df.columns and pd.notna(df[campo].iloc[0]):
            campos_existentes_y_llenos += 1
            
    completitud_metadatos_universal = (campos_existentes_y_llenos / num_campos_totales_base) * 100
    df['completitud_metadatos_universal'] = completitud_metadatos_universal
    
    df['prioridad_riesgo_score'] = (
        df['riesgo_datos_incompletos'] + 
        df['riesgo_consistencia_tipo'] +
        df['riesgo_duplicado']
    )
    
    avg_file_risk = df['prioridad_riesgo_score'].mean()
    quality_score = 100 - (avg_file_risk / RIESGO_MAXIMO_TEORICO_UNIVERSAL * 100)
    
    df['calidad_total_score'] = np.clip(quality_score, 0, 100)

    return df

# 🟢 NUEVA FUNCIÓN: CHEQUEO DINÁMICO DE INCUMPLIMIENTO
@st.cache_data
def apply_dynamic_compliance_check(df):
    """
    Dynamically checks for non-compliance (Incumplimiento de Actualización) 
    based on data age (antiguedad_datos_dias).
    Threshold: > 365 days old (1 año) = Non-Compliance.
    """
    df_copy = df.copy()
    
    # Aseguramos que la columna antiguedad_datos_dias exista
    if 'antiguedad_datos_dias' in df_copy.columns:
        # La columna estado_actualizacion es ahora dinámica
        df_copy['estado_actualizacion'] = np.where(
            df_copy['antiguedad_datos_dias'] > 365,
            '🔴 INCUMPLIMIENTO',
            '🟢 CUMPLE'
        )
    else:
        # Si la columna clave no existe, no se puede realizar el chequeo
        st.warning("Columna 'antiguedad_datos_dias' no encontrada para el chequeo de cumplimiento dinámico. Se usa el estado original si existe.")
        
    return df_copy
# 🟢 FIN NUEVA FUNCIÓN

# 🚀 ADICIÓN: FUNCIÓN PARA DETECCIÓN DE ANOMALÍAS CON ISOLATION FOREST
@st.cache_data
def apply_anomaly_detection(df):
    """
    Detecta anomalías en los activos de datos utilizando Isolation Forest
    basado en métricas clave (Riesgo, Completitud, Antigüedad, Popularidad).
    Asigna -1 para anomalía (outlier) y 1 para normal (inlier).
    """
    df_copy = df.copy()
    
    # 1. Definir features
    # Las columnas deben existir en el DataFrame cargado (pre-procesado)
    # Se usan las columnas que vienen del procesamiento general.
    features = ['prioridad_riesgo_score', 'completitud_score', 'antiguedad_datos_dias', 'popularidad_score']
    
    # 2. Preparar los datos
    # Solo usamos filas donde las features existen.
    df_model = df_copy[features].dropna().astype(float)
    
    if len(df_model) < 10: # Mínimo recomendado para Isolation Forest
        st.sidebar.warning("Advertencia: Menos de 10 filas de datos completos. ML Anomaly Detection se omitirá.")
        df_copy['anomalia_score'] = 1 # Por defecto, no es una anomalía
        return df_copy
    
    # 3. Inicializar y entrenar Isolation Forest
    # contamination='auto' permite al modelo estimar la proporción de outliers
    iso_forest = IsolationForest(
        random_state=42, 
        contamination='auto',
        n_estimators=100
    )
    
    # 4. Ajustar y predecir: 1 (inlier) o -1 (outlier/anomalía)
    predictions = iso_forest.fit_predict(df_model)
    
    # 5. Mapear las predicciones al DataFrame original
    # Inicializar la columna anomalia_score en el df_copy
    df_copy['anomalia_score'] = 1 # Valor por defecto (no es anomalía)
    
    # Mapear las predicciones de vuelta usando el índice
    df_copy.loc[df_model.index, 'anomalia_score'] = predictions
    
    # Reportar el número de anomalías detectadas
    num_anomalies = (df_copy['anomalia_score'] == -1).sum()
    st.sidebar.markdown(f"**🔍 Detección ML:** {num_anomalies} anomalías detectadas.")
    
    return df_copy
# 🚀 FIN ADICIÓN

# --- FUNCIÓN PARA CHEQUEOS AVANZADOS (Implementa la lógica solicitada) ---
@st.cache_data
def apply_advanced_risk_checks(df):
    """
    Calcula nuevos scores de riesgo avanzados (inconsistencias, semántica, vacíos) 
    y los añade al score de riesgo existente para el análisis general.
    """
    df_copy = df.copy()
    
    # 1. Detección de Inconsistencia de Metadatos (Proxy: Riesgo alto A PESAR de ser reciente)
    # Asume: Si un activo tiene un score de riesgo UNIVERSAL alto (> UMBRAL_RIESGO_ALTO) pero se actualizó 
    # hace menos de 1 año (< 365 días), hay una posible inconsistencia entre su estado reportado 
    # (reciente) y su calidad medida (pobre).
    
    # Usamos el UMBRAL_RIESGO_ALTO universal para ser más estrictos
    df_copy['riesgo_inconsistencia_metadatos'] = np.where(
        (df_copy['prioridad_riesgo_score'] > RIESGO_MAXIMO_TEORICO_UNIVERSAL) & 
        (df_copy['antiguedad_datos_dias'] < 365), 
        PENALIZACION_INCONSISTENCIA_METADATOS, 
        0.0
    )
    
    # 2. Detección de Anomalía Silenciosa (Anomalía ML + Baja Popularidad)
    # Una anomalía silenciosa es un activo que el ML marcó como anómalo (-1) Y tiene baja popularidad.
    # Se considera 'baja popularidad' si el score está por debajo del percentil 25.
    
    if 'anomalia_score' in df_copy.columns:
        percentil_25_popularidad = df_copy['popularidad_score'].quantile(0.25)
        
        df_copy['riesgo_anomalia_silenciosa'] = np.where(
            (df_copy['anomalia_score'] == -1) & 
            (df_copy['popularidad_score'] < percentil_25_popularidad),
            PENALIZACION_ANOMALIA_SILENCIOSA,
            0.0
        )
    else:
        # Esto ocurre si la detección de anomalías se omite por falta de datos
        df_copy['riesgo_anomalia_silenciosa'] = 0.0

    # 3. Detección de Activo Vacío en Categoría Popular
    # Activo con completitud muy baja (ej. < 50%) en una categoría que en general es muy popular 
    # (ej. Top 5 categorías por número de activos).
    
    top_categorias = df_copy['categoria'].value_counts().nlargest(5).index
    
    df_copy['riesgo_activo_vacio'] = np.where(
        (df_copy['completitud_score'] < 50) & 
        (df_copy['categoria'].isin(top_categorias)),
        PENALIZACION_ACTIVO_VACIO,
        0.0
    )

    # 4. Actualizar el Score de Riesgo Final
    df_copy['prioridad_riesgo_score_avanzado'] = (
        df_copy['prioridad_riesgo_score'] + 
        df_copy['riesgo_inconsistencia_metadatos'] + 
        df_copy['riesgo_anomalia_silenciosa'] + 
        df_copy['riesgo_activo_vacio']
    )
    
    # Reemplazar el score de riesgo universal por el avanzado para el análisis final
    df_copy['prioridad_riesgo_score'] = df_copy['prioridad_riesgo_score_avanzado']
    df_copy.drop(columns=['prioridad_riesgo_score_avanzado'], inplace=True)
    
    return df_copy

# ----------------------------------------------------------------------
# FUNCIONES DE UI Y VISUALIZACIÓN
# ----------------------------------------------------------------------

@st.cache_data
def get_table_download_link(df, filename, link_text):
    """Genera un link para descargar el DataFrame como CSV."""
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()  # Codifica a base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def display_kpi_card(title, value, delta=None, color='normal'):
    """Muestra una tarjeta KPI con un delta opcional."""
    # Uso de columnas para centrar y estilizar ligeramente
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        # Añadir un div con estilo para el fondo
        st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; text-align: center;">
                <p style="font-size: 16px; margin: 0; color: #4F80B9;">{title}</p>
                <h3 style="font-size: 24px; margin: 5px 0 0 0;">{value}</h3>
            </div>
        """, unsafe_allow_html=True)
        # st.metric(title, value, delta=delta, delta_color=color)

# Función para generar el reporte de riesgo
def generate_risk_report(df):
    """
    Genera un DataFrame detallado de las dimensiones de riesgo aplicadas al inventario.
    """
    total_activos = len(df)
    
    # Definiciones de Riesgo (Basadas en las columnas de penalización)
    riesgos_data = [
        {
            'Dimensión de Riesgo': 'Datos Incompletos (Universal)',
            'Descripción': 'Activos con menos del 70% de campos llenos.',
            'Conteo de Activos Afectados': (df['riesgo_datos_incompletos'] > 0).sum(),
            'Penalización Score': PENALIZACION_DATOS_INCOMPLETOS
        },
        {
            'Dimensión de Riesgo': 'Inconsistencia de Tipo (Universal)',
            'Descripción': 'Activos con mezcla de tipos de datos en la misma columna.',
            'Conteo de Activos Afectados': (df['riesgo_consistencia_tipo'] > 0).sum(),
            'Penalización Score': PENALIZACION_INCONSISTENCIA_TIPO
        },
        {
            'Dimensión de Riesgo': 'Duplicidad Exacta (Universal)',
            'Descripción': 'Activos que son duplicados exactos en el inventario.',
            'Conteo de Activos Afectados': (df['riesgo_duplicado'] > 0).sum(),
            'Penalización Score': PENALIZACION_DUPLICADO
        },
        {
            'Dimensión de Riesgo': 'Incumplimiento Actualización (Dinámico)',
            'Descripción': 'Activos con antigüedad de datos mayor a 365 días.',
            'Conteo de Activos Afectados': (df['estado_actualizacion'] == '🔴 INCUMPLIMIENTO').sum(),
            'Penalización Score': 'No aplica directamente'
        },
        {
            'Dimensión de Riesgo': 'Inconsistencia Metadatos (Avanzado)',
            'Descripción': f'Activo con riesgo universal > {RIESGO_MAXIMO_TEORICO_UNIVERSAL} pero actualizado hace < 365 días.',
            'Conteo de Activos Afectados': (df['riesgo_inconsistencia_metadatos'] > 0).sum(),
            'Penalización Score': PENALIZACION_INCONSISTENCIA_METADATOS
        },
        {
            'Dimensión de Riesgo': 'Anomalía Silenciosa (ML Avanzado)',
            'Descripción': 'Activo detectado como Anómalo por ML y de baja popularidad.',
            'Conteo de Activos Afectados': (df['riesgo_anomalia_silenciosa'] > 0).sum(),
            'Penalización Score': PENALIZACION_ANOMALIA_SILENCIOSA
        },
        {
            'Dimensión de Riesgo': 'Activo Vacío en Categoría Popular (Avanzado)',
            'Descripción': 'Activo con Completitud < 50% en una de las Top 5 categorías.',
            'Conteo de Activos Afectados': (df['riesgo_activo_vacio'] > 0).sum(),
            'Penalización Score': PENALIZACION_ACTIVO_VACIO
        }
    ]
    
    riesgos_df = pd.DataFrame(riesgos_data)
    
    # Calcular % de Activos Afectados
    riesgos_df['% Activos Afectados'] = (riesgos_df['Conteo de Activos Afectados'] / total_activos) * 100
    riesgos_df['% Activos Afectados'] = riesgos_df['% Activos Afectados'].map('{:.2f}%'.format)
    riesgos_df['Penalización Score'] = riesgos_df['Penalización Score'].apply(lambda x: f'{x:.1f}' if isinstance(x, (int, float)) else x)

    return riesgos_df[['Dimensión de Riesgo', 'Descripción', 'Conteo de Activos Afectados', '% Activos Afectados', 'Penalización Score']]

# Función para generar recomendaciones
def generate_recommendation_text(df):
    """
    Genera texto de recomendación basado en los hallazgos de riesgo.
    """
    reporte = generate_risk_report(df)
    
    # Criterio de Riesgo Crítico (Umbral dinámico)
    activos_alto_riesgo = df[df['prioridad_riesgo_score'] >= UMBRAL_RIESGO_ALTO]
    num_criticos = len(activos_alto_riesgo)
    porc_criticos = (num_criticos / len(df)) * 100 if len(df) > 0 else 0
    
    # Generar encabezado
    if porc_criticos > 30:
        header = f"## 🚨 Alerta Crítica: {num_criticos:,} Activos ({porc_criticos:.1f}%) en Alto Riesgo."
        base_rec = "Se requiere una **intervención inmediata** para abordar la alta proporción de activos de datos críticos. La calidad y confiabilidad del inventario está severamente comprometida."
    elif porc_criticos > 10:
        header = f"## ⚠️ Advertencia: {num_criticos:,} Activos ({porc_criticos:.1f}%) en Riesgo Elevado."
        base_rec = "El riesgo es notable y debe ser gestionado prioritariamente. Enfóquese en los hallazgos principales para reducir el impacto."
    else:
        header = f"## ✅ Monitoreo: {num_criticos:,} Activos ({porc_criticos:.1f}%) en Alto Riesgo."
        base_rec = "El inventario presenta un nivel de riesgo bajo o manejable. Mantenga el monitoreo continuo para asegurar el cumplimiento."

    recomendaciones = []
    
    # Recomendaciones por Dimensión
    
    # 1. Incumplimiento de Actualización
    incumplimiento_data = reporte[reporte['Dimensión de Riesgo'] == 'Incumplimiento Actualización (Dinámico)'].iloc[0]
    num_incumplimiento = incumplimiento_data['Conteo de Activos Afectados']
    if num_incumplimiento > 0:
        recomendaciones.append(f"**Actualización:** El {incumplimiento_data['% Activos Afectados']} del inventario está en incumplimiento por antigüedad. Priorice la **actualización o desmantelamiento** de estos {num_incumplimiento:,} activos ({incumplimiento_data['Descripción'].split('(')[1].replace(')', '')}).")

    # 2. Datos Incompletos
    incompletos_data = reporte[reporte['Dimensión de Riesgo'] == 'Datos Incompletos (Universal)'].iloc[0]
    if incompletos_data['Conteo de Activos Afectados'] > 0:
        recomendaciones.append(f"**Completitud:** El {incompletos_data['% Activos Afectados']} presenta baja completitud de datos. Implemente **validaciones de esquema** que exijan un llenado mínimo del 70% de los campos.")

    # 3. Anomalía Silenciosa / Duplicidad
    silenciosa_data = reporte[reporte['Dimensión de Riesgo'] == 'Anomalía Silenciosa (ML Avanzado)'].iloc[0]
    duplicados_data = reporte[reporte['Dimensión de Riesgo'] == 'Duplicidad Exacta (Universal)'].iloc[0]
    
    if silenciosa_data['Conteo de Activos Afectados'] > 0:
        recomendaciones.append(f"**Detección ML:** Revise los {silenciosa_data['Conteo de Activos Afectados']:,} activos marcados como **Anomalías Silenciosas**. Estos podrían ser duplicados semánticos o activos obsoletos que pasan desapercibidos.")
    elif duplicados_data['Conteo de Activos Afectados'] > 0:
        recomendaciones.append(f"**Unicidad:** Revise la duplicidad, el {duplicados_data['% Activos Afectados']} son duplicados exactos. Esto puede inflar métricas y generar inconsistencias en los reportes.")
        
    # 4. Inconsistencia de Metadatos
    metadatos_data = reporte[reporte['Dimensión de Riesgo'] == 'Inconsistencia Metadatos (Avanzado)'].iloc[0]
    if metadatos_data['Conteo de Activos Afectados'] > 0:
        recomendaciones.append(f"**Metadatos:** Investigar la **confiabilidad del proceso de actualización** para los {metadatos_data['Conteo de Activos Afectados']:,} activos con Inconsistencia de Metadatos. Su calidad es mala a pesar de ser recientes.")

    final_recomendacion = f"""
{header}
---
<p style="font-size: 16px;">{base_rec}</p>
<ol>
    {''.join([f'<li>{rec}</li>' for rec in recomendaciones])}
</ol>
"""
    return final_recomendacion


# Función para configurar el Asistente de IA
def setup_data_assistant(df):
    """
    Configura y maneja la lógica para el Asistente de IA (NLP) en una pestaña lateral.
    """
    # ----------------------------------------------------------------------
    # --- PREPARACIÓN DEL CONTEXTO (Knowledge Base) ---
    # ----------------------------------------------------------------------
    
    knowledge_base_content = None

    # Intentar cargar el archivo de la Knowledge Base
    try:
        if os.path.exists(KNOWLEDGE_FILE):
            with open(KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
                knowledge_base_content = f.read()
        else:
            # Si no existe, se crea un mensaje de advertencia
            st.sidebar.warning(f"Archivo de Knowledge Base '{KNOWLEDGE_FILE}' no encontrado.")
            return

    except Exception as e:
        st.sidebar.error(f"Error al leer la Knowledge Base: {e}")
        return

    # ----------------------------------------------------------------------
    # --- CONFIGURACIÓN DEL ASISTENTE EN LA BARRA LATERAL ---
    # ----------------------------------------------------------------------
    st.sidebar.title("🤖 Asistente de Inventario (IA)")
    st.sidebar.info("Este asistente puede responder a preguntas complejas de análisis basadas en la última Knowledge Base generada por la aplicación (archivo: `knowledge_base.txt`).")
    
    # 1. Lógica de Historial de Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Usamos un contenedor simple ya que estamos en el cuerpo principal
    chat_history_container = st.sidebar.container()
    
    with chat_history_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 2. Lógica de Interacción (Chat Input - en el cuerpo principal)
    if prompt := st.sidebar.chat_input("Escribe aquí tu pregunta de análisis complejo:", key="main_chat_input_key", disabled=(knowledge_base_content is None)):
        
        # --- Agregar el mensaje del usuario y simular la respuesta inmediata ---
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Para que el mensaje del usuario aparezca inmediatamente en el historial
        with chat_history_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            # Placeholder para la respuesta del Asistente (se llenará en la función)
            model_response_placeholder = st.empty() 
            
            # Llamar a la función de generación
            generate_ai_response(prompt, knowledge_base_content, model_response_placeholder)

def generate_ai_response(prompt, knowledge_base_content, placeholder):
    """
    Genera la respuesta del modelo Gemini utilizando el contexto.
    """
    try:
        # Inicializar el cliente de Gemini
        client = genai.Client(api_key=GEMINI_API_SECRET_VALUE)
        
        # Construir el prompt con el contexto
        system_prompt = f"""
        Eres un Asistente de Análisis de Inventario de Datos. Tu objetivo es responder preguntas sobre el inventario utilizando **solo** el contexto proporcionado en la 'Knowledge Base'.
        Debes actuar como un analista de datos y director de calidad, proporcionando respuestas precisas, concisas y orientadas a la acción, manteniendo un tono formal y profesional.
        No debes inventar información, si la respuesta no está en el contexto, indica amablemente que no tienes la información.
        
        ---
        Knowledge Base (Contexto de Análisis del Inventario):
        {knowledge_base_content}
        ---
        
        Pregunta del usuario: {prompt}
        """

        # Llamar al modelo
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=system_prompt
        )

        # Mostrar la respuesta
        full_response_text = response.text
        placeholder.chat_message("assistant").markdown(full_response_text)
        
        # Guardar en el historial de sesión
        st.session_state.messages.append({"role": "assistant", "content": full_response_text})

    except Exception as e:
        error_message = f"Error en la llamada a la API de Gemini: {e}. Por favor, verifica tu clave API."
        placeholder.chat_message("assistant").error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})


# =================================================================
# 2. ESTRUCTURA PRINCIPAL DE STREAMLIT
# =================================================================

st.set_page_config(
    page_title="Data Asset Quality & Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# --- CABECERA Y SUBIDA DE ARCHIVO ---
# ----------------------------------------------------------------------
st.title("📊 Data Asset Quality & Risk Dashboard (v1.0)")
st.caption("Herramienta de diagnóstico de calidad y riesgo basada en métricas universales, cumplimiento y ML.")

uploaded_file = st.sidebar.file_uploader(
    "1. Sube tu Inventario de Datos (CSV/TXT)", 
    type=['csv', 'txt'],
    help="El archivo debe contener el inventario de activos de datos. Columnas clave esperadas: 'uid', 'titulo', 'dueño', 'antiguedad_datos_dias', 'completitud_score', 'popularidad_score', 'categoria', 'commoncore_theme', 'Público\\naudience'."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Opciones de Análisis")

# ----------------------------------------------------------------------
# --- LÓGICA DE CARGA Y PROCESAMIENTO ---
# ----------------------------------------------------------------------

df_analisis_completo = pd.DataFrame() # Inicializar el DataFrame final

# 1. Intentar cargar el archivo ya procesado
try:
    df_analisis = load_processed_data(ARCHIVO_PROCESADO)
    if not df_analisis.empty:
        st.success(f"✅ Se cargó el análisis previo (Archivo: **{ARCHIVO_PROCESADO}**).")
        # Asignar al dataframe final si no hay un archivo nuevo subido
        df_analisis_completo = df_analisis
except Exception:
    df_analisis = pd.DataFrame()

# 2. Cargar y procesar el archivo subido por el usuario (sobreescribe si existe)
if uploaded_file is not None:
    try:
        uploaded_filename = uploaded_file.name
        # Leer el archivo subido
        df_uploaded = pd.read_csv(uploaded_file, encoding='utf-8', low_memory=False)
        st.success(f"✅ Archivo **{uploaded_filename}** cargado con éxito.")
        
        # Intentar aplicar el procesamiento universal (solo riesgo básico)
        try:
            df_analisis_completo = process_external_data(df_uploaded.copy())
            st.success("✅ Procesamiento Universal de Riesgo aplicado.")
            
            # --- VALIDACIÓN DE COLUMNAS CLAVE PARA ANÁLISIS AVANZADO ---
            columnas_requeridas = ['antiguedad_datos_dias', 'completitud_score', 'popularidad_score']
            faltan_columnas_avanzadas = [col for col in columnas_requeridas if col not in df_analisis_completo.columns]
            
            if faltan_columnas_avanzadas:
                st.warning(f"⚠️ Columnas faltantes para el Análisis Avanzado/ML: {', '.join(faltan_columnas_avanzadas)}. Se muestra solo el Riesgo Universal.")
            else:
                st.info("⚙️ Iniciando Análisis Avanzado (Cumplimiento, ML y Riesgo Compuesto)...")
                
                # ------------------------------------------------------------------
                # --- MODIFICACIÓN CLAVE: Aplicación de las Nuevas Funciones ---
                # ------------------------------------------------------------------

                # 1. Aplicar Chequeo Dinámico de Cumplimiento (ACTUALIZA estado_actualizacion)
                df_analisis_completo = apply_dynamic_compliance_check(df_analisis_completo)
                
                # 2. Aplicar Detección de Anomalías (Necesita antiguedad_datos_dias, completitud_score, prioridad_riesgo_score, popularidad_score)
                df_analisis_completo = apply_anomaly_detection(df_analisis_completo)

                # 3. Aplicar Chequeos de Riesgo Avanzado (Necesita todo lo anterior)
                df_analisis_completo = apply_advanced_risk_checks(df_analisis_completo)

                st.success("✅ Análisis Avanzado, Cumplimiento Dinámico y ML completados.")
                
        except Exception as e:
            st.error(f"❌ Error durante el procesamiento de datos: {e}")
            df_analisis_completo = pd.DataFrame()

    except Exception as e:
        st.error(f"❌ Error al leer o procesar el archivo CSV: {e}")
        st.warning("Asegúrate de que el archivo es un CSV válido y tiene un formato consistente.")

# ----------------------------------------------------------------------
# --- SECCIÓN PRINCIPAL DEL DASHBOARD ---
# ----------------------------------------------------------------------

if not df_analisis_completo.empty:
    
    # ----------------------------------------------------------------------
    # --- PESTAÑAS PRINCIPALES ---
    # ----------------------------------------------------------------------
    tab_dashboard, tab_detail, tab_ml_insights = st.tabs(["Dashboard General", "Inventario Detallado", "ML & Clustering"])
    
    with tab_dashboard:
        st.header("Resumen de Calidad y Riesgo")
        
        # --- FILTROS DE RIESGO ---
        # Reorganización de columnas para incluir los nuevos filtros
        col_f1, col_f2, col_f3 = st.columns(3)
        
        min_risk = df_analisis_completo['prioridad_riesgo_score'].min()
        max_risk = df_analisis_completo['prioridad_riesgo_score'].max()
        
        # 1. Filtro de Riesgo Mínimo (col_f1)
        risk_threshold = col_f1.slider(
            'Filtro de Riesgo Mínimo (Prioridad Score)',
            min_value=min_risk, 
            max_value=max_risk, 
            value=UMBRAL_RIESGO_ALTO, 
            step=0.1
        )
        
        # 2. Filtro de Cumplimiento (col_f2)
        compliance_filter = col_f2.multiselect(
            'Filtro de Estado de Cumplimiento',
            options=df_analisis_completo['estado_actualizacion'].unique(),
            default=['🟢 CUMPLE', '🔴 INCUMPLIMIENTO']
        )
        
        # 3. Filtro Público/Audiencia (col_f3) - Usa la nueva columna solicitada
        audience_filter_value = None
        if COL_AUDIENCIA in df_analisis_completo.columns:
            audience_options = df_analisis_completo[COL_AUDIENCIA].dropna().unique().tolist()
            if not audience_options:
                col_f3.warning(f"Columna '{COL_AUDIENCIA}' vacía.")
            else:
                audience_filter_value = col_f3.multiselect(
                    'Filtro: Público/Audiencia',
                    options=audience_options,
                    default=audience_options
                )
        else:
            col_f3.warning(f"Columna '{COL_AUDIENCIA}' no encontrada.")
            
        # Nueva fila para el filtro de Tema (Theme)
        col_f4, col_f5 = st.columns(2)
        
        # 4. Filtro de Common Core Theme (col_f4)
        theme_filter_value = None
        if COL_TEMA in df_analisis_completo.columns:
            theme_options = df_analisis_completo[COL_TEMA].dropna().unique().tolist()
            if not theme_options:
                col_f4.warning(f"Columna '{COL_TEMA}' vacía.")
            else:
                theme_filter_value = col_f4.multiselect(
                    'Filtro: Common Core Theme',
                    options=theme_options,
                    default=theme_options # Default to select all
                )
        else:
            col_f4.warning(f"Columna '{COL_TEMA}' no encontrada.")

        # --- APLICAR FILTROS ---
        df_filtered = df_analisis_completo[
            (df_analisis_completo['prioridad_riesgo_score'] >= risk_threshold) &
            (df_analisis_completo['estado_actualizacion'].isin(compliance_filter))
        ]
        
        # Aplicar filtro de Público/Audiencia
        if audience_filter_value and COL_AUDIENCIA in df_analisis_completo.columns:
            df_filtered = df_filtered[df_filtered[COL_AUDIENCIA].isin(audience_filter_value)]

        # Aplicar filtro de Theme
        if theme_filter_value and COL_TEMA in df_analisis_completo.columns:
            df_filtered = df_filtered[df_filtered[COL_TEMA].isin(theme_filter_value)]
        
        st.caption(f"Mostrando {len(df_filtered):,} de {len(df_analisis_completo):,} activos aplicando los filtros.")

        # --- REPORTE DE RECOMENDACIONES ---
        with st.expander("Recomendaciones de Acción Prioritaria", expanded=True):
            recomendacion_final_md = generate_recommendation_text(df_analisis_completo)
            st.markdown(recomendacion_final_md, unsafe_allow_html=True)


        # --- KPIs Y MÉTRICAS GLOBALES ---
        st.markdown("#### Métricas Clave (Inventario Completo)")
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        
        total_activos = len(df_analisis_completo)
        
        # 1. Riesgo Promedio General
        avg_risk = df_analisis_completo['prioridad_riesgo_score'].mean()
        col_kpi1.metric("Riesgo Promedio General", f"{avg_risk:.2f}")

        # 2. Porcentaje de Activos en Alto Riesgo (Usando UMBRAL_RIESGO_ALTO)
        activos_alto_riesgo = df_analisis_completo[df_analisis_completo['prioridad_riesgo_score'] >= UMBRAL_RIESGO_ALTO]
        perc_alto_riesgo = (len(activos_alto_riesgo) / total_activos) * 100 if total_activos > 0 else 0
        col_kpi2.metric("Activos en Alto Riesgo", f"{perc_alto_riesgo:.1f}%", delta=f"{len(activos_alto_riesgo):,} Activos")

        # 3. Cumplimiento de Actualización (Dinámico)
        incumplimiento_count = (df_analisis_completo['estado_actualizacion'] == '🔴 INCUMPLIMIENTO').sum()
        perc_cumplimiento = 100 - (incumplimiento_count / total_activos) * 100 if total_activos > 0 else 0
        col_kpi3.metric("Activos que CUMPLEN (Antigüedad)", f"{perc_cumplimiento:.1f}%", delta=f"{incumplimiento_count:,} INCUMPLEN")

        # 4. Antigüedad Promedio (Días)
        avg_antiguedad = df_analisis_completo['antiguedad_datos_dias'].mean()
        col_kpi4.metric("Antigüedad Promedio Datos", f"{avg_antiguedad:,.0f} días")
        
        st.markdown("---")

        # --- GRÁFICOS DE DISTRIBUCIÓN ---
        col_chart1, col_chart2 = st.columns(2)
        
        # Gráfico 1: Riesgo vs. Completitud (Scatter Plot)
        fig_risk_comp = px.scatter(
            df_filtered,
            x='completitud_score',
            y='prioridad_riesgo_score',
            color='estado_actualizacion',
            hover_data=['titulo', 'dueño', 'categoria'],
            title='Riesgo vs. Completitud (Activos Filtrados)',
            labels={'completitud_score': 'Completitud Score (%)', 'prioridad_riesgo_score': 'Prioridad Riesgo Score'},
            template='plotly_white'
        )
        col_chart1.plotly_chart(fig_risk_comp, use_container_width=True)

        # Gráfico 2: Distribución de Riesgo por Categoría (Bar Plot)
        risk_by_cat = df_filtered.groupby('categoria')['prioridad_riesgo_score'].mean().sort_values(ascending=False).nlargest(10).reset_index()
        fig_cat_risk = px.bar(
            risk_by_cat,
            x='categoria',
            y='prioridad_riesgo_score',
            color='prioridad_riesgo_score',
            color_continuous_scale=px.colors.sequential.Reds,
            title='Top 10 Categorías por Riesgo Promedio (Filtrado)',
            labels={'categoria': 'Categoría', 'prioridad_riesgo_score': 'Riesgo Promedio'},
            template='plotly_white'
        )
        col_chart2.plotly_chart(fig_cat_risk, use_container_width=True)
        
        st.markdown("---")

        # --- REPORTE DE RIESGOS DETALLADO (Tabla) ---
        st.markdown("#### 🔬 Desglose Detallado de Dimensiones de Riesgo (Inventario Completo)")
        riesgos_reporte = generate_risk_report(df_analisis_completo)
        st.dataframe(riesgos_reporte.set_index('Dimensión de Riesgo'), use_container_width=True)

        # --- RANKINGS ---
        st.markdown("#### 🏆 Top Activos y Dueños con Mayor Riesgo")
        col_rank1, col_rank2 = st.columns(2)
        
        # Ranking 1: Top 10 Activos por Riesgo Score
        top_risky_assets = df_analisis_completo.sort_values('prioridad_riesgo_score', ascending=False).head(10)
        col_rank1.markdown("##### 🥇 Top 10 Activos más Críticos")
        col_rank1.dataframe(top_risky_assets[['titulo', 'dueño', 'prioridad_riesgo_score', 'antiguedad_datos_dias', 'estado_actualizacion']].rename(columns={'prioridad_riesgo_score': 'Riesgo Score', 'antiguedad_datos_dias': 'Antigüedad (Días)'}), use_container_width=True)
        
        # Ranking 2: Top 5 Dueños por Riesgo Promedio
        top_risky_owners = df_analisis_completo.groupby('dueño')['prioridad_riesgo_score'].mean().sort_values(ascending=False).head(5).reset_index()
        col_rank2.markdown("##### 🥈 Top 5 Dueños por Riesgo Promedio")
        col_rank2.dataframe(top_risky_owners.rename(columns={'prioridad_riesgo_score': 'Riesgo Promedio'}), use_container_width=True)


    with tab_detail:
        st.header("Inventario Completo y Detallado")
        st.caption(f"Vista de todos los {total_activos:,} activos con las métricas de calidad y riesgo calculadas.")
        
        st.download_button(
            label="Descargar Inventario Procesado como CSV",
            data=df_analisis_completo.to_csv(index=False).encode('utf-8'),
            file_name=f'Inventario_Analizado_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv',
            help="Descarga el DataFrame completo, incluyendo todos los scores de riesgo, completitud y antigüedad calculados."
        )

        st.dataframe(df_analisis_completo, use_container_width=True)

    with tab_ml_insights:
        st.header("ML y Agrupación de Activos")
        
        if 'anomalia_score' in df_analisis_completo.columns:
            st.markdown("#### 🔍 Detección de Anomalías (Isolation Forest)")
            num_anomalies = (df_analisis_completo['anomalia_score'] == -1).sum()
            st.info(f"Se detectaron **{num_anomalies:,} anomalías** ({num_anomalies/total_activos*100:.2f}%) en el inventario.")

            # Gráfico: Riesgo de Anomalías
            df_anomalia = df_analisis_completo.copy()
            df_anomalia['Tipo'] = df_anomalia['anomalia_score'].apply(lambda x: '🔴 Anomalía' if x == -1 else '🟢 Normal')
            
            fig_anomalias = px.scatter(
                df_anomalia,
                x='completitud_score',
                y='prioridad_riesgo_score',
                color='Tipo',
                symbol='Tipo',
                hover_data=['titulo', 'dueño'],
                title='Activos Anómalos (ML) en el Espacio Riesgo/Completitud',
                labels={'completitud_score': 'Completitud Score (%)', 'prioridad_riesgo_score': 'Prioridad Riesgo Score'},
                color_discrete_map={'🔴 Anomalía': 'red', '🟢 Normal': 'blue'},
                template='plotly_white'
            )
            st.plotly_chart(fig_anomalias, use_container_width=True)
            
            # Tabla de Anomalías
            st.markdown("##### Top 10 Activos Marcados como Anomalías")
            anomalies_df = df_anomalia[df_anomalia['Tipo'] == '🔴 Anomalía'].sort_values('prioridad_riesgo_score', ascending=False).head(10)
            st.dataframe(anomalies_df[['titulo', 'dueño', 'prioridad_riesgo_score', 'completitud_score', 'popularidad_score']], use_container_width=True)
            
        else:
            st.warning("La detección de anomalías (ML) no pudo ejecutarse. Asegúrate de tener las columnas requeridas y suficientes datos.")

        st.markdown("---")

        # --- K-MEANS CLUSTERING ---
        st.markdown("#### 🧩 Agrupación No Supervisada (K-Means)")
        
        try:
            # Seleccionar y normalizar las características para clustering
            features_cluster = ['prioridad_riesgo_score', 'completitud_score', 'antiguedad_datos_dias', 'popularidad_score']
            df_cluster = df_analisis_completo[features_cluster].dropna()

            if len(df_cluster) > 100:
                # Normalizar los datos
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(df_cluster)
                
                # Definir número de clusters
                NUM_CLUSTERS = 4
                
                # Aplicar K-Means
                kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init='auto')
                clusters = kmeans.fit_predict(data_scaled)
                
                # Añadir la columna de cluster al DataFrame de análisis
                df_analisis_completo.loc[df_cluster.index, 'cluster_kmeans'] = clusters
                
                st.success(f"Clustering K-Means completado con {NUM_CLUSTERS} grupos.")
                
                # Análisis de los clusters
                cluster_summary = df_analisis_completo.groupby('cluster_kmeans')[features_cluster].mean().reset_index()
                cluster_summary['Conteo'] = df_analisis_completo['cluster_kmeans'].value_counts().sort_index()
                
                st.markdown("##### Resumen de Características por Cluster")
                st.dataframe(cluster_summary.style.background_gradient(cmap='Blues'), use_container_width=True)
                
                # Visualización de Clusters (2D Projection - Riesgo vs. Completitud)
                fig_clusters = px.scatter(
                    df_analisis_completo.dropna(subset=['cluster_kmeans']),
                    x='completitud_score',
                    y='prioridad_riesgo_score',
                    color='cluster_kmeans',
                    hover_data=['titulo', 'dueño', 'categoria'],
                    title='Clusters de Activos (Riesgo vs. Completitud)',
                    labels={'completitud_score': 'Completitud Score (%)', 'prioridad_riesgo_score': 'Prioridad Riesgo Score'},
                    template='plotly_white'
                )
                st.plotly_chart(fig_clusters, use_container_width=True)
                
            else:
                st.warning("Se requieren al menos 100 filas de datos completos para realizar el Clustering K-Means.")
                
        except Exception as e:
            st.error(f"Error al realizar el K-Means Clustering: {e}")


    # ----------------------------------------------------------------------
    # --- GENERACIÓN DE KNOWLEDGE BASE (Para el Asistente de IA) ---
    # ----------------------------------------------------------------------
    with st.spinner("Generando Knowledge Base para el Asistente de IA..."):
        
        # Generar Resumen Global
        total_activos = len(df_analisis_completo)
        
        # 🟢 MODIFICACIÓN: Cálculo de Activos Públicos usando la columna 'Público\naudience'
        num_public = 0
        if COL_AUDIENCIA in df_analisis_completo.columns:
            # Se normaliza para capturar 'public' o 'público' y evitar errores de capitalización
            df_temp = df_analisis_completo[COL_AUDIENCIA].astype(str).str.lower()
            num_public = (df_temp.str.contains('público|publico')).sum()
        # ---------------------------------------------------------------------------------

        summary_kpis = {
            "Total de Activos Registrados": total_activos,
            # Se actualiza el label y el cálculo del KPI con la nueva columna
            "Activos con Acceso Público ('Público\\naudience')": f"{num_public} ({num_public/total_activos*100:.1f}%)" if total_activos > 0 else "N/A",
            "Riesgo Promedio General (Score)": f"{df_analisis_completo['prioridad_riesgo_score'].mean():.2f}",
            "Completitud Promedio General (%)": f"{df_analisis_completo['completitud_score'].mean():.2f}%" if 'completitud_score' in df_analisis_completo.columns else "N/A",
            "Antigüedad Promedio (Días)": f"{df_analisis_completo['antiguedad_datos_dias'].mean():.0f}" if 'antiguedad_datos_dias' in df_analisis_completo.columns else "N/A",
            "Conteo Activos en Alto Riesgo (Score >= 3.5)": len(activos_alto_riesgo),
            "Conteo Activos en Incumplimiento de Actualización": incumplimiento_count
        }

        # Análisis por Dueño
        owner_analysis = df_analisis_completo.groupby('dueño').agg(
            Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
            Completitud_Promedio=('completitud_score', 'mean'),
            Antiguedad_Promedio_Dias=('antiguedad_datos_dias', 'mean'),
            Activos_Totales=('uid', 'count')
        ).reset_index().round(2)
        
        top_risky_owners_kb = owner_analysis.sort_values('Riesgo_Promedio', ascending=False).head(5).to_markdown(index=False)
        
        # Análisis por Categoría
        category_analysis = df_analisis_completo.groupby('categoria').agg(
            Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
            Activos_Totales=('uid', 'count')
        ).reset_index().round(2)
        
        top_risky_categories_kb = category_analysis.sort_values('Riesgo_Promedio', ascending=False).head(5).to_markdown(index=False)
        
        # Generar Knowledge Base
        knowledge_base_content = f"""
## A. CONTEXTO Y FECHA DE GENERACIÓN
Este resumen se generó el: **{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}**.
El agente debe usar estos datos para responder preguntas, cálculos y rankings de alto nivel.

---
## B. RESUMEN GLOBAL DEL INVENTARIO (KPIs CLAVE)
{summary_kpis}

---

## C. ANÁLISIS POR ENTIDAD RESPONSABLE ('dueño')
### Top 5 Entidades con Mayor Riesgo Promedio:
{top_risky_owners_kb}

---

## D. ANÁLISIS POR CATEGORÍA ('categoria')
### Top 5 Categorías con Mayor Riesgo Promedio:
{top_risky_categories_kb}

---

## E. ESQUEMA DEL DATAFRAME ORIGINAL (Columnas y Tipos)
{df_analisis_completo.info(buf=io.StringIO()).getvalue()}
"""
        
        # Guardar el archivo para el Asistente
        with open(KNOWLEDGE_FILE, 'w', encoding='utf-8') as f:
            f.write(knowledge_base_content)

        # Guardar el archivo procesado (opcional, para recarga)
        df_analisis_completo.to_csv(ARCHIVO_PROCESADO, index=False)
        
        st.sidebar.success(f"Knowledge Base ({KNOWLEDGE_FILE}) generada.")
        
    # ----------------------------------------------------------------------
    # --- LLAMADA A LA NUEVA SECCIÓN: ASISTENTE DE DATOS (NLP) ---
    # ----------------------------------------------------------------------
    setup_data_assistant(df_analisis_completo) 

# Si no hay datos, mostrar mensaje inicial
else:
    st.info("Por favor, sube un archivo CSV de inventario de datos en la barra lateral izquierda para comenzar el análisis de calidad y riesgo.")
