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
import base64
import requests # NECESARIO para la API
from google import genai 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Ocultar advertencias de Pandas/Streamlit
warnings.filterwarnings('ignore') 

# =================================================================
# 0. VARIABLES GLOBALES Y CONFIGURACIÓN
# =================================================================

# --- NUEVA FUENTE DE DATOS: API Socrata ---
API_URL = "https://www.datos.gov.co/resource/uzcf-b9dh.json"
API_LIMIT = 100000 # Límite para obtener un conjunto grande de datos

# CRITERIO DE RIESGO
UMBRAL_RIESGO_ALTO = 3.5 
RIESGO_MAXIMO_TEORICO_AVANZADO = 10.0 # Se mantiene para consistencia en la UI

# CLAVE SECRETA DE GEMINI (Asegúrate de configurar esto como secreto en Streamlit)
GEMINI_API_SECRET_VALUE = "AIzaSyDvuJPAAK8AVIS-VQIe39pPgVNb8xlJw3g"

# Columnas esperadas de la API y columnas críticas para la evaluación
# 'dueño' es la columna de la entidad responsable
COLUMNAS_CLAVE = ['uid', 'dueño', 'tema', 't_tulo', 'descripci_n', 'fecha_actualizaci_n', 'formato', 'vistas', 'descargas'] 

# =================================================================
# 1. Funciones de Carga de Datos (Desde API)
# =================================================================

@st.cache_data(ttl=3600) # Caching por 1 hora
def load_data_from_api(url, limit):
    """
    Carga los datos directamente desde la API de Socrata y los cachea.
    """
    st.info(f"Cargando datos de la API: {url}?$limit={limit}. Esto puede tardar unos segundos...")
    try:
        # Petición a la API con el límite
        response = requests.get(f"{url}?$limit={limit}")
        response.raise_for_status() # Lanza una excepción para errores 4xx/5xx
        
        data = response.json()
        df = pd.DataFrame(data)
        
        if df.empty:
            st.error("La API devolvió un conjunto de datos vacío.")
            return pd.DataFrame()

        # Renombrar columnas para facilitar el análisis (si es necesario)
        df.rename(columns={'t_tulo': 'titulo', 'descripci_n': 'descripcion', 'fecha_actualizaci_n': 'fecha_actualizacion'}, inplace=True)
        
        # Filtrar para conservar solo las columnas clave (y las que ya existan)
        df = df[[col for col in COLUMNAS_CLAVE if col in df.columns]]
        
        return df
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error al conectar o recibir datos de la API: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error inesperado durante la carga o procesamiento: {e}")
        return pd.DataFrame()

# =================================================================
# 2. Funciones de Evaluación de Calidad (Guía y ML)
# =================================================================

def calculate_quality_metrics(df_base):
    """
    Calcula las métricas de calidad fundamentales basadas en la Guía 2025 (Completitud, Unicidad, Actualidad, Popularidad).
    """
    df = df_base.copy()
    n_cols = df.shape[1]
    today = datetime.now(df.fecha_actualizacion.dt.tz)

    # 1. CRITERIO DE COMPLETITUD (3.8) - Por fila (Densidad de datos)
    # Puntuación de 0 a 100
    df['completitud_score'] = (df.notna().sum(axis=1) / n_cols) * 100

    # 2. CRITERIO DE UNICIDAD (3.15) - Detección de duplicados
    # Se detectan duplicados en las columnas 'titulo' y 'dueño'
    df['es_duplicado_semantico'] = df.duplicated(subset=['titulo', 'dueño'], keep='first')
    df['unicidad_score'] = np.where(df['es_duplicado_semantico'], 0, 100) # 0 si es duplicado, 100 si es único

    # 3. CRITERIO DE ACTUALIDAD (3.4) - Días desde la última actualización
    # Puntuación basada en la antigüedad (Antigüedad máxima aceptable = 365 días)
    df['fecha_actualizacion'] = pd.to_datetime(df['fecha_actualizacion'], errors='coerce')
    df['dias_antiguedad'] = (today - df['fecha_actualizacion']).dt.days
    df['actualidad_score'] = np.clip(100 - (df['dias_antiguedad'] / 365 * 100), 0, 100)

    # 4. CRITERIO DE RELEVANCIA/POTENCIAL DE USO (3.3 / 5.1.3)
    # Se simula el CTR (Click-Through Rate) con la fórmula de la guía (vistas y descargas)
    df['vistas'] = pd.to_numeric(df['vistas'], errors='coerce').fillna(0)
    df['descargas'] = pd.to_numeric(df['descargas'], errors='coerce').fillna(0)
    
    # CTR = Descargas / Vistas (ajustado para evitar división por cero y capping en 1)
    df['ctr'] = np.where(df['vistas'] > 0, df['descargas'] / df['vistas'], 0)
    df['ctr'] = np.clip(df['ctr'], 0, 1) # Capping en 1, como indica la guía [cite: 1794]
    df['relevancia_score'] = df['ctr'] * 10 # Calificación de 0 a 10 
    
    # Rellenar cualquier NaN que haya podido quedar en scores (debería ser 0)
    df['actualidad_score'].fillna(0, inplace=True)
    df['relevancia_score'].fillna(0, inplace=True)
    
    return df

def apply_ml_evaluation(df_metrics):
    """
    Aplica Detección de Anomalías (IsolationForest) y Clustering de Calidad (KMeans).
    """
    df = df_metrics.copy()
    
    # --- 1. PREPARACIÓN DE CARACTERÍSTICAS PARA ML ---
    # Usar métricas de calidad y popularidad como features
    features = ['completitud_score', 'unicidad_score', 'actualidad_score', 'relevancia_score', 'descargas', 'vistas']
    df_ml = df[features].copy()
    
    # Estandarización para que todas las características tengan peso similar
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_ml)
    
    # --- 2. DETECCIÓN DE ANOMALÍAS (ML - IsolationForest) ---
    # Se ajusta el contamination al 5% de los datos como anomalías
    # Esto cumple con el uso de ML para identificar fallas de calidad o Credibilidad/Consistencia [cite: 1883, 1884, 1895]
    if len(df_scaled) > 10:
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly_score'] = iso_forest.fit_predict(df_scaled)
        # anomaly_score: -1 (anomalía/outlier), 1 (normal)
        df['penalizacion_anomalia'] = np.where(df['anomaly_score'] == -1, 2.0, 0.0)
    else:
        df['penalizacion_anomalia'] = 0.0
    
    # --- 3. CLUSTERING DE CALIDAD (ML - KMeans) ---
    # Features de Clustering: Invertimos scores para que un valor ALTO en el cluster
    # represente una calidad ALTA (menos riesgo/anomalía).
    # Completitud: 100 (Completo) a 0 (Incompleto)
    # Anomaly_Penalty_Invertida: 2.0 (Normal) a 0.0 (Anómalo)
    df['completo_inverso'] = df['completitud_score']
    df['anomalia_inverso'] = np.where(df['anomaly_score'] == 1, 10, 0) # 10 para normal, 0 para anomalía
    
    clustering_features = ['completo_inverso', 'anomalia_inverso', 'unicidad_score']
    df_cluster_data = df[clustering_features].copy()
    df_cluster_scaled = StandardScaler().fit_transform(df_cluster_data)
    
    # 3 Clusters para Completo, Aceptable, Incompleto (Alineado con Sellos 3, 2/1, 0)
    K = 3
    if len(df_cluster_scaled) >= K:
        kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
        df['cluster_label'] = kmeans.fit_predict(df_cluster_scaled)

        # Asignar etiquetas semánticas a los clusters basándose en el score de completitud promedio
        cluster_map = df.groupby('cluster_label')['completitud_score'].mean().sort_values(ascending=False)
        
        # Etiquetado: El de mayor score es 'Completo', el menor es 'Incompleto'
        cluster_to_label = {
            cluster_map.index[0]: 'Completo',
            cluster_map.index[1]: 'Aceptable',
            cluster_map.index[2]: 'Incompleto'
        }
        
        df['estado_calidad_ml'] = df['cluster_label'].map(cluster_to_label)
    else:
        df['estado_calidad_ml'] = 'No Evaluado (Pocos Datos)'

    return df

def calculate_final_risk(df_ml):
    """
    Calcula el score de riesgo final.
    """
    df = df_ml.copy()
    
    # 1. Score de Penalización (RIESGO)
    # La penalización se acumula. Se usa una escala de 0 a 10.
    
    # a. Penalización por Completitud (Inverso del score de 0 a 10)
    df['p_completitud'] = (100 - df['completitud_score']) / 10 # 0-10
    
    # b. Penalización por Unicidad (Duplicados)
    df['p_unicidad'] = np.where(df['es_duplicado_semantico'], 2.0, 0.0)
    
    # c. Penalización por Antigüedad (Inverso del score de Actualidad de 0 a 10)
    df['p_antiguedad'] = (100 - df['actualidad_score']) / 10
    
    # d. Penalización por Baja Relevancia (Inverso del score de Relevancia de 0 a 10)
    df['p_relevancia'] = 10 - df['relevancia_score']
    
    # e. Penalización por Anomalía (ML)
    df['p_anomalia'] = df['penalizacion_anomalia']
    
    # Score de Riesgo (Suma de Penalizaciones)
    df['prioridad_riesgo_score'] = (
        df['p_completitud'] + 
        df['p_unicidad'] + 
        df['p_antiguedad'] + 
        df['p_relevancia'] +
        df['p_anomalia']
    )
    
    # Normalizar el riesgo de 0 a RIESGO_MAXIMO_TEORICO_AVANZADO (10.0) para la UI
    max_theoretical_risk = df[['p_completitud', 'p_unicidad', 'p_antiguedad', 'p_relevancia', 'p_anomalia']].sum(axis=1).max()
    if max_theoretical_risk > 0:
        df['prioridad_riesgo_score'] = (df['prioridad_riesgo_score'] / max_theoretical_risk) * RIESGO_MAXIMO_TEORICO_AVANZADO
    
    # Asignación del estado de incumplimiento (Crítico si supera el umbral)
    df['estado_incumplimiento'] = np.where(
        df['prioridad_riesgo_score'] >= UMBRAL_RIESGO_ALTO, 
        'CRÍTICO', 
        'Bajo/Medio'
    )
    
    return df

@st.cache_data
def process_full_evaluation(df_loaded):
    """Función maestra para la evaluación completa, incluyendo ML."""
    if df_loaded.empty:
        return pd.DataFrame()
    
    # 1. Calcular métricas básicas
    df_metrics = calculate_quality_metrics(df_loaded)
    
    # 2. Aplicar evaluación de ML (Anomalías y Clustering)
    df_ml = apply_ml_evaluation(df_metrics)
    
    # 3. Calcular score de riesgo final
    df_final = calculate_final_risk(df_ml)
    
    return df_final.sort_values(by='prioridad_riesgo_score', ascending=False)

# =================================================================
# 3. Diagnóstico Rápido Universal (Independiente)
# =================================================================

def check_universals_external(df):
    """
    Calcula métricas de calidad universal (Completitud, Unicidad)
    para el diagnóstico rápido, de forma independiente del resto del código.
    """
    df_copy = df.copy() 
    n_cols = df_copy.shape[1]
    
    # 1. COMPLETITUD: Datos por Fila (Densidad)
    df_copy['datos_por_fila_score'] = (df_copy.notna().sum(axis=1) / n_cols) * 100
    
    # 2. UNICIDAD: Detección de duplicados
    df_copy['es_duplicado_semantico'] = df_copy.duplicated(subset=['titulo', 'dueño'], keep='first')
    
    # 3. RIESGO TOTAL (Universal Básico)
    PENALIZACION_DATOS_INCOMPLETOS = 2.0  
    PENALIZACION_DUPLICADO = 1.0 
    
    df_copy['riesgo_datos_incompletos'] = np.where(
        df_copy['datos_por_fila_score'] < 70, PENALIZACION_DATOS_INCOMPLETOS, 0.0
    )
    df_copy['riesgo_duplicado'] = np.where(
        df_copy['es_duplicado_semantico'], PENALIZACION_DUPLICADO, 0.0
    )
    
    df_copy['riesgo_total_universal'] = df_copy['riesgo_datos_incompletos'] + df_copy['riesgo_duplicado']
    
    return df_copy.sort_values(by='riesgo_total_universal', ascending=False)

# =================================================================
# 4. Funciones de Inteligencia Artificial (Métricas en Tiempo Real)
# =================================================================

def get_knowledge_base_content(df_analisis_completo):
    """
    Genera la base de conocimiento con métricas en tiempo real a partir del DataFrame analizado.
    Esta función es CRÍTICA para la consulta de métricas en tiempo real.
    """
    if df_analisis_completo.empty:
        return "No hay datos disponibles para el análisis."

    # Métricas Clave
    total_activos = len(df_analisis_completo)
    riesgo_promedio = df_analisis_completo['prioridad_riesgo_score'].mean()
    criticos = (df_analisis_completo['estado_incumplimiento'] == 'CRÍTICO').sum()
    
    # Estadísticas por Calidad ML (Clustering)
    conteo_calidad = df_analisis_completo['estado_calidad_ml'].value_counts().to_dict()
    
    # Ranking de Entidades (Top 5 peor riesgo promedio)
    ranking_riesgo_entidad = df_analisis_completo.groupby('dueño')['prioridad_riesgo_score'].mean().sort_values(ascending=False).head(5)
    
    # Ranking de Activos (Top 5 peor riesgo individual)
    ranking_riesgo_activo = df_analisis_completo[['titulo', 'dueño', 'prioridad_riesgo_score']].head(5)

    knowledge_content = f"""
    --- MÉTRICAS GLOBALES DEL INVENTARIO DE DATOS (Tiempo Real) ---
    - Total de Activos Analizados: {total_activos}
    - Riesgo de Calidad Promedio Global (Escala 0-10): {riesgo_promedio:.2f}
    - Activos en Estado CRÍTICO (Riesgo >= {UMBRAL_RIESGO_ALTO}): {criticos}
    
    --- CLASIFICACIÓN DE CALIDAD POR ML (K-Means - Sellos) ---
    - Activos Completos: {conteo_calidad.get('Completo', 0)}
    - Activos Aceptables: {conteo_calidad.get('Aceptable', 0)}
    - Activos Incompletos: {conteo_calidad.get('Incompleto', 0)}
    
    --- TOP 5 ENTIDADES CON MAYOR RIESGO PROMEDIO ---
    {ranking_riesgo_entidad.to_string(float_format='%.2f')}
    
    --- TOP 5 ACTIVOS INDIVIDUALES CON MAYOR RIESGO ---
    {ranking_riesgo_activo.to_string(index=False, float_format=lambda x: f'{x:.2f}')}
    
    --- DEFINICIONES DE MÉTRICAS (Basado en la Guía de Calidad) ---
    - El 'Riesgo de Calidad' es un score acumulativo basado en penalizaciones por Baja Completitud (Datos faltantes), Duplicidad Semántica (Unicidad), Desactualización (Actualidad), Baja Popularidad (Relevancia/CTR) y Detección de Anomalías (ML).
    - 'Estado Crítico' indica que el activo supera el umbral de riesgo de {UMBRAL_RIESGO_ALTO}.
    - La 'Clasificación de Calidad ML' diferencia los activos Incompletos, Aceptables y Completos, asimilando a los Sellos de Calidad de la Guía, basada en el clustering de las métricas de Completitud, Unicidad y Penalización por Anomalías.
    """
    return knowledge_content

# --- (El resto de las funciones de Chat (generate_ai_response) permanecen similares, 
# pero usan la salida de get_knowledge_base_content directamente) ---

# ... (Insertar aquí la función generate_ai_response y la lógica de inicialización de Gemini)

def generate_ai_response(prompt, knowledge_base_content, placeholder):
    # Inicialización del cliente (asumiendo que está en st.secrets o configuración)
    try:
        # Reemplazar la línea de inicialización con tu método preferido
        # client = genai.Client(api_key=st.secrets.get("GEMINI_API_SECRET_VALUE", GEMINI_API_SECRET_VALUE))
        client = genai.Client(api_key=GEMINI_API_SECRET_VALUE)
    except Exception as e:
        placeholder.error(f"Error al inicializar el cliente de la IA: {e}")
        return

    # Construcción del prompt de sistema
    system_prompt = (
        "Eres un Asistente de Análisis de Calidad de Datos basado en la 'Guía de Calidad e Interoperabilidad 2025'. "
        "Tu tarea es responder preguntas sobre las métricas de calidad de los activos analizados. "
        "Tu respuesta debe ser concisa, profesional y **estar basada estrictamente en la 'Base de Conocimiento' proporcionada, que contiene métricas en tiempo real.** "
        "Si la información no está en la base de conocimiento, indica que no puedes responder la pregunta específica sobre las métricas. "
        f"\n\nBASE DE CONOCIMIENTO (Métricas de Datos en Tiempo Real):\n{knowledge_base_content}"
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                {"role": "user", "parts": [{"text": prompt}]},
            ],
            config={"system_instruction": system_prompt}
        )
        
        # Muestra la respuesta en el placeholder y la guarda en la sesión
        placeholder.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})
        
    except Exception as e:
        placeholder.error(f"Error al generar la respuesta de la IA: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"Lo siento, ocurrió un error al procesar tu solicitud: {e}"})

# =================================================================
# 5. Función de Generación de Reporte (Actualizada con nuevas métricas)
# =================================================================

def generate_report(df_analisis_completo):
    """
    Genera un informe HTML profesional con las nuevas métricas de calidad y ML.
    """
    if df_analisis_completo.empty:
        return "<h1>No hay datos para generar el informe.</h1>"
        
    # Agregación de nuevas métricas de ML
    conteo_calidad = df_analisis_completo['estado_calidad_ml'].value_counts().to_frame().reset_index()
    conteo_calidad.columns = ['Estado de Calidad (ML)', 'Conteo']
    
    total_activos = len(df_analisis_completo)
    riesgo_promedio = df_analisis_completo['prioridad_riesgo_score'].mean()
    
    # Treemap (Gráfico de Conteo por Calidad ML)
    fig_treemap = px.treemap(
        conteo_calidad,
        path=['Estado de Calidad (ML)'],
        values='Conteo',
        title='Distribución de Activos por Estado de Calidad (ML Clustering)',
        color='Conteo',
        color_continuous_scale='RdYlGn_r'
    )
    treemap_html = fig_treemap.to_html(full_html=False, include_plotlyjs='cdn')

    # Tabla de Activos Prioritarios (CRÍTICOS)
    df_activos_prioritarios = df_analisis_completo[df_analisis_completo['estado_incumplimiento'] == 'CRÍTICO']
    df_activos_prioritarios = df_activos_prioritarios[['titulo', 'dueño', 'estado_calidad_ml', 'prioridad_riesgo_score', 'p_anomalia', 'dias_antiguedad']].head(20)

    # Construcción del HTML Profesional (Asegurar que los datos sean visibles y coherentes)
    html_content = f""" 
    <!DOCTYPE html> 
    <html> 
    <head> 
        <title>Reporte de Análisis de Inventario de Datos Abiertos</title>
        <style>...</style>
    </head>
    <body>
        <div class="header">
            <h1>Reporte de Calidad y Riesgo de Datos Abiertos (API)</h1>
            <p>Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="section-box">
            <h2>Métricas Clave de la Guía de Calidad 2025</h2>
            <p>Evaluación basada en criterios de la Guía (Completitud, Actualidad, Unicidad, Relevancia/CTR) y Machine Learning.</p>
            <ul>
                <li><strong>Total de Activos Analizados:</strong> {total_activos}</li>
                <li><strong>Riesgo Promedio (0-{RIESGO_MAXIMO_TEORICO_AVANZADO:.1f}):</strong> {riesgo_promedio:.2f}</li>
                <li><strong>Activos Críticos ({UMBRAL_RIESGO_ALTO:.1f}+):</strong> {len(df_activos_prioritarios)}</li>
            </ul>
        </div>
        
        <div class="section-box">
            <h2>Clasificación de Calidad por Machine Learning (K-Means)</h2>
            <p>Clasificación automática en tres estados de calidad para alineación con los Sellos de la Guía.</p>
            {treemap_html}
        </div>

        <div class="section-box">
            <h2>Top 20 Activos con Riesgo CRÍTICO (Prioridad Inmediata)</h2>
            <p>Estos activos superan el umbral de riesgo de {UMBRAL_RIESGO_ALTO} y requieren atención inmediata. La columna <code>p_anomalia</code> indica si ML detectó una anomalía.</p>
            {df_activos_prioritarios.to_html(index=False, float_format=lambda x: f'{x:.2f}') if not df_activos_prioritarios.empty else "<p>¡Excelente! No se detectaron activos en riesgo crítico.</p>"}
        </div>
        
    </body>
    </html>
    """
    return html_content

# =================================================================
# 6. Lógica Principal de la Aplicación (Streamlit)
# =================================================================

def main_app_logic():
    st.set_page_config(layout="wide", page_title="Evaluador de Calidad de Datos Abiertos (API)")

    st.title("Sistema de Evaluación de Calidad y Riesgo (API) 📊")
    st.markdown("---")

    # 1. Carga de Datos desde la API
    df_loaded = load_data_from_api(API_URL, API_LIMIT)

    if df_loaded.empty:
        st.error("No se pudo cargar la data de la API. El sistema no puede continuar.")
        return

    # 2. Evaluación Completa (Automática y con Caching)
    df_analisis_completo = process_full_evaluation(df_loaded)

    if df_analisis_completo.empty:
        st.error("La evaluación completa de los datos falló.")
        return
        
    # --- Pestañas de la Aplicación ---
    tab1, tab2, tab3, tab4 = st.tabs(["Evaluación Avanzada (ML)", "Diagnóstico Rápido Universal", "Asistente IA (Métricas en Tiempo Real)", "Informe y Descarga"])

    with tab1:
        st.header("Evaluación Avanzada y Clustering de Calidad (ML)")
        st.info(f"Mostrando {len(df_analisis_completo)} activos. El clustering de calidad se realiza con **KMeans (3 clusters)** y la penalización de anomalías con **IsolationForest**.")

        # Visualización de la Clasificación de Calidad ML
        conteo_calidad = df_analisis_completo['estado_calidad_ml'].value_counts().reset_index()
        conteo_calidad.columns = ['Estado de Calidad ML', 'Conteo']
        
        fig_calidad = px.bar(
            conteo_calidad, 
            x='Estado de Calidad ML', 
            y='Conteo', 
            color='Estado de Calidad ML',
            title='Distribución de Activos por Calidad (ML Clustering)',
            color_discrete_map={'Completo': 'green', 'Aceptable': 'orange', 'Incompleto': 'red'}
        )
        st.plotly_chart(fig_calidad, use_container_width=True) # 
        st.subheader("Tabla de Resultados de la Evaluación (Ordenado por Riesgo)")
        st.dataframe(df_analisis_completo[['titulo', 'dueño', 'prioridad_riesgo_score', 'estado_incumplimiento', 'estado_calidad_ml', 'p_anomalia', 'completitud_score', 'unicidad_score']].head(50), use_container_width=True)

    with tab2:
        st.header("Diagnóstico Rápido Universal (No dependiente)")
        st.markdown("Esta sección utiliza la función independiente `check_universals_external` para evaluar métricas básicas y universales de forma rápida.")
        
        df_quick_check = check_universals_external(df_loaded)
        
        st.subheader("Activos con Mayor Riesgo Básico Universal")
        st.dataframe(df_quick_check[['titulo', 'dueño', 'riesgo_total_universal', 'datos_por_fila_score', 'es_duplicado_semantico']].head(20), use_container_width=True)

    with tab3:
        st.header("Asistente IA (Consulta de Métricas en Tiempo Real) 🧠")
        st.markdown("Pregunta por los **KPIs, rankings o diagnósticos** basados en las **métricas procesadas directamente de la API**.")
        
        knowledge_base_content = get_knowledge_base_content(df_analisis_completo)
        
        # Inicialización de historial de chat
        if "messages" not in st.session_state:
            st.session_state.messages = []

        chat_history_container = st.container()
        
        with chat_history_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if prompt := st.chat_input("Escribe aquí tu pregunta de análisis complejo sobre las métricas:", key="main_chat_input_key"):
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with chat_history_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                model_response_placeholder = st.empty() 
                
                generate_ai_response(prompt, knowledge_base_content, model_response_placeholder)

    with tab4:
        st.header("Generación y Descarga del Informe")
        
        # Generar el reporte HTML con las nuevas métricas
        reporte_html = generate_report(df_analisis_completo)
        
        st.subheader("Vista Previa del Informe")
        st.components.v1.html(reporte_html, height=500, scrolling=True)

        # Función de descarga del reporte
        b64 = base64.b64encode(reporte_html.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="Reporte_Calidad_{datetime.now().strftime("%Y%m%d")}.html">Descargar Reporte HTML</a>'
        st.markdown(href, unsafe_allow_html=True)
        
# Ejecución
if __name__ == "__main__":
    main_app_logic()
