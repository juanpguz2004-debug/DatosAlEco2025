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

# =================================================================
# 1. Funciones de Carga y Procesamiento (Se mantienen igual)
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
    features = ['prioridad_riesgo_score', 'completitud_score', 'antiguedad_datos_dias', 'popularidad_score']
    
    # 2. Preparar los datos
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
    
    df_copy['riesgo_inconsistencia_metadatos'] = np.where(
        (df_copy['prioridad_riesgo_score'] > UMBRAL_RIESGO_ALTO) & (df_copy['antiguedad_datos_dias'] < 365), 
        PENALIZACION_INCONSISTENCIA_METADATOS, 
        0.0
    )

    # 2. Duplicidad Semántica/Cambios Abruptos (Proxy: Anomalía detectada pero baja popularidad)
    # ⚠️ Esta lógica ahora usa el score generado por Isolation Forest.
    df_copy['riesgo_semantico_actualizacion'] = np.where(
        (df_copy['anomalia_score'] == -1) & (df_copy['popularidad_score'] < 0.1),
        PENALIZACION_ANOMALIA_SILENCIOSA,
        0.0
    )

    # 3. Activos Vacíos en Categorías Populares
    # Asume: Un score de completitud muy bajo (< 20%) en una de las 5 categorías con más activos 
    # (proxy para 'dominio clave') indica un 'activo vacío' en un área crítica.
    
    top_categories = df_copy['categoria'].value_counts().nlargest(5).index.tolist()
    
    df_copy['riesgo_activos_vacios'] = np.where(
        (df_copy['completitud_score'] < 20.0) & (df_copy['categoria'].isin(top_categories)),
        PENALIZACION_ACTIVO_VACIO,
        0.0
    )
    
    # Actualizar el score de riesgo principal con las nuevas penalizaciones
    df_copy['prioridad_riesgo_score_v2'] = (
        df_copy['prioridad_riesgo_score'] +
        df_copy['riesgo_inconsistencia_metadatos'] +
        df_copy['riesgo_semantico_actualizacion'] +
        df_copy['riesgo_activos_vacios']
    )
    
    # Sustituir el score principal
    df_copy['prioridad_riesgo_score'] = df_copy['prioridad_riesgo_score_v2']
    df_copy.drop(columns=['prioridad_riesgo_score_v2'], inplace=True, errors='ignore')
    
    return df_copy
# --- FIN CHEQUEOS AVANZADOS ---

# 🚀 Función de Generación de Reporte HTML (Se mantiene igual)
def generate_report_html(df_filtrado, umbral_riesgo):
# ... (Contenido de generate_report_html se mantiene sin cambios)
    """
    Genera el contenido HTML del reporte final que compila insights, tablas y visualizaciones.
    """
    
    # 1. Preparación de Datos
    
    # Datos Principales
    total_activos = len(df_filtrado)
    riesgo_promedio_general = df_filtrado['prioridad_riesgo_score'].mean()
    completitud_promedio_general = df_filtrado['completitud_score'].mean()
    
    # Top Activos de Alto Riesgo (Para Riesgos y Activos Prioritarios)
    df_top_riesgo = df_filtrado.sort_values(by='prioridad_riesgo_score', ascending=False).head(10).copy()
    df_top_riesgo = df_top_riesgo[['titulo', 'prioridad_riesgo_score', 'completitud_score', 'dueño']].rename(columns={'prioridad_riesgo_score': 'Riesgo Score', 'completitud_score': 'Completitud Score', 'dueño': 'Entidad'}).reset_index(drop=True)
    df_top_riesgo['Riesgo'] = df_top_riesgo['Riesgo Score'].apply(lambda x: '🔴 Alto' if x > umbral_riesgo else '🟢 Bajo/Medio')
    
    # Riesgo por Entidad (Para Recomendaciones)
    df_riesgo_entidad = df_filtrado.groupby('dueño').agg(
        Activos_Totales=('uid', 'count'),
        Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
        Completitud_Promedio=('completitud_score', 'mean')
    ).reset_index().sort_values(by='Riesgo_Promedio', ascending=False).head(5)
    
    # Riesgo por Categoría (Para Recomendaciones por Sector)
    df_riesgo_categoria = df_filtrado.groupby('categoria').agg(
        Activos_Totales=('uid', 'count'),
        Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
        Completitud_Promedio=('completitud_score', 'mean')
    ).reset_index().sort_values(by='Riesgo_Promedio', ascending=False).head(5)

    # 2. Lógica del K-Means (Para Activos Prioritarios)
    cluster_html = "No se pudo generar el clustering (menos de 3 activos)."
    df_activos_prioritarios = pd.DataFrame()
    
    if len(df_filtrado) >= 3:
        features = ['prioridad_riesgo_score', 'completitud_score']
        df_cluster = df_filtrado[features].dropna().copy()
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_cluster)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_cluster['cluster'] = kmeans.fit_predict(data_scaled)
        
        centers_scaled = kmeans.cluster_centers_
        centers = scaler.inverse_transform(centers_scaled)
        centers_df = pd.DataFrame(centers, columns=features)
        centers_df['sort_score'] = centers_df['completitud_score'] - centers_df['prioridad_riesgo_score']
        centers_df = centers_df.sort_values(by='sort_score', ascending=False).reset_index()
        
        cluster_map = {}
        # Los clusters se mapean por su "sort_score": Alto (0) -> Medio (1) -> Bajo (2)
        cluster_map[centers_df.loc[0, 'index']] = '🟢 Completo/Riesgo Bajo'
        cluster_map[centers_df.loc[1, 'index']] = '🟡 Aceptable/Mejora Necesaria'
        cluster_map[centers_df.loc[2, 'index']] = '🔴 Incompleto/Riesgo Alto'

        df_cluster['Calidad_Cluster'] = df_cluster['cluster'].map(cluster_map)
        
        df_viz2 = df_cluster.merge(df_filtrado[['titulo', 'dueño', 'categoria']], left_index=True, right_index=True)
        
        # Filtro de activos prioritarios
        df_activos_prioritarios = df_viz2[df_viz2['Calidad_Cluster'] == '🔴 Incompleto/Riesgo Alto'].sort_values(by='prioridad_riesgo_score', ascending=False).head(10)[['titulo', 'dueño', 'prioridad_riesgo_score', 'completitud_score']].rename(columns={'prioridad_riesgo_score': 'Riesgo Score', 'completitud_score': 'Completitud Score', 'dueño': 'Entidad'})
        
        # Generar figura K-Means (para incrustar)
        color_map = {
            '🟢 Completo/Riesgo Bajo': 'green',
            '🟡 Aceptable/Mejora Necesaria': 'orange',
            '🔴 Incompleto/Riesgo Alto': 'red'
        }
        fig2 = px.scatter(
            df_viz2, 
            x='prioridad_riesgo_score', 
            y='completitud_score', 
            color='Calidad_Cluster',
            color_discrete_map=color_map,
            hover_data=['titulo', 'dueño', 'categoria'],
            title='Segmentación de Activos por Calidad (K-Means)',
            labels={
                'prioridad_riesgo_score': 'Riesgo Promedio del Activo (Peor →)', 
                'completitud_score': 'Completitud Score del Activo (Mejor ↑)'
            }
        )
        cluster_html = fig2.to_html(full_html=False, include_plotlyjs='cdn')
        
    # 3. Generar Treemap (para incrustar)
    treemap_html = "No se pudo generar el Treemap (datos insuficientes)."
    if 'categoria' in df_filtrado.columns and len(df_filtrado) > 0 and not df_filtrado['categoria'].isnull().all():
        df_treemap = df_filtrado.groupby('categoria').agg(
            Num_Activos=('uid', 'count'),
            Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
        ).reset_index()
        fig_treemap = px.treemap(
            df_treemap,
            path=['categoria'],
            values='Num_Activos',
            color='Riesgo_Promedio', 
            color_continuous_scale=px.colors.sequential.Reds, 
            title='Matriz Treemap: Cobertura Temática vs. Riesgo Promedio'
        )
        treemap_html = fig_treemap.to_html(full_html=False, include_plotlyjs='cdn')

    # 4. Construcción del HTML
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporte de Análisis de Inventario</title>
        <meta charset="utf-8">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; }}
            h1 {{ color: #004d99; border-bottom: 3px solid #004d99; padding-bottom: 10px; }}
            h2 {{ color: #333; border-bottom: 2px solid #ccc; padding-bottom: 5px; margin-top: 30px; }}
            h3 {{ color: #555; margin-top: 20px; }}
            .metric {{ background-color: #f0f8ff; border: 1px solid #cceeff; padding: 15px; border-radius: 5px; margin-bottom: 15px; display: inline-block; width: 30%; margin-right: 1%; }}
            .high-risk {{ color: red; font-weight: bold; }}
            .low-risk {{ color: green; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .recommendation {{ background-color: #fffacd; border-left: 5px solid #ffcc00; padding: 15px; margin-top: 15px; }}
        </style>
    </head>
    <body>

    <h1>📋 Reporte Final de Análisis de Inventario de Datos</h1>
    <p><strong>Fecha de Generación:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p><strong>Umbral de Riesgo Alto (Crítico):</strong> > {umbral_riesgo:.1f}</p>

    <h2>📊 Hallazgos clave</h2>
    <div class="metric">
        <h3>Activos Analizados</h3>
        <p>{total_activos}</p>
    </div>
    <div class="metric">
        <h3>Riesgo Promedio General</h3>
        <p class="{'high-risk' if riesgo_promedio_general > umbral_riesgo else 'low-risk'}">{riesgo_promedio_general:.2f}</p>
    </div>
    <div class="metric">
        <h3>Completitud Promedio</h3>
        <p>{completitud_promedio_general:.2f}%</p>
    </div>

    <p>El activo con **Mayor Riesgo** es: {df_top_riesgo.iloc[0]['titulo']} (Riesgo: {df_top_riesgo.iloc[0]['Riesgo Score']:.2f}, Entidad: {df_top_riesgo.iloc[0]['Entidad']}).</p>
    <p>La entidad con **Mayor Riesgo Promedio** es: {df_riesgo_entidad.iloc[0]['dueño']} (Riesgo Promedio: {df_riesgo_entidad.iloc[0]['Riesgo_Promedio']:.2f}).</p>
    
    <h2>🔥 Riesgos</h2>
    <h3>Top 10 Activos con Mayor Riesgo</h3>
    <p>Activos individuales con la mayor puntuación de riesgo, indicando fallas en calidad universal y avanzada.</p>
    {df_top_riesgo[['titulo', 'Entidad', 'Riesgo Score', 'Riesgo']].to_html(index=False)}
    
    <h3>Top 5 Entidades con Mayor Riesgo Promedio</h3>
    {df_riesgo_entidad.to_html(index=False, float_format=lambda x: f'{x:.2f}')}

    <h2>🚨 Activos prioritarios</h2>
    <p>Lista de activos clasificados en el cluster **"🔴 Incompleto/Riesgo Alto"** mediante K-Means Clustering. Estos requieren atención inmediata.</p>
    {df_activos_prioritarios.to_html(index=False, float_format=lambda x: f'{x:.2f}') if not df_activos_prioritarios.empty else "<p>No se identificaron activos en el cluster de Riesgo Alto con los filtros actuales.</p>"}

    <h3>Visualización de Priorización (K-Means)</h3>
    <p>Distribución de Activos por Riesgo vs. Completitud.</p>
    {cluster_html}

    <h2>💡 Recomendaciones por sector</h2>
    <p>Análisis de las categorías (sectores) con mayor Riesgo Promedio, indicando áreas temáticas críticas.</p>
    
    {df_riesgo_categoria.to_html(index=False, float_format=lambda x: f'{x:.2f}')}

    <div class="recommendation">
        <h3>Recomendación General:</h3>
        <p>Priorizar la revisión de metadatos (completitud) y consistencia de tipos de datos en la Categoría <strong>'{df_riesgo_categoria.iloc[0]['categoria']}'</strong>, ya que presenta el mayor Riesgo Promedio ({df_riesgo_categoria.iloc[0]['Riesgo_Promedio']:.2f}).</p>
        <p>Asegurarse de que los activos más antiguos y menos usados en esta categoría no estén generando ruido o inconsistencias silenciosas.</p>
    </div>

    <h3>Visualización de Cobertura y Riesgo (Treemap)</h3>
    <p>El tamaño del bloque indica el número de activos y el color (rojo) indica el Riesgo Promedio.</p>
    {treemap_html}
    
    </body>
    </html>
    """
    return html_content

def get_table_download_link(html_content, filename, text):
    """Genera el link de descarga para el contenido HTML/PDF"""
    b64 = base64.b64encode(html_content.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 5px;">{text}</a>'
    return href

def generate_specific_recommendation(risk_dimension):
    """Genera pasos de acción específicos para la dimensión de riesgo más alta."""
    
    if 'Datos Incompletos' in risk_dimension:
        return """
**Identificación:** Localiza las columnas o filas con un alto porcentaje de valores **Nulos (NaN)**. El umbral de alerta se activa si el promedio de datos por fila es **menor al 70%**.

**Acción:** Revisa los procesos de ingesta de datos. Si el campo es **obligatorio**, asegúrate de que todos los registros lo contengan. Si el campo es **opcional**, considera si es crucial para el análisis antes de llenarlo con un valor por defecto.
        """
    elif 'Duplicados Exactos' in risk_dimension:
        return """
**Identificación:** Encuentra las filas que son **copias exactas** (duplicados de todo el registro).

**Acción:** Revisa tu proceso de extracción/carga. Un duplicado exacto generalmente indica un error de procesamiento o ingesta. **Elimina las copias** y asegúrate de que exista una **clave única** (UID) para cada registro que evite la re-ingesta accidental.
        """
    elif 'Consistencia de Tipo' in risk_dimension:
        return """
**Identificación:** Una columna contiene **datos mezclados** (ej. números, fechas, y texto en una columna que debería ser solo números). Esto afecta seriamente el análisis.

**Acción:** Normaliza el tipo de dato para la columna afectada. Si es una columna numérica, **elimina los valores de texto** o conviértelos a `NaN` para una limpieza posterior. Define el **tipo de dato esperado** (Schema) para cada columna y aplica una validación estricta al inicio del proceso.
        """
    else:
        return "No se requiere una acción específica o el riesgo detectado es demasiado bajo."


def load_knowledge_base(file_path):
    """Carga el contenido del archivo de texto como contexto del sistema."""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # st.warning() no funciona en esta parte del código que se ejecuta antes de la interfaz.
            return None
    except Exception as e:
        # st.error() no funciona aquí.
        return None

# =================================================================
# 2. FUNCIÓN ROBUSTA DEL AGENTE DE IA (USANDO RAG) (Se mantiene igual)
# =================================================================

def generate_ai_response(user_query, knowledge_base_content, model_placeholder):
    """
    Función robusta que interactúa con la API de Gemini utilizando la Base de Conocimiento (RAG).
    """
    
    # 1. Manejo de error de base de conocimiento (mostrar mensaje en el placeholder)
    if knowledge_base_content is None:
        error_msg = "No puedo responder. La base de conocimiento no ha sido cargada."
        st.session_state.messages.append({"role": "user", "content": user_query})
        with model_placeholder.chat_message("assistant"):
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        return

    # --- 2. CONFIGURACIÓN DEL CLIENTE GEMINI ---
    try:
        client = genai.Client(api_key=GEMINI_API_SECRET_VALUE)
    except Exception as e:
        error_msg = f"❌ Error al inicializar el Cliente Gemini. Verifica tu clave API. Detalle: {e}"
        st.session_state.messages.append({"role": "user", "content": user_query})
        with model_placeholder.chat_message("assistant"):
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        return

    # --- 3. CONSTRUIR PROMPT ROBUSTO ---
    system_prompt = (
        "Eres un **Analista de Inventario de Datos experto**, especializado en el análisis de calidad y riesgo de activos. "
        "Tu objetivo es responder a las preguntas del usuario basándote **ÚNICA Y EXCLUSIVAMENTE** en la 'BASE DE CONOCIMIENTO ROBUSTA' proporcionada. "
        "Utiliza la información de las tablas (KPIs, Rankings, Desgloses) para hacer inferencias, diagnósticos y sugerencias. "
        "NO uses datos brutos, solo las métricas pre-calculadas y los rankings.\n\n"
        
        "**BASE DE CONOCIMIENTO ROBUSTA (RAG CONTEXTO):**\n"
        f"```txt\n{knowledge_base_content}\n```\n\n"
        
        "**REGLAS DE RESPUESTA:**\n"
        "1. **Analiza, no solo cites:** Utiliza los datos de las tablas para dar respuestas completas y con valor. Por ejemplo, si te preguntan por el peor riesgo, cita el valor, la entidad y explícalo.\n"
        "2. **Sé conciso y profesional:** Usa un tono de experto. Incluye los valores numéricos con dos decimales cuando sea apropiado (ej: 3.14). Cita el nombre de las entidades y activos directamente de las tablas.\n"
        "3. **Si no está en el contexto:** Si la pregunta no se puede responder con la información del archivo, responde honestamente: 'La base de conocimiento no contiene la métrica o el ranking específico para responder a esa pregunta'."
    )

    # Añadir la pregunta del usuario al historial
    # NOTA: Ya fue añadido antes de llamar a esta función para que aparezca inmediatamente.

    # Generar la respuesta
    with model_placeholder.chat_message("assistant"):
        with st.spinner("Analizando la Base de Conocimiento Robustizada para generar un diagnóstico experto..."):
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[
                        {"role": "user", "parts": [{"text": user_query}]},
                    ],
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.1
                    )
                )
                
                ai_response = response.text
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})

            except Exception as e:
                error_msg = f"❌ Error en la API de Gemini: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# =================================================================
# 3. Ejecución Principal del Dashboard
# =================================================================

st.set_page_config(page_title="Asistente de Análisis de Inventario", layout="wide") # Asegura layout wide

st.title("📊 Dashboard de Priorización de Activos de Datos (Análisis Completo)")

try:
    with st.spinner(f'Cargando archivo procesado: **{ARCHIVO_PROCESADO}**...'):
        df_analisis_completo = load_processed_data(ARCHIVO_PROCESADO) 

    if df_analisis_completo.empty:
        st.error(f"🛑 Error: No se pudo cargar el archivo **{ARCHIVO_PROCESADO}**. Asegúrate de que existe y se ejecutó `preprocess.py`.")
    else:
        # 🚀 ADICIÓN: APLICAR DETECCIÓN DE ANOMALÍAS CON ML (Isolation Forest)
        df_analisis_completo = apply_anomaly_detection(df_analisis_completo)
        # 🚀 FIN ADICIÓN: DETECCIÓN DE ANOMALÍAS
        
        # --- APLICAR CHEQUEOS DE RIESGO AVANZADOS (NUEVA LÓGICA) ---
        # Ahora apply_advanced_risk_checks usa el resultado de anomalia_score
        df_analisis_completo = apply_advanced_risk_checks(df_analisis_completo) 
        # --- FIN DE APLICACIÓN DE CHEQUEOS AVANZADOS ---
        
        st.success(f'✅ Archivo pre-procesado cargado. Total de activos: **{len(df_analisis_completo)}**')

        # --- Carga de la Base de Conocimiento (Inicialización) ---
        if "knowledge_content" not in st.session_state:
            st.session_state.knowledge_content = load_knowledge_base(KNOWLEDGE_FILE)

        knowledge_base_content = st.session_state.knowledge_content
        
        # --- Inicialización de variables de estado (Main) ---
        if "messages" not in st.session_state:
            st.session_state.messages = []

        
        # ----------------------------------------------------------------------
        # --- FILTROS EN EL SIDEBAR (SE MANTIENEN AQUÍ) ---
        # ----------------------------------------------------------------------
        with st.sidebar:
            st.header("⚙️ Filtros para Visualizaciones")
            
            filtro_acceso_publico = False 
            
            if 'audience' in df_analisis_completo.columns:
                filtro_acceso_publico = st.checkbox(
                    "Mostrar Solo Activos 'public'",
                    value=False,
                    help="Si está marcado, solo se mostrarán los activos cuyo nivel de acceso sea 'public'."
                )
            
            filtro_categoria = "Mostrar Todos"
            if 'categoria' in df_analisis_completo.columns:
                categories = df_analisis_completo['categoria'].dropna().unique().tolist()
                categories.sort()
                categories.insert(0, "Mostrar Todos")
                filtro_categoria = st.selectbox("Filtrar por Categoría:", categories)
                
            # 🚀 Botón de Descarga del Reporte en el Sidebar
            st.markdown("---")
            st.subheader("📥 Generar Reporte Final")
            
            if st.button("Generar y Descargar Reporte (HTML)"):
                report_html = generate_report_html(df_analisis_completo, UMBRAL_RIESGO_ALTO) 
                
                filename = f"Reporte_Inventario_Datos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                
                st.markdown(
                    get_table_download_link(report_html, filename, "Click para Descargar Reporte"), 
                    unsafe_allow_html=True
                )


        # ----------------------------------------------------------------------
        # --- CONTENIDO PRINCIPAL ---
        # ----------------------------------------------------------------------
        
        # --- SECCIÓN DE SELECCIÓN Y DESGLOSE DE ENTIDAD ---
        owners = df_analisis_completo['dueño'].dropna().unique().tolist()
        owners.sort()
        owners.insert(0, "Mostrar Análisis General")
        
        filtro_dueño = st.selectbox(
            "Selecciona una Entidad para ver su Desglose de Estadísticas:",
            owners
        )
        
        # --- DESGLOSE DE ESTADÍSTICAS (KPIs) ---
        if filtro_dueño != "Mostrar Análisis General":
            df_entidad_seleccionada = df_analisis_completo[df_analisis_completo['dueño'] == filtro_dueño]
            
            if not df_entidad_seleccionada.empty:
                st.subheader(f"Estadísticas Clave para: **{filtro_dueño}**")
                
                total_activos = len(df_entidad_seleccionada)
                incumplimiento = (df_entidad_seleccionada['estado_actualizacion'] == '🔴 INCUMPLIMIENTO').sum()
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                col1.metric("Activos Totales", total_activos)
                col2.metric("Completitud Promedio", f"{df_entidad_seleccionada['completitud_score'].mean():.2f}%")
                col3.metric("Riesgo Promedio", f"{df_entidad_seleccionada['prioridad_riesgo_score'].mean():.2f}")
                col4.metric("Incumplimiento Absoluto", f"{incumplimiento} / {total_activos}")
                
                if 'antiguedad_datos_dias' in df_entidad_seleccionada.columns:
                    col5.metric("Antigüedad Promedio", f"{df_entidad_seleccionada['antiguedad_datos_dias'].mean():.0f} días")
                else:
                    col5.metric("Antigüedad Promedio", "N/A")
                
                st.markdown("---")
            else:
                st.warning(f"⚠️ No se encontraron activos para la entidad: {filtro_dueño}")
                st.markdown("---")

        # --- APLICAR FILTROS (Para las Visualizaciones) ---
        df_filtrado = df_analisis_completo.copy()
        
        if filtro_dueño != "Mostrar Análisis General":
             df_filtrado = df_filtrado[df_filtrado['dueño'] == filtro_dueño]

        # --- LÓGICA DE FILTRO CLAVE ---
        if filtro_acceso_publico:
             df_filtrado = df_filtrado[df_filtrado['common_core_public_access_level'] == 'public']
        
        if filtro_categoria != "Mostrar Todos":
            df_filtrado = df_filtrado[df_filtrado['categoria'] == filtro_categoria]

            
        st.header("📊 Visualizaciones y Rankings")
        
        # Actualizar el texto informativo
        info_acceso = "solo Activos Públicos" if filtro_acceso_publico else "Todos los Niveles de Acceso"
        st.info(f"Vista actual de gráficos: **{len(df_filtrado)} activos** (Filtro de Entidad: {filtro_dueño}; Acceso: {info_acceso}; Categoría: {filtro_categoria})")

        if df_filtrado.empty:
            st.warning("⚠️ No hay datos para mostrar en los gráficos con los filtros seleccionados.")
        else:
            
            # --- 3. Métricas de la Vista Actual ---
            st.subheader("Métricas de la Vista Actual")
            col_metrica1, col_metrica2, col_metrica3 = st.columns(3)
            col_metrica1.metric("Completitud Promedio", f"{df_filtrado['completitud_score'].mean():.2f}%")
            col_metrica2.metric("Activos en Incumplimiento", f"{(df_filtrado['estado_actualizacion'] == '🔴 INCUMPLIMIENTO').sum()} / {len(df_filtrado)}")
            # ⚠️ Uso del score generado por Isolation Forest:
            col_metrica3.metric("Anomalías Detectadas (ML)", f"{(df_filtrado['anomalia_score'] == -1).sum()}")
            
            st.markdown("---")

            # --- 4. Tabla de Búsqueda y Diagnóstico ---
            st.header("🔍 4. Tabla de Búsqueda y Diagnóstico")

            
            # Lógica Condicional para mostrar la tabla
            
            # Determinar si se debe mostrar el detalle de activos individuales:
            # 1. Si se filtra por activos públicos (filtro_acceso_publico)
            # 2. O si se ha seleccionado una entidad específica (filtro_dueño)
            show_asset_detail = filtro_acceso_publico or (filtro_dueño != "Mostrar Análisis General")

            if show_asset_detail:
                # Caso: Activos Públicos O Entidad Específica (Mostrar detalle por ACTIVO)
                
                # Lógica para personalizar el encabezado
                if filtro_dueño != "Mostrar Análisis General":
                    st.subheader(f"Detalle de Activos Individuales para la Entidad: **{filtro_dueño}**")
                    info_text = f"""
                        **Vista Detallada:** Se muestran los **{len(df_filtrado)} activos individuales** de la entidad **{filtro_dueño}**, ordenados por su Score de Riesgo (más alto primero).
                        * 🟢 **Verde:** Riesgo $\le {UMBRAL_RIESGO_ALTO:.1f}$
                        * 🔴 **Rojo:** Riesgo $> {UMBRAL_RIESGO_ALTO:.1f}$ (Prioridad Máxima)
                        
                        **NOTA:** Este riesgo ahora incluye penalizaciones avanzadas por **Inconsistencia de Metadatos**, **Duplicidad Semántica/Cambios Abruptos** y **Activos Vacíos**. El riesgo máximo teórico es **{RIESGO_MAXIMO_TEORICO_AVANZADO:.1f}**.
                    """
                else:
                    st.subheader("Detalle por Activo Público (Priorización Individual)")
                    info_text = f"""
                        **Vista Detallada:** Se muestran los **activos individuales públicos** filtrados, ordenados por su Score de Riesgo (más alto primero).
                        * 🟢 **Verde:** Riesgo $\le {UMBRAL_RIESGO_ALTO:.1f}$
                        * 🔴 **Rojo:** Riesgo $> {UMBRAL_RIESGO_ALTO:.1f}$ (Prioridad Máxima)
                        
                        **NOTA:** Este riesgo ahora incluye penalizaciones avanzadas por **Inconsistencia de Metadatos**, **Duplicidad Semántica/Cambios Abruptos** y **Activos Vacíos**. El riesgo máximo teórico es **{RIESGO_MAXIMO_TEORICO_AVANZADO:.1f}**.
                    """

                # Definir las columnas a mostrar
                cols_common = ['titulo', 'prioridad_riesgo_score', 'completitud_score', 'antiguedad_datos_dias']
                
                if filtro_dueño == "Mostrar Análisis General":
                    # Activos Públicos de Múltiples Dueños: Mostrar el dueño.
                    cols_to_show = ['dueño'] + cols_common
                    column_config_map = {
                        'dueño': st.column_config.TextColumn("Entidad Responsable"),
                        'titulo': st.column_config.TextColumn("Título del Activo"),
                        'prioridad_riesgo_score': st.column_config.NumberColumn("Riesgo Score", help=f"Rojo > {UMBRAL_RIESGO_ALTO:.1f}."),
                        'completitud_score': st.column_config.NumberColumn("Completitud Score", format="%.2f%%"),
                        'antiguedad_datos_dias': st.column_config.NumberColumn("Antigüedad (Días)", format="%d"),
                    }
                else: 
                    # Entidad Específica Seleccionada: El dueño es único, no se muestra.
                    cols_to_show = cols_common
                    column_config_map = {
                        'titulo': st.column_config.TextColumn("Título del Activo"),
                        'prioridad_riesgo_score': st.column_config.NumberColumn("Riesgo Score", help=f"Rojo > {UMBRAL_RIESGO_ALTO:.1f}."),
                        'completitud_score': st.column_config.NumberColumn("Completitud Score", format="%.2f%%"),
                        'antiguedad_datos_dias': st.column_config.NumberColumn("Antigüedad (Días)", format="%d"),
                    }
                
                df_tabla_activos = df_filtrado[cols_to_show].copy()
                
                # Renombrar columnas para la tabla
                rename_map = {
                    'titulo': 'Activo',
                    'prioridad_riesgo_score': 'Riesgo_Score',
                    'completitud_score': 'Completitud_Score',
                    'antiguedad_datos_dias': 'Antiguedad_Dias'
                }
                if 'dueño' in df_tabla_activos.columns:
                    rename_map['dueño'] = 'Entidad Responsable'
                
                df_tabla_activos = df_tabla_activos.rename(columns=rename_map).sort_values(by='Riesgo_Score', ascending=False)
                
                
                def color_riesgo_score(val):
                    # ⚠️ USO DEL UMBRAL (3.5)
                    color = 'background-color: #f79999' if val > UMBRAL_RIESGO_ALTO else 'background-color: #a9dfbf'
                    return color
                
                styled_df = df_tabla_activos.style.applymap(
                    color_riesgo_score, 
                    subset=['Riesgo_Score']
                ).format({
                    'Riesgo_Score': '{:.2f}',
                    'Completitud_Score': '{:.2f}%',
                    'Antiguedad_Dias': '{:.0f}'
                })
                
                st.info(info_text)

                # Ajustar column_config para quitar el dueño si no está en la tabla
                if 'Entidad Responsable' not in df_tabla_activos.columns:
                    column_config_map.pop('Entidad Responsable', None) # Quitar si no existe
                    
                st.dataframe(
                    styled_df, 
                    use_container_width=True,
                    column_config=column_config_map,
                    hide_index=True
                )
                
            else:
                # Caso: Activos No Públicos o Todos Y Análisis General (Mostrar resumen AGRUPADO por ENTIDAD)
                st.subheader("Resumen Agrupado por Entidad Responsable")
                
                st.info(f"""
                    La columna **Riesgo Promedio** tiene un formato de color:
                    * 🟢 **Verde:** El riesgo promedio es **menor o igual a {UMBRAL_RIESGO_ALTO:.1f}**. Intervención no urgente.
                    * 🔴 **Rojo:** El riesgo promedio es **mayor a {UMBRAL_RIESGO_ALTO:.1f}**. Se requiere **intervención/actualización prioritaria**.

                    **NOTA:** Este riesgo ahora incluye penalizaciones avanzadas por **Inconsistencia de Metadatos**, **Duplicidad Semántica/Cambios Abruptos** y **Activos Vacíos**. El riesgo máximo teórico es **{RIESGO_MAXIMO_TEORICO_AVANZADO:.1f}**.
                """)

                resumen_entidades_busqueda = df_filtrado.groupby('dueño').agg(
                    Activos_Totales=('uid', 'count'),
                    Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
                    Completitud_Promedio=('completitud_score', 'mean'),
                    Antiguedad_Promedio_Dias=('antiguedad_datos_dias', 'mean'),
                    Incumplimiento_Absoluto=('estado_actualizacion', lambda x: (x == '🔴 INCUMPLIMIENTO').sum())
                ).reset_index()

                resumen_entidades_busqueda['%_Incumplimiento'] = (resumen_entidades_busqueda['Incumplimiento_Absoluto'] / resumen_entidades_busqueda['Activos_Totales']) * 100
                resumen_entidades_busqueda = resumen_entidades_busqueda.rename(columns={'dueño': 'Entidad Responsable'})
                resumen_entidades_busqueda = resumen_entidades_busqueda.sort_values(by='Riesgo_Promedio', ascending=False)
                
                def color_riesgo_promedio(val):
                    # ⚠️ USO DEL UMBRAL (3.5)
                    color = 'background-color: #f79999' if val > UMBRAL_RIESGO_ALTO else 'background-color: #a9dfbf'
                    return color
                
                styled_df = resumen_entidades_busqueda.style.applymap(
                    color_riesgo_promedio, 
                    subset=['Riesgo_Promedio']
                ).format({
                    'Riesgo_Promedio': '{:.2f}',
                    'Completitud_Promedio': '{:.2f}%',
                    'Antiguedad_Promedio_Dias': '{:.0f}',
                    '%_Incumplimiento': '{:.2f}%'
                })


                st.dataframe(
                    styled_df, 
                    use_container_width=True,
                    column_config={
                        'Entidad Responsable': st.column_config.TextColumn("Entidad Responsable"),
                        'Activos_Totales': st.column_config.NumberColumn("Activos Totales"),
                        # ⚠️ USO DEL UMBRAL (3.5) en la ayuda
                        'Riesgo_Promedio': st.column_config.NumberColumn("Riesgo Promedio (Score)", help=f"Rojo > {UMBRAL_RIESGO_ALTO:.1f}."),
                        'Completitud_Promedio': st.column_config.NumberColumn("Completitud Promedio", format="%.2f%%"),
                        'Antiguedad_Promedio_Dias': st.column_config.NumberColumn("Antigüedad Promedio (Días)", format="%d"),
                        'Incumplimiento_Absoluto': st.column_config.NumberColumn("Activos en Incumplimiento (Count)"),
                        '%_Incumplimiento': st.column_config.TextColumn("% Incumplimiento")
                    },
                    hide_index=True
                )

            st.markdown("---")
            
            # ----------------------------------------------------------------------
            # --- BLOQUE CLAVE DE PESTAÑAS (GRÁFICOS) ---
            # ----------------------------------------------------------------------
            
            if filtro_acceso_publico:
                # 📌 CASO: Activos Públicos (Priorización)
                tab1, tab2, tab3, tab4 = st.tabs(["1. Ranking de Priorización (Riesgo/Incompletitud)", "2. K-Means Clustering", "3. Activos Menos Actualizados (Antigüedad)", "4. Treemap de Cobertura y Calidad"])
            else:
                # 📌 CASO: Vista General (Completitud/Riesgo)
                tab1, tab2, tab3, tab4 = st.tabs(["1. Ranking de Completitud", "2. K-Means Clustering (Priorización)", "3. Cobertura Temática", "4. Treemap de Cobertura y Calidad"])

            with tab1:
                # --- Visualización 1: Ranking de Priorización (Combinado o por Entidad) ---
                
                if filtro_acceso_publico:
                    st.subheader("1. 🔴 Ranking Top 10 Activos Públicos (Incompletos y Riesgo Alto)")
                    st.info("Este ranking prioriza activos públicos con el **peor rendimiento combinado**: Bajo Score de Completitud y Alto Score de Riesgo. La puntuación de visualización es un promedio simple de estos dos factores normalizados.")
                    
                    df_viz1_public = df_filtrado.copy()
                    
                    df_viz1_public['prioridad_riesgo_score'] = df_viz1_public['prioridad_riesgo_score'].astype(float)
                    df_viz1_public['completitud_score_inv'] = (100 - df_viz1_public['completitud_score']).astype(float)

                    max_riesgo = df_viz1_public['prioridad_riesgo_score'].max()
                    df_viz1_public['riesgo_norm'] = df_viz1_public['prioridad_riesgo_score'] / max_riesgo if max_riesgo > 0 else 0
                    
                    max_incomp = df_viz1_public['completitud_score_inv'].max()
                    df_viz1_public['incomp_norm'] = df_viz1_public['completitud_score_inv'] / max_incomp if max_incomp > 0 else 0
                    
                    df_viz1_public['prioridad_combinada'] = (df_viz1_public['riesgo_norm'] + df_viz1_public['incomp_norm']) / 2
                    
                    df_viz1 = df_viz1_public.sort_values(by='prioridad_combinada', ascending=False).head(10)
                    EJE_Y = 'titulo'
                    X_COLUMN = 'prioridad_combinada'
                    TITULO = 'Top 10 Activos Públicos: Peor Prioridad (Riesgo/Incompletitud)'
                    Y_TITLE = 'Activo'
                    X_TITLE = 'Score de Prioridad Combinada (0=Bajo, 1=Alto)'
                    
                else:
                    st.subheader("1. 📉 Ranking de Entidades por Completitud Promedio (Peor Rendimiento)")
                    COLUMNA_ENTIDAD = 'dueño'
                    resumen_completitud = df_filtrado.groupby(COLUMNA_ENTIDAD).agg(
                        Total_Activos=('uid', 'count'),
                        Completitud_Promedio=('completitud_score', 'mean')
                    ).reset_index()
                    entidades_volumen = resumen_completitud[resumen_completitud['Total_Activos'] >= 5]
                    df_viz1 = entidades_volumen.sort_values(by='Completitud_Promedio', ascending=True).head(10)
                    EJE_Y = COLUMNA_ENTIDAD
                    X_COLUMN = 'Completitud_Promedio'
                    TITULO = 'Top 10 Entidades con Peor Completitud Promedio'
                    Y_TITLE = 'Entidad Responsable'
                    X_TITLE = 'Score de Completitud Promedio (%)'


                try:
                    
                    if not df_viz1.empty:
                        fig1 = px.bar(
                            df_viz1, 
                            x=X_COLUMN, 
                            y=EJE_Y, 
                            orientation='h',
                            title=TITULO,
                            labels={X_COLUMN: X_TITLE, EJE_Y: Y_TITLE},
                            color=X_COLUMN,
                            color_continuous_scale=px.colors.sequential.Reds_r, 
                            height=500
                        )
                        fig1.update_layout(xaxis_title=X_TITLE, yaxis_title=Y_TITLE)
                        st.plotly_chart(fig1, use_container_width=True) 
                    else:
                        st.warning("No hay suficientes datos para generar el ranking con los filtros seleccionados.")
                except Exception as e:
                    st.error(f"❌ ERROR [Visualización 1]: Falló la generación del Gráfico de Priorización. Detalle: {e}")

            with tab2:
                # --- Visualización 2: K-Means Clustering para Segmentación de Calidad ---
                st.subheader("2. 💡 K-Means Clustering: Segmentación de Calidad (3 Grupos)")
                st.markdown("Se aplica el algoritmo K-Means para segmentar los activos en **3 grupos de calidad** basándose en su **Riesgo** y **Completitud**.")
                
                try:
                    # ... (Lógica de K-Means, se mantiene)
                    features = ['prioridad_riesgo_score', 'completitud_score']
                    df_cluster = df_filtrado[features].dropna().copy()
                    
                    if len(df_cluster) < 3:
                        st.warning("Se requieren al menos 3 activos para ejecutar K-Means.")
                    else:
                        scaler = StandardScaler()
                        data_scaled = scaler.fit_transform(df_cluster)
                        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                        df_cluster['cluster'] = kmeans.fit_predict(data_scaled)
                        
                        centers_scaled = kmeans.cluster_centers_
                        centers = scaler.inverse_transform(centers_scaled)
                        centers_df = pd.DataFrame(centers, columns=features)
                        centers_df['sort_score'] = centers_df['completitud_score'] - centers_df['prioridad_riesgo_score']
                        centers_df = centers_df.sort_values(by='sort_score', ascending=False).reset_index()
                        
                        cluster_map = {}
                        cluster_map[centers_df.loc[0, 'index']] = '🟢 Completo/Riesgo Bajo'
                        cluster_map[centers_df.loc[1, 'index']] = '🟡 Aceptable/Mejora Necesaria'
                        cluster_map[centers_df.loc[2, 'index']] = '🔴 Incompleto/Riesgo Alto'

                        df_cluster['Calidad_Cluster'] = df_cluster['cluster'].map(cluster_map)

                        color_map = {
                            '🟢 Completo/Riesgo Bajo': 'green',
                            '🟡 Aceptable/Mejora Necesaria': 'orange',
                            '🔴 Incompleto/Riesgo Alto': 'red'
                        }
                        
                        df_viz2 = df_cluster.merge(df_filtrado[['titulo', 'dueño', 'categoria']], left_index=True, right_index=True)
                        
                        fig2 = px.scatter(
                            df_viz2, 
                            x='prioridad_riesgo_score', 
                            y='completitud_score', 
                            color='Calidad_Cluster',
                            color_discrete_map=color_map,
                            hover_data=['titulo', 'dueño', 'categoria'],
                            title='Segmentación de Activos por Calidad (K-Means)',
                            labels={
                                'prioridad_riesgo_score': 'Riesgo Promedio del Activo (Peor →)', 
                                'completitud_score': 'Completitud Score del Activo (Mejor ↑)',
                                'Calidad_Cluster': 'Segmento de Calidad'
                            },
                            height=600
                        )
                        
                        fig2.update_layout(
                            xaxis_title='Riesgo Promedio del Activo',
                            yaxis_title='Completitud Score del Activo (%)'
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ ERROR [Visualización 2]: Falló la generación del K-Means Clustering. Detalle: Asegúrate de tener suficientes datos ({len(df_cluster)}) para el clustering. Error técnico: {e}")


            with tab3:
                # --- Visualización 3: Cobertura Temática (General) o Activos Menos Actualizados (Público) ---
                
                if filtro_acceso_publico:
                    # 📌 NUEVO GRÁFICO: Activos Menos Actualizados (Antigüedad)
                    st.subheader("3. ⏰ Ranking Top 10 Activos Públicos Menos Actualizados")
                    st.info("Estos activos requieren una revisión inmediata de su proceso de recolección de datos, ya que su antigüedad es la más alta en el inventario público.")
                    
                    df_viz3 = df_filtrado.sort_values(by='antiguedad_datos_dias', ascending=False).head(10)
                    EJE_Y = 'titulo'
                    X_COLUMN = 'antiguedad_datos_dias'
                    TITULO = 'Top 10 Activos Públicos con Mayor Antigüedad (Menos Actualizados)'
                    X_TITLE = 'Antigüedad (Días)'
                    Y_TITLE = 'Activo'
                    COLOR_SCALE = px.colors.sequential.YlOrRd # Escala que va a rojo (peor)

                else:
                    # 📌 GRÁFICO EXISTENTE: Cobertura Temática (General)
                    st.subheader("3. 🗺️ Cobertura Temática por Categoría (Mayor a Menor)")
                    
                    COLUMNA_CATEGORIA = 'categoria'
                    if COLUMNA_CATEGORIA in df_filtrado.columns:
                        conteo_categoria = df_filtrado[COLUMNA_CATEGORIA].value_counts().head(10).reset_index()
                        conteo_categoria.columns = ['Categoria', 'Numero_de_Activos']
                        conteo_categoria = conteo_categoria.sort_values(by='Numero_de_Activos', ascending=False)
                    else:
                        conteo_categoria = pd.DataFrame({'Categoria': [], 'Numero_de_Activos': []})
                        
                    df_viz3 = conteo_categoria
                    EJE_Y = 'Categoria'
                    X_COLUMN = 'Numero_de_Activos'
                    TITULO = 'Top 10 Categorías con Mayor Cobertura Temática'
                    X_TITLE = 'Número de Activos'
                    Y_TITLE = 'Categoría'
                    COLOR_SCALE = px.colors.sequential.Viridis
                    

                try:
                    if not df_viz3.empty:
                        fig3 = px.bar(
                            df_viz3, 
                            x=X_COLUMN, 
                            y=EJE_Y, 
                            orientation='h',
                            title=TITULO,
                            labels={X_COLUMN: X_TITLE, EJE_Y: Y_TITLE},
                            color=X_COLUMN,
                            color_continuous_scale=COLOR_SCALE,
                            height=500
                        )
                        fig3.update_layout(xaxis_title=X_TITLE, yaxis_title=Y_TITLE)
                        st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.warning("La columna 'antiguedad_datos_dias' o 'categoria' no contiene suficientes valores para generar la visualización.")
                except Exception as e:
                    st.error(f"❌ ERROR [Visualización 3]: Falló la generación del Bar Plot. Detalle: {e}")

            # 🚀 ADICIÓN DEL BLOQUE DE CÓDIGO PARA EL TREEMAP (tab4)
            with tab4:
                # --- Visualización 4: Treemap de Cobertura y Calidad ---
                st.subheader("4. 🌳 Matriz Treemap: Cobertura Temática vs. Riesgo Promedio")
                st.info("El tamaño de cada bloque representa el **Número de Activos** en esa Categoría, y el color indica el **Riesgo Promedio** (más rojo = Riesgo Alto).")
                
                try:
                    # 1. Agrupación por categoría para el Treemap
                    COLUMNA_CATEGORIA = 'categoria'
                    
                    # Se necesita al menos una fila y la columna de categoría
                    if COLUMNA_CATEGORIA in df_filtrado.columns and len(df_filtrado) > 0 and not df_filtrado[COLUMNA_CATEGORIA].isnull().all():
                        df_treemap = df_filtrado.groupby(COLUMNA_CATEGORIA).agg(
                            Num_Activos=('uid', 'count'),
                            Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
                            Completitud_Promedio=('completitud_score', 'mean')
                        ).reset_index()

                        # 2. Crear el Treemap
                        fig_treemap = px.treemap(
                            df_treemap,
                            path=[COLUMNA_CATEGORIA],
                            values='Num_Activos',
                            color='Riesgo_Promedio',  # Color por Riesgo Promedio (indicador de Calidad)
                            color_continuous_scale=px.colors.sequential.Reds, # Escala de color: Rojo = Riesgo Alto
                            hover_data=['Riesgo_Promedio', 'Completitud_Promedio', 'Num_Activos'],
                            title='Matriz Treemap: Cobertura Temática vs. Riesgo Promedio'
                        )
                        
                        fig_treemap.update_layout(margin=dict(t=50, l=25, r=25, b=25))
                        st.plotly_chart(fig_treemap, use_container_width=True)
                    
                    else:
                        st.warning("No hay suficientes datos o la columna 'categoria' no está disponible para generar el Treemap.")

                except Exception as e:
                    st.error(f"❌ ERROR [Visualización 4]: Falló la generación del Treemap. Detalle: {e}")
            # 🚀 FIN DE ADICIÓN DEL BLOQUE DE CÓDIGO PARA EL TREEMAP (tab4)


            
            # ----------------------------------------------------------------------
            # --- SECCIÓN 5: DIAGNÓSTICO DE ARCHIVO EXTERNO
            # ----------------------------------------------------------------------
            st.markdown("<hr style='border: 4px solid #f0f2f6;'>", unsafe_allow_html=True)
            st.header("💾 Diagnóstico de Archivo CSV Externo (Calidad Universal)")
            st.markdown(f"Sube un archivo CSV. La **Calidad Total** se calcula en base a 3 dimensiones universales (Riesgo Máximo: **{RIESGO_MAXIMO_TEORICO_UNIVERSAL:.1f}**).")

            uploaded_file = st.file_uploader(
                "Selecciona el Archivo CSV", 
                type="csv"
            )

            if uploaded_file is not None:
                with st.spinner('Analizando archivo...'):
                    try:
                        uploaded_filename = uploaded_file.name
                        # Lógica de lectura robusta con detección de delimitadores
                        uploaded_file.seek(0)
                        file_contents = uploaded_file.getvalue().decode("utf-8")
                        
                        # Intentos de lectura con diferentes separadores
                        try:
                            uploaded_df = pd.read_csv(io.StringIO(file_contents), low_memory=False)
                        except Exception:
                            try:
                                uploaded_df = pd.read_csv(io.StringIO(file_contents), low_memory=False, sep=';')
                            except Exception:
                                try:
                                    uploaded_df = pd.read_csv(io.StringIO(file_contents), low_memory=False, sep='\t')
                                except Exception:
                                    st.error("❌ No se pudo determinar el delimitador del archivo.")
                                    uploaded_df = pd.DataFrame() # Vaciar si falla
                                    
                        if uploaded_df.empty:
                            st.warning(f"⚠️ El archivo subido **{uploaded_filename}** está vacío o es ilegible.")
                        else:
                            df_diagnostico = process_external_data(uploaded_df.copy())
                            
                            if not df_diagnostico.empty:
                                
                                # Métricas consolidadas
                                calidad_total_final = df_diagnostico['calidad_total_score'].iloc[0] 
                                completitud_universal_promedio = df_diagnostico['completitud_metadatos_universal'].iloc[0] 
                                riesgo_promedio_total = df_diagnostico['prioridad_riesgo_score'].mean()

                                # Desglose de Riesgos Promedio
                                riesgos_reporte = pd.DataFrame({
                                    'Dimensión de Riesgo': [
                                        '1. Datos Incompletos (Completitud)',
                                        '2. Duplicados Exactos (Unicidad)',
                                        '3. Consistencia de Tipo (Coherencia)',
                                    ],
                                    'Riesgo Promedio (0-Máx)': [
                                        df_diagnostico['riesgo_datos_incompletos'].mean(),
                                        df_diagnostico['riesgo_duplicado'].mean(),
                                        df_diagnostico['riesgo_consistencia_tipo'].mean(),
                                    ]
                                })
                                riesgos_reporte = riesgos_reporte.sort_values(by='Riesgo Promedio (0-Máx)', ascending=False)
                                riesgos_reporte['Riesgo Promedio (0-Máx)'] = riesgos_reporte['Riesgo Promedio (0-Máx)'].round(2)
                                
                                
                                # === LÓGICA DE RECOMENDACIÓN PRÁCTICA ===
                                
                                recomendacion_final_md = ""
                                
                                riesgo_max_reportado = riesgos_reporte.iloc[0]['Riesgo Promedio (0-Máx)']
                                
                                if riesgo_max_reportado > 0.15:
                                    riesgo_dimension_max = riesgos_reporte.iloc[0]['Dimensión de Riesgo']
                                    explicacion_especifica = generate_specific_recommendation(riesgo_dimension_max)
                                    
                                    recomendacion_final_md = f"""
El riesgo más alto es por **{riesgo_dimension_max}** ({riesgo_max_reportado:.2f}). Enfoca tu esfuerzo en corregir este problema primero.

<br>

**Detalle y Acciones:**

{explicacion_especifica}
"""

                                if not recomendacion_final_md:
                                    recomendacion_final_md = "La Calidad es excelente. No se requieren mejoras prioritarias en las dimensiones analizadas."
                                    estado = "🟢 CALIDAD ALTA"
                                    color = "green"
                                else:
                                    if calidad_total_final < 60:
                                        estado = "🔴 CALIDAD BAJA (URGENTE)"
                                        color = "red"
                                    elif calidad_total_final < 85:
                                        estado = "🟡 CALIDAD MEDIA (MEJORA REQUERIDA)"
                                        color = "orange"
                                    else:
                                        estado = "🟢 CALIDAD ACEPTABLE"
                                        color = "green"
                                
                                # === FIN LÓGICA DE RECOMENDACIÓN ===
                                
                                st.subheader("Resultados del Diagnóstico Rápido")
                                
                                # --- DESPLIEGUE DE MÉTRICAS SIMPLIFICADO ---
                                col_calidad, col_meta, col_riesgo = st.columns(3)
                                
                                col_calidad.metric("⭐ Calidad Total del Archivo", f"{calidad_total_final:.1f}%")
                                col_meta.metric("Completitud Metadatos (Avg)", f"{completitud_universal_promedio:.2f}%") 
                                col_riesgo.metric("Riesgo Promedio Total", f"{riesgo_promedio_total:.2f}")

                                # Despliegue de la Recomendación
                                st.markdown(f"""
                                    <div style='border: 2px solid {color}; padding: 15px; border-radius: 5px; background-color: #f9f9f9;'>
                                        <h4 style='color: {color}; margin-top: 0;'>Diagnóstico General: {estado}</h4>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown("#### 🔬 Desglose de Riesgos (Auditoría)")
                                
                                st.dataframe(
                                    riesgos_reporte.set_index('Dimensión de Riesgo'),
                                    use_container_width=True
                                )

                                st.markdown(f"#### ✨ Recomendación de Acciones:")
                                st.markdown(recomendacion_final_md, unsafe_allow_html=True)

                            else:
                                st.error(f"❌ El archivo subido **{uploaded_filename}** no pudo ser procesado.")
                                
                    except Exception as e:
                        st.error(f"❌ Error al leer o procesar el archivo CSV: {e}")
                        st.warning("Asegúrate de que el archivo es un CSV válido y tiene un formato consistente.")
            
            # ----------------------------------------------------------------------
            # 🧠 ASISTENTE DE DATOS (AL FINAL DE LA PÁGINA)
            # ----------------------------------------------------------------------
            st.markdown("<hr style='border: 4px solid #38c8f0;'>", unsafe_allow_html=True)
            st.header("🧠 Asistente de Análisis Experto (Base de Conocimiento)")
            st.info(
                "Pregunta por los **KPIs, rankings o diagnósticos** basados en la Base de Conocimiento. "
                "Ej: '¿Qué entidad tiene más activos?', 'Dime el Top 5 peores activos por riesgo', "
                "'¿Cuál es el riesgo promedio en activos en incumplimiento?'"
            )
            
            if knowledge_base_content is None:
                 st.error("La base de conocimiento `knowledge_base.txt` no fue encontrada. El asistente no funcionará.")
            
            # 1. Contenedor para el Historial de Conversación
            # Usamos un contenedor simple ya que estamos en el cuerpo principal
            chat_history_container = st.container()
            
            with chat_history_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            # 2. Lógica de Interacción (Chat Input - en el cuerpo principal)
            if prompt := st.chat_input("Escribe aquí tu pregunta de análisis complejo:", key="main_chat_input_key", disabled=(knowledge_base_content is None)):
                
                # --- Agregar el mensaje del usuario y simular la respuesta inmediata ---\
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Para que el mensaje del usuario aparezca inmediatamente en el historial
                with chat_history_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Placeholder para la respuesta del Asistente (se llenará en la función)
                    model_response_placeholder = st.empty() 
                    
                    # Llamar a la función de generación
                    generate_ai_response(prompt, knowledge_base_content, model_response_placeholder)
                        # =================================================================
                        # --- BLOQUE TEMPORAL DE DEPURACIÓN DE COLUMNAS ---
                        # =================================================================
                        
                        if 'df_analisis_completo' in locals() and not df_analisis_completo.empty:
                            with st.expander("🔍 Mostrar Columnas Disponibles (DEBUGGING TEMPORAL)"):
                                st.info("Estas son todas las columnas en tu DataFrame procesado `df_analisis_completo`:")
                                # Usamos .columns.tolist() para obtener una lista fácil de leer
                                st.code(df_analisis_completo.columns.tolist())
                        
                        # =================================================================
    st.error(f"❌ ERROR FATAL: Ocurrió un error inesperado al iniciar la aplicación: {e}")


