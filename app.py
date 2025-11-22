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
from google import genai 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import requests # Necesario para el fetch de la API Mintic

# Ocultar advertencias de Pandas/Streamlit
warnings.filterwarnings('ignore') 

# =================================================================
# 0. VARIABLES GLOBALES Y CONFIGURACIÓN
# =================================================================

ARCHIVO_PROCESADO = "Asset_Inventory_PROCESSED.csv" 
KNOWLEDGE_FILE = "knowledge_base.txt" 

# CRITERIO DE RIESGO
# Umbral de Riesgo Alto (Crítico)
UMBRAL_RIESGO_ALTO = 3.5 

# --- CONFIGURACIÓN DE RIESGOS UNIVERSALES ---
PENALIZACION_DATOS_INCOMPLETOS = 2.0  
PENALIZACION_INCONSISTENCIA_TIPO = 0.5    
PENALIZACION_DUPLICADO = 1.0          
# RIESGO MÁXIMO TEÓRICO UNIVERSAL BASE: 3.5 (Variable según columnas afectadas)

# --- CONFIGURACIÓN DE RIESGOS AVANZADOS ---
PENALIZACION_INCONSISTENCIA_METADATOS = 1.5 # Inconsistencia de metadatos (ej. frecuencia vs. antigüedad)
PENALIZACION_ANOMALIA_SILENCIOSA = 1.0     # Duplicidad semántica/Cambios abruptos (Anomalía + Baja Popularidad)
PENALIZACION_ACTIVO_VACIO = 2.0          # Activos vacíos en categorías populares

RIESGO_MAXIMO_TEORICO_AVANZADO = 10.0

# CLAVE SECRETA DE GEMINI
GEMINI_API_SECRET_VALUE = "AIzaSyDvuJPAAK8AVIS-VQIe39pPgVNb8xlJw3g"

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
    df_copy = df.copy() 
    n_cols = df_copy.shape[1]
    
    # --- 1. COMPLETITUD: Datos por Fila (Densidad) ---
    df_copy['datos_por_fila_score'] = (df_copy.notna().sum(axis=1) / n_cols) * 100
    df_copy['riesgo_datos_incompletos'] = np.where(
        df_copy['datos_por_fila_score'] < 70, PENALIZACION_DATOS_INCOMPLETOS, 0.0
    )

    # --- 2. CONSISTENCIA: Mezcla de Tipos ---
    # Nota: Esta penalización se acumula por columna, pudiendo superar el 0.5 base total
    df_copy['riesgo_consistencia_tipo'] = 0.0
    
    object_cols_for_check = [col for col in df_copy.select_dtypes(include='object').columns if col not in ['titulo', 'descripcion', 'dueño']]
    
    for col in object_cols_for_check:
        inconsistencies = df_copy[col].apply(lambda x: not isinstance(x, str) and pd.notna(x))
        df_copy.loc[inconsistencies, 'riesgo_consistencia_tipo'] += PENALIZACION_INCONSISTENCIA_TIPO
        
    # --- 3. UNICIDAD: Duplicados Exactos ---
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
    
    # Usamos 10.0 como denominador seguro para evitar porcentajes negativos si el riesgo sube mucho
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
    Calcula nuevos scores de riesgo avanzados (inconsistencias, semántica, vacíos) 
    y los añade al score de riesgo existente.
    """
    df_copy = df.copy()
    
    # 1. Detección de Inconsistencia de Metadatos
    df_copy['riesgo_inconsistencia_metadatos'] = np.where(
        (df_copy['prioridad_riesgo_score'] > UMBRAL_RIESGO_ALTO) & (df_copy['antiguedad_datos_dias'] < 365), 
        PENALIZACION_INCONSISTENCIA_METADATOS, 
        0.0
    )

    # 2. Duplicidad Semántica/Cambios Abruptos
    df_copy['riesgo_semantico_actualizacion'] = np.where(
        (df_copy['anomalia_score'] == -1) & (df_copy['popularidad_score'] < 0.1),
        PENALIZACION_ANOMALIA_SILENCIOSA,
        0.0
    )

    # 3. Activos Vacíos en Categorías Populares
    top_categories = df_copy['categoria'].value_counts().nlargest(5).index.tolist()
    
    df_copy['riesgo_activos_vacios'] = np.where(
        (df_copy['completitud_score'] < 20.0) & (df_copy['categoria'].isin(top_categories)),
        PENALIZACION_ACTIVO_VACIO,
        0.0
    )
    
    # Actualizar el score de riesgo principal
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

# Función de Generación de Reporte HTML Profesional
def generate_report_html(df_filtrado, umbral_riesgo):
    """
    Genera el contenido HTML del reporte final que compila insights, tablas y visualizaciones.
    Estilo profesional y limpio.
    """
    
    # 1. Preparación de Datos
    
    # Datos Principales
    total_activos = len(df_filtrado)
    riesgo_promedio_general = df_filtrado['prioridad_riesgo_score'].mean()
    completitud_promedio_general = df_filtrado['completitud_score'].mean()
    
    # Top Activos de Alto Riesgo
    df_top_riesgo = df_filtrado.sort_values(by='prioridad_riesgo_score', ascending=False).head(10).copy()
    df_top_riesgo = df_top_riesgo[['titulo', 'prioridad_riesgo_score', 'completitud_score', 'dueño']].rename(columns={'prioridad_riesgo_score': 'Riesgo Score', 'completitud_score': 'Completitud Score', 'dueño': 'Entidad'}).reset_index(drop=True)
    # Etiqueta de texto limpia
    df_top_riesgo['Nivel Riesgo'] = df_top_riesgo['Riesgo Score'].apply(lambda x: 'Alto' if x > umbral_riesgo else 'Bajo/Medio')
    
    # Riesgo por Entidad
    df_riesgo_entidad = df_filtrado.groupby('dueño').agg(
        Activos_Totales=('uid', 'count'),
        Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
        Completitud_Promedio=('completitud_score', 'mean')
    ).reset_index().sort_values(by='Riesgo_Promedio', ascending=False).head(5)
    
    # Riesgo por Categoría
    df_riesgo_categoria = df_filtrado.groupby('categoria').agg(
        Activos_Totales=('uid', 'count'),
        Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
        Completitud_Promedio=('completitud_score', 'mean')
    ).reset_index().sort_values(by='Riesgo_Promedio', ascending=False).head(5)

    # 2. Lógica del K-Means
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
        cluster_map[centers_df.loc[0, 'index']] = 'Completo/Riesgo Bajo'
        cluster_map[centers_df.loc[1, 'index']] = 'Aceptable/Mejora Necesaria'
        cluster_map[centers_df.loc[2, 'index']] = 'Incompleto/Riesgo Alto'

        df_cluster['Calidad_Cluster'] = df_cluster['cluster'].map(cluster_map)
        
        df_viz2 = df_cluster.merge(df_filtrado[['titulo', 'dueño', 'categoria']], left_index=True, right_index=True)
        
        # Filtro de activos prioritarios
        df_activos_prioritarios = df_viz2[df_viz2['Calidad_Cluster'] == 'Incompleto/Riesgo Alto'].sort_values(by='prioridad_riesgo_score', ascending=False).head(10)[['titulo', 'dueño', 'prioridad_riesgo_score', 'completitud_score']].rename(columns={'prioridad_riesgo_score': 'Riesgo Score', 'completitud_score': 'Completitud Score', 'dueño': 'Entidad'})
        
        color_map = {
            'Completo/Riesgo Bajo': 'green',
            'Aceptable/Mejora Necesaria': 'orange',
            'Incompleto/Riesgo Alto': 'red'
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
                'prioridad_riesgo_score': 'Riesgo Promedio (Peor ->)', 
                'completitud_score': 'Completitud Score (Mejor ^)'
            }
        )
        cluster_html = fig2.to_html(full_html=False, include_plotlyjs='cdn')
        
    # 3. Generar Treemap
    treemap_html = "No se pudo generar el Treemap (datos insuficientes)."
    
    COLUMNA_TREEMAP = 'categoria'
    if 'common_core_theme' in df_filtrado.columns:
        if 'filtro_tema' in st.session_state and st.session_state.filtro_tema != "Mostrar Todos":
            COLUMNA_TREEMAP = 'common_core_theme'

    if COLUMNA_TREEMAP in df_filtrado.columns and len(df_filtrado) > 0 and not df_filtrado[COLUMNA_TREEMAP].isnull().all():
        df_treemap = df_filtrado.groupby(COLUMNA_TREEMAP).agg(
            Num_Activos=('uid', 'count'),
            Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
        ).reset_index()
        fig_treemap = px.treemap(
            df_treemap,
            path=[COLUMNA_TREEMAP], 
            values='Num_Activos',
            color='Riesgo_Promedio', 
            color_continuous_scale=px.colors.sequential.Reds, 
            title=f'Matriz Treemap: Cobertura por {COLUMNA_TREEMAP.capitalize()} vs. Riesgo Promedio'
        )
        treemap_html = fig_treemap.to_html(full_html=False, include_plotlyjs='cdn')
        

    # 4. Construcción del HTML Profesional
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporte de Análisis de Inventario</title>
        <meta charset="utf-8">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; margin: 40px; color: #333; line-height: 1.6; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; margin-bottom: 30px; }}
            h2 {{ color: #34495e; margin-top: 40px; margin-bottom: 20px; border-left: 5px solid #3498db; padding-left: 10px; }}
            h3 {{ color: #7f8c8d; margin-top: 25px; }}
            
            .metrics-container {{ display: flex; justify-content: space-between; margin-bottom: 30px; }}
            .metric {{ background-color: #f8f9fa; border: 1px solid #e9ecef; padding: 20px; border-radius: 8px; width: 30%; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
            .metric h3 {{ margin: 0 0 10px 0; font-size: 1.1em; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; }}
            .metric p {{ font-size: 2em; font-weight: bold; margin: 0; color: #2c3e50; }}
            
            .high-risk {{ color: #e74c3c !important; }}
            .low-risk {{ color: #27ae60 !important; }}
            
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 0.9em; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f8f9fa; color: #333; font-weight: bold; text-transform: uppercase; }}
            tr:hover {{ background-color: #f1f1f1; }}
            
            .recommendation {{ background-color: #fff3cd; border: 1px solid #ffeeba; border-left: 5px solid #ffc107; padding: 20px; margin-top: 25px; border-radius: 4px; }}
            .footer {{ margin-top: 50px; font-size: 0.8em; color: #999; text-align: center; border-top: 1px solid #eee; padding-top: 20px; }}
        </style>
    </head>
    <body>

    <h1>Reporte Final de Análisis de Inventario de Datos</h1>
    <p><strong>Fecha de Generación:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p><strong>Umbral de Riesgo Alto:</strong> > {umbral_riesgo:.1f}</p>

    <h2>Hallazgos Clave</h2>
    <div class="metrics-container">
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
    </div>

    <p>El activo con <strong>Mayor Riesgo</strong> es: {df_top_riesgo.iloc[0]['titulo']} (Riesgo: {df_top_riesgo.iloc[0]['Riesgo Score']:.2f}, Entidad: {df_top_riesgo.iloc[0]['Entidad']}).</p>
    <p>La entidad con <strong>Mayor Riesgo Promedio</strong> es: {df_riesgo_entidad.iloc[0]['dueño']} (Riesgo Promedio: {df_riesgo_entidad.iloc[0]['Riesgo_Promedio']:.2f}).</p>
    
    <h2>Análisis de Riesgos</h2>
    <h3>Top 10 Activos con Mayor Riesgo</h3>
    <p>Activos individuales con la mayor puntuación de riesgo, indicando fallas en calidad universal y avanzada.</p>
    {df_top_riesgo[['titulo', 'Entidad', 'Riesgo Score', 'Nivel Riesgo']].to_html(index=False)}
    
    <h3>Top 5 Entidades con Mayor Riesgo Promedio</h3>
    {df_riesgo_entidad.to_html(index=False, float_format=lambda x: f'{x:.2f}')}

    <h2>Activos Prioritarios</h2>
    <p>Lista de activos clasificados en el cluster <strong>"Incompleto/Riesgo Alto"</strong> mediante K-Means Clustering. Estos requieren atención inmediata.</p>
    {df_activos_prioritarios.to_html(index=False, float_format=lambda x: f'{x:.2f}') if not df_activos_prioritarios.empty else "<p>No se identificaron activos en el cluster de Riesgo Alto con los filtros actuales.</p>"}

    <h3>Visualización de Priorización (K-Means)</h3>
    <p>Distribución de Activos por Riesgo vs. Completitud.</p>
    {cluster_html}

    <h2>Recomendaciones por Sector</h2>
    <p>Análisis de las categorías (sectores) con mayor Riesgo Promedio, indicando áreas temáticas críticas.</p>
    
    {df_riesgo_categoria.to_html(index=False, float_format=lambda x: f'{x:.2f}')}

    <div class="recommendation">
        <h3>Recomendación General:</h3>
        <p>Priorizar la revisión de metadatos (completitud) y consistencia de tipos de datos en la Categoría <strong>'{df_riesgo_categoria.iloc[0]['categoria']}'</strong>, ya que presenta el mayor Riesgo Promedio ({df_riesgo_categoria.iloc[0]['Riesgo_Promedio']:.2f}).</p>
        <p>Asegurarse de que los activos más antiguos y menos usados en esta categoría no estén generando ruido o inconsistencias silenciosas.</p>
    </div>

    <h3>Visualización de Cobertura y Riesgo (Treemap)</h3>
    <p>El tamaño del bloque indica el número de activos y el color (intensidad) indica el Riesgo Promedio.</p>
    {treemap_html}
    
    <div class="footer">
        Generado por Sistema de Análisis de Inventario de Datos
    </div>
    </body>
    </html>
    """
    return html_content

def get_table_download_link(html_content, filename, text):
    """Genera el link de descarga para el contenido HTML/PDF"""
    b64 = base64.b64encode(html_content.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 5px; font-family: Arial, sans-serif;">{text}</a>'
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
            return None
    except Exception as e:
        return None

# =================================================================
# 2. FUNCIÓN ROBUSTA DEL AGENTE DE IA (USANDO RAG)
# =================================================================

def generate_ai_response(user_query, knowledge_base_content, model_placeholder):
    """
    Función robusta que interactúa con la API de Gemini utilizando la Base de Conocimiento (RAG).
    """
    
    if knowledge_base_content is None:
        error_msg = "No puedo responder. La base de conocimiento no ha sido cargada."
        st.session_state.messages.append({"role": "user", "content": user_query})
        with model_placeholder.chat_message("assistant"):
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        return

    try:
        client = genai.Client(api_key=GEMINI_API_SECRET_VALUE)
    except Exception as e:
        error_msg = f"Error al inicializar el Cliente Gemini. Verifica tu clave API. Detalle: {e}"
        st.session_state.messages.append({"role": "user", "content": user_query})
        with model_placeholder.chat_message("assistant"):
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        return

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

    with model_placeholder.chat_message("assistant"):
        with st.spinner("Analizando la Base de Conocimiento para generar un diagnóstico experto..."):
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
                error_msg = f"Error en la API de Gemini: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# =================================================================
# 4. FUNCIONES DE CÁLCULO DE CALIDAD NORMATIVA MINTIC (ACTUALIZADO)
# =================================================================

MINTIC_API_URL = "https://www.datos.gov.co/resource/uzcf-b9dh.json?$limit=10000"

@st.cache_data
def fetch_mintic_data():
    """Realiza la llamada a la API y devuelve un DataFrame, intentando la conversión de tipos."""
    try:
        response = requests.get(MINTIC_API_URL, timeout=30)
        response.raise_for_status() # Lanza un error para códigos de estado HTTP malos
        data = response.json()
        if not data:
            return pd.DataFrame(), "No se recuperaron datos de la API."
        
        df = pd.DataFrame(data)
        
        # Eliminar columnas de metadatos de Socrata para la lógica del cálculo (ej. ':sid')
        cols_to_drop = [col for col in df.columns if col.startswith(':')]
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        
        # Intentar forzar tipos numéricos para cálculos de varianza
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
                
        return df, f"Datos cargados. {len(df)} filas, {len(df.columns)} columnas."
    except requests.exceptions.RequestException as e:
        return pd.DataFrame(), f"Error al hacer fetch de la API: {e}"
    except Exception as e:
        return pd.DataFrame(), f"Error desconocido al cargar o procesar los datos: {e}"

def calculate_mintic_quality_score(df):
    """
    Implementa las 17 fórmulas de calidad normativa Mintic, utilizando valores derivados 
    del DataFrame donde es posible y proxies razonables donde se requiere metadato externo.
    """
    
    if df.empty:
        return pd.Series(dtype=float)
        
    dfColumnas = len(df.columns)
    totalCeldas = df.size
    totalNulos = df.isnull().sum().sum()
    numFilas = len(df)
    
    if dfColumnas == 0 or totalCeldas == 0:
        # Retorno ajustado para DataFrame vacío
        return pd.Series(0.0, index=['Confidencialidad', 'Relevancia', 'Actualidad', 'Completitud', 'Comprensibilidad', 'Conformidad', 'Consistencia', 'Credibilidad', 'Disponibilidad', 'Eficiencia', 'Exactitud', 'Portabilidad', 'Precisión', 'Recuperabilidad', 'Accesibilidad', 'Trazabilidad', 'Unicidad', 'Score_Calidad_Normativa_Mintic'])

    # =================================================================
    # PASO 1: CÁLCULO DE SUB-MÉTRICAS BASADAS EN EL DATAFRAME
    # =================================================================
    
    # 1. Confidencialidad (numColConfidencial, riesgo_total)
    cols_risk_keywords = ['nombre', 'identificacion', 'cedula', 'email', 'direccion', 'telefono', 'rut']
    sensitive_cols = [col for col in df.columns if any(k in col.lower() for k in cols_risk_keywords)]
    numColConfidencial = len(sensitive_cols)
    riesgo_total = 0.0
    for col in sensitive_cols:
        if any(k in col.lower() for k in ['identificacion', 'cedula', 'rut']):
            riesgo_total += 3.0 # Riesgo Alto
        elif any(k in col.lower() for k in ['nombre', 'email', 'direccion', 'telefono']):
            riesgo_total += 2.0 # Riesgo Medio
        else:
            riesgo_total += 1.0 # Riesgo Bajo
            
    # 2. Completitud (numColPorcNulos, medidaColNoVacias)
    col_null_percentages = df.isnull().mean()
    numColPorcNulos = (col_null_percentages > 0.7).sum() # Columnas con más del 70% nulas
    numColNoVaciasAbs = (col_null_percentages < 0.01).sum() # Columnas con menos del 1% nulas
    medidaColNoVacias = 10.0 * (numColNoVaciasAbs / dfColumnas) 

    # 3. Consistencia (columnas_cumplen_criterios)
    columnas_cumplen_criterios = 0
    for col in df.columns:
        unique_vals = df[col].nunique()
        has_unique_criteria = unique_vals >= 2

        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        col_variance = df[col].var() if is_numeric and df[col].count() > 1 else 0.0
        has_variance_criteria = col_variance >= 0.1

        if has_unique_criteria and has_variance_criteria:
            columnas_cumplen_criterios += 1
            
    # 4. Unicidad (unicidad_score)
    numFilasDuplicadas = df.duplicated().sum()
    propFilasDuplicadas = numFilasDuplicadas / numFilas if numFilas > 0 else 0.0
    unicidad_score = 10.0 * (1 - propFilasDuplicadas)

    # 5. Relevancia (medidaFilas)
    medidaFilas = 0.0
    if numFilas >= 50:
        score_filas_min = 10.0 # Puntaje base por tamaño adecuado
        prop_nulos_data = totalNulos / totalCeldas
        score_nulos_inv = 10.0 * (1 - prop_nulos_data) # Inverso de la proporción de nulos
        medidaFilas = (score_filas_min * 0.5) + (score_nulos_inv * 0.5)
    medidaFilas = min(medidaFilas, 10.0)
    
    # 6. Comprensibilidad (Proxy: Longitud de nombres de columna)
    avg_col_len = np.mean([len(col) for col in df.columns])
    # Penaliza la longitud: 10 - (longitud/factor). Factor 4.0 penaliza nombres de 40+ caracteres.
    comprensibilidad = max(0.0, 10.0 - (avg_col_len / 4.0)) 
    comprensibilidad = min(comprensibilidad, 9.5) # Cap at 9.5

    # =================================================================
    # PASO 2: MÉTODOS ARBITRARIOS/PROXIES (ASUMIMOS CALIDAD BASE DE LA API)
    # =================================================================

    # Métricas que requieren metadatos o ML avanzado (usamos valores altos razonables)
    medidaCategoria = 8.5            # Relevancia: Asumimos buena categorización
    medidaMetadatosCompletos = 8.5   # Credibilidad: Asumimos 85% de metadatos obligatorios
    medidaPublicadorValido = 10.0    # Credibilidad: Asumimos datos.gov.co es válido
    medidaColDescValida = 7.0        # Credibilidad: Asumimos 70% de columnas con descripción

    metadatosAuditados = 8.0         # Recuperabilidad: Score base para auditoría

    accesibilidad_score = 10.0       # Accesibilidad: API en JSON/formato abierto = 10
    actualidad_score = 9.0           # Actualidad: Asumimos que la API está bien mantenida

    conformidad_score = 9.0          # Conformidad: Asumimos buena adherencia a estándares
    
    # Exactitud Sintáctica/Semántica (difícil de calcular sin ML)
    numColValoresUnicosSimilares = int(dfColumnas * 0.05) # Asumimos 5% de columnas tienen valores similares
    numColNoSimSemantica = 0         # Simplificación: 0 fallos semánticos

    trazabilidad = 9.5               # Trazabilidad: Asumimos buen historial/versionamiento en la API

    # =================================================================
    # PASO 3: APLICACIÓN DE LAS 17 FÓRMULAS
    # =================================================================
    
    # 1. Confidencialidad (MÍN: 0, MÁX: 10)
    if numColConfidencial == 0:
        confidencialidad = 10.0
    else:
        # Fórmula: 10 - (riesgo_total / dfColumnas * numColConfidencial * 3)
        penalizacion = (riesgo_total / dfColumnas) * numColConfidencial * 3
        confidencialidad = max(0.0, 10.0 - penalizacion)

    # 2. Relevancia (MÍN: 0, MÁX: 10)
    relevancia = (medidaCategoria + medidaFilas) / 2.0

    # 3. Actualidad (MÍN: 0, MÁX: 10)
    actualidad = actualidad_score 

    # 4. Completitud (MÍN: 0, MÁX: 10)
    
    # a) Completitud de datos
    prop_nulos = totalNulos / totalCeldas
    medidaCompletitudDatos = 10.0 * (1 - (prop_nulos)**1.5)
        
    # b) Completitud por columnas
    prop_col_nulos = numColPorcNulos / dfColumnas
    medidaCompletitudCol = 10.0 * (1 - (prop_col_nulos)**2)
    
    completitud = (medidaCompletitudDatos + medidaCompletitudCol + medidaColNoVacias) / 3.0

    # 5. Comprensibilidad (MÍN: 0, MÁX: 10)
    comprensibilidad_score = comprensibilidad

    # 6. Conformidad (MÍN: 0, MÁX: 10)
    conformidad = conformidad_score 

    # 7. Consistencia (MÍN: 0, MÁX: 10)
    consistencia = 10.0 * (columnas_cumplen_criterios / dfColumnas)

    # 8. Credibilidad (MÍN: 0, MÁX: 10)
    credibilidad = (
        0.70 * medidaMetadatosCompletos +
        0.05 * medidaPublicadorValido +
        0.25 * medidaColDescValida
    )

    # 9. Disponibilidad (MÍN: 0, MÁX: 10)
    disponibilidad = (accesibilidad_score + actualidad) / 2.0

    # 10. Eficiencia (MÍN: 0, MÁX: 10)
    # Derivación: Combina completitud, unicidad y la penalización por columnas nulas
    eficiencia = (completitud + unicidad_score + (10.0 - (numColPorcNulos / dfColumnas * 10.0))) / 3.0
    eficiencia = np.clip(eficiencia, 0.0, 10.0)

    # 11. Exactitud (MÍN: 0, MÁX: 10)
    
    # a) Exactitud sintáctica
    exactitudSintactica = 10.0 * (1 - (numColValoresUnicosSimilares / dfColumnas)**2)
        
    # b) Exactitud semántica
    exactitudSemantica = 10.0 - (10.0 * (1 - (numColNoSimSemantica / dfColumnas)**2))
        
    exactitud = (exactitudSintactica + exactitudSemantica) / 2.0

    # 12. Portabilidad (MÍN: 0, MÁX: 10)
    portabilidad_base = accesibilidad_score # Proxy de portabilidad: Si es accesible y en formato abierto
    portabilidad = (
        0.50 * portabilidad_base +
        0.25 * conformidad +
        0.25 * completitud
    )

    # 13. Precisión (MÍN: 0, MÁX: 10)
    # Derivación: Se asume que es similar a Consistencia
    precision = consistencia 

    # 14. Recuperabilidad (MÍN: 0, MÁX: 10)
    recuperabilidad = (
        accesibilidad_score +
        medidaMetadatosCompletos +
        metadatosAuditados
    ) / 3.0

    # 15. Accesibilidad (MÍN: 0, MÁX: 10)
    accesibilidad = accesibilidad_score 

    # 16. Trazabilidad (MÍN: 0, MÁX: 10)
    trazabilidad_score = trazabilidad 

    # 17. Unicidad (MÍN: 0, MÁX: 10)
    unicidad = unicidad_score 
    
    # Consolidar Resultados
    results = pd.Series({
        'Confidencialidad': confidencialidad,
        'Relevancia': relevancia,
        'Actualidad': actualidad,
        'Completitud': completitud,
        'Comprensibilidad': comprensibilidad_score,
        'Conformidad': conformidad,
        'Consistencia': consistencia,
        'Credibilidad': credibilidad,
        'Disponibilidad': disponibilidad,
        'Eficiencia': eficiencia,
        'Exactitud': exactitud,
        'Portabilidad': portabilidad,
        'Precisión': precision,
        'Recuperabilidad': recuperabilidad,
        'Accesibilidad': accesibilidad,
        'Trazabilidad': trazabilidad_score,
        'Unicidad': unicidad
    })
    
    # Calcular Score de Calidad Normativa (Promedio Simple de los 17)
    score_calidad_normativa = results.mean()
    results['Score_Calidad_Normativa_Mintic'] = score_calidad_normativa
    
    return results

def display_mintic_analysis(df):
    """Muestra los resultados del análisis normativo Mintic en Streamlit."""
    
    results = calculate_mintic_quality_score(df)
    
    if 'Score_Calidad_Normativa_Mintic' not in results:
        st.error("No se pueden calcular las métricas: el DataFrame está vacío o la lógica falló.")
        return

    score = results['Score_Calidad_Normativa_Mintic']
    
    st.markdown(f"""
        <div style='border: 3px solid #00A651; padding: 20px; border-radius: 8px; background-color: #e6fff0; text-align: center;'>
            <h2 style='color: #00A651; margin-top: 0;'>Puntuación de Calidad Normativa Mintic</h2>
            <p style='font-size: 4em; font-weight: bold; margin: 0;'>{score:.2f} / 10.00</p>
            <p style='font-size: 1.2em; color: #333;'>Promedio de los 17 Criterios</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Desglose de los 17 Criterios de Calidad")
    
    # Eliminar el score final de la tabla de desglose
    results_display = results.drop('Score_Calidad_Normativa_Mintic', errors='ignore').reset_index()
    results_display.columns = ['Criterio', 'Puntuación (0-10)']
    results_display['Puntuación (0-10)'] = results_display['Puntuación (0-10)'].round(2)
    
    # Aplicar formato de color para visualizar el rendimiento
    def color_score(val):
        if val >= 9.0:
            color = '#a9dfbf' # Verde
        elif val >= 7.0:
            color = '#fdfd96' # Amarillo
        else:
            color = '#f79999' # Rojo
        return f'background-color: {color}'

    styled_df = results_display.style.applymap(
        color_score, 
        subset=['Puntuación (0-10)']
    ).hide(axis="index")

    st.dataframe(styled_df, use_container_width=True)
    
    st.info("""
        **NOTA DE IMPLEMENTACIÓN:** Los cálculos reflejan las **fórmulas exactas** proporcionadas. 
        Para sub-métricas que requieren análisis de metadatos externos, ML (ej. `medidaCategoria`, `medidaMetadatosCompletos`), se han utilizado **proxies razonables** o valores base altos, asumiendo una calidad decente de la fuente `datos.gov.co`.
    """)

# =================================================================
# 3. Ejecución Principal del Dashboard
# =================================================================

st.set_page_config(page_title="Asistente de Análisis de Inventario", layout="wide") 

st.title("Dashboard de Priorización de Activos de Datos (Análisis Completo)")

try:
    with st.spinner(f'Cargando archivo procesado: {ARCHIVO_PROCESADO}...'):
        df_analisis_completo = load_processed_data(ARCHIVO_PROCESADO) 

    if df_analisis_completo.empty:
        st.error(f"Error: No se pudo cargar el archivo {ARCHIVO_PROCESADO}. Asegúrate de que existe y se ejecutó preprocess.py.")
    else:
        # ADICIÓN: APLICAR DETECCIÓN DE ANOMALÍAS CON ML
        df_analisis_completo = apply_anomaly_detection(df_analisis_completo)
        
        # APLICAR CHEQUEOS DE RIESGO AVANZADOS
        df_analisis_completo = apply_advanced_risk_checks(df_analisis_completo) 
        
        st.success(f'Archivo pre-procesado cargado. Total de activos: {len(df_analisis_completo)}')

        # --- Carga de la Base de Conocimiento ---
        if "knowledge_content" not in st.session_state:
            st.session_state.knowledge_content = load_knowledge_base(KNOWLEDGE_FILE)

        knowledge_base_content = st.session_state.knowledge_content
        
        # --- Inicialización de variables de estado ---
        if "messages" not in st.session_state:
            st.session_state.messages = []

        
        # ----------------------------------------------------------------------
        # --- FILTROS EN EL SIDEBAR ---
        # ----------------------------------------------------------------------
        with st.sidebar:
            st.header("Filtros para Visualizaciones")
            
            filtro_acceso_publico = False 
            
            if 'publico' in df_analisis_completo.columns:
                filtro_acceso_publico = st.checkbox(
                    "Mostrar Solo Activos públicos",
                    value=False,
                    help="Si está marcado, solo se mostrarán los activos cuyo nivel de acceso sea 'public' (columna 'publico')."
                )
            
            filtro_categoria = "Mostrar Todos"
            if 'categoria' in df_analisis_completo.columns:
                categories = df_analisis_completo['categoria'].dropna().unique().tolist()
                categories.sort()
                categories.insert(0, "Mostrar Todos")
                filtro_categoria = st.selectbox("Filtrar por Categoría:", categories)
                
            filtro_tema = "Mostrar Todos" 
            if 'common_core_theme' in df_analisis_completo.columns:
                themes = df_analisis_completo['common_core_theme'].dropna().unique().tolist()
                themes.sort()
                themes.insert(0, "Mostrar Todos")
                filtro_tema = st.selectbox("Tema:", themes)
                
            st.session_state.filtro_tema = filtro_tema
            
            st.markdown("---")
            st.subheader("Generar Reporte Final")
            
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
                st.subheader(f"Estadísticas Clave para: {filtro_dueño}")
                total_activos = len(df_entidad_seleccionada)
                # Mantenemos la lógica de búsqueda intacta para que coincida con los datos, pero el texto visible será limpio
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
                st.warning(f"No se encontraron activos para la entidad: {filtro_dueño}")
        
        st.markdown("---")

        # --- APLICAR FILTROS ---
        df_filtrado = df_analisis_completo.copy()
        
        if filtro_dueño != "Mostrar Análisis General":
            df_filtrado = df_filtrado[df_filtrado['dueño'] == filtro_dueño]
            
        if filtro_acceso_publico:
            df_filtrado = df_filtrado[df_filtrado['publico'] == 'public']
            
        if filtro_categoria != "Mostrar Todos":
            df_filtrado = df_filtrado[df_filtrado['categoria'] == filtro_categoria]
            
        if 'common_core_theme' in df_analisis_completo.columns and filtro_tema != "Mostrar Todos":
            df_filtrado = df_filtrado[df_filtrado['common_core_theme'] == filtro_tema]

        st.header("Visualizaciones y Rankings")
        info_acceso = "solo Activos Públicos" if filtro_acceso_publico else "Todos los Niveles de Acceso"
        info_tema = filtro_tema if 'filtro_tema' in locals() and filtro_tema != "Mostrar Todos" else "Todos los Temas"
        st.info(f"Vista actual de gráficos: **{len(df_filtrado)} activos** (Filtro de Entidad: {filtro_dueño}; Acceso: {info_acceso}; Categoría: {filtro_categoria}; Tema: {info_tema})")

        if df_filtrado.empty:
            st.warning("No hay datos para mostrar en los gráficos con los filtros seleccionados.")
        else:
            # --- 3. Métricas de la Vista Actual ---
            st.subheader("Métricas de la Vista Actual")
            col_metrica1, col_metrica2, col_metrica3 = st.columns(3)
            col_metrica1.metric("Completitud Promedio", f"{df_filtrado['completitud_score'].mean():.2f}%")
            col_metrica2.metric("Activos en Incumplimiento", f"{(df_filtrado['estado_actualizacion'] == '🔴 INCUMPLIMIENTO').sum()} / {len(df_filtrado)}")
            col_metrica3.metric("Anomalías Detectadas (ML)", f"{(df_filtrado['anomalia_score'] == -1).sum()}")
            st.markdown("---")

            # --- 4. Tabla de Búsqueda y Diagnóstico ---
            st.header("Tabla de Búsqueda y Diagnóstico")
            show_asset_detail = filtro_acceso_publico or (filtro_dueño != "Mostrar Análisis General") or (filtro_tema != "Mostrar Todos")
            
            if show_asset_detail:
                # Caso: Activos Públicos O Entidad Específica O Tema Específico (Mostrar detalle por ACTIVO)
                if filtro_dueño != "Mostrar Análisis General":
                    st.subheader(f"Detalle de Activos Individuales para la Entidad: {filtro_dueño}")
                    info_text = f"""
                        **Vista Detallada:** Se muestran los **{len(df_filtrado)} activos individuales** de la entidad **{filtro_dueño}**, ordenados por su Score de Riesgo (más alto primero).
                        * **Color Rojo:** Riesgo > {UMBRAL_RIESGO_ALTO:.1f} (Prioridad Máxima)
                        **NOTA:** Este riesgo ahora incluye penalizaciones avanzadas por **Inconsistencia de Metadatos**, **Duplicidad Semántica/Cambios Abruptos** y **Activos Vacíos**. El riesgo máximo teórico ajustado es **{RIESGO_MAXIMO_TEORICO_AVANZADO:.1f}** (para permitir múltiples inconsistencias por columna).
                    """
                elif filtro_tema != "Mostrar Todos":
                    st.subheader(f"Detalle por Activo Individual para el Tema: {filtro_tema}")
                    info_text = f"""
                        **Vista Detallada:** Se muestran los **{len(df_filtrado)} activos individuales** del tema **{filtro_tema}**, ordenados por su Score de Riesgo (más alto primero).
                        * **Color Rojo:** Riesgo > {UMBRAL_RIESGO_ALTO:.1f} (Prioridad Máxima)
                        **NOTA:** Este riesgo ahora incluye penalizaciones avanzadas por **Inconsistencia de Metadatos**, **Duplicidad Semántica/Cambios Abruptos** y **Activos Vacíos**. El riesgo máximo teórico ajustado es **{RIESGO_MAXIMO_TEORICO_AVANZADO:.1f}** (para permitir múltiples inconsistencias por columna).
                    """
                else:
                    st.subheader("Detalle por Activo Público (Priorización Individual)")
                    info_text = f"""
                        **Vista Detallada:** Se muestran los **activos individuales públicos** filtrados, ordenados por su Score de Riesgo (más alto primero).
                        * **Color Rojo:** Riesgo > {UMBRAL_RIESGO_ALTO:.1f} (Prioridad Máxima)
                        **NOTA:** Este riesgo ahora incluye penalizaciones avanzadas por **Inconsistencia de Metadatos**, **Duplicidad Semántica/Cambios Abruptos** y **Activos Vacíos**. El riesgo máximo teórico ajustado es **{RIESGO_MAXIMO_TEORICO_AVANZADO:.1f}** (para permitir múltiples inconsistencias por columna).
                    """
                    
                st.markdown(info_text)

                df_display = df_filtrado[[
                    'titulo', 
                    'dueño', 
                    'prioridad_riesgo_score', 
                    'completitud_score', 
                    'antiguedad_datos_dias'
                ]].sort_values(by='prioridad_riesgo_score', ascending=False)

                def color_risk_row(row):
                    if row['prioridad_riesgo_score'] > UMBRAL_RIESGO_ALTO:
                        return ['background-color: #ffcccc'] * 5 
                    return [''] * 5

                st.dataframe(
                    df_display.style.apply(color_risk_row, axis=1).format(
                        {'prioridad_riesgo_score': '{:.2f}', 'completitud_score': '{:.2f}%', 'antiguedad_datos_dias': '{:.0f} días'}
                    ).set_table_attributes('style="font-size: 14px"'),
                    use_container_width=True,
                    hide_index=True
                )
                
            else:
                # Caso: Análisis General (Mostrar detalle por Entidad)
                st.subheader("Detalle de Riesgo y Completitud por Entidad (Vista General)")
                st.info(f"""
                    **Vista General:** Se agrupan los activos por Entidad. La Entidad con mayor **Riesgo Promedio** es la más prioritaria.
                    * **Color Rojo:** Riesgo Promedio > {UMBRAL_RIESGO_ALTO * 0.7:.1f} (Umbral suave para promedio).
                """)
                df_display_grouped = df_filtrado.groupby('dueño').agg(
                    Activos_Totales=('uid', 'count'),
                    Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
                    Completitud_Promedio=('completitud_score', 'mean'),
                    Antigüedad_Promedio=('antiguedad_datos_dias', 'mean')
                ).reset_index().sort_values(by='Riesgo_Promedio', ascending=False)
                
                def color_risk_group(val):
                    if val > UMBRAL_RIESGO_ALTO * 0.7:
                        color = '#f79999' # Rojo suave
                    else:
                        color = ''
                    return f'background-color: {color}'

                st.dataframe(
                    df_display_grouped.style.applymap(color_risk_group, subset=['Riesgo_Promedio']).format(
                        {'Riesgo_Promedio': '{:.2f}', 'Completitud_Promedio': '{:.2f}%', 'Antigüedad_Promedio': '{:.0f} días'}
                    ).set_table_attributes('style="font-size: 14px"'),
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown("---")


            # ----------------------------------------------------------------------
            # --- BLOQUE CLAVE DE PESTAÑAS (GRÁFICOS) ---
            # ----------------------------------------------------------------------
            
            # Nueva lógica de pestañas: siempre incluye el Análisis Mintic como la primera pestaña
            
            mintic_tab, *rest_tabs = st.tabs([
                "Atributos según Calidad Normativa Mintic",
                "1. Ranking de Priorización (Riesgo/Incompletitud)", 
                "2. K-Means Clustering (Priorización)", 
                "3. Cobertura Temática", 
                "4. Treemap de Cobertura y Calidad"
            ])
            
            # Mapeo de pestañas restantes
            tab1, tab2, tab3, tab4 = rest_tabs[0], rest_tabs[1], rest_tabs[2], rest_tabs[3]

            # --- NUEVA PESTAÑA: ANÁLISIS NORMATIVO MINTIC ---
            with mintic_tab:
                st.header("Análisis de Atributos de Calidad Normativa Mintic (Datos.gov.co)")
                
                st.markdown("""
                    Esta sección calcula una Puntuación de Calidad basándose en el análisis de una **muestra de datos** obtenida directamente de la API abierta de datos.gov.co: 
                    `https://www.datos.gov.co/resource/uzcf-b9dh.json`.
                    
                    El cálculo utiliza las **17 fórmulas de criterios** definidas en la Guía de Calidad e Interoperabilidad 2025.
                """)
                
                with st.spinner("Conectando y recuperando datos de la API de datos.gov.co..."):
                    df_mintic, message = fetch_mintic_data()
                    
                if not df_mintic.empty:
                    st.success(message)
                    st.markdown("---")
                    display_mintic_analysis(df_mintic)
                else:
                    st.error(f"Fallo al cargar la muestra de datos de la API: {message}")
                    st.warning("Asegúrate de tener conexión a Internet y que la API esté disponible.")


            # --- PESTAÑAS EXISTENTES ---
            
            with tab1:
                # 1. Ranking de Priorización (Riesgo/Incompletitud)
                if filtro_acceso_publico:
                    st.subheader("1. Ranking Top 10 Activos Públicos (Incompletos y Riesgo Alto)")
                    df_viz1_public = df_filtrado[df_filtrado['publico'] == 'public'].sort_values(
                        by='prioridad_riesgo_score', 
                        ascending=False
                    ).head(10).copy()
                    
                    df_viz1_public['prioridad_riesgo_score'] = df_viz1_public['prioridad_riesgo_score'].astype(float)
                    df_viz1_public['completitud_score'] = df_viz1_public['completitud_score'].astype(float)
                    
                    fig1 = px.bar(
                        df_viz1_public.melt(
                            id_vars=['titulo'], 
                            value_vars=['prioridad_riesgo_score', 'completitud_score'],
                            var_name='Métrica', 
                            value_name='Valor'
                        ), 
                        x='titulo', 
                        y='Valor', 
                        color='Métrica',
                        barmode='group',
                        height=500,
                        title="Riesgo (Barra Oscura) vs. Completitud (Barra Clara)",
                        labels={'titulo': 'Activo de Datos', 'Valor': 'Puntuación'},
                    )
                    fig1.update_xaxes(tickangle=45)
                    st.plotly_chart(fig1, use_container_width=True)
                    
                else:
                    st.subheader("1. Ranking de Entidades por Completitud Promedio (Peor Rendimiento)")
                    df_viz1 = df_filtrado.groupby('dueño').agg(
                        Completitud_Promedio=('completitud_score', 'mean'),
                        Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
                        Num_Activos=('uid', 'count')
                    ).reset_index().sort_values(by='Completitud_Promedio', ascending=True).head(10)
                    
                    fig1 = px.bar(
                        df_viz1.melt(
                            id_vars=['dueño'], 
                            value_vars=['Completitud_Promedio', 'Riesgo_Promedio'], 
                            var_name='Métrica', 
                            value_name='Valor'
                        ), 
                        x='dueño', 
                        y='Valor', 
                        color='Métrica',
                        barmode='group',
                        height=500,
                        title="Completitud Promedio (Peor ->) vs. Riesgo Promedio (Mejor ->)",
                        labels={'dueño': 'Entidad', 'Valor': 'Puntuación'},
                    )
                    fig1.update_xaxes(tickangle=45)
                    st.plotly_chart(fig1, use_container_width=True)


            with tab2:
                # 2. K-Means Clustering (Priorización)
                st.subheader("2. Análisis de Priorización mediante K-Means Clustering")
                
                if len(df_filtrado) < 3:
                    st.warning("Se requieren al menos 3 activos para ejecutar el algoritmo K-Means.")
                else:
                    features = ['prioridad_riesgo_score', 'completitud_score']
                    df_cluster = df_filtrado[features].dropna().copy()
                    
                    if df_cluster.empty:
                        st.warning("No hay suficientes datos completos (Riesgo y Completitud) para el clustering.")
                    else:
                        try:
                            scaler = StandardScaler()
                            data_scaled = scaler.fit_transform(df_cluster)
                            
                            # Usamos 3 clusters: Riesgo Bajo/Alto, Completitud Alta/Baja
                            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                            df_cluster['cluster'] = kmeans.fit_predict(data_scaled)
                            
                            # Interpretación de Clusters (Basado en la media de los centros)
                            centers_scaled = kmeans.cluster_centers_
                            centers = scaler.inverse_transform(centers_scaled)
                            centers_df = pd.DataFrame(centers, columns=features)
                            
                            # Un score simple para ordenar: Completitud - Riesgo
                            centers_df['sort_score'] = centers_df['completitud_score'] - centers_df['prioridad_riesgo_score']
                            centers_df = centers_df.sort_values(by='sort_score', ascending=False).reset_index()
                            
                            cluster_map = {}
                            # Cluster con mejor score (Alta Completitud, Bajo Riesgo)
                            cluster_map[centers_df.loc[0, 'index']] = 'Completo/Riesgo Bajo'
                            # Cluster Intermedio
                            cluster_map[centers_df.loc[1, 'index']] = 'Aceptable/Mejora Necesaria'
                            # Cluster con peor score (Baja Completitud, Alto Riesgo) - PRIORIDAD
                            cluster_map[centers_df.loc[2, 'index']] = 'Incompleto/Riesgo Alto'
                            
                            df_cluster['Calidad_Cluster'] = df_cluster['cluster'].map(cluster_map)
                            
                            # Unir de nuevo los títulos para el hover
                            df_viz2 = df_cluster.merge(
                                df_filtrado[['titulo', 'dueño', 'categoria']], 
                                left_index=True, 
                                right_index=True
                            )
                            
                            st.markdown("""
                                Este gráfico utiliza el algoritmo **K-Means** para agrupar automáticamente los activos en 3 categorías 
                                (clusters) basado en su **Score de Riesgo** y su **Completitud**. 
                                
                                * **Rojo (Incompleto/Riesgo Alto):** Máxima prioridad de corrección.
                                * **Naranja (Aceptable/Mejora Necesaria):** Revisión regular.
                                * **Verde (Completo/Riesgo Bajo):** Baja prioridad.
                            """)
                            
                            color_map = {
                                'Completo/Riesgo Bajo': 'green',
                                'Aceptable/Mejora Necesaria': 'orange',
                                'Incompleto/Riesgo Alto': 'red'
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
                                    'prioridad_riesgo_score': 'Riesgo Promedio (Peor ->)', 
                                    'completitud_score': 'Completitud Score (Mejor ^)'
                                }
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                            
                            
                            st.subheader("Top 10 Activos de Máxima Prioridad (Cluster Rojo)")
                            df_prioridad = df_viz2[df_viz2['Calidad_Cluster'] == 'Incompleto/Riesgo Alto']
                            
                            if not df_prioridad.empty:
                                df_prioridad_display = df_prioridad.sort_values(by='prioridad_riesgo_score', ascending=False).head(10)[['titulo', 'dueño', 'prioridad_riesgo_score', 'completitud_score']]
                                
                                def color_risk_row(row):
                                    if row['prioridad_riesgo_score'] > UMBRAL_RIESGO_ALTO:
                                        return ['background-color: #ffcccc'] * 4 
                                    return [''] * 4
                                    
                                st.dataframe(
                                    df_prioridad_display.style.apply(color_risk_row, axis=1).format(
                                        {'prioridad_riesgo_score': '{:.2f}', 'completitud_score': '{:.2f}%'}
                                    ).set_table_attributes('style="font-size: 14px"'),
                                    use_container_width=True,
                                    hide_index=True
                                )
                            else:
                                st.info("¡Excelente! No hay activos clasificados actualmente como Incompleto/Riesgo Alto.")
                            
                        except Exception as e:
                            st.error(f"Error al ejecutar K-Means: {e}")


            with tab3:
                # 3. Cobertura Temática
                st.subheader("3. Cobertura Temática y Distribución de Riesgos")
                
                # Determinar si usar 'common_core_theme' o 'categoria'
                COLUMNA_COBERTURA = 'categoria'
                if 'common_core_theme' in df_filtrado.columns:
                    if filtro_tema != "Mostrar Todos":
                        COLUMNA_COBERTURA = 'common_core_theme'

                
                df_viz3 = df_filtrado.groupby(COLUMNA_COBERTURA).agg(
                    Num_Activos=('uid', 'count'),
                    Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
                    Completitud_Promedio=('completitud_score', 'mean')
                ).reset_index().sort_values(by='Num_Activos', ascending=False)
                
                col_viz3_1, col_viz3_2 = st.columns(2)
                
                with col_viz3_1:
                    st.markdown(f"**Distribución de Activos por {COLUMNA_COBERTURA.capitalize()}**")
                    fig3_1 = px.pie(
                        df_viz3, 
                        values='Num_Activos', 
                        names=COLUMNA_COBERTURA, 
                        title=f'Distribución de Activos por {COLUMNA_COBERTURA.capitalize()}'
                    )
                    fig3_1.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig3_1, use_container_width=True)
                    
                with col_viz3_2:
                    st.markdown(f"**Riesgo vs. Completitud por {COLUMNA_COBERTURA.capitalize()}**")
                    fig3_2 = px.scatter(
                        df_viz3, 
                        x='Riesgo_Promedio', 
                        y='Completitud_Promedio', 
                        size='Num_Activos', 
                        color=COLUMNA_COBERTURA,
                        hover_name=COLUMNA_COBERTURA,
                        title=f'Riesgo vs. Completitud por {COLUMNA_COBERTURA.capitalize()}',
                        labels={
                            'Riesgo_Promedio': 'Riesgo Promedio (Peor ->)', 
                            'Completitud_Promedio': 'Completitud Promedio (Mejor ^)'
                        }
                    )
                    st.plotly_chart(fig3_2, use_container_width=True)
                    
                st.markdown("---")
                st.subheader("Detalle de Riesgos por Categoría")
                
                df_viz3_display = df_viz3.sort_values(by='Riesgo_Promedio', ascending=False).format({
                    'Riesgo_Promedio': '{:.2f}', 
                    'Completitud_Promedio': '{:.2f}%'
                })
                
                def color_risk_cat(val):
                    if val > UMBRAL_RIESGO_ALTO * 0.7: # Umbral más suave para promedio de categoría
                        color = '#f79999' # Rojo suave
                    else:
                        color = ''
                    return f'background-color: {color}'

                st.dataframe(
                    df_viz3_display.style.applymap(color_risk_cat, subset=['Riesgo_Promedio']).set_table_attributes('style="font-size: 14px"'), 
                    use_container_width=True,
                    hide_index=True
                )


            with tab4:
                # 4. Treemap de Cobertura y Calidad
                st.subheader("4. Matriz Treemap de Cobertura, Riesgo y Completitud")
                
                # Determinar si usar 'common_core_theme' o 'categoria'
                COLUMNA_TREEMAP = 'categoria'
                if 'common_core_theme' in df_filtrado.columns:
                    if filtro_tema != "Mostrar Todos":
                        COLUMNA_TREEMAP = 'common_core_theme'
                
                df_treemap = df_filtrado.groupby(COLUMNA_TREEMAP).agg(
                    Num_Activos=('uid', 'count'),
                    Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
                    Completitud_Promedio=('completitud_score', 'mean')
                ).reset_index()
                
                if not df_treemap.empty:
                    st.markdown(f"""
                        El **tamaño** del bloque representa el número de activos ({COLUMNA_TREEMAP.capitalize()}).
                        El **color** (intensidad de rojo) representa el **Riesgo Promedio**.
                        (Más grande y más rojo = Alta Cobertura y Alto Riesgo).
                    """)
                    fig_treemap = px.treemap(
                        df_treemap,
                        path=[COLUMNA_TREEMAP], 
                        values='Num_Activos',
                        color='Riesgo_Promedio', 
                        color_continuous_scale=px.colors.sequential.Reds, 
                        title=f'Matriz Treemap: Cobertura por {COLUMNA_TREEMAP.capitalize()} vs. Riesgo Promedio'
                    )
                    st.plotly_chart(fig_treemap, use_container_width=True)

                    st.markdown("---")
                    st.subheader("Visualización de Completitud (Treemap)")
                    st.markdown(f"""
                        El **color** (intensidad de verde) representa la **Completitud Promedio**.
                        (Más grande y más verde = Alta Cobertura y Alta Calidad).
                    """)
                    fig_treemap_comp = px.treemap(
                        df_treemap,
                        path=[COLUMNA_TREEMAP], 
                        values='Num_Activos',
                        color='Completitud_Promedio', 
                        color_continuous_scale=px.colors.sequential.Greens, 
                        title=f'Matriz Treemap: Cobertura por {COLUMNA_TREEMAP.capitalize()} vs. Completitud Promedio'
                    )
                    st.plotly_chart(fig_treemap_comp, use_container_width=True)
                    
                else:
                    st.warning("No hay suficientes datos para generar el Treemap.")

            st.markdown("---")
            st.header("Asistente de Diagnóstico (IA)")
            st.info(
                "Pregunta por los **KPIs, rankings o diagnósticos** basados en la Base de Conocimiento. "
                "Ej: '¿Qué entidad tiene más activos?', 'Dime el Top 5 peores activos por riesgo', "
                "'¿Cuál es el riesgo promedio en activos en incumplimiento?'"
            )
            
            if knowledge_base_content is None:
                 st.error("La base de conocimiento `knowledge_base.txt` no fue encontrada. El asistente no funcionará.")
            
            chat_history_container = st.container()
            
            with chat_history_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            if prompt := st.chat_input("Escribe aquí tu pregunta de análisis complejo:", key="main_chat_input_key", disabled=(knowledge_base_content is None)):
                
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                with chat_history_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    model_response_placeholder = st.empty() 
                    
                    generate_ai_response(prompt, knowledge_base_content, model_response_placeholder)

except Exception as e:
    st.error(f"ERROR FATAL: Ocurrió un error inesperado al iniciar la aplicación: {e}")
