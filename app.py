import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import requests
import io
from datetime import datetime, timedelta
import re
import warnings
import os
import base64
import math
from google import genai

# =================================================================
# 0. CONFIGURACIÓN Y VARIABLES GLOBALES
# =================================================================

# Ocultar advertencias
warnings.filterwarnings('ignore')

# URL API Socrata (Datos.gov.co) - Inventario de Activos
API_URL = "https://www.datos.gov.co/resource/uzcf-b9dh.json?$limit=100000"

# CLAVE SECRETA DE GEMINI (Asegúrate de protegerla en producción)
GEMINI_API_SECRET_VALUE = "AIzaSyDvuJPAAK8AVIS-VQIe39pPgVNb8xlJw3g"

# Configuración de la Guía 2025
UMBRAL_CALIDAD_MINIMA = 7.0  # Para Sello de Calidad 1
PESO_COMPLETITUD = 0.25
PESO_CONFORMIDAD = 0.25
PESO_PORTABILIDAD = 0.50

# Listas de Referencia para Confidencialidad (Guía Sec 3.3)
RIESGO_ALTO_KEYWORDS = ['tarjeta de identidad', 'cedula', 'historial medico', 'diagnostico', 'cuenta bancaria', 'ingresos']
RIESGO_MEDIO_KEYWORDS = ['direccion domicilio', 'telefono personal', 'celular']
RIESGO_BAJO_KEYWORDS = ['fecha nacimiento', 'edad']

st.set_page_config(page_title="Evaluador Calidad Guía 2025", layout="wide")

# =================================================================
# 1. MOTOR DE INGESTA Y PREPROCESAMIENTO (API LIVE)
# =================================================================

@st.cache_data(ttl=3600)  # Cache de 1 hora para no saturar la API
def fetch_data_from_api():
    """
    Consume los datos directamente de la API de Datos Abiertos.
    Realiza la normalización de columnas para cumplir con la Guía.
    """
    try:
        with st.spinner('Conectando con API de Datos Abiertos (Socrata)...'):
            response = requests.get(API_URL)
            if response.status_code != 200:
                st.error(f"Error API: {response.status_code}")
                return pd.DataFrame()
            
            data = response.json()
            df = pd.DataFrame(data)
            
            # Normalización de Nombres de Columnas para el Motor de Evaluación
            # Mapeo basado en la estructura común de Socrata
            column_mapping = {
                'nombre': 'titulo',
                'descripcion': 'descripcion',
                'entidad': 'dueno',
                'fecha_emision': 'fecha_creacion',
                'fecha_actualizacion': 'fecha_actualizacion',
                'categoria': 'categoria',
                'palabras_clave': 'etiquetas',
                'url': 'enlace',
                'correo_electronico': 'contacto_email',
                'frecuencia_actualizacion': 'frecuencia'
            }
            
            # Renombrar si existen, si no, crear vacías
            for api_col, int_col in column_mapping.items():
                if api_col in df.columns:
                    df.rename(columns={api_col: int_col}, inplace=True)
                elif int_col not in df.columns:
                    df[int_col] = np.nan

            # Conversión de Tipos
            date_cols = ['fecha_creacion', 'fecha_actualizacion']
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Asegurar columnas de texto
            text_cols = ['titulo', 'descripcion', 'dueno', 'categoria', 'etiquetas', 'frecuencia']
            for col in text_cols:
                if col in df.columns:
                    df[col] = df[col].fillna('').astype(str)
            
            return df
            
    except Exception as e:
        st.error(f"Fallo crítico en la ingesta de datos: {e}")
        return pd.DataFrame()

# =================================================================
# 2. MOTOR DE EVALUACIÓN (GUÍA 2025 ESTRICTA)
# =================================================================

def calcular_confidencialidad(row):
    """
    Sec 3.3.2: Cálculo de Confidencialidad basado en riesgo.
    Formula: 10 - (riesgo_total / (num_col_conf * 3)) * factor
    Nota: Al evaluar metadatos (inventario), buscamos palabras clave en titulo/descripcion
    ya que no tenemos acceso a las columnas internas del dataset en esta vista.
    """
    texto_analisis = (str(row['titulo']) + " " + str(row['descripcion'])).lower()
    
    riesgo_total = 0
    hallazgos = 0
    
    # Riesgo Alto (Peso 3)
    for kw in RIESGO_ALTO_KEYWORDS:
        if kw in texto_analisis:
            riesgo_total += 3
            hallazgos += 1
            
    # Riesgo Medio (Peso 2)
    for kw in RIESGO_MEDIO_KEYWORDS:
        if kw in texto_analisis:
            riesgo_total += 2
            hallazgos += 1
            
    if hallazgos == 0:
        return 10.0
    
    # Aplicación simplificada de la fórmula de la guía para metadatos
    penalizacion = (riesgo_total / (hallazgos * 3)) * 10
    return max(0.0, 10.0 - penalizacion)

def calcular_comprensibilidad(row):
    """
    Sec 3.14.1: Criterio de Comprensibilidad.
    Usa funciones logarítmicas y exponenciales sobre la longitud de descripción y etiquetas.
    """
    desc = str(row['descripcion'])
    length = len(desc)
    
    # Fórmula Guía: 10 * (1 - exp(-0.05 * length))
    puntaje_desc = 10 * (1 - math.exp(-0.05 * length))
    
    # Etiquetas (Tags)
    tags = str(row['etiquetas'])
    tag_len = len(tags)
    max_len_esperado = 100 # Umbral referencia
    
    # Fórmula Logarítmica Guía (aprox): 10 * log(1 + len) / log(1 + max)
    if tag_len < 2:
        puntaje_tags = 0
    else:
        val = math.log(1 + (tag_len - 2))
        denom = math.log(1 + (max_len_esperado - 2))
        puntaje_tags = 10 * (val / denom) if denom > 0 else 0
        
    return min(10.0, (puntaje_desc * 0.6 + puntaje_tags * 0.4))

def calcular_actualidad(row):
    """
    Sec 3.5.1: Criterio de Actualidad.
    Formula: 0 si (FechaActual - FechaAct) > Frecuencia, sino 10.
    """
    if pd.isnull(row['fecha_actualizacion']):
        return 0.0
    
    frecuencia_txt = str(row['frecuencia']).lower()
    dias_limite = 365 # Default Anual
    
    if 'mensual' in frecuencia_txt:
        dias_limite = 30
    elif 'semestral' in frecuencia_txt:
        dias_limite = 180
    elif 'trimestral' in frecuencia_txt:
        dias_limite = 90
    elif 'diaria' in frecuencia_txt:
        dias_limite = 1
        
    delta = datetime.now() - row['fecha_actualizacion']
    
    if delta.days > dias_limite:
        return 0.0
    else:
        return 10.0

def calcular_trazabilidad(row, total_columns_expected=10):
    """
    Sec 3.6.1: Criterio de Trazabilidad.
    Evalúa metadatos diligenciados con penalización cuadrática.
    """
    # Campos críticos de metadatos según Guía
    campos_revisar = ['titulo', 'descripcion', 'dueno', 'categoria', 'enlace', 'fecha_creacion', 'contacto_email']
    vacios = 0
    for c in campos_revisar:
        if c not in row or pd.isnull(row[c]) or str(row[c]).strip() == '':
            vacios += 1
            
    missing_prop = vacios / len(campos_revisar)
    
    # Penalización cuadrática: penalty = missing_prop^2
    # Score = 10 * (1 - penalty) (Simplificado de la guia compleja)
    # La guia usa ponderados: 75% metadatos, 20% acceso auditado, 5% titulo fecha
    
    puntaje_meta = 10 * (1 - (missing_prop ** 2))
    
    # Chequeo Título Sin Fecha (5%)
    tiene_fecha_titulo = bool(re.search(r'\d{4}', str(row['titulo'])))
    puntaje_titulo = 0 if tiene_fecha_titulo else 10 # Penaliza si NO tiene fecha según interpretación o al revés (Guia: Penaliza si HAY referencias temporales que limiten) -> Guia dice: "Identifica si en el título se encuentran referencias a fechas... y aplica penalización". Entonces si tiene fecha, penaliza.
    
    final_score = (puntaje_meta * 0.95) + (puntaje_titulo * 0.05) # Simplificación auditado
    return max(0.0, final_score)

def calcular_credibilidad(row):
    """
    Sec 3.13.1: Criterio de Credibilidad.
    Valida publicador, email y enlaces.
    """
    # 1. Publicador Válido
    tiene_dueno = 1 if len(str(row['dueno'])) > 3 else 0
    tiene_email = 1 if '@' in str(row.get('contacto_email', '')) else 0
    
    score_publicador = 10 if (tiene_dueno and tiene_email) else 0
    
    # 2. Metadatos Completos (reutiliza lógica parcial)
    tiene_url = 1 if str(row['enlace']).startswith('http') else 0
    
    # Ponderación Guía: 70% Meta, 5% Publicador, 25% Descripciones validas
    # Como no podemos validar columnas internas, ajustamos al metadato
    return (score_publicador * 0.3) + (tiene_url * 10 * 0.7)

def calcular_accesibilidad(row):
    """
    Sec 3.15.1: Criterio de Accesibilidad.
    Suma de puntajes por tags y links.
    """
    tiene_tags = 1 if len(str(row['etiquetas'])) > 2 else 0
    tiene_link = 1 if str(row['enlace']).startswith('http') else 0
    
    score = (5 * tiene_tags) + (5 * tiene_link)
    return float(score)

def evaluar_cumplimiento_guia_2025(df):
    """
    Aplica todas las funciones de evaluación fila por fila.
    """
    if df.empty:
        return df
    
    st.toast("Iniciando evaluación estricta Guía 2025...", icon="🚀")
    
    # 1. Evaluaciones Individuales
    df['Score_Confidencialidad'] = df.apply(calcular_confidencialidad, axis=1)
    df['Score_Comprensibilidad'] = df.apply(calcular_comprensibilidad, axis=1)
    df['Score_Actualidad'] = df.apply(calcular_actualidad, axis=1)
    df['Score_Trazabilidad'] = df.apply(calcular_trazabilidad, axis=1)
    df['Score_Credibilidad'] = df.apply(calcular_credibilidad, axis=1)
    df['Score_Accesibilidad'] = df.apply(calcular_accesibilidad, axis=1)
    
    # 2. Disponibilidad (Promedio Accesibilidad + Actualidad) - Sec 3.19
    df['Score_Disponibilidad'] = (df['Score_Accesibilidad'] + df['Score_Actualidad']) / 2
    
    # 3. Recuperabilidad (Promedio Accesibilidad + Trazabilidad) - Sec 3.18 (Aprox)
    df['Score_Recuperabilidad'] = (df['Score_Accesibilidad'] + df['Score_Trazabilidad']) / 2
    
    # 4. CALIFICACIÓN FINAL PONDERADA
    # Promedio de los criterios críticos
    columnas_score = [
        'Score_Confidencialidad', 'Score_Comprensibilidad', 'Score_Actualidad',
        'Score_Trazabilidad', 'Score_Credibilidad', 'Score_Accesibilidad'
    ]
    
    df['Calidad_Global_2025'] = df[columnas_score].mean(axis=1)
    
    # 5. ASIGNACIÓN DE SELLOS DE CALIDAD (Sec 4)
    def asignar_sello(row):
        # Sello 0: No cumple mínimos
        if row['Score_Actualidad'] < 10 or row['Score_Trazabilidad'] < 7 or row['Score_Credibilidad'] < 7:
            return "Sello 0 (Sin Calidad)"
        
        # Sello 1: Cumple mínimos básicos
        if row['Calidad_Global_2025'] >= 7.0:
            # Sello 2: Sello 1 + Conformidad alta (simulado aqui como > 8.5 global)
            if row['Calidad_Global_2025'] >= 8.5:
                 return "Sello 2 (Plata)"
            return "Sello 1 (Bronce)"
        
        return "Sello 0 (Sin Calidad)"

    df['Sello_Calidad'] = df.apply(asignar_sello, axis=1)
    
    return df

# =================================================================
# 3. DETECCIÓN DE ANOMALÍAS (IA)
# =================================================================

@st.cache_data
def apply_anomaly_detection(df):
    """
    Usa Isolation Forest sobre los puntajes calculados.
    """
    if df.empty or len(df) < 50:
        df['anomalia_score'] = 1
        return df
        
    features = ['Calidad_Global_2025', 'Score_Actualidad', 'Score_Comprensibilidad']
    df_model = df[features].fillna(0)
    
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['anomalia_score'] = iso_forest.fit_predict(df_model)
    
    return df

# =================================================================
# 4. GENERACIÓN DE REPORTES CON LLM (RAG)
# =================================================================

def get_dataset_context_string(df):
    """Genera un resumen estadístico para el LLM."""
    stats = df.describe().to_string()
    top_errors = df[df['Calidad_Global_2025'] < 5].head(5)[['titulo', 'dueno', 'Calidad_Global_2025']].to_string()
    sellos = df['Sello_Calidad'].value_counts().to_string()
    
    context = f"""
    RESUMEN ESTADÍSTICO (GUÍA 2025):
    {stats}
    
    DISTRIBUCIÓN DE SELLOS:
    {sellos}
    
    EJEMPLOS DE BAJA CALIDAD:
    {top_errors}
    """
    return context

def generate_ai_response(user_query, df_context, model_placeholder):
    """
    Analista experto usando Gemini.
    """
    try:
        client = genai.Client(api_key=GEMINI_API_SECRET_VALUE)
        
        context_str = get_dataset_context_string(df_context)
        
        system_prompt = (
            "Eres un Auditor Senior de Datos Abiertos del gobierno colombiano. "
            "Tu trabajo es analizar el cumplimiento estricto de la Guía de Calidad e Interoperabilidad 2025. "
            "Usa el siguiente contexto estadístico de los datos evaluados para responder. "
            "Sé crítico, profesional y cita los criterios (Actualidad, Trazabilidad, etc.).\n\n"
            f"CONTEXTO DATOS:\n{context_str}"
        )

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[user_query],
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2
            )
        )
        model_placeholder.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})

    except Exception as e:
        model_placeholder.error(f"Error AI: {str(e)}")

# =================================================================
# 5. INTERFAZ DE USUARIO (STREAMLIT)
# =================================================================

def main():
    st.title("🇨🇴 Auditoría de Calidad de Datos Abiertos 2025")
    st.markdown("### Sistema de Evaluación Automática basado en la Guía Oficial")
    
    # Carga de Datos
    df_raw = fetch_data_from_api()
    
    if df_raw.empty:
        st.warning("No se pudieron cargar datos. Verifique la conexión a la API.")
        return

    # Procesamiento
    df_evaluated = evaluar_cumplimiento_guia_2025(df_raw)
    df_final = apply_anomaly_detection(df_evaluated)
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://www.datos.gov.co/assets/images/logo-datos-abiertos.png", width=200)
        st.header("Filtros de Auditoría")
        
        # Filtro Entidad
        entidades = ['Todas'] + sorted(df_final['dueno'].unique().tolist())
        filtro_entidad = st.selectbox("Entidad:", entidades)
        
        # Filtro Sello
        sellos = ['Todos'] + sorted(df_final['Sello_Calidad'].unique().tolist())
        filtro_sello = st.selectbox("Sello de Calidad:", sellos)
        
        st.divider()
        st.info(f"Total Activos Analizados: {len(df_final)}")
        st.info(f"Fecha Auditoría: {datetime.now().strftime('%Y-%m-%d')}")

    # --- FILTRADO ---
    df_view = df_final.copy()
    if filtro_entidad != 'Todas':
        df_view = df_view[df_view['dueno'] == filtro_entidad]
    if filtro_sello != 'Todos':
        df_view = df_view[df_view['Sello_Calidad'] == filtro_sello]

    # --- DASHBOARD PRINCIPAL ---
    
    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    avg_score = df_view['Calidad_Global_2025'].mean()
    criticos = len(df_view[df_view['Calidad_Global_2025'] < 6.0])
    cumplimiento = (len(df_view[df_view['Sello_Calidad'] != 'Sello 0 (Sin Calidad)']) / len(df_view) * 100) if len(df_view) > 0 else 0
    
    kpi1.metric("Índice de Calidad Global (IQA)", f"{avg_score:.2f}/10", delta_color="normal" if avg_score > 7 else "inverse")
    kpi2.metric("Activos Críticos (<6.0)", criticos, delta_color="inverse")
    kpi3.metric("% Cumplimiento Estándar", f"{cumplimiento:.1f}%")
    kpi4.metric("Anomalías Detectadas", len(df_view[df_view['anomalia_score'] == -1]), help="Detección IA Isolation Forest")

    st.divider()

    # PESTAÑAS
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Ranking & Diagnóstico", "🔍 Explorador de Datos", "🤖 Auditoría IA", "📈 Clusters Calidad"])

    with tab1:
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Distribución de Sellos de Calidad")
            fig_pie = px.pie(df_view, names='Sello_Calidad', title='Porcentaje por Nivel de Certificación', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_chart2:
            st.subheader("Puntajes Promedio por Criterio (Guía 2025)")
            cols_radar = ['Score_Confidencialidad', 'Score_Comprensibilidad', 'Score_Actualidad', 'Score_Trazabilidad', 'Score_Credibilidad', 'Score_Accesibilidad']
            radar_data = pd.DataFrame(dict(
                r=df_view[cols_radar].mean().values,
                theta=[c.replace('Score_', '') for c in cols_radar]
            ))
            fig_radar = px.line_polar(radar_data, r='r', theta='theta', line_close=True, range_r=[0,10], title="Perfil de Calidad Promedio")
            fig_radar.update_traces(fill='toself')
            st.plotly_chart(fig_radar, use_container_width=True)
            
        st.subheader("🚨 Top 10 Activos con Peor Calidad (Requieren Acción Inmediata)")
        worst_assets = df_view.sort_values('Calidad_Global_2025').head(10)[['titulo', 'dueno', 'Calidad_Global_2025', 'Score_Actualidad', 'Score_Trazabilidad', 'Sello_Calidad']]
        st.dataframe(worst_assets.style.background_gradient(cmap='Reds_r', subset=['Calidad_Global_2025']), use_container_width=True)

    with tab2:
        st.subheader("Explorador Detallado de Evaluación")
        
        # Buscador
        search = st.text_input("Buscar por palabra clave en Título o Descripción:", "")
        if search:
            df_view = df_view[df_view['titulo'].str.contains(search, case=False) | df_view['descripcion'].str.contains(search, case=False)]
            
        cols_show = ['titulo', 'dueno', 'Calidad_Global_2025', 'Sello_Calidad', 'Score_Actualidad', 'Score_Confidencialidad', 'anomalia_score']
        
        def highlight_row(row):
            if row.Calidad_Global_2025 < 5:
                return ['background-color: #ffcccc'] * len(row)
            return [''] * len(row)

        st.dataframe(
            df_view[cols_show].style.format({'Calidad_Global_2025': "{:.2f}"}),
            use_container_width=True,
            height=600
        )

    with tab3:
        st.subheader("🤖 Asistente de Auditoría (Powered by Gemini)")
        st.markdown("Pregunta sobre el estado de cumplimiento, entidades rezagadas o detalles técnicos de la Guía.")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("Ej: ¿Cuáles son las entidades con peor puntaje de actualidad?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            placeholder = st.empty()
            generate_ai_response(prompt, df_view, placeholder)

    with tab4:
        st.subheader("Segmentación de Activos (K-Means)")
        st.markdown("Agrupación automática basada en comportamiento de métricas.")
        
        if len(df_view) > 10:
            features_cluster = ['Score_Actualidad', 'Score_Trazabilidad', 'Score_Comprensibilidad']
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(df_view[features_cluster].fillna(0))
            
            kmeans = KMeans(n_clusters=3, random_state=42)
            df_view['Cluster'] = kmeans.fit_predict(data_scaled)
            
            fig_cluster = px.scatter_3d(
                df_view, x='Score_Actualidad', y='Score_Trazabilidad', z='Score_Comprensibilidad',
                color='Cluster', hover_data=['titulo', 'dueno'],
                title="Clusters de Calidad (3D)", opacity=0.7
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
        else:
            st.info("Datos insuficientes para clustering (min 10 registros).")

if __name__ == "__main__":
    main()
