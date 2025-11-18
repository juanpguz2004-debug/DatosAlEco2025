# ==============================================================================
# 1. IMPORTS Y CONFIGURACIÓN INICIAL
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import IsolationForest
import warnings
import os

warnings.filterwarnings('ignore')

# Configuración de la página (amplia para dashboards)
st.set_page_config(layout="wide")

# ==============================================================================
# 2. FUNCIONES DE PROCESAMIENTO Y CACHEO (ETL + MÉTRICAS + ML)
# ==============================================================================

# Función para estandarizar nombres de columnas (de la Celda 3)
def clean_col_name(col):
    name = col.lower().strip()
    name = name.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
    name = name.replace(' ', '_').replace('.', '').replace('/', '_').replace(':', '').replace('(', '').replace(')', '')
    return name

# La función principal de procesamiento que se cachea
@st.cache_data
def load_and_process_data(file_path):
    
    # ----------------------------------------
    # I. Carga y Limpieza (Celdas 2 y 3)
    # ----------------------------------------
    try:
        df = pd.read_csv(file_path, low_memory=False)
        df.columns = [clean_col_name(col) for col in df.columns]
    except Exception as e:
        st.error(f"Error al cargar o limpiar el CSV: {e}")
        return pd.DataFrame() # Devuelve vacío si falla
    
    # ----------------------------------------
    # II. Métrica de Completitud (Celda 5 original)
    # ----------------------------------------
    campos_minimos = [
        'titulo', 'descripcion', 'dueño', 'correo_electronico_de_contacto', 
        'licencia', 'dominio', 'categoria', 'informacion_de_datos_frecuencia_de_actualizacion', 
        'common_core_public_access_level', 'informacion_de_datos_cobertura_geografica'
    ]
    campos_existentes = [col for col in campos_minimos if col in df.columns]
    num_campos_totales = len(campos_existentes)
    
    if num_campos_totales > 0:
        df['campos_diligenciados'] = df[campos_existentes].notna().sum(axis=1)
        df['completitud_score'] = (df['campos_diligenciados'] / num_campos_totales) * 100
    else:
        df['completitud_score'] = 0

    # ----------------------------------------
    # III. Métricas de Tiempo y Uso (Celda 6 original)
    # ----------------------------------------
    COLUMNA_FECHA_ACTUALIZACION = 'fecha_de_ultima_actualizacion_de_datos_utc' 
    COLUMNA_FRECUENCIA = 'informacion_de_datos_frecuencia_de_actualizacion'
    
    # Antigüedad
    df[COLUMNA_FECHA_ACTUALIZACION] = pd.to_datetime(df[COLUMNA_FECHA_ACTUALIZACION], errors='coerce', utc=True)
    HOY = pd.Timestamp.now(tz='utc') 
    df['antiguedad_datos_dias'] = (HOY - df[COLUMNA_FECHA_ACTUALIZACION]).dt.days.fillna(9999) 

    # Frecuencia y Cumplimiento
    mapa_frecuencia = {'diaria': 1, 'diario': 1, 'continuamente': 1, 'semanal': 7, 'quincenal': 15, 'mensual': 30, 'mensualmente': 30, 'bimestral': 60, 'trimestral': 90, 'semestral': 182, 'anual': 365, 'anualmente': 365, 'no aplica': 365 * 10, 'null': 365 * 10 }
    df['frecuencia_esperada_dias'] = df[COLUMNA_FRECUENCIA].astype(str).str.lower().str.strip().map(mapa_frecuencia).fillna(365 * 10) 
    UMBRAL_GRACIA_DIAS = 15 
    df['estado_actualizacion'] = np.where(
        df['antiguedad_datos_dias'] > (df['frecuencia_esperada_dias'] + UMBRAL_GRACIA_DIAS),
        '🔴 INCUMPLIMIENTO',
        '🟢 CUMPLE'
    )
    
    # Popularidad
    df['vistas'] = pd.to_numeric(df['vistas'], errors='coerce')
    df['descargas'] = pd.to_numeric(df['descargas'], errors='coerce')
    df['popularidad_score'] = (df['vistas'].fillna(0) * 0.6) + (df['descargas'].fillna(0) * 0.4)

    # ----------------------------------------
    # IV. Detección de Anomalías (Isolation Forest - Celda 4 modificada)
    # ----------------------------------------
    df['anomalia_score'] = 0 # Inicializar
    df_modelo = df[(df['antiguedad_datos_dias'] < 9999) & (df['popularidad_score'] > 0)].copy()
    
    if not df_modelo.empty:
        features = df_modelo[['antiguedad_datos_dias', 'popularidad_score', 'completitud_score']]
        model = IsolationForest(contamination=0.01, random_state=42)
        model.fit(features)
        anomalias = model.predict(features)
        df.loc[df_modelo.index, 'anomalia_score'] = anomalias

    # ----------------------------------------
    # V. Score de Prioridad/Riesgo (Celda 6 modificada)
    # ----------------------------------------
    df['riesgo_incumplimiento'] = np.where(df['estado_actualizacion'] == '🔴 INCUMPLIMIENTO', 3.0, 0.0)
    df['riesgo_completitud'] = np.where(df['completitud_score'] < 50, 1.5, 0.0)
    max_popularidad = df['popularidad_score'].max()
    df['riesgo_demanda'] = (df['popularidad_score'] / max_popularidad) * 1.0 if max_popularidad > 0 else 0.0
    df['riesgo_anomalia'] = np.where(df['anomalia_score'] == -1, 2.0, 0.0)
    
    df['prioridad_riesgo_score'] = (
        df['riesgo_incumplimiento'] + 
        df['riesgo_completitud'] + 
        df['riesgo_demanda'] +
        df['riesgo_anomalia']
    )
    
    # ----------------------------------------
    # VI. Filtrado Público (Celda 7)
    # ----------------------------------------
    df_publico = df[df['publico'].astype(str).str.lower().str.strip() == 'public'].copy()
    
    return df_publico

# ==============================================================================
# 3. EJECUCIÓN DEL PROCESAMIENTO Y MANEJO DE ARCHIVOS
# ==============================================================================

# Se asume que el archivo CSV está en la carpeta 'data/' del repositorio
FILE_PATH = 'data/inventory.csv' # ¡Asegúrate de que este nombre sea correcto!

try:
    df_publico = load_and_process_data(FILE_PATH)
except FileNotFoundError:
    st.error(f"Error: No se encontró el archivo '{FILE_PATH}'. Asegúrate de subirlo a la carpeta 'data/' en GitHub.")
    df_publico = pd.DataFrame()

if df_publico.empty:
    st.warning("No hay datos públicos para mostrar. Revisa la carga de datos o los filtros.")
    st.stop() # Detiene la ejecución si no hay datos

# ==============================================================================
# 4. DISEÑO DE LA INTERFAZ STREAMLIT
# ==============================================================================

st.title("🛡️ Agente de Datos Abiertos: Diagnóstico de Calidad y Riesgo")
st.markdown("---")

# --- BARRA LATERAL Y FILTROS ---
st.sidebar.header("Filtros de Análisis")
entidad_seleccionada = st.sidebar.selectbox(
    "Filtrar Entidad Dueña:", 
    options=['Todas'] + df_publico['dueño'].dropna().unique().tolist()
)

# Aplicar filtro a la vista principal
if entidad_seleccionada != 'Todas':
    df_filtrado = df_publico[df_publico['dueño'] == entidad_seleccionada]
else:
    df_filtrado = df_publico.copy()

# --- PESTAÑAS PRINCIPALES ---
tab1, tab2, tab3 = st.tabs(["🚨 Prioridad de Intervención (OE2)", "🔍 Diagnóstico (OE1)", "💡 Agente de Datos (LLM)"])

# ------------------------------------------------------------------------------
# PESTAÑA 1: PRIORIDAD DE INTERVENCIÓN (OE2/ML)
# ------------------------------------------------------------------------------
with tab1:
    st.header("1. Score de Riesgo y Prioridad de Intervención")
    
    # KPIs Rápidos
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    num_incumplimiento = (df_filtrado['estado_actualizacion'] == '🔴 INCUMPLIMIENTO').sum()
    pct_incumplimiento = num_incumplimiento / len(df_filtrado) * 100 if len(df_filtrado) > 0 else 0
    
    num_anomalias = (df_filtrado['anomalia_score'] == -1).sum()
    avg_completitud = df_filtrado['completitud_score'].mean()
    avg_riesgo = df_filtrado['prioridad_riesgo_score'].mean()
    
    col_kpi1.metric("Activos en Incumplimiento", f"{num_incumplimiento} ({pct_incumplimiento:.1f}%)")
    col_kpi2.metric("Anomalías ML Detectadas", num_anomalias)
    col_kpi3.metric("Completitud Promedio", f"{avg_completitud:.1f}%")
    col_kpi4.metric("Riesgo Promedio (Score)", f"{avg_riesgo:.2f}")

    st.markdown("---")
    
    # Gráfico de Prioridad (Celda 8)
    st.subheader("Visualización de Riesgo: Antigüedad vs. Score de Prioridad")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x='antiguedad_datos_dias', 
        y='prioridad_riesgo_score', 
        data=df_filtrado, 
        hue='estado_actualizacion', 
        palette={'🔴 INCUMPLIMIENTO': 'red', '🟢 CUMPLE': 'green'}, 
        size='popularidad_score',
        sizes=(50, 800),
        alpha=0.7,
        ax=ax
    )
    ax.set_title('Antigüedad vs. Score de Prioridad de Intervención')
    ax.set_xlabel('Antigüedad de Datos (Días)')
    ax.set_ylabel('Score de Prioridad de Riesgo')
    ax.axhline(y=df_filtrado['prioridad_riesgo_score'].quantile(0.75), color='red', linestyle='--', label='Prioridad Alta (Q3)')
    st.pyplot(fig)
    
    # Tabla de Top 20 Riesgosos
    st.subheader("Top 20 Activos de Mayor Riesgo (Prioridad de Intervención)")
    top_riesgo = df_filtrado.sort_values('prioridad_riesgo_score', ascending=False).head(20)
    st.dataframe(top_riesgo[[
        'titulo', 'dueño', 'antiguedad_datos_dias', 'completitud_score', 
        'popularidad_score', 'prioridad_riesgo_score', 'anomalia_score'
    ]].style.background_gradient(cmap='Reds', subset=['prioridad_riesgo_score']))


# ------------------------------------------------------------------------------
# PESTAÑA 2: DIAGNÓSTICO DE COHERENCIA Y COBERTURA (OE1)
# ------------------------------------------------------------------------------
with tab2:
    st.header("2. Diagnóstico de Coherencia y Cobertura")
    
    col_viz2_1, col_viz2_2 = st.columns(2)
    
    # Incumplimiento por Entidad (Celda 9)
    with col_viz2_1:
        st.subheader("Incumplimiento por Entidad")
        resumen_entidad = df_publico.groupby('dueño').agg(
            Total_Activos=('titulo', 'count'),
            Activos_Incumplimiento=('estado_actualizacion', lambda x: (x == '🔴 INCUMPLIMIENTO').sum())
        ).reset_index()

        resumen_entidad['Porcentaje_Incumplimiento'] = (resumen_entidad['Activos_Incumplimiento'] / resumen_entidad['Total_Activos']) * 100
        resumen_entidad_top = resumen_entidad[resumen_entidad['Total_Activos'] >= 5].sort_values('Porcentaje_Incumplimiento', ascending=False).head(10)
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Porcentaje_Incumplimiento', y='dueño', data=resumen_entidad_top, palette='Reds_d', ax=ax2)
        ax2.set_xlabel('Porcentaje de Activos en INCUMPLIMIENTO (%)')
        ax2.set_ylabel('Entidad Responsable')
        st.pyplot(fig2)

    # Cobertura Temática (Celda 10 original)
    with col_viz2_2:
        st.subheader("Cobertura Temática (por Categoría)")
        conteo_categoria = df_publico['categoria'].value_counts().head(10)
        
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.barplot(x=conteo_categoria.values, y=conteo_categoria.index, palette='viridis', ax=ax3)
        ax3.set_xlabel('Número de Activos')
        ax3.set_ylabel('Categoría')
        st.pyplot(fig3)

# ------------------------------------------------------------------------------
# PESTAÑA 3: ASISTENTE LLM (SIMULACIÓN)
# ------------------------------------------------------------------------------
with tab3:
    st.header("3. Asistente Inteligente Agente de Datos")
    st.markdown("""
        Esta sección simula el **Agente de Datos** basado en LLM.
        En el proyecto final, un modelo (como GPT4All) usaría las métricas calculadas 
        (Riesgo, Completitud, Cumplimiento) para responder preguntas en lenguaje natural 
        y generar recomendaciones accionables.
    """)
    
    st.subheader("Simulador de Recomendaciones Clave")
    
    # En un entorno real, el LLM generaría este texto. Aquí lo generamos con pandas.
    
    # 1. Hallazgo de Ries
