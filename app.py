import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from datetime import datetime, date 

# Desactivar advertencias de Matplotlib en Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)
# Configuración inicial de la página
st.set_page_config(
    page_title="Dashboard de Diagnóstico de Activos",
    layout="wide"
)

# --- Nombre del archivo CSV que Streamlit debe encontrar ---
ARCHIVO_CSV = "Asset_Inventory_-_Public_20251118.csv"

## --- 1. Funciones de Procesamiento de Datos ---

# Función para limpiar y estandarizar nombres de columnas (snake_case)
def clean_col_name(col):
    name = col.lower().strip()
    name = name.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
    name = name.replace(' ', '_').replace('.', '').replace('/', '_').replace(':', '').replace('(', '').replace(')', '')
    return name

# Funciones de Soporte para el Score de Riesgo (Antigüedad y Estado de Actualización)
def calculate_antiguedad_y_estado(df_temp):
    COL_FECHA_ACTUALIZACION = 'fecha_de_ultima_actualizacion_de_datos_utc'
    COL_FRECUENCIA = 'informacion_de_datos_frecuencia_de_actualizacion'

    # Convertir fechas a formato datetime (asumiendo que vienen como strings)
    df_temp[COL_FECHA_ACTUALIZACION] = pd.to_datetime(df_temp[COL_FECHA_ACTUALIZACION], errors='coerce', utc=True)
    
    # Calcular Antigüedad (días)
    hoy = pd.to_datetime(datetime.now().date(), utc=True)
    df_temp['antiguedad_datos_dias'] = (hoy - df_temp[COL_FECHA_ACTUALIZACION]).dt.days
    
    # Mapeo de frecuencia (Simplificado para el ejemplo)
    mapa_frecuencia = {
        'diario': 1, 'semanal': 7, 'quincenal': 15, 'mensual': 30, 
        'trimestral': 90, 'semestral': 180, 'anual': 365
    }
    df_temp['frecuencia_esperada_dias'] = df_temp[COL_FRECUENCIA].astype(str).str.lower().str.strip().map(mapa_frecuencia).fillna(9999)

    # Calcular Estado de Actualización
    df_temp['estado_actualizacion'] = np.where(
        (df_temp['antiguedad_datos_dias'] > df_temp['frecuencia_esperada_dias']) & 
        (df_temp['frecuencia_esperada_dias'] < 9999), 
        '🔴 INCUMPLIMIENTO', 
        '🟢 CUMPLE'
    )
    return df_temp

# Usamos st.cache_data para que el procesamiento solo se ejecute una vez
@st.cache_data
def process_data(df):
    
    # 1. Limpieza de nombres de columnas
    df.columns = [clean_col_name(col) for col in df.columns]
    
    # 2. Asunción de CÁLCULOS PREVIOS (Antigüedad y Estado de Actualización)
    df['popularidad_score'] = df['vistas'] + df['descargas'] # Score de Popularidad simple
    df = calculate_antiguedad_y_estado(df.copy()) 
    
    # 3. CÁLCULO DE MÉTRICA DE COMPLETITUD
    campos_minimos = [
        'titulo', 'descripcion', 'dueño', 'correo_electronico_de_contacto',
        'licencia', 'dominio', 'categoria', 'informacion_de_datos_frecuencia_de_actualizacion',
        'common_core_public_access_level', 'informacion_de_datos_cobertura_geografica'
    ]
    campos_existentes = [col for col in campos_minimos if col in df.columns]
    num_campos_totales = len(campos_existentes)
    df['campos_diligenciados'] = df[campos_existentes].notna().sum(axis=1)
    df['completitud_score'] = (df['campos_diligenciados'] / num_campos_totales) * 100
    
    # 4. DETECCIÓN DE ANOMALÍAS
    df['anomalia_score'] = 0 
    df_modelo = df[(df['antiguedad_datos_dias'] < 9999) & (df['popularidad_score'] > 0)].copy()
    
    if not df_modelo.empty:
        features = df_modelo[['antiguedad_datos_dias', 'popularidad_score', 'completitud_score']]
        model = IsolationForest(contamination=0.01, random_state=42)
        model.fit(features)
        anomalias = model.predict(features)
        df.loc[df_modelo.index, 'anomalia_score'] = anomalias
    
    # 5. CÁLCULO DE SCORE DE RIESGO/PRIORIDAD
    max_popularidad = df['popularidad_score'].max()
    max_popularidad = max_popularidad if max_popularidad > 0 else 1 

    df['riesgo_incumplimiento'] = np.where(df['estado_actualizacion'] == '🔴 INCUMPLIMIENTO', 3.0, 0.0)
    df['riesgo_completitud'] = np.where(df['completitud_score'] < 50, 1.5, 0.0)
    df['riesgo_demanda'] = (df['popularidad_score'] / max_popularidad) * 1.0
    df['riesgo_anomalia'] = np.where(df['anomalia_score'] == -1, 2.0, 0.0)
    
    df['prioridad_riesgo_score'] = (
        df['riesgo_incumplimiento'] +
        df['riesgo_completitud'] +
        df['riesgo_demanda'] +
        df['riesgo_anomalia']
    )
    
    # 6. Filtrar Públicos
    VALOR_PUBLICO_REAL = 'public'
    df_publico = df[df['publico'].astype(str).str.lower().str.strip() == VALOR_PUBLICO_REAL.lower().strip()].copy()

    return df_publico

## --- 2. Título y Carga Directa ---
st.title("📊 Dashboard de Priorización de Activos de Datos")

try:
    with st.spinner(f'Cargando y procesando el archivo: **{ARCHIVO_CSV}**...'):
        # Carga directa del archivo
        df = pd.read_csv(ARCHIVO_CSV, low_memory=False)
        df_publico = process_data(df.copy())

    st.success(f'✅ Archivo **{ARCHIVO_CSV}** cargado y procesamiento completado. Mostrando resultados para activos PÚBLICOS.')
    st.info(f"El procesamiento se realiza con el modelo ML Isolation Forest para la detección de anomalías.")
    st.write(f"Total de activos en el inventario: **{len(df)}**")
    st.write(f"Total de activos de modalidad PÚBLICA analizados: **{len(df_publico)}**")
    
    if df_publico.empty:
        st.warning("⚠️ El DataFrame público está vacío. Revisa la columna 'publico' y su valor para filtrado.")
    else:
        
        # --- 3. Métricas y Visualizaciones ---
        st.header("🔍 Resultados Clave de Calidad y Prioridad")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Completitud Promedio", f"{df_publico['completitud_score'].mean():.2f}%")
        col2.metric("Activos en Incumplimiento", f"{(df_publico['estado_actualizacion'] == '🔴 INCUMPLIMIENTO').sum()} / {len(df_publico)}")
        col3.metric("Anomalías Detectadas (ML)", f"{(df_publico['anomalia_score'] == -1).sum()}")
        
        
        # --- Visualización 1: Prioridad de Intervención ---
        st.subheader("1. Prioridad de Intervención (Score ML)")
        st.markdown("Este gráfico identifica activos que requieren atención urgente, considerando **Antigüedad**, **Riesgo** (Score ML), y **Demanda** (Tamaño del punto).")
        
        fig1, ax1 = plt.subplots(figsize=(12, 7))
        sns.scatterplot(
            x='antiguedad_datos_dias',
            y='prioridad_riesgo_score', 
            data=df_publico,
            hue='estado_actualizacion',
            palette={'🔴 INCUMPLIMIENTO': 'red', '🟢 CUMPLE': 'green'},
            size='popularidad_score',
            sizes=(20, 400),
            alpha=0.7,
            ax=ax1
        )
        ax1.set_title('Prioridad de Intervención vs. Antigüedad (Score ML)', fontsize=16)
        ax1.set_xlabel('Antigüedad de Datos (Días desde la última actualización)', fontsize=12)
        ax1.set_ylabel('Score de Prioridad de Intervención (Riesgo)', fontsize=12)
        ax1.axhline(y=df_publico['prioridad_riesgo_score'].quantile(0.75), color='red', linestyle='--', label='Prioridad Alta (Q3)')
        ax1.legend(title='Estado de Actualización')
        ax1.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig1)

        st.markdown("---")


        # --- Visualización 2: Top 10 Entidades con Incumplimiento ---
        st.subheader("2. Top 10 Entidades con Mayor Porcentaje de Incumplimiento")
        
        COLUMNA_ENTIDAD = 'dueño'
        resumen_entidad = df_publico.groupby(COLUMNA_ENTIDAD).agg(
            Total_Activos=('uid', 'count'),
            Activos_Incumplimiento=('estado_actualizacion', lambda x: (x == '🔴 INCUMPLIMIENTO').sum())
        ).reset_index()

        resumen_entidad['Porcentaje_Incumplimiento'] = (resumen_entidad['Activos_Incumplimiento'] / resumen_entidad['Total_Activos']) * 100
        resumen_entidad_top = resumen_entidad[resumen_entidad['Total_Activos'] >= 5].sort_values(by='Porcentaje_Incumplimiento', ascending=False).head(10)
        
        if not resumen_entidad_top.empty:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x='Porcentaje_Incumplimiento',
                y=COLUMNA_ENTIDAD,
                data=resumen_entidad_top,
                palette='Reds_d',
                ax=ax2
            )
            ax2.set_title('Top 10 Entidades con Mayor Porcentaje de Incumplimiento de Actualización', fontsize=14)
            ax2.set_xlabel('Porcentaje de Activos en INCUMPLIMIENTO (%)', fontsize=12)
            ax2.set_ylabel('Entidad Responsable', fontsize=12)
            ax2.grid(axis='x', linestyle='--', alpha=0.6)
            st.pyplot(fig2)
            
            st.markdown("### Resumen de Entidades")
            st.dataframe(resumen_entidad_top)
        else:
            st.info("No hay entidades con suficiente volumen (>= 5 activos) o incumplimiento para mostrar el top 10.")
        
        st.markdown("---")

        # --- Visualización 3: Top 10 Categorías ---
        st.subheader("3. Top 10 Categorías con Mayor Cobertura Temática")
        
        COLUMNA_CATEGORIA = 'categoria'
        conteo_categoria = df_publico[COLUMNA_CATEGORIA].value_counts().head(10)
        
        if not conteo_categoria.empty:
            fig3, ax3 = plt.subplots(figsize=(10, 7))
            sns.barplot(x=conteo_categoria.values, y=conteo_categoria.index, palette='viridis', ax=ax3)

            ax3.set_title('Top 10 Categorías con Mayor Cobertura Temática', fontsize=16)
            ax3.set_xlabel('Número de Activos', fontsize=12)
            ax3.set_ylabel('Categoría', fontsize=12)
            st.pyplot(fig3)
            
            st.markdown("### Conteo de Categorías")
            st.dataframe(conteo_categoria.to_frame())
        else:
            st.info("La columna 'categoria' no contiene valores para generar la visualización.")

except FileNotFoundError:
    st.error(f"❌ Error: No se encontró el archivo **{ARCHIVO_CSV}**.")
    st.info("Asegúrate de que el archivo CSV esté en la misma carpeta que `app.py`.")
except Exception as e:
    st.error(f"❌ Ocurrió un error inesperado al procesar el archivo: {e}")
    st.info("Verifica la integridad del archivo CSV y que todas las librerías estén instaladas.")
