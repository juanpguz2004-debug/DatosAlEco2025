import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import io 

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Diagnóstico de Activos",
    layout="wide"
)

# --- Variables Globales ---
ARCHIVO_PROCESADO = "Asset_Inventory_PROCESSED.csv" # Usamos el archivo pre-calculado
UMBRAL_RIESGO_ALTO = 1.0 

# =================================================================
# 1. Funciones de Carga y Procesamiento (MUY SIMPLIFICADO)
# =================================================================

@st.cache_data
def load_processed_data(file_path):
    """Carga el archivo CSV YA PROCESADO y lo cachea."""
    try:
        # Cargamos solo las columnas necesarias para el dashboard para ahorrar memoria
        df = pd.read_csv(file_path, low_memory=False)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

def process_external_data(df):
    """Lógica de riesgo universal para el archivo externo (DEBE PERMANECER AQUÍ)."""
    
    # 1. Asegurar la existencia de las columnas mínimas para filtros/títulos
    if 'titulo' not in df.columns:
        df['titulo'] = 'Activo sin título'
    if 'dueño' not in df.columns:
        df['dueño'] = 'Desconocido'

    # 2. CÁLCULO DE RIESGO UNIVERSAL
    # Completitud de Metadatos (usando el mismo set de 10)
    campos_minimos = [
        'titulo', 'descripcion', 'dueño', 'correo_electronico_de_contacto',
        'licencia', 'dominio', 'categoria', 'informacion_de_datos_frecuencia_de_actualizacion',
        'common_core_public_access_level', 'informacion_de_datos_cobertura_geografica'
    ]
    campos_existentes = [col for col in campos_minimos if col in df.columns]
    num_campos_totales_base = len(campos_minimos)

    df['campos_diligenciados'] = df[campos_existentes].notna().sum(axis=1)
    df['completitud_score'] = (df['campos_diligenciados'] / num_campos_totales_base) * 100
    
    # Completitud de Datos por Fila (universal)
    df['datos_por_fila_score'] = (df.notna().sum(axis=1) / df.shape[1]) * 100
    
    # Penalización 1: Score bajo de Datos por Fila (riesgo de datos incompletos)
    df['riesgo_datos_incompletos'] = np.where(df['datos_por_fila_score'] < 70, 2.0, 0.0)
    
    # Penalización 2: Completitud de Metadatos (si es menor a 10%)
    df['riesgo_metadatos_nulo'] = np.where(df['completitud_score'] < 10, 1.0, 0.0)
    
    df['prioridad_riesgo_score'] = df['riesgo_datos_incompletos'] + df['riesgo_metadatos_nulo']
    
    return df


# =================================================================
# 2. Ejecución Principal
# =================================================================

st.title("📊 Dashboard de Priorización de Activos de Datos (Análisis Completo)")

try:
    with st.spinner(f'Cargando archivo procesado: **{ARCHIVO_PROCESADO}**...'):
        df_analisis_completo = load_processed_data(ARCHIVO_PROCESADO) 

    if df_analisis_completo.empty:
        st.error(f"🛑 Error: No se pudo cargar el archivo **{ARCHIVO_PROCESADO}**. Ejecuta `preprocess.py` primero.")
    else:
        st.success(f'✅ Archivo pre-procesado cargado. Total de activos: **{len(df_analisis_completo)}**')

        # --- SECCIÓN DE SELECCIÓN Y DESGLOSE DE ENTIDAD ---
        # ... (Todo el código de filtros y visualizaciones se mantiene igual, 
        # usando df_analisis_completo y df_filtrado) ...
        
        # --- FILTROS DE ENTIDAD ---
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
                # Nota: 'estado_actualizacion' ya viene calculado desde el pre-proceso
                incumplimiento = (df_entidad_seleccionada['estado_actualizacion'] == '🔴 INCUMPLIMIENTO').sum()
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                col1.metric("Activos Totales", total_activos)
                col2.metric("Completitud Promedio", f"{df_entidad_seleccionada['completitud_score'].mean():.2f}%")
                col3.metric("Riesgo Promedio", f"{df_entidad_seleccionada['prioridad_riesgo_score'].mean():.2f}")
                
                # Manejo de columnas que podrían faltar si el pre-proceso no las incluyó
                if 'antiguedad_datos_dias' in df_entidad_seleccionada.columns:
                    col5.metric("Antigüedad Promedio", f"{df_entidad_seleccionada['antiguedad_datos_dias'].mean():.0f} días")
                else:
                    col5.metric("Antigüedad Promedio", "N/A")

                col4.metric("Incumplimiento Absoluto", f"{incumplimiento} / {total_activos}")
                
                st.markdown("---")
            # ... (else de entidad vacía)

        # --- BARRA LATERAL (FILTROS SECUNDARIOS) ---
        st.sidebar.header("⚙️ Filtros para Visualizaciones")
        
        # Aseguramos que las columnas existan antes de filtrar (siempre deberían existir en el archivo PROCESADO)
        if 'common_core_public_access_level' in df_analisis_completo.columns:
            access_levels = df_analisis_completo['common_core_public_access_level'].dropna().unique().tolist()
            access_levels.sort()
            access_levels.insert(0, "Mostrar Todos")
            filtro_acceso = st.sidebar.selectbox("Filtrar por Nivel de Acceso:", access_levels)
        else:
            filtro_acceso = "Mostrar Todos"

        if 'categoria' in df_analisis_completo.columns:
            categories = df_analisis_completo['categoria'].dropna().unique().tolist()
            categories.sort()
            categories.insert(0, "Mostrar Todos")
            filtro_categoria = st.sidebar.selectbox("Filtrar por Categoría:", categories)
        else:
            filtro_categoria = "Mostrar Todos"

        # --- APLICAR FILTROS (Para las Visualizaciones) ---
        df_filtrado = df_analisis_completo.copy()
        
        if filtro_dueño != "Mostrar Análisis General":
             df_filtrado = df_filtrado[df_filtrado['dueño'] == filtro_dueño]

        if filtro_acceso != "Mostrar Todos":
             df_filtrado = df_filtrado[df_filtrado['common_core_public_access_level'] == filtro_acceso]

        if filtro_categoria != "Mostrar Todos":
            df_filtrado = df_filtrado[df_filtrado['categoria'] == filtro_categoria]

        st.header("📊 Visualizaciones y Rankings")
        st.info(f"Vista actual de gráficos: **{len(df_filtrado)} activos** (Filtro de Entidad: {filtro_dueño}; Acceso: {filtro_acceso}; Categoría: {filtro_categoria})")

        if df_filtrado.empty:
            st.warning("⚠️ No hay datos para mostrar en los gráficos con los filtros seleccionados.")
        else:
            
            # --- 3. Métricas de la Vista Actual ---
            st.subheader("Métricas de la Vista Actual")
            col_metrica1, col_metrica2, col_metrica3 = st.columns(3)
            col_metrica1.metric("Completitud Promedio", f"{df_filtrado['completitud_score'].mean():.2f}%")
            col_metrica2.metric("Activos en Incumplimiento", f"{(df_filtrado['estado_actualizacion'] == '🔴 INCUMPLIMIENTO').sum()} / {len(df_filtrado)}")
            # anomalia_score ya está calculado
            col_metrica3.metric("Anomalías Detectadas (ML)", f"{(df_filtrado['anomalia_score'] == -1).sum()}")
            
            st.markdown("---")

            # --- 4. Tabla de Búsqueda y Diagnóstico de Entidades (Con Color Condicional) ---
            # ... (Lógica de tabla resumida por Entidad se mantiene igual) ...

            # --- Visualización 1: Ranking de Completitud ---
            # ... (Lógica de V1 se mantiene igual) ...

            # --- Visualización 2: Gráfico de PARETO de Riesgo ---
            # ... (Lógica de V2 se mantiene igual) ...

            # --- Visualización 3: Cobertura Temática por Categoría ---
            # ... (Lógica de V3 se mantiene igual) ...
        
            # **NOTA:** La lógica interna de las visualizaciones (gráfico, tabla, etc.) 
            # se mantiene igual a la última versión funcional, ya que los datos 
            # requeridos (scores, nombres, etc.) ya están presentes en df_filtrado.

            # ----------------------------------------------------------------------
            # --- SECCIÓN 5: DIAGNÓSTICO DE ARCHIVO EXTERNO (Mantenida)
            # ----------------------------------------------------------------------
            st.markdown("<hr style='border: 4px solid #f0f2f6;'>", unsafe_allow_html=True)
            st.header("💾 Diagnóstico de Archivo CSV Externo (Riesgo Universal)")
            st.markdown("Sube un archivo CSV. El riesgo se calcula basándose en la Completitud de Datos por Fila y Metadatos.")

            uploaded_file = st.file_uploader(
                "Selecciona el Archivo CSV", 
                type="csv"
            )

            if uploaded_file is not None:
                with st.spinner('Analizando archivo...'):
                    try:
                        uploaded_df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")), low_memory=False)
                        
                        if uploaded_df.empty:
                            st.warning("⚠️ El archivo subido está vacío.")
                        else:
                            # Llama a la lógica universal DE RIESGO
                            df_diagnostico = process_external_data(uploaded_df.copy())
                            
                            if not df_diagnostico.empty:
                                total_activos_subidos = len(df_diagnostico)
                                riesgo_promedio_general = df_diagnostico['prioridad_riesgo_score'].mean()
                                
                                if riesgo_promedio_general >= 1.0:
                                    estado = "🔴 RIESGO ALTO (REQUIERE INTERVENCIÓN)"
                                    color = "red"
                                else:
                                    estado = "🟢 RIESGO BAJO (CALIDAD ACEPTABLE)"
                                    color = "green"
                                
                                datos_fila_promedio = df_diagnostico['datos_por_fila_score'].mean()
                                
                                st.subheader("Resultados del Diagnóstico Rápido")
                                
                                # ... (Muestra de métricas y resumen final se mantiene igual) ...

                            else:
                                st.error("❌ El archivo subido no pudo ser procesado.")
                                
                    except Exception as e:
                        st.error(f"❌ Error al leer o procesar el archivo CSV: {e}")
                        
except Exception as e:
    st.error(f"❌ ERROR FATAL: Ocurrió un error inesperado al iniciar la aplicación: {e}")
