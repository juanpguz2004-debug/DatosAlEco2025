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
# 1. Funciones de Carga y Procesamiento
# =================================================================

@st.cache_data
def load_processed_data(file_path):
    """Carga el archivo CSV YA PROCESADO y lo cachea."""
    try:
        # Cargamos el DF que ya tiene todos los scores calculados
        df = pd.read_csv(file_path, low_memory=False)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

def process_external_data(df):
    """
    Lógica de riesgo universal para el archivo externo subido. 
    Se basa en métricas de calidad de datos que aplican a cualquier DataFrame,
    independiente del esquema del Inventario Principal.
    """
    
    # --- AJUSTE PRELIMINAR: Asegurar columnas MÍNIMAS para el despliegue ---
    if 'titulo' not in df.columns:
        df['titulo'] = 'Activo sin título'
    if 'dueño' not in df.columns:
        df['dueño'] = 'Desconocido'

    # 1. Métrica Universal: Completitud de Datos por Fila (Densidad de datos)
    df['datos_por_fila_score'] = (df.notna().sum(axis=1) / df.shape[1]) * 100
    
    # 2. Métrica Universal: Completitud de Metadatos (Verifica si existen columnas básicas de descripción)
    # Usamos tres columnas comunes a cualquier metadata de catálogo como referencia base.
    campos_clave_universal = ['titulo', 'descripcion', 'dueño'] 
    
    # Calcular cuántos de estos campos existen en el DF subido
    campos_existentes = [col for col in campos_clave_universal if col in df.columns]
    num_campos_totales_base = len(campos_clave_universal) # Base 3
    
    # Calcular el score de completitud basado SOLO en los campos que existen en este archivo
    df['campos_diligenciados_universal'] = df[campos_existentes].notna().sum(axis=1)
    df['completitud_metadatos_universal'] = (df['campos_diligenciados_universal'] / num_campos_totales_base) * 100

    
    # 3. CÁLCULO DE SCORE DE RIESGO UNIVERSAL (Máximo teórico 3.5)
    
    # Penalización 1: Score bajo de Datos por Fila (riesgo de datos incompletos)
    # Penaliza si la fila está menos del 70% llena (Riesgo 2.0)
    df['riesgo_datos_incompletos'] = np.where(df['datos_por_fila_score'] < 70, 2.0, 0.0)
    
    # Penalización 2: Metadatos insuficientes (Riesgo 1.5)
    # Penaliza si la completitud de metadatos universal está por debajo del 50%.
    df['riesgo_metadatos_nulo'] = np.where(df['completitud_metadatos_universal'] < 50, 1.5, 0.0)
    
    # Score de riesgo universal
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
        st.error(f"🛑 Error: No se pudo cargar el archivo **{ARCHIVO_PROCESADO}**. Asegúrate de que existe y se ejecutó `preprocess.py`.")
    else:
        st.success(f'✅ Archivo pre-procesado cargado. Total de activos: **{len(df_analisis_completo)}**')

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

        # --- BARRA LATERAL (FILTROS SECUNDARIOS) ---
        st.sidebar.header("⚙️ Filtros para Visualizaciones")
        
        filtro_acceso = "Mostrar Todos"
        if 'common_core_public_access_level' in df_analisis_completo.columns:
            access_levels = df_analisis_completo['common_core_public_access_level'].dropna().unique().tolist()
            access_levels.sort()
            access_levels.insert(0, "Mostrar Todos")
            filtro_acceso = st.sidebar.selectbox("Filtrar por Nivel de Acceso:", access_levels)

        filtro_categoria = "Mostrar Todos"
        if 'categoria' in df_analisis_completo.columns:
            categories = df_analisis_completo['categoria'].dropna().unique().tolist()
            categories.sort()
            categories.insert(0, "Mostrar Todos")
            filtro_categoria = st.sidebar.selectbox("Filtrar por Categoría:", categories)


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
            col_metrica3.metric("Anomalías Detectadas (ML)", f"{(df_filtrado['anomalia_score'] == -1).sum()}")
            
            st.markdown("---")

            # --- 4. Tabla de Búsqueda y Diagnóstico de Entidades (Con Color Condicional) ---
            st.header("🔍 4. Tabla de Búsqueda y Diagnóstico de Entidades")

            st.info(f"""
                La columna **Riesgo Promedio** tiene un formato de color:
                * 🟢 **Verde:** El riesgo promedio es **menor o igual a {UMBRAL_RIESGO_ALTO}**. Intervención no urgente.
                * 🔴 **Rojo:** El riesgo promedio es **mayor a {UMBRAL_RIESGO_ALTO}**. Se requiere **intervención/actualización prioritaria**.
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
                    'Riesgo_Promedio': st.column_config.NumberColumn("Riesgo Promedio (Score)", help=f"Rojo > {UMBRAL_RIESGO_ALTO}."),
                    'Completitud_Promedio': st.column_config.NumberColumn("Completitud Promedio", format="%.2f%%"),
                    'Antiguedad_Promedio_Dias': st.column_config.NumberColumn("Antigüedad Promedio (Días)", format="%d"),
                    'Incumplimiento_Absoluto': st.column_config.NumberColumn("Activos en Incumplimiento (Count)"),
                    '%_Incumplimiento': st.column_config.TextColumn("% Incumplimiento")
                },
                hide_index=True
            )

            st.markdown("---")

            # --- Visualización 1: Ranking de Completitud (Peor Rendimiento) ---
            st.subheader("1. 📉 Ranking de Entidades por Completitud Promedio (Peor Rendimiento)")
            
            try:
                COLUMNA_ENTIDAD = 'dueño'
                resumen_completitud = df_filtrado.groupby(COLUMNA_ENTIDAD).agg(
                    Total_Activos=('uid', 'count'),
                    Completitud_Promedio=('completitud_score', 'mean')
                ).reset_index()
                
                entidades_volumen = resumen_completitud[resumen_completitud['Total_Activos'] >= 5]
                df_top_10_peor_completitud = entidades_volumen.sort_values(by='Completitud_Promedio', ascending=True).head(10)
                
                if not df_top_10_peor_completitud.empty:
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Completitud_Promedio', y=COLUMNA_ENTIDAD, data=df_top_10_peor_completitud, palette='Reds_r', ax=ax1)
                    ax1.set_title('Top 10 Entidades con Peor Completitud Promedio', fontsize=14)
                    ax1.set_xlabel('Score de Completitud Promedio (%)', fontsize=12)
                    st.pyplot(fig1)
                else:
                    st.warning("No hay entidades con suficiente volumen (>= 5 activos) para generar el ranking.")
            except Exception as e:
                st.error(f"❌ ERROR [Visualización 1]: Falló la generación del Gráfico de Completitud. Detalle: {e}")

            st.markdown("---")

            # --- Visualización 2: Gráfico de PARETO de Riesgo (Activos más Críticos) ---
            st.subheader("2. 🎯 Gráfico de Pareto de Riesgo (Activos más Críticos)")
            
            try:
                df_riesgo = df_filtrado[df_filtrado['prioridad_riesgo_score'] > 0].sort_values(by='prioridad_riesgo_score', ascending=False).copy()
                
                if not df_riesgo.empty:
                    df_riesgo['Riesgo Acumulado'] = df_riesgo['prioridad_riesgo_score'].cumsum()
                    df_riesgo['Riesgo Total'] = df_riesgo['prioridad_riesgo_score'].sum()
                    df_riesgo['% Riesgo Acumulado'] = (df_riesgo['Riesgo Acumulado'] / df_riesgo['Riesgo Total']) * 100
                    
                    fig2, ax2 = plt.subplots(figsize=(12, 6))
                    ax2.bar(df_riesgo.index, df_riesgo['prioridad_riesgo_score'], color="C0")
                    ax3 = ax2.twinx()
                    ax3.plot(df_riesgo.index, df_riesgo["% Riesgo Acumulado"], color="C1", marker="D", ms=4)
                    ax3.yaxis.set_major_formatter(PercentFormatter())
                    ax3.axhline(80, color='red', linestyle='--', alpha=0.7)
                    ax2.set_title("Gráfico de Pareto: Concentración de Riesgo por Activo", fontsize=16)
                    plt.tight_layout()
                    st.pyplot(fig2)
                else:
                    st.warning("No hay activos con score de riesgo > 0 en la vista actual para generar el Pareto.")

            except Exception as e:
                st.error(f"❌ ERROR [Visualización 2]: Falló la generación del Gráfico de Pareto. Detalle: {e}")
            
            st.markdown("---")

            # --- Visualización 3: Cobertura Temática por Categoría ---
            st.subheader("3. 🗺️ Cobertura Temática por Categoría")
            
            try:
                COLUMNA_CATEGORIA = 'categoria'
                if COLUMNA_CATEGORIA in df_filtrado.columns:
                    conteo_categoria = df_filtrado[COLUMNA_CATEGORIA].value_counts().head(10)
                else:
                    conteo_categoria = pd.Series([], dtype='int')

                if not conteo_categoria.empty:
                    fig3, ax3 = plt.subplots(figsize=(10, 7))
                    sns.barplot(x=conteo_categoria.values, y=conteo_categoria.index, palette='viridis', ax=ax3)
                    ax3.set_title('Top 10 Categorías con Mayor Cobertura Temática', fontsize=16)
                    ax3.set_xlabel('Número de Activos', fontsize=12)
                    st.pyplot(fig3)
                else:
                    st.warning("La columna 'categoria' no contiene suficientes valores para generar la visualización.")
            except Exception as e:
                st.error(f"❌ ERROR [Visualización 3]: Falló la generación del Bar Plot de Categorías. Detalle: {e}")
            
            st.markdown("---")
        
        # ----------------------------------------------------------------------
        # --- SECCIÓN 5: DIAGNÓSTICO DE ARCHIVO EXTERNO
        # ----------------------------------------------------------------------
        st.markdown("<hr style='border: 4px solid #f0f2f6;'>", unsafe_allow_html=True)
        st.header("💾 Diagnóstico de Archivo CSV Externo (Riesgo Universal)")
        st.markdown("Sube un archivo CSV. El riesgo se calcula basándose en la Completitud de Datos por Fila y Metadatos UNIVERSALES.")

        uploaded_file = st.file_uploader(
            "Selecciona el Archivo CSV", 
            type="csv"
        )

        if uploaded_file is not None:
            with st.spinner('Analizando archivo...'):
                try:
                    uploaded_filename = uploaded_file.name
                    uploaded_df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")), low_memory=False)
                    
                    if uploaded_df.empty:
                        st.warning(f"⚠️ El archivo subido **{uploaded_filename}** está vacío.")
                    else:
                        # Llama a la lógica universal DE RIESGO (Agnóstica al esquema)
                        df_diagnostico = process_external_data(uploaded_df.copy())
                        
                        if not df_diagnostico.empty:
                            total_activos_subidos = len(df_diagnostico)
                            riesgo_promedio_general = df_diagnostico['prioridad_riesgo_score'].mean()
                            
                            # Usamos la métrica universal
                            completitud_universal_promedio = df_diagnostico['completitud_metadatos_universal'].mean()
                            
                            if riesgo_promedio_general >= 1.0:
                                estado = "🔴 RIESGO ALTO (REQUIERE INTERVENCIÓN)"
                                color = "red"
                            elif riesgo_promedio_general > 0.0:
                                estado = "🟡 RIESGO MEDIO (ATENCIÓN REQUERIDA)"
                                color = "orange"
                            else:
                                estado = "🟢 RIESGO BAJO (CALIDAD ACEPTABLE)"
                                color = "green"
                            
                            datos_fila_promedio = df_diagnostico['datos_por_fila_score'].mean()
                            
                            st.subheader("Resultados del Diagnóstico Rápido")
                            
                            # --- DESPLIEGUE DE MÉTRICAS ---
                            col_info1, col_info2, col_info3 = st.columns(3)
                            col_info1.metric("Activos Analizados", total_activos_subidos)
                            col_info2.metric("Completitud de Metadatos", f"{completitud_universal_promedio:.2f}%") 
                            col_info3.metric("Riesgo Promedio Universal", f"{riesgo_promedio_general:.2f}")

                            st.markdown(f"""
                                <div style='border: 2px solid {color}; padding: 15px; border-radius: 5px; background-color: #f9f9f9;'>
                                    <h4 style='color: {color}; margin-top: 0;'>Diagnóstico General: {estado}</h4>
                                    <p>El Score de Riesgo Universal de **{riesgo_promedio_general:.2f}** indica la prioridad de intervención. (Máximo teórico: 3.5)</p>
                                    <p><b>Promedio de Datos por Fila:</b> {datos_fila_promedio:.2f}% (Indica cuántas celdas están llenas).</p>
                                </div>
                            """, unsafe_allow_html=True)

                            st.markdown("---")
                            st.subheader("Desglose de Calidad de las Filas (Top 10 Riesgo)")
                            
                            # Función de estilo para garantizar que el texto sea negro (Arreglo de Visibilidad)
                            def make_text_black(s):
                                return ['color: black' for v in s]

                            cols_diagnostico = ['prioridad_riesgo_score', 'datos_por_fila_score', 'riesgo_datos_incompletos', 'riesgo_metadatos_nulo']
                            df_cols_disponibles = df_diagnostico[[col for col in cols_diagnostico if col in df_diagnostico.columns]]
                            
                            st.dataframe(
                                df_cols_disponibles.sort_values(by='prioridad_riesgo_score', ascending=False).head(10).style.apply(make_text_black, axis=1), 
                                use_container_width=True
                            )

                        else:
                            st.error(f"❌ El archivo subido **{uploaded_filename}** no pudo ser procesado.")
                            
                except Exception as e:
                    st.error(f"❌ Error al leer o procesar el archivo CSV: {e}")
                    
except Exception as e:
    st.error(f"❌ ERROR FATAL: Ocurrió un error inesperado al iniciar la aplicación: {e}")
