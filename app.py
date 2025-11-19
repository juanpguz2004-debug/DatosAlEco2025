import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import io 
from datetime import datetime
import re 
import warnings
warnings.filterwarnings('ignore') # Ocultar advertencias de Pandas/Streamlit

# --- Variables Globales ---
ARCHIVO_PROCESADO = "Asset_Inventory_PROCESSED.csv" 
# CRITERIO CORREGIDO: Subimos el umbral a 3.0 para balancear la nueva escala de riesgo (~7.5 máx)
UMBRAL_RIESGO_ALTO = 3.0 

# --- CONFIGURACIÓN DE RIESGOS UNIVERSALES (DEBE COINCIDIR CON PREPROCESS.PY) ---
PENALIZACION_DATOS_INCOMPLETOS = 2.0  
PENALIZACION_METADATOS_NULOS = 1.5      
PENALIZACION_INCONSISTENCIA_TIPO = 0.5   
PENALIZACION_DUPLICADO = 1.0             
# Suma del riesgo universal máximo posible: 2.0 + 1.5 + 0.5 + 1.0 = 5.0
RIESGO_MAXIMO_TEORICO_UNIVERSAL = 5.0 

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
        
    # --- 5. UNICIDAD: Duplicados Exactos ---
    df['es_duplicado'] = df.duplicated(keep=False) 
    df['riesgo_duplicado'] = np.where(
        df['es_duplicado'], PENALIZACION_DUPLICADO, 0.0
    )
    
    return df

def process_external_data(df):
    """
    Lógica de riesgo universal para el archivo externo subido.
    """
    
    # --- 1. EVALUACIÓN DE UNIVERSALES (Completitud, Consistencia, Unicidad) ---
    df = check_universals_external(df)
    
    # --- 2. EVALUACIÓN DE METADATOS A NIVEL DE ARCHIVO ---
    
    campos_clave_universal = ['titulo', 'descripcion', 'dueño'] 
    riesgo_metadatos = 0.0
    
    campos_existentes_y_llenos = 0
    num_campos_totales_base = len(campos_clave_universal)

    for campo in campos_clave_universal:
        # Penalización si la columna no existe O si existe pero el primer valor es nulo
        if campo not in df.columns or pd.isna(df[campo].iloc[0]):
            riesgo_metadatos = PENALIZACION_METADATOS_NULOS
        else:
            campos_existentes_y_llenos += 1
            
    df['riesgo_metadatos_nulo'] = riesgo_metadatos
    completitud_metadatos_universal = (campos_existentes_y_llenos / num_campos_totales_base) * 100
    df['completitud_metadatos_universal'] = completitud_metadatos_universal
    
    # --- 3. CÁLCULO FINAL DE RIESGO Y CALIDAD ---
    
    # Score de riesgo universal
    df['prioridad_riesgo_score'] = (
        df['riesgo_datos_incompletos'] + 
        df['riesgo_metadatos_nulo'] +
        df['riesgo_consistencia_tipo'] +
        df['riesgo_duplicado']
    )
    
    # CÁLCULO DE CALIDAD TOTAL DEL ARCHIVO (0% a 100%)
    # Usamos la media del riesgo para la calidad total del archivo
    avg_file_risk = df['prioridad_riesgo_score'].mean()
    quality_score = 100 - (avg_file_risk / RIESGO_MAXIMO_TEORICO_UNIVERSAL * 100)
    
    df['calidad_total_score'] = np.clip(quality_score, 0, 100)

    return df


# =================================================================
# 2. Ejecución Principal del Dashboard
# =================================================================

st.title("📊 Dashboard de Priorización de Activos de Datos (Análisis Completo)")

try:
    with st.spinner(f'Cargando archivo procesado: **{ARCHIVO_PROCESADO}**...'):
        df_analisis_completo = load_processed_data(ARCHIVO_PROCESADO) 

    if df_analisis_completo.empty:
        st.error(f"🛑 Error: No se pudo cargar el archivo **{ARCHIVO_PROCESADO}**. Asegúrate de que existe y se ejecutó `preprocess.py`.")
    else:
        st.success(f'✅ Archivo pre-procesado cargado. Total de activos: **{len(df_analisis_completo)}**')

        # [ ... Bloque de selección de entidad, filtros y visualizaciones se mantiene sin cambios ... ]
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

            # TEXTO CORREGIDO PARA EL NUEVO UMBRAL
            st.info(f"""
                La columna **Riesgo Promedio** tiene un formato de color:
                * 🟢 **Verde:** El riesgo promedio es **menor o igual a {UMBRAL_RIESGO_ALTO:.1f}**. Intervención no urgente.
                * 🔴 **Rojo:** El riesgo promedio es **mayor a {UMBRAL_RIESGO_ALTO:.1f}**. Se requiere **intervención/actualización prioritaria**.
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
                # Aplicamos el nuevo umbral 
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
                    'Riesgo_Promedio': st.column_config.NumberColumn("Riesgo Promedio (Score)", help=f"Rojo > {UMBRAL_RIESGO_ALTO:.1f}."),
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
        st.header("💾 Diagnóstico de Archivo CSV Externo (Calidad Universal)")
        st.markdown(f"Sube un archivo CSV. La **Calidad Total** se calcula en base a las 4 dimensiones universales principales (Riesgo Máximo: **{RIESGO_MAXIMO_TEORICO_UNIVERSAL:.1f}**).")

        uploaded_file = st.file_uploader(
            "Selecciona el Archivo CSV", 
            type="csv"
        )

        if uploaded_file is not None:
            with st.spinner('Analizando archivo...'):
                try:
                    uploaded_filename = uploaded_file.name
                    # Intentar lectura robusta (asumiendo que podría usar otro delimitador)
                    uploaded_df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")), low_memory=False)
                    if len(uploaded_df.columns) <= 1:
                        uploaded_file.seek(0)
                        uploaded_df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")), low_memory=False, sep=';')
                        if len(uploaded_df.columns) <= 1:
                            uploaded_file.seek(0)
                            uploaded_df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")), low_memory=False, sep='\t')


                    if uploaded_df.empty:
                        st.warning(f"⚠️ El archivo subido **{uploaded_filename}** está vacío.")
                    else:
                        df_diagnostico = process_external_data(uploaded_df.copy())
                        
                        if not df_diagnostico.empty:
                            
                            # Métricas consolidadas
                            calidad_total_final = df_diagnostico['calidad_total_score'].iloc[0] 
                            completitud_universal_promedio = df_diagnostico['completitud_metadatos_universal'].iloc[0] 
                            datos_fila_promedio = df_diagnostico['datos_por_fila_score'].mean()
                            riesgo_promedio_total = df_diagnostico['prioridad_riesgo_score'].mean()

                            # Desglose de Riesgos Promedio
                            riesgos_reporte = pd.DataFrame({
                                'Dimensión de Riesgo': [
                                    '1. Datos Incompletos (Completitud)',
                                    '2. Metadatos Faltantes (Gobernanza)',
                                    '3. Duplicados Exactos (Unicidad)',
                                    '4. Consistencia de Tipo (Coherencia)',
                                ],
                                'Riesgo Promedio (0-Máx)': [
                                    df_diagnostico['riesgo_datos_incompletos'].mean(),
                                    df_diagnostico['riesgo_metadatos_nulo'].mean(),
                                    df_diagnostico['riesgo_duplicado'].mean(),
                                    df_diagnostico['riesgo_consistencia_tipo'].mean(),
                                ]
                            })
                            riesgos_reporte = riesgos_reporte.sort_values(by='Riesgo Promedio (0-Máx)', ascending=False)
                            riesgos_reporte['Riesgo Promedio (0-Máx)'] = riesgos_reporte['Riesgo Promedio (0-Máx)'].round(2)
                            
                            
                            # === LÓGICA DE RECOMENDACIÓN PRÁCTICA (Final) ===
                            
                            recomendacion_lista = []
                            
                            if riesgos_reporte.iloc[0]['Riesgo Promedio (0-Máx)'] > 0.15: 
                                recomendacion_lista.append(f"El riesgo más alto es por **{riesgos_reporte.iloc[0]['Dimensión de Riesgo']}** ({riesgos_reporte.iloc[0]['Riesgo Promedio (0-Máx)']:.2f}). Enfoca tu esfuerzo en corregir este problema primero.")

                            # 2. Recomendación: Metadatos (Siempre que no sea 100%)
                            if completitud_universal_promedio < 100.0:
                                recomendacion_lista.append(f"Asegure el **Metadato Universal**: La Completitud de Metadatos es de **{completitud_universal_promedio:.2f}%**. Diligencie las columnas (`titulo`, `descripcion`, `dueño`) para evitar que el activo sea huérfano.")
                            
                            if not recomendacion_lista:
                                recomendacion_final = "La **Calidad** es excelente. No se requieren mejoras prioritarias."
                                estado = "🟢 CALIDAD ALTA"
                                color = "green"
                            else:
                                recomendaciones_md = "\n".join([f"* {r}" for r in recomendacion_lista])
                                recomendacion_final = f"Para aumentar la Calidad Total, se requiere **atención prioritaria** en los siguientes aspectos:\n\n{recomendaciones_md}"
                                
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
                                riesgos_reporte,
                                use_container_width=True,
                                hide_index=True
                            )

                            st.markdown(f"#### ✨ Recomendación de Acciones:")
                            st.markdown(recomendacion_final)

                        else:
                            st.error(f"❌ El archivo subido **{uploaded_filename}** no pudo ser procesado.")
                            
                except Exception as e:
                    st.error(f"❌ Error al leer o procesar el archivo CSV: {e}")
                    
except Exception as e:
    st.error(f"❌ ERROR FATAL: Ocurrió un error inesperado al iniciar la aplicación: {e}")
