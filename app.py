import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from datetime import datetime, date 
from matplotlib.ticker import PercentFormatter
import io 

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Diagnóstico de Activos",
    layout="wide"
)

# --- Variables Globales ---
ARCHIVO_CSV = "Asset_Inventory_-_Public_20251118.csv"
UMBRAL_RIESGO_ALTO = 1.0 

# =================================================================
# 1. Funciones de Soporte
# =================================================================

def clean_col_name(col):
    """Limpia y normaliza los nombres de las columnas."""
    name = col.lower().strip()
    name = name.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
    name = name.replace(' ', '_').replace('.', '').replace('/', '_').replace(':', '').replace('(', '').replace(')', '')
    return name

def calculate_antiguedad_y_estado(df_temp):
    """Calcula la antigüedad y el estado de cumplimiento (coherencia) para el INVENTARIO PRINCIPAL."""
    try:
        COL_FECHA_ACTUALIZACION = 'fecha_de_ultima_actualizacion_de_datos_utc'
        COL_FRECUENCIA = 'informacion_de_datos_frecuencia_de_actualizacion'

        df_temp[COL_FECHA_ACTUALIZACION] = pd.to_datetime(df_temp.get(COL_FECHA_ACTUALIZACION), errors='coerce', utc=True)
        
        hoy = pd.to_datetime(datetime.now().date(), utc=True)
        df_temp['antiguedad_datos_dias'] = (hoy - df_temp[COL_FECHA_ACTUALIZACION]).dt.days
        
        mapa_frecuencia = {
            'diario': 1, 'semanal': 7, 'quincenal': 15, 'mensual': 30, 
            'trimestral': 90, 'semestral': 180, 'anual': 365
        }
        df_temp['frecuencia_esperada_dias'] = df_temp[COL_FRECUENCIA].astype(str).str.lower().str.strip().map(mapa_frecuencia).fillna(9999)

        df_temp['estado_actualizacion'] = np.where(
            (df_temp['antiguedad_datos_dias'].fillna(9999) > df_temp['frecuencia_esperada_dias']) & 
            (df_temp['frecuencia_esperada_dias'] < 9999), 
            '🔴 INCUMPLIMIENTO', 
            '🟢 CUMPLE'
        )
        return df_temp
    except KeyError:
        # Si el archivo subido no tiene las columnas, se establecen valores por defecto.
        df_temp['antiguedad_datos_dias'] = 9999
        df_temp['estado_actualizacion'] = 'NO APLICA'
        return df_temp
    except Exception as e:
        st.error(f"❌ ERROR INESPERADO [Paso Antigüedad]: Falló el cálculo de antigüedad: {e}.")
        raise

# =================================================================
# 2. Funciones de Carga y Procesamiento (Cacheado)
# =================================================================

@st.cache_data
def load_data(file_path):
    """Carga el archivo CSV del inventario principal y lo cachea."""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def process_data(df, is_external_file=False):
    """Aplica todos los cálculos de métricas y scores de riesgo."""
    
    # 1. Limpieza de nombres de columnas
    df.columns = [clean_col_name(col) for col in df.columns]

    # --- Popularidad (Vistas/Descargas) CORREGIDO ---
    try:
        # Función utilitaria para obtener Series con valor 0 si la columna no existe
        def get_series(col_name, df_local):
            if col_name in df_local.columns:
                return df_local[col_name]
            else:
                return pd.Series(0, index=df_local.index)

        df['vistas'] = pd.to_numeric(get_series('vistas', df), errors='coerce').fillna(0)
        df['descargas'] = pd.to_numeric(get_series('descargas', df), errors='coerce').fillna(0)
        df['popularidad_score'] = df['vistas'] + df['descargas'] 
    except Exception as e:
        st.error(f"❌ ERROR [Paso Popularidad]: Falló la conversión o suma de 'vistas'/'descargas'. Detalle: {e}")
        return pd.DataFrame() 

    # 2. Antigüedad y Estado de Actualización
    try:
        df = calculate_antiguedad_y_estado(df.copy()) 
    except Exception:
        return pd.DataFrame() 
    
    # 3. Completitud de Metadatos (Se usa la base de 10 campos para el inventario principal)
    try:
        campos_minimos = [
            'titulo', 'descripcion', 'dueño', 'correo_electronico_de_contacto',
            'licencia', 'dominio', 'categoria', 'informacion_de_datos_frecuencia_de_actualizacion',
            'common_core_public_access_level', 'informacion_de_datos_cobertura_geografica'
        ]
        campos_existentes = [col for col in campos_minimos if col in df.columns]
        num_campos_totales_base = len(campos_minimos)
        
        if num_campos_totales_base == 0:
            df['completitud_score'] = 0
        else:
            df['campos_diligenciados'] = df[campos_existentes].notna().sum(axis=1)
            df['completitud_score'] = (df['campos_diligenciados'] / num_campos_totales_base) * 100
    except Exception as e:
        st.error(f"❌ ERROR [Paso Completitud]: Falló el cálculo de 'completitud_score'. Detalle: {e}")
        return pd.DataFrame()
    
    # 4. Detección de Anomalías (Isolation Forest RESTAURADO)
    df['anomalia_score'] = 0 
    if not is_external_file: # Solo ejecutamos en el inventario principal
        try:
            df_modelo = df[(df['antiguedad_datos_dias'] < 9999) & (df['popularidad_score'] > 0)].copy()
            
            if not df_modelo.empty and len(df_modelo) > 1: 
                features = df_modelo[['antiguedad_datos_dias', 'popularidad_score', 'completitud_score']]
                model = IsolationForest(contamination=0.01, random_state=42)
                model.fit(features)
                anomalias = model.predict(features)
                df.loc[df_modelo.index, 'anomalia_score'] = np.where(anomalias == -1, -1, 0)
        except Exception as e:
            # Captura de errores para evitar que caiga toda la app por el modelo ML
            st.warning(f"⚠️ Aviso [Paso Anomalías]: El modelo Isolation Forest no se ejecutó. Detalle: {e}")
            df['anomalia_score'] = 0


    # 5. CÁLCULO DE SCORE DE RIESGO/PRIORIDAD
    if is_external_file:
        # --- LÓGICA DE RIESGO UNIVERSAL PARA ARCHIVO SUBIDO ---
        df['datos_por_fila_score'] = (df.notna().sum(axis=1) / df.shape[1]) * 100
        
        # Penalización 1: Score bajo de Datos por Fila (riesgo de datos incompletos)
        df['riesgo_datos_incompletos'] = np.where(df['datos_por_fila_score'] < 70, 2.0, 0.0)
        
        # Penalización 2: Completitud de Metadatos (se asume 0 si las columnas no están, por lo que un score < 10% es penalizado)
        df['riesgo_metadatos_nulo'] = np.where(df['completitud_score'] < 10, 1.0, 0.0)
        
        # El score de riesgo para el archivo externo es la suma de estas dos métricas universales
        df['prioridad_riesgo_score'] = df['riesgo_datos_incompletos'] + df['riesgo_metadatos_nulo']
        
    else:
        # --- LÓGICA DE RIESGO COMPLEJA PARA EL INVENTARIO PRINCIPAL ---
        try:
            max_popularidad = df['popularidad_score'].max()
            max_popularidad = max_popularidad if max_popularidad > 0 else 1 

            df['riesgo_incumplimiento'] = np.where(df['estado_actualizacion'] == '🔴 INCUMPLIMIENTO', 3.0, 0.0)
            df['riesgo_completitud'] = np.where(df['completitud_score'] < 50, 1.5, 0.0)
            df['riesgo_demanda'] = (df['popularidad_score'] / max_popularidad) * 1.0 if max_popularidad > 1 else 0.0
            df['riesgo_anomalia'] = np.where(df['anomalia_score'] == -1, 2.0, 0.0)
            
            df['prioridad_riesgo_score'] = (
                df['riesgo_incumplimiento'] +
                df['riesgo_completitud'] +
                df['riesgo_demanda'] +
                df['riesgo_anomalia']
            )
        except Exception as e:
            st.error(f"❌ ERROR [Paso Score Riesgo Principal]: Falló el cálculo del score final. Detalle: {e}")
            return pd.DataFrame() 
    
    return df

# =================================================================
# 3. Ejecución Principal
# =================================================================

st.title("📊 Dashboard de Priorización de Activos de Datos (Análisis Completo)")

try:
    with st.spinner(f'Cargando archivo: **{ARCHIVO_CSV}**...'):
        df = load_data(ARCHIVO_CSV) 

    if df.empty:
        st.error(f"🛑 Error: No se pudo cargar el archivo **{ARCHIVO_CSV}**. Asegúrate de que existe.")
    else:
        with st.spinner('Procesando datos y calculando métricas...'):
            # Llama a process_data con el valor por defecto is_external_file=False
            df_analisis_completo = process_data(df.copy()) 
            
        if df_analisis_completo.empty:
            st.error("🛑 Proceso de datos detenido debido a errores en los cálculos.")
        else:
            st.success(f'✅ Archivo **{ARCHIVO_CSV}** cargado y procesamiento completado. Total de activos: **{len(df_analisis_completo)}**')

            # --- SECCIÓN DE SELECCIÓN Y DESGLOSE DE ENTIDAD ---
            # ... (Lógica de selección de entidad, filtros y métricas se mantiene) ...

            owners = df_analisis_completo['dueño'].dropna().unique().tolist()
            owners.sort()
            owners.insert(0, "Mostrar Análisis General")
            
            filtro_dueño = st.selectbox(
                "Selecciona una Entidad para ver su Desglose de Estadísticas:",
                owners
            )
            
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
                    col5.metric("Antigüedad Promedio", f"{df_entidad_seleccionada['antiguedad_datos_dias'].mean():.0f} días")
                    
                    st.markdown("---")
                else:
                    st.warning(f"⚠️ No se encontraron activos para la entidad: {filtro_dueño}")
                    st.markdown("---")

            # --- BARRA LATERAL (FILTROS SECUNDARIOS) ---
            st.sidebar.header("⚙️ Filtros para Visualizaciones")
            
            access_levels = df_analisis_completo['common_core_public_access_level'].dropna().unique().tolist()
            access_levels.sort()
            access_levels.insert(0, "Mostrar Todos")
            
            filtro_acceso = st.sidebar.selectbox("Filtrar por Nivel de Acceso:", access_levels)
            
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

                # --- Visualización 1: Ranking de Completitud ---
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

                # --- Visualización 2: Gráfico de PARETO de Riesgo ---
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
                    conteo_categoria = df_filtrado[COLUMNA_CATEGORIA].value_counts().head(10)
                    
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
            st.markdown("""
                Sube un archivo CSV. El riesgo se calcula basándose en dos métricas universales de calidad de datos:
                1.  **Datos por Fila:** Porcentaje de celdas completas en el archivo.
                2.  **Completitud de Metadatos:** Presencia de columnas de metadatos clave (si aplican).
            """)

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
                            # Llama a process_data con is_external_file=True
                            df_diagnostico = process_data(uploaded_df.copy(), is_external_file=True)
                            
                            if not df_diagnostico.empty:
                                total_activos_subidos = len(df_diagnostico)
                                riesgo_promedio_general = df_diagnostico['prioridad_riesgo_score'].mean()
                                
                                # El umbral de riesgo universal es 1.0 (máximo riesgo 3.0)
                                if riesgo_promedio_general >= 1.0:
                                    estado = "🔴 RIESGO ALTO (REQUIERE INTERVENCIÓN)"
                                    color = "red"
                                else:
                                    estado = "🟢 RIESGO BAJO (CALIDAD ACEPTABLE)"
                                    color = "green"
                                
                                # Nueva métrica de datos por fila (media)
                                datos_fila_promedio = df_diagnostico['datos_por_fila_score'].mean()
                                
                                st.subheader("Resultados del Diagnóstico Rápido")
                                
                                col_info1, col_info2, col_info3 = st.columns(3)
                                col_info1.metric("Activos Analizados", total_activos_subidos)
                                col_info2.metric("Completitud de Metadatos", f"{df_diagnostico['completitud_score'].mean():.2f}%")
                                col_info3.metric("Riesgo Promedio Universal", f"{riesgo_promedio_general:.2f}")

                                st.markdown(f"""
                                    <div style='border: 2px solid {color}; padding: 15px; border-radius: 5px; background-color: #f9f9f9;'>
                                        <h4 style='color: {color}; margin-top: 0;'>Diagnóstico General: {estado}</h4>
                                        <p>El Score de Riesgo Universal de **{riesgo_promedio_general:.2f}** indica la prioridad de intervención. (Máximo teórico: 3.0)</p>
                                        <p><b>Promedio de Datos por Fila:</b> {datos_fila_promedio:.2f}% (Indica cuántas celdas están llenas).</p>
                                    </div>
                                """, unsafe_allow_html=True)

                                st.markdown("---")
                                st.subheader("Desglose de Calidad de las Filas (Top 10 Riesgo)")
                                # Usamos el riesgo universal y el score de datos por fila
                                st.dataframe(df_diagnostico[['prioridad_riesgo_score', 'datos_por_fila_score', 'riesgo_datos_incompletos', 'riesgo_metadatos_nulo']].sort_values(by='prioridad_riesgo_score', ascending=False).head(10), use_container_width=True)

                            else:
                                st.error("❌ El archivo subido no pudo ser procesado. Revisa los mensajes de error anteriores.")
                                
                    except Exception as e:
                        st.error(f"❌ Error al leer o procesar el archivo CSV: {e}")
                        
except Exception as e:
    st.error(f"❌ ERROR FATAL: Ocurrió un error inesperado al iniciar la aplicación: {e}")
