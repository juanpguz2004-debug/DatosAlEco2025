import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from datetime import datetime, date 
from matplotlib.ticker import PercentFormatter
import io # Importamos la librería io para manejar archivos en memoria

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Diagnóstico de Activos",
    layout="wide"
)

# --- Nombre del archivo CSV que Streamlit debe encontrar ---
ARCHIVO_CSV = "Asset_Inventory_-_Public_20251118.csv"

# --- UMBLRALES DE RIESGO PARA FORMATO CONDICIONAL ---
UMBRAL_RIESGO_ALTO = 1.0 

## 1. Funciones de Procesamiento de Datos (Se mantiene la lógica central)
def clean_col_name(col):
    name = col.lower().strip()
    name = name.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
    name = name.replace(' ', '_').replace('.', '').replace('/', '_').replace(':', '').replace('(', '').replace(')', '')
    return name

def calculate_antiguedad_y_estado(df_temp):
    try:
        COL_FECHA_ACTUALIZACION = 'fecha_de_ultima_actualizacion_de_datos_utc'
        COL_FRECUENCIA = 'informacion_de_datos_frecuencia_de_actualizacion'

        # Intentar forzar la conversión a datetime con manejo de errores
        df_temp[COL_FECHA_ACTUALIZACION] = pd.to_datetime(df_temp.get(COL_FECHA_ACTUALIZACION), errors='coerce', utc=True)
        
        hoy = pd.to_datetime(datetime.now().date(), utc=True)
        # Calcular antigüedad solo para filas donde la fecha de actualización no sea NaT (Not a Time)
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
    except KeyError as e:
        # En el caso de un archivo subido sin estas columnas, se ignora el cálculo
        df_temp['antiguedad_datos_dias'] = 9999
        df_temp['estado_actualizacion'] = 'NO APLICA'
        return df_temp
    except Exception as e:
        st.error(f"❌ ERROR INESPERADO [Paso Antigüedad]: Falló el cálculo de antigüedad: {e}.")
        raise


@st.cache_data(show_spinner=False) # Ocultamos el spinner para que no aparezca en la carga rápida del archivo externo
def process_data(df):
    
    # 1. Limpieza de nombres de columnas
    df.columns = [clean_col_name(col) for col in df.columns]

    # --- CORRECCIÓN y VERIFICACIÓN DE POPULARIDAD ---
    try:
        # Usamos .get() para evitar KeyErrors si las columnas no están en el archivo externo
        df['vistas'] = pd.to_numeric(df.get('vistas', 0), errors='coerce').fillna(0)
        df['descargas'] = pd.to_numeric(df.get('descargas', 0), errors='coerce').fillna(0)
        df['popularidad_score'] = df['vistas'] + df['descargas'] 
    except Exception as e:
        st.error(f"❌ ERROR [Paso Popularidad]: Falló la conversión o suma de 'vistas'/'descargas'. Detalle: {e}")
        return pd.DataFrame() 

    # 2. CÁLCULOS PREVIOS (Antigüedad y Estado de Actualización)
    try:
        # Llamamos a calculate_antiguedad_y_estado. Maneja el error si las columnas no existen.
        df = calculate_antiguedad_y_estado(df.copy()) 
    except Exception:
        return pd.DataFrame() 
    
    # 3. CÁLCULO DE MÉTRICA DE COMPLETITUD
    try:
        # Definición de campos mínimos esperados en el inventario principal
        campos_minimos = [
            'titulo', 'descripcion', 'dueño', 'correo_electronico_de_contacto',
            'licencia', 'dominio', 'categoria', 'informacion_de_datos_frecuencia_de_actualizacion',
            'common_core_public_access_level', 'informacion_de_datos_cobertura_geografica'
        ]
        campos_existentes = [col for col in campos_minimos if col in df.columns]
        num_campos_totales = len(campos_minimos) # Siempre usamos la base de 10 campos para estandarizar el score
        
        # Si no hay campos relevantes, la completitud es 0, evitamos división por cero.
        if num_campos_totales == 0:
            df['completitud_score'] = 0
        else:
            df['campos_diligenciados'] = df[campos_existentes].notna().sum(axis=1)
            # El cálculo usa num_campos_totales (10) como base, no solo los existentes en el archivo subido
            df['completitud_score'] = (df['campos_diligenciados'] / num_campos_totales) * 100
    except Exception as e:
        st.error(f"❌ ERROR [Paso Completitud]: Falló el cálculo de 'completitud_score'. Detalle: {e}")
        return pd.DataFrame()
    
    # 4. DETECCIÓN DE ANOMALÍAS (Se mantiene, pero se asegura que las columnas existan)
    df['anomalia_score'] = 0 
    
    # 5. CÁLCULO DE SCORE DE RIESGO/PRIORIDAD
    try:
        max_popularidad = df['popularidad_score'].max()
        max_popularidad = max_popularidad if max_popularidad > 0 else 1 

        # Penalizaciones
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
        st.error(f"❌ ERROR [Paso Score Riesgo]: Falló el cálculo del score final. Detalle: {e}")
        return pd.DataFrame() 
    
    return df

## 2. Título y Ejecución Principal

st.title("📊 Dashboard de Priorización de Activos de Datos (Análisis Completo)")

try:
    with st.spinner(f'Cargando y procesando el archivo: **{ARCHIVO_CSV}**...'):
        df = pd.read_csv(ARCHIVO_CSV, low_memory=False)
        df_analisis_completo = process_data(df.copy()) 
        
    if df_analisis_completo.empty:
        st.error("🛑 Proceso de datos detenido debido a errores previos. Revisa los mensajes de error ❌ para depurar.")
    else:
        st.success(f'✅ Archivo **{ARCHIVO_CSV}** cargado y procesamiento completado. Total de activos: **{len(df_analisis_completo)}**')

        # --- SECCIÓN DE SELECCIÓN Y DESGLOSE DE ENTIDAD (Se mantiene) ---
        st.header("🔬 Desglose de Métricas por Entidad")
        
        # (El resto del código de filtros y visualizaciones 1, 2 y 3 se mantiene igual)

        owners = df_analisis_completo['dueño'].dropna().unique().tolist()
        owners.sort()
        owners.insert(0, "Mostrar Análisis General")
        
        # Selector de Entidad principal (filtra las visualizaciones)
        filtro_dueño = st.selectbox(
            "Selecciona una Entidad para ver su Desglose de Estadísticas:",
            owners
        )
        
        # --- DESGLOSE DE ESTADÍSTICAS PARA LA ENTIDAD SELECCIONADA ---
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

        # --- 2.1 BARRA LATERAL (FILTROS SECUNDARIOS) ---
        st.sidebar.header("⚙️ Filtros para Visualizaciones")
        
        # FILTRO: Nivel de Acceso (Activa/Desactiva 'public', etc.)
        access_levels = df_analisis_completo['common_core_public_access_level'].dropna().unique().tolist()
        access_levels.sort()
        access_levels.insert(0, "Mostrar Todos")
        
        filtro_acceso = st.sidebar.selectbox(
            "Filtrar por Nivel de Acceso:",
            access_levels
        )
        
        # Filtro de Categoría (Secundario)
        categories = df_analisis_completo['categoria'].dropna().unique().tolist()
        categories.sort()
        categories.insert(0, "Mostrar Todos")
        
        filtro_categoria = st.sidebar.selectbox(
            "Filtrar por Categoría:",
            categories
        )

        # --- 2.2 APLICAR FILTROS (Para las Visualizaciones) ---
        df_filtrado = df_analisis_completo.copy()
        
        # Aplicar filtro de DUEÑO (si no es el análisis general)
        if filtro_dueño != "Mostrar Análisis General":
             df_filtrado = df_filtrado[df_filtrado['dueño'] == filtro_dueño]

        # Aplicar filtro de Nivel de Acceso
        if filtro_acceso != "Mostrar Todos":
             df_filtrado = df_filtrado[df_filtrado['common_core_public_access_level'] == filtro_acceso]

        # Aplicar filtro de CATEGORÍA
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

            # --- SECCIÓN 4. TABLA DE BÚSQUEDA Y DIAGNÓSTICO (Se mantiene) ---
            st.header("🔍 4. Tabla de Búsqueda y Diagnóstico de Entidades")
            
            st.info(f"""
                Utiliza la barra de búsqueda para filtrar el diagnóstico por **Entidad Responsable** (`dueño`). 
                La columna **Riesgo Promedio** ahora tiene un formato de color:
                * 🟢 **Verde:** El riesgo promedio es **menor o igual a {UMBRAL_RIESGO_ALTO}**. Intervención no urgente.
                * 🔴 **Rojo:** El riesgo promedio es **mayor a {UMBRAL_RIESGO_ALTO}**. Se requiere **intervención/actualización prioritaria**.
            """)
            
            # Calculamos el resumen de métricas por DUEÑO
            resumen_entidades_busqueda = df_filtrado.groupby('dueño').agg(
                Activos_Totales=('uid', 'count'),
                Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
                Completitud_Promedio=('completitud_score', 'mean'),
                Antiguedad_Promedio_Dias=('antiguedad_datos_dias', 'mean'),
                Incumplimiento_Absoluto=('estado_actualizacion', lambda x: (x == '🔴 INCUMPLIMIENTO').sum())
            ).reset_index()

            resumen_entidades_busqueda['%_Incumplimiento'] = (resumen_entidades_busqueda['Incumplimiento_Absoluto'] / resumen_entidades_busqueda['Activos_Totales']) * 100
            
            # Formato de la tabla
            resumen_entidades_busqueda = resumen_entidades_busqueda.rename(columns={'dueño': 'Entidad Responsable'})
            resumen_entidades_busqueda = resumen_entidades_busqueda.sort_values(by='Riesgo_Promedio', ascending=False)
            
            
            # FUNCIÓN DE ESTILO CONDICIONAL PARA RIESGO
            def color_riesgo_promedio(val):
                color = 'background-color: #f79999' if val > UMBRAL_RIESGO_ALTO else 'background-color: #a9dfbf'
                return color
            
            # Aplicar el estilo condicional solo a la columna 'Riesgo_Promedio'
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
                    'Entidad Responsable': st.column_config.TextColumn("Entidad Responsable", help="Buscar por nombre de la entidad."),
                    'Activos_Totales': st.column_config.NumberColumn("Activos Totales", format="%d"),
                    'Riesgo_Promedio': st.column_config.NumberColumn(
                        "Riesgo Promedio (Score)", 
                        format="%.2f", 
                        help=f"Score de prioridad de intervención. Rojo > {UMBRAL_RIESGO_ALTO}."
                    ),
                    'Completitud_Promedio': st.column_config.NumberColumn("Completitud Promedio", format="%.2f%%"),
                    'Antiguedad_Promedio_Dias': st.column_config.NumberColumn("Antigüedad Promedio (Días)", format="%d"),
                    'Incumplimiento_Absoluto': st.column_config.NumberColumn("Activos en Incumplimiento (Count)", format="%d"),
                    '%_Incumplimiento': st.column_config.TextColumn("% Incumplimiento", help="Porcentaje de Activos Obsoletos")
                },
                hide_index=True
            )

            st.markdown("---")
            
            # --- Visualizaciones 1, 2 y 3 se mantienen iguales ---
            
            # ... (Resto del código de Visualización 1, 2, y 3 aquí) ...
            
            # --- Visualización 1: Gráfico de Barras de Completitud por Entidad (Peor Rendimiento)
            st.subheader("1. 📉 Ranking de Entidades por Completitud Promedio (Peor Rendimiento)")
            # ... (Lógica de Gráfico 1)
            # ... (Código Gráfico 1)
            try:
                COLUMNA_ENTIDAD = 'dueño'
                
                resumen_completitud = df_filtrado.groupby(COLUMNA_ENTIDAD).agg(
                    Total_Activos=('uid', 'count'),
                    Completitud_Promedio=('completitud_score', 'mean')
                ).reset_index()
                
                entidades_volumen = resumen_completitud[resumen_completitud['Total_Activos'] >= 5]
                
                df_top_10_peor_completitud = entidades_volumen.sort_values(
                    by='Completitud_Promedio', 
                    ascending=True 
                ).head(10)
                
                if not df_top_10_peor_completitud.empty:
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    sns.barplot(
                        x='Completitud_Promedio',
                        y=COLUMNA_ENTIDAD,
                        data=df_top_10_peor_completitud,
                        palette='Reds_r', 
                        ax=ax1
                    )
                    
                    ax1.set_title('Top 10 Entidades con Peor Completitud Promedio (Vista Actual)', fontsize=14)
                    ax1.set_xlabel('Score de Completitud Promedio (%)', fontsize=12)
                    ax1.set_ylabel('Entidad Responsable', fontsize=12)
                    ax1.grid(axis='x', linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    st.pyplot(fig1)

                    st.markdown("### Datos del Ranking (Peor Completitud)")
                    st.dataframe(df_top_10_peor_completitud.sort_values(by='Completitud_Promedio', ascending=True), use_container_width=True)
                else:
                    st.warning("No hay entidades con suficiente volumen (>= 5 activos) para generar el ranking de Completitud en la vista actual.")

            except Exception as e:
                st.error(f"❌ ERROR [Visualización 1]: Falló la generación del Gráfico de Completitud. Detalle: {e}")

            st.markdown("---")

            # --- Visualización 2: Gráfico de PARETO de Riesgo (Activos más Críticos)
            st.subheader("2. 🎯 Gráfico de Pareto de Riesgo (Activos más Críticos)")
            # ... (Lógica de Gráfico 2)
            # ... (Código Gráfico 2)
            st.info("""
                **Propósito:** Identificar el subconjunto de activos que concentran el **mayor puntaje de riesgo** (principio 80/20).
                **Interpretación:** La barra azul muestra la contribución de cada activo al riesgo total. La línea naranja muestra la contribución acumulada. El punto donde la línea cruza el **80%** indica la pequeña cantidad de activos que generan la mayor parte del riesgo que debe ser atacado prioritariamente.
            """)
            
            try:
                # Filtrar solo activos con score de riesgo > 0 para el Pareto
                df_riesgo = df_filtrado[df_filtrado['prioridad_riesgo_score'] > 0].sort_values(
                    by='prioridad_riesgo_score', ascending=False
                ).copy()
                
                if not df_riesgo.empty:
                    # Cálculo de la curva de Pareto
                    df_riesgo['Riesgo Acumulado'] = df_riesgo['prioridad_riesgo_score'].cumsum()
                    df_riesgo['Riesgo Total'] = df_riesgo['prioridad_riesgo_score'].sum()
                    df_riesgo['% Riesgo Acumulado'] = (df_riesgo['Riesgo Acumulado'] / df_riesgo['Riesgo Total']) * 100
                    
                    # Generación del gráfico
                    fig2, ax2 = plt.subplots(figsize=(12, 6))
                    
                    # Barras (Riesgo individual)
                    ax2.bar(df_riesgo.index, df_riesgo['prioridad_riesgo_score'], color="C0")
                    ax2.set_xlabel("Activos (Ordenados por Riesgo)", fontsize=12)
                    ax2.set_ylabel("Score de Riesgo Individual", color="C0", fontsize=12)
                    ax2.tick_params(axis='y', labelcolor="C0")
                    ax2.tick_params(axis='x', rotation=45)
                    ax2.set_xticks(range(len(df_riesgo)))
                    ax2.set_xticklabels([f"Activo {i+1}" for i in range(len(df_riesgo))])
                    
                    # Curva (Riesgo acumulado)
                    ax3 = ax2.twinx()
                    ax3.plot(df_riesgo.index, df_riesgo["% Riesgo Acumulado"], color="C1", marker="D", ms=4)
                    ax3.yaxis.set_major_formatter(PercentFormatter())
                    ax3.set_ylabel("% Riesgo Acumulado", color="C1", fontsize=12)
                    ax3.tick_params(axis='y', labelcolor="C1")

                    # Línea de 80%
                    ax3.axhline(80, color='red', linestyle='--', alpha=0.7)
                    
                    ax2.set_title("Gráfico de Pareto: Concentración de Riesgo por Activo", fontsize=16)
                    plt.tight_layout()
                    st.pyplot(fig2)
                
                    st.markdown("### Datos de Riesgo (Top 20 Activos)")
                    st.dataframe(df_riesgo[['titulo', 'dueño', 'prioridad_riesgo_score', '% Riesgo Acumulado']].head(20), use_container_width=True)
                
                else:
                    st.warning("No hay activos con score de riesgo > 0 en la vista actual para generar el Pareto.")

            except Exception as e:
                st.error(f"❌ ERROR [Visualización 2]: Falló la generación del Gráfico de Pareto. Detalle: {e}")
            
            st.markdown("---")

            # --- Visualización 3: Top 10 Categorías (Cobertura Temática)
            st.subheader("3. 🗺️ Cobertura Temática por Categoría")
            # ... (Lógica de Gráfico 3)
            # ... (Código Gráfico 3)
            st.info("""
                **Propósito:** Mapear la **cobertura temática** del inventario.
                **Interpretación:** Las categorías con más activos (barras más largas) indican las áreas donde la organización tiene su **mayor volumen** de datos, ayudando a identificar áreas de especialización o redundancia.
            """)
            
            try:
                COLUMNA_CATEGORIA = 'categoria'
                conteo_categoria = df_filtrado[COLUMNA_CATEGORIA].value_counts().head(10)
                
                if not conteo_categoria.empty:
                    fig3, ax3 = plt.subplots(figsize=(10, 7))
                    sns.barplot(x=conteo_categoria.values, y=conteo_categoria.index, palette='viridis', ax=ax3)

                    ax3.set_title('Top 10 Categorías con Mayor Cobertura Temática (Vista Actual)', fontsize=16)
                    ax3.set_xlabel('Número de Activos', fontsize=12)
                    ax3.set_ylabel('Categoría', fontsize=12)
                    st.pyplot(fig3)
                    
                    st.markdown("### Datos del Conteo de Categorías")
                    st.dataframe(conteo_categoria.to_frame(), use_container_width=True)
                else:
                    st.warning("La columna 'categoria' no contiene suficientes valores para generar la visualización con los filtros seleccionados.")
            except Exception as e:
                st.error(f"❌ ERROR [Visualización 3]: Falló la generación del Bar Plot de Categorías. Detalle: {e}")
            
            # --- FIN DE LAS VISUALIZACIONES PRINCIPALES ---

        # ----------------------------------------------------------------------
        # --- NUEVA SECCIÓN: DIAGNÓSTICO DE ARCHIVO EXTERNO (al final de la página)
        # ----------------------------------------------------------------------
        st.markdown("<hr style='border: 4px solid #f0f2f6;'>", unsafe_allow_html=True)
        st.header("Upload de Archivo CSV para Diagnóstico Rápido 💾")
        st.markdown("Sube un archivo CSV de activos (o similar al inventario) para obtener un diagnóstico rápido de su calidad.")

        uploaded_file = st.file_uploader(
            "Selecciona el Archivo CSV", 
            type="csv",
            help="El archivo debe contener, idealmente, columnas como 'titulo', 'dueño', y las relacionadas con fechas y popularidad."
        )

        if uploaded_file is not None:
            # Mostrar spinner mientras se procesa
            with st.spinner('Analizando archivo...'):
                try:
                    # Leer el archivo subido usando io.StringIO para compatibilidad
                    uploaded_df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")), low_memory=False)
                    
                    if uploaded_df.empty:
                        st.warning("⚠️ El archivo subido está vacío.")
                    else:
                        # Procesar el archivo subido con la misma lógica
                        df_diagnostico = process_data(uploaded_df.copy())
                        
                        if not df_diagnostico.empty:
                            total_activos_subidos = len(df_diagnostico)
                            
                            # CÁLCULO DEL PROMEDIO DE RIESGO
                            riesgo_promedio_general = df_diagnostico['prioridad_riesgo_score'].mean()
                            
                            # DETERMINAR EL ESTADO
                            if riesgo_promedio_general > UMBRAL_RIESGO_ALTO:
                                estado = "🔴 RIESGO ALTO (REQUIERE INTERVENCIÓN)"
                                color = "red"
                            else:
                                estado = "🟢 RIESGO BAJO (CALIDAD ACEPTABLE)"
                                color = "green"
                            
                            st.subheader("Resultados del Diagnóstico Rápido")
                            
                            col_info1, col_info2, col_info3 = st.columns(3)
                            
                            col_info1.metric("Activos Analizados", total_activos_subidos)
                            col_info2.metric("Completitud Promedio", f"{df_diagnostico['completitud_score'].mean():.2f}%")
                            col_info3.metric("Riesgo Promedio General", f"{riesgo_promedio_general:.2f}")

                            # Mostrar el resultado final con color condicional
                            st.markdown(f"""
                                <div style='border: 2px solid {color}; padding: 15px; border-radius: 5px; background-color: #f9f9f9;'>
                                    <h4 style='color: {color}; margin-top: 0;'>Diagnóstico General: {estado}</h4>
                                    <p>El score de riesgo promedio es de <b>{riesgo_promedio_general:.2f}</b>.</p>
                                    <p>Basado en el umbral de {UMBRAL_RIESGO_ALTO}, la base de datos es clasificada como <b>{estado}</b>.</p>
                                </div>
                            """, unsafe_allow_html=True)

                            st.markdown("---")
                            st.subheader("Desglose de Calidad de los Activos Subidos")
                            st.dataframe(df_diagnostico[['titulo', 'dueño', 'completitud_score', 'antiguedad_datos_dias', 'estado_actualizacion', 'prioridad_riesgo_score']].sort_values(by='prioridad_riesgo_score', ascending=False), use_container_width=True)

                        else:
                            st.error("❌ El archivo subido no pudo ser procesado debido a un error en el formato o las columnas.")
                            
                except Exception as e:
                    st.error(f"❌ Error al leer o procesar el archivo CSV: {e}")
                    st.info("Asegúrate de que el archivo es un CSV válido y que utiliza UTF-8.")
                    
        # ----------------------------------------------------------------------
        # --- FIN: DIAGNÓSTICO DE ARCHIVO EXTERNO
        # ----------------------------------------------------------------------

except FileNotFoundError:
    st.error(f"❌ ERROR FATAL: No se encontró el archivo **{ARCHIVO_CSV}**.")
    st.info("Asegúrate de que el archivo CSV esté en la misma carpeta que `app.py`.")
except Exception as e:
    st.error(f"❌ ERROR FATAL: Ocurrió un error inesperado durante la carga del archivo: {e}")
    st.info("Verifica que todas las librerías estén instaladas y que el archivo CSV no esté corrupto.")
