import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from datetime import datetime, date 

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Diagnóstico de Activos",
    layout="wide"
)

# --- Nombre del archivo CSV que Streamlit debe encontrar ---
ARCHIVO_CSV = "Asset_Inventory_-_Public_20251118.csv"

## 1. Funciones de Procesamiento de Datos (Sin Cambios)
def clean_col_name(col):
    name = col.lower().strip()
    name = name.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
    name = name.replace(' ', '_').replace('.', '').replace('/', '_').replace(':', '').replace('(', '').replace(')', '')
    return name

def calculate_antiguedad_y_estado(df_temp):
    try:
        COL_FECHA_ACTUALIZACION = 'fecha_de_ultima_actualizacion_de_datos_utc'
        COL_FRECUENCIA = 'informacion_de_datos_frecuencia_de_actualizacion'

        df_temp[COL_FECHA_ACTUALIZACION] = pd.to_datetime(df_temp[COL_FECHA_ACTUALIZACION], errors='coerce', utc=True)
        
        hoy = pd.to_datetime(datetime.now().date(), utc=True)
        df_temp['antiguedad_datos_dias'] = (hoy - df_temp[COL_FECHA_ACTUALIZACION]).dt.days
        
        mapa_frecuencia = {
            'diario': 1, 'semanal': 7, 'quincenal': 15, 'mensual': 30, 
            'trimestral': 90, 'semestral': 180, 'anual': 365
        }
        df_temp['frecuencia_esperada_dias'] = df_temp[COL_FRECUENCIA].astype(str).str.lower().str.strip().map(mapa_frecuencia).fillna(9999)

        df_temp['estado_actualizacion'] = np.where(
            (df_temp['antiguedad_datos_dias'] > df_temp['frecuencia_esperada_dias']) & 
            (df_temp['frecuencia_esperada_dias'] < 9999), 
            '🔴 INCUMPLIMIENTO', 
            '🟢 CUMPLE'
        )
        return df_temp
    except KeyError as e:
        st.error(f"❌ ERROR [Paso Antigüedad]: No se encontró la columna de fecha o frecuencia requerida: {e}. Revisa el nombre.")
        raise
    except Exception as e:
        st.error(f"❌ ERROR INESPERADO [Paso Antigüedad]: Falló el cálculo de antigüedad: {e}.")
        raise


@st.cache_data
def process_data(df):
    
    # 1. Limpieza de nombres de columnas
    df.columns = [clean_col_name(col) for col in df.columns]

    # --- CORRECCIÓN y VERIFICACIÓN DE POPULARIDAD ---
    try:
        df['vistas'] = pd.to_numeric(df.get('vistas'), errors='coerce')
        df['descargas'] = pd.to_numeric(df.get('descargas'), errors='coerce')
        df['popularidad_score'] = df['vistas'].fillna(0) + df['descargas'].fillna(0) 
    except Exception as e:
        st.error(f"❌ ERROR [Paso Popularidad]: Falló la conversión o suma de 'vistas'/'descargas'. Detalle: {e}")
        return pd.DataFrame() 

    # 2. CÁLCULOS PREVIOS (Antigüedad y Estado de Actualización)
    try:
        df = calculate_antiguedad_y_estado(df.copy()) 
    except Exception as e:
        return pd.DataFrame() 
    
    # 3. CÁLCULO DE MÉTRICA DE COMPLETITUD
    try:
        campos_minimos = [
            'titulo', 'descripcion', 'dueño', 'correo_electronico_de_contacto',
            'licencia', 'dominio', 'categoria', 'informacion_de_datos_frecuencia_de_actualizacion',
            'common_core_public_access_level', 'informacion_de_datos_cobertura_geografica'
        ]
        campos_existentes = [col for col in campos_minimos if col in df.columns]
        num_campos_totales = len(campos_existentes)
        df['campos_diligenciados'] = df[campos_existentes].notna().sum(axis=1)
        df['completitud_score'] = (df['campos_diligenciados'] / num_campos_totales) * 100
    except Exception as e:
        st.error(f"❌ ERROR [Paso Completitud]: Falló el cálculo de 'completitud_score'. Detalle: {e}")
        return pd.DataFrame()
    
    # 4. DETECCIÓN DE ANOMALÍAS (Isolation Forest)
    try:
        df['anomalia_score'] = 0 
        df_modelo = df[(df['antiguedad_datos_dias'] < 9999) & (df['popularidad_score'] > 0)].copy()
        
        if not df_modelo.empty and len(df_modelo) > 1: 
            features = df_modelo[['antiguedad_datos_dias', 'popularidad_score', 'completitud_score']]
            model = IsolationForest(contamination=0.01, random_state=42)
            model.fit(features)
            anomalias = model.predict(features)
            df.loc[df_modelo.index, 'anomalia_score'] = anomalias
    except ImportError:
        st.error("❌ ERROR [Paso Anomalías]: `scikit-learn` no está instalado. Instala: `pip install scikit-learn`.")
    except Exception as e:
        st.error(f"❌ ERROR [Paso Anomalías]: Falló el modelo Isolation Forest. Detalle: {e}")

    # 5. CÁLCULO DE SCORE DE RIESGO/PRIORIDAD
    try:
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

        # --- SECCIÓN DE SELECCIÓN Y DESGLOSE DE ENTIDAD ---
        st.header("🔬 Desglose de Métricas por Entidad")
        
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
        
        # NUEVO FILTRO: Nivel de Acceso (Activa/Desactiva 'public', etc.)
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
            
            # --- 3. Métricas y Visualizaciones ---
            
            st.subheader("Métricas de la Vista Actual")
            col_metrica1, col_metrica2, col_metrica3 = st.columns(3)
            col_metrica1.metric("Completitud Promedio", f"{df_filtrado['completitud_score'].mean():.2f}%")
            col_metrica2.metric("Activos en Incumplimiento", f"{(df_filtrado['estado_actualizacion'] == '🔴 INCUMPLIMIENTO').sum()} / {len(df_filtrado)}")
            col_metrica3.metric("Anomalías Detectadas (ML)", f"{(df_filtrado['anomalia_score'] == -1).sum()}")
            
            st.markdown("---")
            
            
            # --- Visualización 1: Gráfico de Barras de Completitud por Entidad ---
            st.subheader("1. 📉 Ranking de Entidades por Completitud Promedio (Peor Rendimiento)")
            
            st.info("""
                **Propósito:** Identificar las entidades (`dueño`) que tienen la **peor calidad de documentación**.
                **Interpretación:** Las barras más cortas (más a la izquierda) indican un menor `Score de Completitud Promedio`. Estas entidades deben ser **priorizadas** para mejorar el llenado de metadatos.
                *Solo se incluyen entidades con 5 o más activos para asegurar un ranking significativo.*
            """)
            
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


            # --- Visualización 2: Top 10 Entidades con Incumplimiento ---
            st.subheader("2. 🚨 Ranking de Entidades por Porcentaje de Incumplimiento")
            
            st.info("""
                **Propósito:** Determinar qué entidades tienen el mayor porcentaje de activos que **no se actualizan** con la frecuencia prometida.
                **Interpretación:** Las entidades con mayor porcentaje de incumplimiento (barras más largas) representan el **mayor riesgo operacional** debido a datos obsoletos. Este ranking utiliza el **conjunto TOTAL** de activos para el ranking general.
            """)
            
            df_para_ranking = df_analisis_completo.copy() 

            try:
                COLUMNA_ENTIDAD = 'dueño'
                entidades_con_volumen = df_para_ranking.groupby(COLUMNA_ENTIDAD).filter(lambda x: len(x) >= 5)

                if not entidades_con_volumen.empty:
                    resumen_entidad = entidades_con_volumen.groupby(COLUMNA_ENTIDAD).agg(
                        Total_Activos=('uid', 'count'),
                        Activos_Incumplimiento=('estado_actualizacion', lambda x: (x == '🔴 INCUMPLIMIENTO').sum())
                    ).reset_index()

                    resumen_entidad['Porcentaje_Incumplimiento'] = (resumen_entidad['Activos_Incumplimiento'] / resumen_entidad['Total_Activos']) * 100
                    resumen_entidad_top = resumen_entidad.sort_values(by='Porcentaje_Incumplimiento', ascending=False).head(10)
                    
                    if not resumen_entidad_top.empty:
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        sns.barplot(
                            x='Porcentaje_Incumplimiento',
                            y=COLUMNA_ENTIDAD,
                            data=resumen_entidad_top,
                            palette='Reds_d',
                            ax=ax2
                        )
                        ax2.set_title('Top 10 Entidades con Mayor % de Incumplimiento (Ranking Global)', fontsize=14)
                        ax2.set_xlabel('Porcentaje de Activos en INCUMPLIMIENTO (%)', fontsize=12)
                        ax2.set_ylabel('Entidad Responsable', fontsize=12)
                        ax2.grid(axis='x', linestyle='--', alpha=0.6)
                        st.pyplot(fig2)
                        
                        st.markdown("### Datos del Ranking (Incumplimiento)")
                        st.dataframe(resumen_entidad_top, use_container_width=True)
                    else:
                        st.warning("No hay entidades con suficiente volumen (>= 5 activos) o incumplimiento para mostrar el top 10.")
                else:
                    st.warning("No hay entidades que cumplan el volumen mínimo de 5 activos para el ranking.")
            except Exception as e:
                st.error(f"❌ ERROR [Visualización 2]: Falló la generación del Bar Plot de Entidades. Detalle: {e}")
            
            st.markdown("---")

            # --- Visualización 3: Top 10 Categorías ---
            st.subheader("3. 🗺️ Cobertura Temática por Categoría")
            
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

except FileNotFoundError:
    st.error(f"❌ ERROR FATAL: No se encontró el archivo **{ARCHIVO_CSV}**.")
    st.info("Asegúrate de que el archivo CSV esté en la misma carpeta que `app.py`.")
except Exception as e:
    st.error(f"❌ ERROR FATAL: Ocurrió un error inesperado durante la carga del archivo: {e}")
    st.info("Verifica que todas las librerías estén instaladas y que el archivo CSV no esté corrupto.")
