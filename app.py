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

## 1. Funciones de Procesamiento de Datos

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

st.title("📊 Dashboard de Priorización de Activos de Datos (Todos los Activos)")

try:
    with st.spinner(f'Cargando y procesando el archivo: **{ARCHIVO_CSV}**...'):
        df = pd.read_csv(ARCHIVO_CSV, low_memory=False)
        df_analisis_completo = process_data(df.copy()) 
        
    if df_analisis_completo.empty:
        st.error("🛑 Proceso de datos detenido debido a errores previos. Revisa los mensajes de error ❌ para depurar.")
    else:
        st.success(f'✅ Archivo **{ARCHIVO_CSV}** cargado y procesamiento completado. Total de activos: **{len(df_analisis_completo)}**')
        
        # --- 2.1 BARRA LATERAL (FILTROS) ---
        st.sidebar.header("⚙️ Filtros de Análisis")
        
        # Opción para filtrar por DUEÑO (Entidad)
        owners = df_analisis_completo['dueño'].dropna().unique().tolist()
        owners.sort()
        owners.insert(0, "Mostrar Todos los Activos")
        
        filtro_dueño = st.sidebar.selectbox(
            "Filtrar por Entidad Responsable:",
            owners
        )
        
        # Opción para filtrar por CATEGORÍA
        categories = df_analisis_completo['categoria'].dropna().unique().tolist()
        categories.sort()
        categories.insert(0, "Mostrar Todos")
        
        filtro_categoria = st.sidebar.selectbox(
            "Filtrar por Categoría:",
            categories
        )

        # --- 2.2 APLICAR FILTROS ---
        df_filtrado = df_analisis_completo.copy()
        
        if filtro_dueño != "Mostrar Todos los Activos":
            df_filtrado = df_filtrado[df_filtrado['dueño'] == filtro_dueño]
            st.info(f"Filtro aplicado: **Entidad = {filtro_dueño}**")

        if filtro_categoria != "Mostrar Todos":
            df_filtrado = df_filtrado[df_filtrado['categoria'] == filtro_categoria]
            st.info(f"Filtro aplicado: **Categoría = {filtro_categoria}**")

        st.markdown(f"---")
        st.write(f"Activos en la vista actual: **{len(df_filtrado)}**")

        if df_filtrado.empty:
            st.warning("⚠️ No hay datos para mostrar con los filtros seleccionados.")
        else:
            
            # --- 3. Métricas y Visualizaciones ---
            
            st.header("🔍 Resultados Clave de Calidad y Prioridad")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Completitud Promedio", f"{df_filtrado['completitud_score'].mean():.2f}%")
            col2.metric("Activos en Incumplimiento", f"{(df_filtrado['estado_actualizacion'] == '🔴 INCUMPLIMIENTO').sum()} / {len(df_filtrado)}")
            col3.metric("Anomalías Detectadas (ML)", f"{(df_filtrado['anomalia_score'] == -1).sum()}")
            
            
            # --- Visualización 1: Gráfico de Barras de Completitud por Entidad ---
            st.subheader("1. 📉 Completitud Promedio por Entidad (Top 10 Peor Rendimiento)")
            st.markdown("Este gráfico muestra las **10 entidades** (`dueño`) en la vista actual con el **Score de Completitud Promedio más bajo**. Estos sectores requieren la mayor atención para mejorar la documentación de sus activos.")
            
            try:
                COLUMNA_ENTIDAD = 'dueño'
                
                # Agrupar por Entidad y calcular el Score de Completitud Promedio
                resumen_completitud = df_filtrado.groupby(COLUMNA_ENTIDAD).agg(
                    Total_Activos=('uid', 'count'),
                    Completitud_Promedio=('completitud_score', 'mean')
                ).reset_index()
                
                # Filtrar entidades con volumen mínimo (ejemplo: 5 activos)
                entidades_volumen = resumen_completitud[resumen_completitud['Total_Activos'] >= 5]
                
                # Ordenar para obtener el TOP 10 con la PEOR COMPLETITUD (el promedio más bajo)
                df_top_10_peor_completitud = entidades_volumen.sort_values(
                    by='Completitud_Promedio', 
                    ascending=True # Orden ascendente para mostrar lo peor primero
                ).head(10)
                
                if not df_top_10_peor_completitud.empty:
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    sns.barplot(
                        x='Completitud_Promedio',
                        y=COLUMNA_ENTIDAD,
                        data=df_top_10_peor_completitud,
                        palette='Reds_r', # Paleta que destaca los valores bajos
                        ax=ax1
                    )
                    
                    ax1.set_title('Top 10 Entidades con Peor Completitud Promedio (Mín. 5 activos)', fontsize=14)
                    ax1.set_xlabel('Score de Completitud Promedio (%)', fontsize=12)
                    ax1.set_ylabel('Entidad Responsable', fontsize=12)
                    ax1.grid(axis='x', linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    st.pyplot(fig1)

                    st.markdown("### Resumen de Completitud (Top 10 Peor)")
                    st.dataframe(df_top_10_peor_completitud.sort_values(by='Completitud_Promedio', ascending=True), use_container_width=True)
                else:
                    st.info("No hay entidades con suficiente volumen (>= 5 activos) para generar el ranking de Completitud en la vista actual.")

            except Exception as e:
                st.error(f"❌ ERROR [Visualización 1]: Falló la generación del Gráfico de Completitud. Detalle: {e}")

            st.markdown("---")


            # --- Visualización 2: Top 10 Entidades con Incumplimiento ---
            st.subheader("2. Top 10 Entidades con Mayor Porcentaje de Incumplimiento (Vista Total)")
            
            # Esta visualización usa el análisis COMPLETO (df_analisis_completo) para mostrar el ranking general.
            df_para_ranking = df_analisis_completo.copy() 

            try:
                COLUMNA_ENTIDAD = 'dueño'
                # Asegurar que la entidad tenga al menos 5 activos para ser relevante
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
                        ax2.set_title('Top 10 Entidades con Mayor % de Incumplimiento (Min. 5 activos)', fontsize=14)
                        ax2.set_xlabel('Porcentaje de Activos en INCUMPLIMIENTO (%)', fontsize=12)
                        ax2.set_ylabel('Entidad Responsable', fontsize=12)
                        ax2.grid(axis='x', linestyle='--', alpha=0.6)
                        st.pyplot(fig2)
                        
                        st.markdown("### Resumen de Entidades (Top Global)")
                        st.dataframe(resumen_entidad_top, use_container_width=True)
                    else:
                        st.info("No hay entidades con suficiente volumen (>= 5 activos) o incumplimiento para mostrar el top 10.")
                else:
                    st.info("No hay entidades que cumplan el volumen mínimo de 5 activos para el ranking.")
            except Exception as e:
                st.error(f"❌ ERROR [Visualización 2]: Falló la generación del Bar Plot de Entidades. Detalle: {e}")
            
            st.markdown("---")

            # --- Visualización 3: Top 10 Categorías ---
            st.subheader("3. Top 10 Categorías con Mayor Cobertura Temática (Vista Actual)")
            
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
                    
                    st.markdown("### Conteo de Categorías (Vista Actual)")
                    st.dataframe(conteo_categoria.to_frame(), use_container_width=True)
                else:
                    st.info("La columna 'categoria' no contiene valores para generar la visualización con los filtros seleccionados.")
            except Exception as e:
                st.error(f"❌ ERROR [Visualización 3]: Falló la generación del Bar Plot de Categorías. Detalle: {e}")

except FileNotFoundError:
    st.error(f"❌ ERROR FATAL: No se encontró el archivo **{ARCHIVO_CSV}**.")
    st.info("Asegúrate de que el archivo CSV esté en la misma carpeta que `app.py`.")
except Exception as e:
    st.error(f"❌ ERROR FATAL: Ocurrió un error inesperado durante la carga del archivo: {e}")
    st.info("Verifica que todas las librerías estén instaladas y que el archivo CSV no esté corrupto.")
