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

        # Convertir fechas 
        df_temp[COL_FECHA_ACTUALIZACION] = pd.to_datetime(df_temp[COL_FECHA_ACTUALIZACION], errors='coerce', utc=True)
        
        # Calcular Antigüedad 
        hoy = pd.to_datetime(datetime.now().date(), utc=True)
        df_temp['antiguedad_datos_dias'] = (hoy - df_temp[COL_FECHA_ACTUALIZACION]).dt.days
        
        # Mapeo de frecuencia
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
        # Conversión a numérico (maneja el error de 'str' y 'int')
        df['vistas'] = pd.to_numeric(df.get('vistas'), errors='coerce')
        df['descargas'] = pd.to_numeric(df.get('descargas'), errors='coerce')
        
        # El fillna(0) asegura que la suma sea numérica y no haya NaN en el score
        df['popularidad_score'] = df['vistas'].fillna(0) + df['descargas'].fillna(0) 
    except Exception as e:
        st.error(f"❌ ERROR [Paso Popularidad]: Falló la conversión o suma de 'vistas'/'descargas'. Detalle: {e}")
        return pd.DataFrame() 

    # 2. CÁLCULOS PREVIOS (Antigüedad y Estado de Actualización)
    try:
        df = calculate_antiguedad_y_estado(df.copy()) 
    except Exception as e:
        # El error ya se maneja en calculate_antiguedad_y_estado
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
        
        if not df_modelo.empty and len(df_modelo) > 1: # Isolation Forest necesita al menos 2 muestras
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
    
    # --- 6. Filtrar Públicos (ESTE PASO HA SIDO ELIMINADO/MODIFICADO) ---
    # Ya no se filtra por 'publico'. Se devuelve el DataFrame completo.
    
    return df

## 2. Título y Ejecución Principal

st.title("📊 Dashboard de Priorización de Activos de Datos (Todos los Activos)")

try:
    with st.spinner(f'Cargando y procesando el archivo: **{ARCHIVO_CSV}**...'):
        # Carga del archivo
        df = pd.read_csv(ARCHIVO_CSV, low_memory=False)
        # Llamamos al DataFrame final 'df_analisis' para diferenciarlo del original
        df_analisis = process_data(df.copy()) 
        
    if df_analisis.empty:
        st.error("🛑 Proceso de datos detenido debido a errores previos. Revisa los mensajes de error ❌ para depurar.")
    else:
        st.success(f'✅ Archivo **{ARCHIVO_CSV}** cargado y procesamiento completado.')
        st.info(f"Analizando **TODOS** los activos en el inventario, incluyendo aquellos no clasificados como públicos.")
        st.write(f"Total de activos analizados: **{len(df_analisis)}**")
        
        # --- 3. Métricas y Visualizaciones ---
        
        st.header("🔍 Resultados Clave de Calidad y Prioridad")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Completitud Promedio", f"{df_analisis['completitud_score'].mean():.2f}%")
        col2.metric("Activos en Incumplimiento", f"{(df_analisis['estado_actualizacion'] == '🔴 INCUMPLIMIENTO').sum()} / {len(df_analisis)}")
        col3.metric("Anomalías Detectadas (ML)", f"{(df_analisis['anomalia_score'] == -1).sum()}")
        
        
        # --- Visualización 1: Prioridad de Intervención ---
        st.subheader("1. Prioridad de Intervención (Score ML)")
        try:
            fig1, ax1 = plt.subplots(figsize=(12, 7))
            sns.scatterplot(
                x='antiguedad_datos_dias',
                y='prioridad_riesgo_score', 
                data=df_analisis, # Usamos df_analisis
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
            ax1.axhline(y=df_analisis['prioridad_riesgo_score'].quantile(0.75), color='red', linestyle='--', label='Prioridad Alta (Q3)')
            ax1.legend(title='Estado de Actualización')
            ax1.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig1)
        except Exception as e:
            st.error(f"❌ ERROR [Visualización 1]: Falló la generación del Scatter Plot. Detalle: {e}")

        st.markdown("---")


        # --- Visualización 2: Top 10 Entidades con Incumplimiento ---
        st.subheader("2. Top 10 Entidades con Mayor Porcentaje de Incumplimiento")
        try:
            COLUMNA_ENTIDAD = 'dueño'
            resumen_entidad = df_analisis.groupby(COLUMNA_ENTIDAD).agg(
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
        except Exception as e:
            st.error(f"❌ ERROR [Visualización 2]: Falló la generación del Bar Plot de Entidades. Detalle: {e}")
        
        st.markdown("---")

        # --- Visualización 3: Top 10 Categorías ---
        st.subheader("3. Top 10 Categorías con Mayor Cobertura Temática")
        try:
            COLUMNA_CATEGORIA = 'categoria'
            conteo_categoria = df_analisis[COLUMNA_CATEGORIA].value_counts().head(10)
            
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
        except Exception as e:
            st.error(f"❌ ERROR [Visualización 3]: Falló la generación del Bar Plot de Categorías. Detalle: {e}")

except FileNotFoundError:
    st.error(f"❌ ERROR FATAL: No se encontró el archivo **{ARCHIVO_CSV}**.")
    st.info("Asegúrate de que el archivo CSV esté en la misma carpeta que `app.py`.")
except Exception as e:
    st.error(f"❌ ERROR FATAL: Ocurrió un error inesperado durante la carga del archivo: {e}")
    st.info("Verifica que todas las librerías estén instaladas y que el archivo CSV no esté corrupto.")
