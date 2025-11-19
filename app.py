import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import io
from datetime import datetime
import re
import warnings
import os # Necesario para la conexión con el LLM
from openai import OpenAI # Importar la librería del LLM
import pandas as pd
import streamlit as st
warnings.filterwarnings('ignore') # Ocultar advertencias de Pandas/Streamlit

# --- Variables Globales ---
ARCHIVO_PROCESADO = "Asset_Inventory_PROCESSED.csv"
# CRITERIO DE RIESGO
UMBRAL_RIESGO_ALTO = 3.0

# --- CONFIGURACIÓN DE RIESGOS UNIVERSALES ---
PENALIZACION_DATOS_INCOMPLETOS = 2.0
PENALIZACION_INCONSISTENCIA_TIPO = 0.5
PENALIZACION_DUPLICADO = 1.0
PENALIZACION_METADATO_CLAVE_FALTA = 1.0
PENALIZACION_VENCIMIENTO_ML = 1.5 # Penalización por anomalía detectada

# RIESGO MÁXIMO TEÓRICO (Universal + Metadatos + ML): 2.0 + 0.5 + 1.0 + 1.0 + 1.5 = 6.0
RIESGO_MAXIMO_TEORICO_UNIVERSAL_COMPLETO = 6.0

# =================================================================
# 1. Funciones de Carga y Procesamiento
# =================================================================

@st.cache_data
def load_processed_data(file_path):
    """Carga el archivo CSV YA PROCESADO y lo cachea."""
    try:
        # Asegúrate de que el archivo preprocess.py haya generado este archivo
        df = pd.read_csv(file_path, low_memory=False)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

def clean_and_convert_types_external(df):
    """Fuerza a las columnas a ser tipo object para asegurar la detección de inconsistencias."""
    
    # Columnas que suelen ser de tipo 'object' (string)
    object_cols = ['titulo', 'descripcion', 'dueño']
    
    # Columnas que contienen los datos que queremos chequear por tipo mixto
    data_cols = [col for col in df.columns if col not in object_cols]
    
    for col in data_cols:
        if df[col].dtype != 'object':
            try:
                df[col] = df[col].astype(object)
            except:
                pass

    return df

def check_universals_external(df):
    """
    Calcula métricas de calidad universal: Completitud (Datos), Consistencia, Unicidad.
    """
    n_cols = df.shape[1]
    
    # --- 1. COMPLETITUD: Datos por Fila (Densidad) ---
    df['datos_por_fila_score'] = (df.notna().sum(axis=1) / n_cols) * 100
    df['riesgo_datos_incompletos'] = np.where(
        df['datos_por_fila_score'] < 70, PENALIZACION_DATOS_INCOMPLETOS, 0.0
    )

    # --- 2. CONSISTENCIA: Mezcla de Tipos (LÓGICA CORREGIDA) ---
    df['riesgo_consistencia_tipo'] = 0.0
    
    for col in df.columns:
        if df[col].dtype == 'object':
            inconsistencies_mask = df[col].apply(lambda x: pd.notna(x) and not isinstance(x, str))
            df.loc[inconsistencies_mask, 'riesgo_consistencia_tipo'] += PENALIZACION_INCONSISTENCIA_TIPO
    
    df['riesgo_consistencia_tipo'] = np.clip(df['riesgo_consistencia_tipo'], 0.0, PENALIZACION_INCONSISTENCIA_TIPO)
        
    # --- 3. UNICIDAD: Duplicados Exactos ---
    df['es_duplicado'] = df.duplicated(keep=False)
    df['riesgo_duplicado'] = np.where(
        df['es_duplicado'], PENALIZACION_DUPLICADO, 0.0
    )
    
    return df

# =================================================================
# 2. Funciones de Riesgos Específicos (Restaurado del Código Viejo)
# =================================================================

def check_specific_metadata(df):
    """
    Evalúa la calidad de metadatos clave que definen el activo:
    1. Existencia de campos clave.
    2. Formato de la fecha de creación/modificación.
    """
    
    campos_clave_metadatos = ['fecha_creacion', 'formato_archivo', 'categoria']
    
    # Inicializa la penalización de metadatos
    df['riesgo_metadato_faltante'] = 0.0
    
    # --- 1. Penalización por Metadato Clave Faltante/Incompleto ---
    
    for campo in campos_clave_metadatos:
        if campo in df.columns:
            # Penaliza si el campo existe pero está vacío (NaN)
            df.loc[df[campo].isna(), 'riesgo_metadato_faltante'] += PENALIZACION_METADATO_CLAVE_FALTA / len(campos_clave_metadatos)
        else:
            # Penaliza a todos los registros si el campo clave NO existe
            df['riesgo_metadato_faltante'] += PENALIZACION_METADATO_CLAVE_FALTA / len(campos_clave_metadatos)
            
    # Asegura que la penalización máxima no exceda el límite
    df['riesgo_metadato_faltante'] = np.clip(df['riesgo_metadato_faltante'], 0.0, PENALIZACION_METADATO_CLAVE_FALTA)

    # --- 2. Validación de Formato de Fecha (Solo para métrica, no para riesgo) ---
    # Esto impacta el 'completitud_score' general, no el 'riesgo_metadato_faltante'
    
    def is_valid_date(date_str):
        if pd.isna(date_str):
            return False
        try:
            # Intenta parsear la fecha. Se puede ajustar el formato
            datetime.strptime(str(date_str).split(' ')[0], '%Y-%m-%d')
            return True
        except:
            return False

    if 'fecha_creacion' in df.columns:
        df['fecha_valida'] = df['fecha_creacion'].apply(is_valid_date)
    else:
        df['fecha_valida'] = False # Si la columna no existe

    return df

def run_ml_anomaly_detection(df):
    """
    Simula una Detección de Anomalías (Machine Learning) basada en la antigüedad.
    En un entorno real, esto sería un modelo como Isolation Forest o One-Class SVM.
    """
    
    # Si la columna ya existe, se asume que el pre-proceso ya la corrió.
    if 'anomalia_score' in df.columns:
        return df

    # Simulación de detección:
    # 1. Calcular la antigüedad promedio.
    # 2. Asignar un valor anómalo (ej. -1) si el activo tiene una antigüedad muy baja (ej. 1 día)
    #    y el score de riesgo es bajo (lo que indicaría un 'activo fantasma' o un error de carga).
    
    if 'antiguedad_datos_dias' in df.columns:
        # Penaliza activos muy nuevos (antigüedad <= 1 día) que tienen un riesgo universal muy bajo (casi perfecto)
        # Esto podría indicar una carga de datos falsa o de prueba.
        anomalia_mask = (df['antiguedad_datos_dias'] <= 1) & (df['prioridad_riesgo_score'] < 0.5)
        df['anomalia_score'] = np.where(anomalia_mask, -1, 1) # -1 indica anomalía
    else:
        # Si la columna no está disponible, no penalizamos
        df['anomalia_score'] = 1 
        
    # Asignar penalización por anomalía
    df['riesgo_anomalia_ml'] = np.where(
        df['anomalia_score'] == -1, PENALIZACION_VENCIMIENTO_ML, 0.0
    )
    
    return df

# =================================================================
# 3. Función de Procesamiento Global (Integra todos los riesgos)
# =================================================================

def process_external_data(df):
    """
    Lógica de riesgo universal completa para el archivo externo subido.
    """
    
    # PASO 1: Limpieza y Evaluación de UNIVERSALES (Completitud, Consistencia, Unicidad)
    df = clean_and_convert_types_external(df)
    df = check_universals_external(df)
    
    # PASO 2: Evaluación de Metadatos Específicos
    df = check_specific_metadata(df)

    # PASO 3: Cálculo del Score de Riesgo Preliminar (para ML)
    df['prioridad_riesgo_score_pre_ml'] = (
        df['riesgo_datos_incompletos'] + 
        df['riesgo_consistencia_tipo'] +
        df['riesgo_duplicado'] +
        df['riesgo_metadato_faltante']
    )
    
    # PASO 4: Detección de Anomalías (ML)
    df = run_ml_anomaly_detection(df)
    
    # --- 5. CÁLCULO FINAL DE RIESGO Y CALIDAD (INTEGRACIÓN COMPLETA) ---
    
    # Score de riesgo total (las 4 dimensiones + ML)
    df['prioridad_riesgo_score'] = (
        df['prioridad_riesgo_score_pre_ml'] + 
        df['riesgo_anomalia_ml']
    )
    
    # CÁLCULO DE CALIDAD TOTAL DEL ARCHIVO (0% a 100%) - BASADO EN EL RIESGO MÁXIMO GLOBAL
    avg_file_risk = df['prioridad_riesgo_score'].mean()
    quality_score = 100 - (avg_file_risk / RIESGO_MAXIMO_TEORICO_UNIVERSAL_COMPLETO * 100)
    
    df['calidad_total_score'] = np.clip(quality_score, 0, 100)

    return df

# =================================================================
# 4. Funciones de Soporte y Recomendación
# =================================================================

def generate_specific_recommendation(risk_dimension):
    """Genera pasos de acción específicos para la dimensión de riesgo más alta."""
    
    # 1. Datos Incompletos (Completitud)
    if 'Datos Incompletos' in risk_dimension:
        return """
**Identificación:** Localiza las columnas o filas con un alto porcentaje de valores **Nulos (NaN)**. El umbral de alerta se activa si el promedio de datos por fila es **menor al 70%**.

**Acción:** Revisa los procesos de ingesta de datos. Si el campo es **obligatorio**, asegúrate de que todos los registros lo contengan. Si el campo es **opcional**, considera si es crucial para el análisis antes de llenarlo con un valor por defecto.
        """
    # 2. Duplicados Exactos (Unicidad)
    elif 'Duplicados Exactos' in risk_dimension:
        return """
**Identificación:** Encuentra las filas que son **copias exactas** (duplicados de todo el registro).

**Acción:** Revisa tu proceso de extracción/carga. Un duplicado exacto generalmente indica un error de procesamiento o ingesta. **Elimina las copias** y asegúrate de que exista una **clave única** (UID) para cada registro que evite la re-ingesta accidental.
        """
    # 3. Consistencia de Tipo (Coherencia)
    elif 'Consistencia de Tipo' in risk_dimension:
        return """
**Identificación:** Una columna contiene **datos mezclados** (ej. números, fechas, y texto en una columna que debería ser solo números). Esto afecta seriamente el análisis.

**Acción:** Normaliza el tipo de dato para la columna afectada. Si es una columna numérica, **elimina los valores de texto** o conviértelos a `NaN` para una limpieza posterior. Define el **tipo de dato esperado** (Schema) para cada columna y aplica una validación estricta al inicio del proceso.
        """
    # 4. Metadatos Clave Faltantes
    elif 'Metadatos Clave Faltantes' in risk_dimension:
        return """
**Identificación:** Faltan campos esenciales del esquema de metadatos (ej., `fecha_creacion`, `formato_archivo`). Estos campos son cruciales para el ciclo de vida del dato.

**Acción:** **Prioriza la documentación**. Asegura que las tres columnas clave (`fecha_creacion`, `formato_archivo`, `categoria`) existan y contengan valores válidos para el activo.
        """
    # 5. Anomalía Detectada (ML)
    elif 'Anomalía Detectada (ML)' in risk_dimension:
        return """
**Identificación:** El sistema de Machine Learning ha detectado que el activo tiene una combinación inusual de atributos (ej. un activo **muy nuevo** pero con un **riesgo de calidad extremadamente bajo**).

**Acción:** **Revisa la fuente y el proceso de ingesta de este activo**. Podría ser un error de carga, un dato de prueba que se quedó en producción, o un problema con la etiqueta de tiempo.
        """
    else:
        return "No se requiere una acción específica o el riesgo detectado es demasiado bajo."

# =================================================================
# 5. ASISTENTE DE CONSULTA DE DATOS (NLP)
# =================================================================

def setup_data_assistant(df):
    """
    Configura el asistente de consulta de datos usando LLM.
    """
    
    st.markdown("---")
    st.header("🧠 Asistente de Consulta de Datos (NLP)")
    st.markdown("#### 💬 Haz una pregunta sobre los Activos (Lenguaje Natural)")
    st.info("Ejemplos: '¿Cuál es la entidad con el riesgo promedio más alto?' o 'Dame el promedio de Completitud por categoría'.")
    
    # --- 1. CONFIGURACIÓN DE CLAVE API ---
    api_key = st.text_input(
        "Ingresa tu clave API de OpenAI o Gemini (si usas otro modelo):", 
        type="password", 
        key="api_key_nlp"
    )
    
    if not api_key:
        st.warning("Por favor, introduce una clave API para activar el asistente.")
        st.markdown("---")
        return

    # --- 2. INTERFAZ DE USUARIO ---
    user_query = st.text_input(
        "Tu pregunta sobre el Inventario de Activos:",
        key="nlp_query"
    )

    if st.button("Consultar Datos", use_container_width=True) and user_query:
        if df.empty:
            st.error("No hay datos cargados para realizar la consulta.")
            return

        with st.spinner(f"El Asistente está analizando: '{user_query}'..."):
            try:
                # Inicializar el cliente (Asegúrate de cambiar a Google-GenAI si usas Gemini)
                client = OpenAI(api_key=api_key)
                
                # Definir el contexto del Agente (System Prompt)
                system_prompt = f"""
                Eres un asistente de datos experto en Python y Pandas. Tu tarea es responder preguntas 
                sobre el DataFrame 'df_analisis_completo'. 
                El DataFrame contiene {len(df)} activos y tiene las siguientes columnas clave: {df.columns.tolist()}.
                
                Genera código Python (pandas) para encontrar la respuesta. Luego, proporciona el resultado
                de la ejecución del código. NO necesitas ejecutar el código, solo simula la respuesta.
                """
                
                # --- SIMULACIÓN AVANZADA DE RESPUESTAS (para demostrar la funcionalidad) ---
                
                if 'riesgo' in user_query.lower() or 'peor' in user_query.lower():
                    if 'dueño' in df.columns and 'prioridad_riesgo_score' in df.columns:
                        simulated_code = "df.groupby('dueño')['prioridad_riesgo_score'].mean().sort_values(ascending=False).head(3)"
                        simulated_result = df.groupby('dueño')['prioridad_riesgo_score'].mean().sort_values(ascending=False).head(3)
                        
                        st.success(f"✅ Resultado de la consulta: Entidades con Mayor Riesgo Promedio")
                        st.code(f"Código ejecutado:\n{simulated_code}", language='python')
                        st.dataframe(simulated_result.reset_index().rename(columns={'prioridad_riesgo_score': 'Riesgo_Promedio'}), hide_index=True)
                    else:
                        st.warning("Columnas 'dueño' o 'prioridad_riesgo_score' no encontradas para esta consulta.")
                    
                elif 'completitud' in user_query.lower() or 'promedio' in user_query.lower():
                    if 'completitud_score' in df.columns:
                        simulated_code = "df['completitud_score'].mean()"
                        simulated_result = df['completitud_score'].mean()
                        
                        st.success(f"✅ Resultado de la consulta: Completitud Promedio Global")
                        st.code(f"Código ejecutado:\n{simulated_code}", language='python')
                        st.write(f"El score de Completitud Promedio Global es: **{simulated_result:.2f}%**")
                    else:
                        st.warning("Columna 'completitud_score' no encontrada para esta consulta.")

                else:
                    st.warning("⚠️ El agente LLM (motor de IA) no ejecutó el código. Esta es una **simulación** que requiere un LLM real y una ejecución segura (por ejemplo, con LangChain) para obtener resultados exactos de consultas complejas.")
                    st.code("Consulta enviada al LLM. El modelo generaría y ejecutaría el código Pandas aquí.")

            except Exception as e:
                st.error(f"❌ Error durante la consulta al LLM: {e}. Asegúrate que la librería **openai** está instalada y la clave API es correcta.")


# =================================================================
# 6. Ejecución Principal del Dashboard
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
            
            # --- 7. Métricas de la Vista Actual ---
            st.subheader("Métricas de la Vista Actual")
            col_metrica1, col_metrica2, col_metrica3 = st.columns(3)
            col_metrica1.metric("Completitud Promedio", f"{df_filtrado['completitud_score'].mean():.2f}%")
            col_metrica2.metric("Activos en Incumplimiento", f"{(df_filtrado['estado_actualizacion'] == '🔴 INCUMPLIMIENTO').sum()} / {len(df_filtrado)}")
            if 'anomalia_score' in df_filtrado.columns:
                 col_metrica3.metric("Anomalías Detectadas (ML)", f"{(df_filtrado['anomalia_score'] == -1).sum()}")
            else:
                 col_metrica3.metric("Anomalías Detectadas (ML)", "N/A")
            
            st.markdown("---")

            # --- 8. Tabla de Búsqueda y Diagnóstico de Entidades (Con Color Condicional) ---
            st.header("🔍 8. Tabla de Búsqueda y Diagnóstico de Entidades")

            # TEXTO CORREGIDO PARA EL NUEVO UMBRAL (3.0)
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
            
            # --- PESTAÑAS PARA EL "CARRUSEL" DE VISUALIZACIONES ---
            tab1, tab2, tab3 = st.tabs(["1. Ranking de Completitud", "2. Burbujas de Riesgo", "3. Cobertura Temática"])

            with tab1:
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
                        ax1.set_ylabel('Entidad Responsable', fontsize=12)
                        st.pyplot(fig1)
                    else:
                        st.warning("No hay entidades con suficiente volumen (>= 5 activos) para generar el ranking.")
                except Exception as e:
                    st.error(f"❌ ERROR [Visualización 1]: Falló la generación del Gráfico de Completitud. Detalle: {e}")

            with tab2:
                # --- Visualización 2: Gráfico de Burbujas de Riesgo ---
                st.subheader("2. 🫧 Burbujas de Priorización de Riesgo por Entidad")
                st.markdown("Este gráfico muestra la **relación entre el riesgo, la completitud de metadatos y el volumen de activos** por entidad.")
                st.markdown("* **Eje X:** Riesgo Promedio (Se debe minimizar, mejor a la izquierda).")
                st.markdown("* **Eje Y:** Completitud Promedio (Se debe maximizar, mejor arriba).")
                st.markdown("* **Tamaño de Burbuja:** Volumen de Activos.")

                try:
                    df_bubble = df_filtrado.groupby('dueño').agg(
                        Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
                        Completitud_Promedio=('completitud_score', 'mean'),
                        Volumen=('uid', 'count')
                    ).reset_index()
                    
                    if not df_bubble.empty:
                        fig2, ax2 = plt.subplots(figsize=(12, 8))
                        
                        max_volumen = df_bubble['Volumen'].max()
                        s_volumen = (df_bubble['Volumen'] / max_volumen) * 2000
                        
                        scatter = ax2.scatter(
                            x=df_bubble['Riesgo_Promedio'], 
                            y=df_bubble['Completitud_Promedio'], 
                            s=s_volumen, 
                            c=df_bubble['Completitud_Promedio'], 
                            cmap='RdYlGn', 
                            alpha=0.6, 
                            edgecolors="w", 
                            linewidth=1
                        )
                        
                        # Anotar las 5 burbujas más grandes (mayor volumen)
                        for i in df_bubble.nlargest(5, 'Volumen').index:
                             ax2.annotate(df_bubble.loc[i, 'dueño'], 
                                         (df_bubble.loc[i, 'Riesgo_Promedio'], df_bubble.loc[i, 'Completitud_Promedio']), 
                                         fontsize=8, alpha=0.8)

                        ax2.axhline(80, color='gray', linestyle='--', alpha=0.5)
                        ax2.axvline(UMBRAL_RIESGO_ALTO, color='red', linestyle=':', alpha=0.7)

                        ax2.set_xlabel('Riesgo Promedio (Peor →)', fontsize=12)
                        ax2.set_ylabel('Completitud Promedio (Mejor ↑)', fontsize=12)
                        ax2.set_title('Matriz de Priorización de Entidades (Riesgo vs. Completitud)', fontsize=16)
                        
                        cbar = fig2.colorbar(scatter, ax=ax2)
                        cbar.set_label('Completitud Promedio (%)')
                        
                        st.pyplot(fig2)
                    else:
                        st.warning("No hay suficientes datos de entidad para generar el Gráfico de Burbujas.")
                        
                except Exception as e:
                    st.error(f"❌ ERROR [Visualización 2]: Falló la generación del Gráfico de Burbujas. Detalle: {e}")


            with tab3:
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
                        ax3.set_ylabel('Categoría', fontsize=12)
                        st.pyplot(fig3)
                    else:
                        st.warning("La columna 'categoria' no contiene suficientes valores para generar la visualización.")
                except Exception as e:
                    st.error(f"❌ ERROR [Visualización 3]: Falló la generación del Bar Plot de Categorías. Detalle: {e}")


        
        # ----------------------------------------------------------------------
        # --- SECCIÓN 9: DIAGNÓSTICO DE ARCHIVO EXTERNO
        # ----------------------------------------------------------------------
        st.markdown("<hr style='border: 4px solid #f0f2f6;'>", unsafe_allow_html=True)
        st.header(f"💾 Diagnóstico de Archivo CSV Externo (Calidad Integral)")
        st.markdown(f"Sube un archivo CSV. La **Calidad Total** se calcula en base a 5 dimensiones integrales (Riesgo Máximo: **{RIESGO_MAXIMO_TEORICO_UNIVERSAL_COMPLETO:.1f}**).")

        uploaded_file = st.file_uploader(
            "Selecciona el Archivo CSV", 
            type="csv"
        )

        if uploaded_file is not None:
            with st.spinner('Analizando archivo...'):
                try:
                    uploaded_filename = uploaded_file.name
                    # Lógica de lectura robusta con detección de delimitadores
                    uploaded_file.seek(0)
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
                            riesgo_promedio_total = df_diagnostico['prioridad_riesgo_score'].mean()

                            # Desglose de Riesgos Promedio
                            riesgos_reporte = pd.DataFrame({
                                'Dimensión de Riesgo': [
                                    '1. Datos Incompletos (Completitud)',
                                    '2. Duplicados Exactos (Unicidad)',
                                    '3. Consistencia de Tipo (Coherencia)',
                                    '4. Metadatos Clave Faltantes',
                                    '5. Anomalía Detectada (ML)',
                                ],
                                'Riesgo Promedio (0-Máx)': [
                                    df_diagnostico['riesgo_datos_incompletos'].mean(),
                                    df_diagnostico['riesgo_duplicado'].mean(),
                                    df_diagnostico['riesgo_consistencia_tipo'].mean(),
                                    df_diagnostico['riesgo_metadato_faltante'].mean(),
                                    df_diagnostico['riesgo_anomalia_ml'].mean(),
                                ]
                            })
                            riesgos_reporte = riesgos_reporte.sort_values(by='Riesgo Promedio (0-Máx)', ascending=False)
                            riesgos_reporte['Riesgo Promedio (0-Máx)'] = riesgos_reporte['Riesgo Promedio (0-Máx)'].round(2)
                            
                            
                            # === LÓGICA DE RECOMENDACIÓN PRÁCTICA ===
                            
                            recomendacion_final_md = ""
                            
                            riesgo_max_reportado = riesgos_reporte.iloc[0]['Riesgo Promedio (0-Máx)']
                            dimension_riesgo_max = riesgos_reporte.iloc[0]['Dimensión de Riesgo']

                            # Determinar la penalización máxima de la dimensión más riesgosa
                            if 'Datos Incompletos' in dimension_riesgo_max:
                                PENALIZACION_MAX = PENALIZACION_DATOS_INCOMPLETOS
                            elif 'Duplicados Exactos' in dimension_riesgo_max:
                                PENALIZACION_MAX = PENALIZACION_DUPLICADO
                            elif 'Consistencia de Tipo' in dimension_riesgo_max:
                                PENALIZACION_MAX = PENALIZACION_INCONSISTENCIA_TIPO
                            elif 'Metadatos Clave Faltantes' in dimension_riesgo_max:
                                PENALIZACION_MAX = PENALIZACION_METADATO_CLAVE_FALTA
                            elif 'Anomalía Detectada (ML)' in dimension_riesgo_max:
                                PENALIZACION_MAX = PENALIZACION_VENCIMIENTO_ML
                            else:
                                PENALIZACION_MAX = 0.0
                            
                            # Umbral: Se activa si el riesgo promedio excede el 10% de la penalización máxima de esa dimensión
                            UMBRAL_ACTIVACION_RECOMENDACION = 0.10 * PENALIZACION_MAX

                            if riesgo_max_reportado > UMBRAL_ACTIVACION_RECOMENDACION:
                                # Identificar el riesgo más alto
                                riesgo_dimension_max = dimension_riesgo_max
                                
                                # Generar la explicación específica
                                explicacion_especifica = generate_specific_recommendation(riesgo_dimension_max)
                                
                                # Formato de salida con bloques de código para claridad
                                recomendacion_final_md = f"""
El riesgo más alto es por **{riesgo_dimension_max}** ({riesgo_max_reportado:.2f}). Enfoca tu esfuerzo en corregir este problema primero.

<br>

**Detalle y Acciones:**

{explicacion_especifica}
"""

                            if not recomendacion_final_md:
                                recomendacion_final_md = "La Calidad es excelente. No se requieren mejoras prioritarias en las dimensiones analizadas."
                                estado = "🟢 CALIDAD ALTA"
                                color = "green"
                            else:
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
                            
                            # --- DESPLIEGUE DE MÉTRICAS SIMPLIFICADO ---
                            st.subheader("Resultados del Diagnóstico Rápido")
                            
                            col_calidad, col_riesgo = st.columns(2)
                            
                            col_calidad.metric("⭐ Calidad Total del Archivo", f"{calidad_total_final:.1f}%")
                            col_riesgo.metric("Riesgo Promedio Total", f"{riesgo_promedio_total:.2f}")

                            # Despliegue de la Recomendación
                            st.markdown(f"""
                                <div style='border: 2px solid {color}; padding: 15px; border-radius: 5px; background-color: #f9f9f9;'>
                                    <h4 style='color: {color}; margin-top: 0;'>Diagnóstico General: {estado}</h4>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("#### 🔬 Desglose de Riesgos (Auditoría)")
                            
                            # VISUALIZACIÓN DE TABLA DE RIESGOS
                            st.dataframe(
                                riesgos_reporte.style.format({'Riesgo Promedio (0-Máx)': '{:.2f}'}),
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            st.markdown("#### 🚀 Recomendación Prioritaria")
                            st.markdown(recomendacion_final_md)


                except Exception as e:
                    st.error(f"❌ Error al procesar el archivo subido: {e}. Asegúrate de que el formato (CSV, delimitador) es correcto.")

        # ----------------------------------------------------------------------
        # --- SECCIÓN 10: ASISTENTE DE CONSULTA DE DATOS (LLM)
        # ----------------------------------------------------------------------
        setup_data_assistant(df_analisis_completo)

except Exception as e:
    st.error(f"Se produjo un error crítico en el flujo principal del dashboard: {e}")
