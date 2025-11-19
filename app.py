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
import os 
# --- Importaciones para el Agente de IA (Usando API nativa de Gemini) ---
from google import genai 
# --- Importación de Plotly Express ---
import plotly.express as px
# --- Importaciones para el Clustering Dinámico/Clasificación (ML) ---
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
# *** NUEVO: Importación para Clasificación Supervisada ***
from sklearn.tree import DecisionTreeClassifier 
# --- Fin de Importaciones para el Clustering Dinámico/Clasificación (ML) ---


warnings.filterwarnings('ignore') # Ocultar advertencias de Pandas/Streamlit

# =================================================================
# 0. VARIABLES GLOBALES Y CONFIGURACIÓN (SIN CAMBIOS)
# =================================================================

ARCHIVO_PROCESADO = "Asset_Inventory_PROCESSED.csv" 
# CRITERIO DE RIESGO
UMBRAL_RIESGO_ALTO = 3.0 

# --- CONFIGURACIÓN DE RIESGOS UNIVERSALES ---
PENALIZACION_DATOS_INCOMPLETOS = 2.0 	
PENALIZACION_INCONSISTENCIA_TIPO = 0.5 	 
PENALIZACION_DUPLICADO = 1.0 	 	 	 
# RIESGO MÁXIMO TEÓRICO CORREGIDO: 2.0 + 0.5 + 1.0 = 3.5
RIESGO_MAXIMO_TEORICO_UNIVERSAL = 3.5 

# ⚠️ CLAVE SECRETA DE GEMINI
# REEMPLAZA ESTE VALOR con tu clave secreta real de Gemini (comienza con AIza...).
# NOTA: En un entorno de producción de Streamlit Cloud, usa st.secrets["GEMINI_API_KEY"]
GEMINI_API_SECRET_VALUE = "Aiza"

# =================================================================
# 1. Funciones de Carga y Procesamiento (SIN CAMBIOS)
# =================================================================

@st.cache_data
def load_processed_data(file_path):
	"""Carga el archivo CSV YA PROCESADO y lo cachea."""
	try:
		df = pd.read_csv(file_path, low_memory=False)
		return df
	except FileNotFoundError:
		return pd.DataFrame()

def clean_and_convert_types_external(df):
	"""Fuerza a las columnas a ser tipo string para asegurar la detección de inconsistencias."""
	
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
		# Aplicar penalización solo si hay inconsistencias en esa columna
		if inconsistencies.any():
			df.loc[inconsistencies, 'riesgo_consistencia_tipo'] = PENALIZACION_INCONSISTENCIA_TIPO
		
	# --- 3. UNICIDAD: Duplicados Exactos ---
	df['es_duplicado'] = df.duplicated(keep=False) 
	df['riesgo_duplicado'] = np.where(
		df['es_duplicado'], PENALIZACION_DUPLICADO, 0.0
	)
	
	return df

def process_external_data(df):
	"""
	Lógica de riesgo universal para el archivo externo subido (AJUSTADA).
	"""
	
	# PASO CLAVE CORREGIDO: Asegurar que los tipos permitan la detección
	df = clean_and_convert_types_external(df)

	# --- 1. EVALUACIÓN DE UNIVERSALES (Completitud, Consistencia, Unicidad) ---
	df = check_universals_external(df)
	
	# --- 2. EVALUACIÓN DE METADATOS A NIVEL DE ARCHIVO (SOLO PARA MÉTRICA) ---
	# MEJORA: Evaluar completitud promedio en TODAS las filas (si las columnas existen)
	campos_clave_universal = ['titulo', 'descripcion', 'dueño'] 
	num_campos_totales_base = len(campos_clave_universal)
	
	df['campos_clave_llenados'] = 0
	
	for campo in campos_clave_universal:
		if campo in df.columns:
			# Sumar 1 si el campo no es NaN para CADA FILA
			df['campos_clave_llenados'] += df[campo].notna().astype(int)
			
	# Calcular la completitud por fila para estas columnas clave y luego el promedio
	df['completitud_metadatos_universal'] = (df['campos_clave_llenados'] / num_campos_totales_base) * 100
	
	# Usar el promedio de esa nueva columna para el score del archivo
	completitud_metadatos_universal_score = df['completitud_metadatos_universal'].mean()
	df['completitud_metadatos_universal'] = completitud_metadatos_universal_score

	
	# --- 3. CÁLCULO FINAL DE RIESGO Y CALIDAD ---
	
	# Score de riesgo universal (SOLO 3 DIMENSIONES)
	df['prioridad_riesgo_score'] = (
		df['riesgo_datos_incompletos'] + 
		df['riesgo_consistencia_tipo'] +
		df['riesgo_duplicado']
	)
	
	# CÁLCULO DE CALIDAD TOTAL DEL ARCHIVO (0% a 100%)
	avg_file_risk = df['prioridad_riesgo_score'].mean()
	quality_score = 100 - (avg_file_risk / RIESGO_MAXIMO_TEORICO_UNIVERSAL * 100)
	
	df['calidad_total_score'] = np.clip(quality_score, 0, 100)

	return df

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
	else:
		return "No se requiere una acción específica o el riesgo detectado es demasiado bajo."

# =================================================================
# FUNCIÓN CORE: CLASIFICACIÓN SUPERVISADA Y PCA
# =================================================================

def run_supervised_segmentation_pca(df_input, MAX_SAMPLE_SIZE=15000):
	"""
	Segmenta activos en 3 grupos (Completos, Aceptables, Incompletos) usando
	Clasificación Supervisada (Decision Tree) entrenado en los scores de riesgo,
	y aplica PCA para visualización interactiva con Plotly.
	"""
	
	# Comprobar columnas críticas
	if 'antiguedad_datos_dias' not in df_input.columns:
		return pd.DataFrame(), None, "La columna 'antiguedad_datos_dias' es necesaria para el ML y no está presente."
	
	if df_input.empty or len(df_input) < 10:
		return pd.DataFrame(), None, "No hay suficientes datos (mínimo 10 filas)."

	# --- 1. MUESTREO (Para rendimiento y visualización clara) ---
	sample_size = min(MAX_SAMPLE_SIZE, len(df_input))
	# Usar un índice limpio para el sample
	df_sample = df_input.reset_index(drop=True).sample(n=sample_size, random_state=42)
	
	# ------------------------------------------------------------
	# 2) CREACIÓN DE LA VARIABLE OBJETIVO (Y) - GROUND TRUTH
	# ------------------------------------------------------------
	# Segmentos basados en el score de riesgo calculado: 'prioridad_riesgo_score'
	
	conditions = [
		(df_sample['prioridad_riesgo_score'] <= 1.0),
		(df_sample['prioridad_riesgo_score'] > 1.0) & (df_sample['prioridad_riesgo_score'] <= 2.0),
		(df_sample['prioridad_riesgo_score'] > 2.0)
	]
	
	choices = ['🟢 Completos', '🟡 Aceptables', '🔴 Incompletos']
	# Esta es la etiqueta de entrenamiento que el modelo debe aprender
	df_sample['TARGET_SEGMENT'] = np.select(conditions, choices, default='Indefinido')

	# ------------------------------------------------------------
	# 3) PREPARACIÓN DE FEATURES (X)
	# ------------------------------------------------------------
	
	# Features que definen el estado de calidad/riesgo
	ML_FEATURES = ['prioridad_riesgo_score', 'datos_por_fila_score', 'completitud_score', 'antiguedad_datos_dias']
	
	df_ml_features = df_sample[ML_FEATURES].fillna(0).copy()
	
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(df_ml_features)
	
	# Preparar Y
	y_train = df_sample['TARGET_SEGMENT']
	
	# ------------------------------------------------------------
	# 4) CLASIFICACIÓN SUPERVISADA (Decision Tree)
	# ------------------------------------------------------------
	# Entrenamos el modelo con las características (X) para predecir los segmentos (Y)
	model = DecisionTreeClassifier(random_state=42)
	model.fit(X_scaled, y_train)
	
	# Predecir los segmentos (en un ambiente real, esto se haría en datos nuevos)
	df_sample['PREDICTED_SEGMENT'] = model.predict(X_scaled)

	# ------------------------------------------------------------
	# 5) PCA (Visualización)
	# ------------------------------------------------------------
	# Usamos el mismo conjunto de características escaladas (X_scaled) para la proyección
	pca = PCA(n_components=2)
	points_2d = pca.fit_transform(X_scaled)

	df_sample['PC1'] = points_2d[:, 0]
	df_sample['PC2'] = points_2d[:, 1]
	
	variance_ratio = pca.explained_variance_ratio_.sum()
	
	return df_sample, variance_ratio, None

# =================================================================
# SECCIÓN 6: ASISTENTE DE CONSULTA DE DATOS (NLP) (SIN CAMBIOS)
# =================================================================

def setup_data_assistant(df):
	"""
	Configura el asistente de consulta de datos usando la API nativa de Gemini.
	"""
	
	st.markdown("---")
	st.header("🧠 Asistente de Consulta de Datos (Análisis de Lenguaje Natural)")
	st.markdown("#### ✅ Pregunta lo que sea sobre la estructura de tus datos (API nativa de Gemini)")
	st.info("Ejemplos: '¿Qué columnas tenemos disponibles?', 'Describe los valores más comunes en la columna dueño'. Si la pregunta no se puede responder con la estructura de los datos, el modelo te lo dirá y te sugerirá una pregunta alternativa.")
	
	# --- 1. VERIFICACIÓN DE CLAVE API Y CONFIGURACIÓN ---
	# Corregido para usar la notación de diccionario
	if GEMINI_API_SECRET_VALUE == "Aiza":
		st.error("🛑 Error de Configuración: La clave API de Gemini no ha sido configurada.")
		st.markdown("Por favor, **reemplaza el placeholder** en el código por el valor secreto real de tu clave `AIza...`.")
		st.markdown("---")
		return

	# --- 2. INICIALIZAR EL CLIENTE GEMINI ---
	try:
		# Usar la clave API directamente para inicializar el cliente nativo
		client = genai.Client(api_key=GEMINI_API_SECRET_VALUE)
		
	except Exception as e:
		st.error(f"❌ Error al inicializar el Cliente Gemini. Verifica tu clave API. Detalle: {e}")
		st.markdown("---")
		return

	# --- 3. PREPARAR CONTEXTO DE DATOS (SCHEMA) ---
	# Usaremos el encabezado y el resumen de tipos para darle contexto al modelo.
	
	# Capturar la información de tipos (df.info()) en una cadena
	buffer = io.StringIO()
	df.info(buf=buffer)
	df_info_str = buffer.getvalue()
	
	# Capturar el encabezado de datos (df.head())
	data_head = df.head(5).to_markdown(index=False)


	# --- 4. INTERFAZ DE USUARIO ---
	user_query = st.text_input(
		"Tu pregunta sobre la ESTRUCTURA del Inventario de Activos:",
		key="nlp_query_simple"
	)

	if st.button("Consultar Estructura (Gemini Simple)", use_container_width=True) and user_query:
		if df.empty:
			st.error("No hay datos cargados para realizar la consulta.")
			return

		with st.spinner(f"El Asistente de Gemini está analizando la estructura de los datos para responder: '{user_query}'..."):
			
			# --- CONSTRUIR PROMPT ROBUSTO ---
			system_prompt = (
				"Eres un Asistente de Análisis de Datos experto. Tu objetivo es responder preguntas sobre la ESTRUCTURA y las MUESTRAS "
				"de un DataFrame de Pandas. NO PUEDES EJECUTAR CÓDIGO de Python para cálculos complejos (sumas, promedios, filtrados extensos), "
				"solo puedes analizar la información de la MUESTRA y el ESQUEMA que se te proporciona.\n"
				
				"CONTEXTO DEL DATAFRAME:\n"
				f"Esquema (df.info()):\n{df_info_str}\n"
				f"Muestra de Datos (df.head()):\n{data_head}\n"
				
				"REGLAS CRÍTICAS DE RESPUESTA:\n"
				"1. Si la pregunta del usuario puede ser respondida directamente con la MUESTRA o el ESQUEMA (ej: '¿Cuáles son las columnas?', '¿De qué tipo es la columna dueño?', '¿Qué valores aparecen en la muestra para la columna X?'), responde de manera concisa y profesional.\n"
				"2. Si la pregunta requiere CÁLCULOS O AGREGACIONES COMPLEJAS sobre todo el dataset (ej: 'Suma de activos', 'Promedio de riesgo', 'Cuántos hay en la categoría X'), DEBES responder ÚNICAMENTE con el siguiente texto exacto:\n"
				"'No puedo responder esa pregunta basándome en los datos disponibles. Mi funcionalidad actual solo me permite analizar el esquema y una pequeña muestra de los datos. Te sugiero preguntar: [SUGERENCIA DE PREGUNTA ALTERNATIVA].'\n"
				"3. La SUGESTIÓN DE PREGUNTA ALTERNATIVA debe ser una pregunta que SÍ se pueda responder con la muestra o el esquema (ej: '¿Qué columnas son de tipo object?', '¿Qué valores tiene la columna dueño en la muestra?', '¿Qué tan viejo es el activo de la primera fila?')."
			)

			try:
				# LLAMADA A LA API DE GEMINI
				response = client.models.generate_content(
					model='gemini-2.5-flash',
					contents=[
						{"role": "user", "parts": [{"text": user_query}]},
					],
					config=genai.types.GenerateContentConfig(
						system_instruction=system_prompt,
						temperature=0.0 # Bajar la temperatura para respuestas más determinísticas
					)
				)
				
				# Mostrar el resultado
				st.success("✅ Respuesta generada por el Asistente de IA:")
				st.markdown(response.text)

			except Exception as e:
				st.error(f"❌ Error durante la llamada a la API de Gemini. Detalle: {e}")
				st.warning("Verifica tu clave API y la conexión.")


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

		# --- SECCIÓN DE SELECCIÓN Y DESGLOSE DE ENTIDAD (SIN CAMBIOS) ---
		owners = df_analisis_completo['dueño'].dropna().unique().tolist()
		owners.sort()
		owners.insert(0, "Mostrar Análisis General")
		
		filtro_dueño = st.selectbox(
			"Selecciona una Entidad para ver su Desglose de Estadísticas:",
			owners
		)
		
		# --- DESGLOSE DE ESTADÍSTICAS (KPIs) (SIN CAMBIOS) ---
		if filtro_dueño != "Mostrar Análisis General":
			df_entidad_seleccionada = df_analisis_completo[df_analisis_completo['dueño'] == filtro_dueño]
			
			if not df_entidad_seleccionada.empty:
				st.subheader(f"Estadísticas Clave para: **{filtro_dueño}**")
				
				total_activos = len(df_entidad_seleccionada)
				# Asumiendo que 'estado_actualizacion' existe y ya se procesó en el pre-cálculo.
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

		# --- BARRA LATERAL (FILTROS SECUNDARIOS) (SIN CAMBIOS) ---
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


		# --- APLICAR FILTROS (Para las Visualizaciones) (SIN CAMBIOS) ---
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
			
			# --- 3. Métricas de la Vista Actual (SIN CAMBIOS) ---
			st.subheader("Métricas de la Vista Actual")
			col_metrica1, col_metrica2, col_metrica3 = st.columns(3)
			col_metrica1.metric("Completitud Promedio", f"{df_filtrado['completitud_score'].mean():.2f}%")
			col_metrica2.metric("Activos en Incumplimiento", f"{(df_filtrado['estado_actualizacion'] == '🔴 INCUMPLIMIENTO').sum()} / {len(df_filtrado)}")
			
			# Check si 'anomalia_score' existe antes de usarlo
			if 'anomalia_score' in df_filtrado.columns:
				col_metrica3.metric("Anomalías Detectadas (ML)", f"{(df_filtrado['anomalia_score'] == -1).sum()}")
			else:
				col_metrica3.metric("Anomalías Detectadas (ML)", "N/A")
			
			st.markdown("---")

			# --- 4. Tabla de Búsqueda y Diagnóstico de Entidades (SIN CAMBIOS) ---
			st.header("🔍 4. Tabla de Búsqueda y Diagnóstico de Entidades")

			# AÑADIDO: Explicación de las nuevas reglas de color
			st.info(f"""
				La tabla usa color condicional para identificar problemas de calidad rápidamente:
				* 🔴 **Riesgo Promedio** > **{UMBRAL_RIESGO_ALTO:.1f}** (Prioridad Máxima).
				* 🔴 **%_Incumplimiento** > **20%** (Problema Operacional).
				* 🔴 **Antigüedad Promedio** > **180 días** (Riesgo de Obsolescencia).
				* 🔴 **Completitud Promedio** < **85%** (Riesgo de Usabilidad).
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
			
			
			# --- FUNCIÓN DE ESTILO (SIN CAMBIOS) ---
			def highlight_metrics(row):
				"""Aplica el estilo de color a toda la fila según las métricas críticas."""
				styles = [''] * len(row)
				
				# 1. Riesgo Promedio (Columna 2)
				if row['Riesgo_Promedio'] > UMBRAL_RIESGO_ALTO:
					styles[2] = 'background-color: #f79999' # Rojo claro
				else:
					styles[2] = 'background-color: #a9dfbf' # Verde claro

				# 2. % Incumplimiento (Columna 6)
				if row['%_Incumplimiento'] > 20:
					styles[6] = 'background-color: #f79999' # Rojo claro
				
				# 3. Antigüedad Promedio (Columna 4)
				if row['Antiguedad_Promedio_Dias'] > 180:
					styles[4] = 'background-color: #f79999' # Rojo claro

				# 4. Completitud Promedio (Columna 3)
				if row['Completitud_Promedio'] < 85:
					styles[3] = 'background-color: #f79999' # Rojo claro
					
				return styles


			# Aplicar la función de estilo a todas las filas
			styled_df = resumen_entidades_busqueda.style.apply(
				highlight_metrics,
				axis=1
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
					'Completitud_Promedio': st.column_config.NumberColumn("Completitud Promedio", format="%.2f%%", help="Rojo < 85%."),
					'Antiguedad_Promedio_Dias': st.column_config.NumberColumn("Antigüedad Promedio (Días)", format="%d", help="Rojo > 180 días."),
					'Incumplimiento_Absoluto': st.column_config.NumberColumn("Activos en Incumplimiento (Count)"),
					'%_Incumplimiento': st.column_config.TextColumn("% Incumplimiento", help="Rojo > 20%")
				},
				hide_index=True
			)

			st.markdown("---")
			
			# --- PESTAÑAS PARA EL "CARRUSEL" DE VISUALIZACIONES ---
			tab1, tab2, tab3 = st.tabs(["1. Ranking de Completitud", "2. Segmentación (Clasificación Supervisada)", "3. Cobertura Temática"])

			with tab1:
				# --- Visualización 1: Ranking de Completitud (Plotly) (SIN CAMBIOS) ---
				st.subheader("1. 📉 Ranking de Entidades por Completitud Promedio (Peor Rendimiento)")
				st.caption("Gráfico interactivo: Usa el hover para ver valores exactos y la barra de herramientas para zoom.")
				
				try:
					COLUMNA_ENTIDAD = 'dueño'
					resumen_completitud = df_filtrado.groupby(COLUMNA_ENTIDAD).agg(
						Total_Activos=('uid', 'count'),
						Completitud_Promedio=('completitud_score', 'mean')
					).reset_index()
					
					entidades_volumen = resumen_completitud[resumen_completitud['Total_Activos'] >= 5]
					df_top_10_peor_completitud = entidades_volumen.sort_values(by='Completitud_Promedio', ascending=True).head(10)
					
					if not df_top_10_peor_completitud.empty:
						
						# *** Usar Plotly Express ***
						fig1 = px.bar(
							df_top_10_peor_completitud,
							x='Completitud_Promedio', 
							y=COLUMNA_ENTIDAD,
							orientation='h', # Barras horizontales
							title='Top 10 Entidades con Peor Completitud Promedio',
							color='Completitud_Promedio',
							color_continuous_scale=px.colors.sequential.Reds_r, # Usar escala de rojos invertida
							labels={
								'Completitud_Promedio': 'Score de Completitud Promedio (%)',
								COLUMNA_ENTIDAD: 'Entidad Responsable'
							},
							hover_data={
								'Completitud_Promedio': ':.2f', # Formato a 2 decimales en el hover
								'Total_Activos': True
							}
						)
						
						# Ajustar layout para mejor visualización
						fig1.update_layout(yaxis={'categoryorder':'total ascending'}) 
						
						st.plotly_chart(fig1, use_container_width=True)
					else:
						st.warning("No hay entidades con suficiente volumen (>= 5 activos) para generar el ranking.")
				except Exception as e:
					st.error(f"❌ ERROR [Visualización 1]: Falló la generación del Gráfico de Completitud (Plotly). Detalle: {e}")

			with tab2:
				# --- Visualización 2: Segmentación de Riesgo (¡CLASIFICACIÓN SUPERVISADA!) ---
				st.subheader("2. 🤖 Segmentación de Riesgo (Clasificación Supervisada y PCA)")
				st.markdown("Se utiliza un modelo de **Clasificación Supervisada (Árbol de Decisión)**, entrenado en las métricas de calidad y riesgo, para predecir si un activo cae en el segmento **Completos**, **Aceptables** o **Incompletos**.")
				st.caption("Gráfico interactivo: Usa el hover para ver el segmento predicho, el riesgo exacto y la entidad.")
				
				with st.spinner("Ejecutando Modelo de Clasificación Supervisada y PCA..."):
					# Llamada a la función de ML Supervisado
					df_segmented_sample, variance_ratio, error_message = run_supervised_segmentation_pca(df_filtrado)
				
				if error_message:
					st.warning(f"⚠️ {error_message}")
				elif not df_segmented_sample.empty:
					try:
						# Definir el mapeo de colores
						color_map = {
							'🟢 Completos': 'green', 
							'🟡 Aceptables': 'gold', 
							'🔴 Incompletos': 'red'
						}
						
						fig2 = px.scatter(
							df_segmented_sample, 
							x='PC1', 
							y='PC2',
							color='PREDICTED_SEGMENT', # Colorear por el segmento PREDICHO por el ML
							color_discrete_map=color_map,
							title=f'Segmentos Predichos por ML (Proyección PCA, {len(df_segmented_sample)} muestras)',
							hover_data={
								'dueño': True,
								'titulo': True,
								'prioridad_riesgo_score': ':.2f', # Mostrar riesgo con 2 decimales
								'PREDICTED_SEGMENT': True,
								'PC1': False, 
								'PC2': False
							}
						)
						
						fig2.update_traces(marker=dict(size=8, opacity=0.8))
						fig2.update_layout(
							xaxis_title=f"Componente Principal 1 (PC1)",
							yaxis_title=f"Componente Principal 2 (PC2)",
							legend_title="Segmento Predicho (ML)"
						)

						st.plotly_chart(fig2, use_container_width=True)
						st.caption(f"Varianza Explicada por PC1 y PC2: **{variance_ratio*100:.2f}%**")
					except Exception as e:
						st.error(f"❌ ERROR [Visualización 2 - Gráfico]: Falló la generación del Gráfico de Segmentación (Plotly). Detalle: {e}")
						st.code(df_segmented_sample.head())
				else:
					st.warning("No se pudo calcular la Segmentación para los datos filtrados.")


			with tab3:
				# --- Visualización 3: Cobertura Temática por Categoría (Plotly) (SIN CAMBIOS) ---
				st.subheader("3. 🗺️ Cobertura Temática por Categoría")
				st.caption("Gráfico interactivo: Usa el hover para ver valores exactos y la barra de herramientas para zoom.")
				
				try:
					COLUMNA_CATEGORIA = 'categoria'
					if COLUMNA_CATEGORIA in df_filtrado.columns:
						conteo_categoria_df = df_filtrado[COLUMNA_CATEGORIA].value_counts().reset_index()
						conteo_categoria_df.columns = [COLUMNA_CATEGORIA, 'Numero de Activos']
						
						# Tomar solo el Top 10 para la visualización
						conteo_categoria_df = conteo_categoria_df.head(10)
					else:
						conteo_categoria_df = pd.DataFrame()

					if not conteo_categoria_df.empty:
						
						# *** Usar Plotly Express ***
						fig3 = px.bar(
							conteo_categoria_df,
							x='Numero de Activos', 
							y=COLUMNA_CATEGORIA,
							orientation='h', # Barras horizontales
							title='Top 10 Categorías con Mayor Cobertura Temática',
							color='Numero de Activos',
							color_continuous_scale=px.colors.sequential.Viridis,
							labels={
								'Numero de Activos': 'Número de Activos',
								COLUMNA_CATEGORIA: 'Categoría'
							},
							hover_data={
								'Numero de Activos': True
							}
						)
						
						# Ajustar layout para mejor visualización
						fig3.update_layout(yaxis={'categoryorder':'total ascending'}) 
						
						st.plotly_chart(fig3, use_container_width=True)

					else:
						st.warning("La columna 'categoria' no contiene suficientes valores para generar la visualización.")
				except Exception as e:
					st.error(f"❌ ERROR [Visualización 3]: Falló la generación del Bar Plot de Categorías (Plotly). Detalle: {e}")


			
			# ----------------------------------------------------------------------
			# --- SECCIÓN 5: DIAGNÓSTICO DE ARCHIVO EXTERNO (SIN CAMBIOS)
			# ----------------------------------------------------------------------
			st.markdown("<hr style='border: 4px solid #f0f2f6;'>", unsafe_allow_html=True)
			st.header("💾 Diagnóstico de Archivo CSV Externo (Calidad Universal)")
			st.markdown(f"Sube un archivo CSV. La **Calidad Total** se calcula en base a 3 dimensiones universales (Riesgo Máximo: **{RIESGO_MAXIMO_TEORICO_UNIVERSAL:.1f}**).")

			uploaded_file = st.file_uploader(
				"Selecciona el Archivo CSV", 
				type="csv"
			)

			if uploaded_file is not None:
				with st.spinner('Analizando archivo...'):
					try:
						uploaded_filename = uploaded_file.name
						# Lógica de lectura robusta con detección de delimitadores
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
								# CORREGIDO: Usar el valor que es promedio de todas las filas
								completitud_universal_promedio = df_diagnostico['completitud_metadatos_universal'].iloc[0] 
								riesgo_promedio_total = df_diagnostico['prioridad_riesgo_score'].mean()

								# Desglose de Riesgos Promedio
								riesgos_reporte = pd.DataFrame({
									'Dimensión de Riesgo': [
										'1. Datos Incompletos (Completitud)',
										'2. Duplicados Exactos (Unicidad)',
										'3. Consistencia de Tipo (Coherencia)',
									],
									'Riesgo Promedio (0-Máx)': [
										df_diagnostico['riesgo_datos_incompletos'].mean(),
										df_diagnostico['riesgo_duplicado'].mean(),
										df_diagnostico['riesgo_consistencia_tipo'].mean(),
									]
								})
								riesgos_reporte = riesgos_reporte.sort_values(by='Riesgo Promedio (0-Máx)', ascending=False)
								riesgos_reporte['Riesgo Promedio (0-Máx)'] = riesgos_reporte['Riesgo Promedio (0-Máx)'].round(2)
								
								
								# === LÓGICA DE RECOMENDACIÓN PRÁCTICA (SIN CAMBIOS) ===
								
								recomendacion_final_md = ""
								
								riesgo_max_reportado = riesgos_reporte.iloc[0]['Riesgo Promedio (0-Máx)']
								
								if riesgo_max_reportado > 0.15:
									# Identificar el riesgo más alto
									riesgo_dimension_max = riesgos_reporte.iloc[0]['Dimensión de Riesgo']
									
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
								
								st.subheader(f"Resultado del Diagnóstico para {uploaded_filename}")
								
								col_res1, col_res2, col_res3 = st.columns(3)
								col_res1.metric("Calidad Total (0-100%)", f"{calidad_total_final:.2f}%", delta=estado, delta_color=color)
								col_res2.metric("Riesgo Promedio Total", f"{riesgo_promedio_total:.2f}")
								col_res3.metric("Metadatos Base Completitud", f"{completitud_universal_promedio:.2f}%")

								st.markdown("### 📋 Desglose de Riesgos por Dimensión")
								st.dataframe(riesgos_reporte, use_container_width=True, hide_index=True)
								
								st.markdown("### 🛠️ Recomendación de Acción Prioritaria")
								st.markdown(recomendacion_final_md)
								
							else:
								st.error("❌ Falló el procesamiento del archivo subido.")
					
					except Exception as e:
						st.error(f"❌ Error al procesar el archivo CSV. Asegúrate de que el formato sea correcto. Detalle: {e}")
				
			
			# ----------------------------------------------------------------------
			# --- SECCIÓN 6: ASISTENTE DE CONSULTA DE DATOS (SIN CAMBIOS)
			# ----------------------------------------------------------------------
			setup_data_assistant(df_analisis_completo)


except Exception as e_principal:
	st.error(f"❌ ERROR FATAL en la ejecución del Dashboard: {e_principal}")
