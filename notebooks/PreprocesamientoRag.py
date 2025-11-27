import pandas as pd
import numpy as np
import json
import io
from datetime import datetime
import os
from google.colab import files # Módulo de carga para Colab

# =================================================================
# 0. VARIABLES GLOBALES
# =================================================================
ARCHIVO_PROCESADO = "Asset_Inventory_PROCESSED.csv"
KNOWLEDGE_FILE = "knowledge_base.txt"

# =================================================================
# 1. FUNCIÓN DE CARGA DEL ARCHIVO (Manejo de Archivos en Colab)
# =================================================================
def load_processed_data_script(file_path):
    """
    Intenta cargar el archivo por ruta. Si no se encuentra, usa el widget de Colab
    para solicitar al usuario que suba el archivo.
    """

    # 1. Intenta cargar el archivo por la ruta por defecto
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, low_memory=False)
            print(f"✅ Archivo cargado desde la ruta existente: {file_path}")
            return df
        except Exception as e:
             print(f"❌ Error al intentar leer el archivo existente: {e}. Procediendo a solicitar la subida.")


    # 2. Si no existe, usa el widget de carga (esto abre la ventana de diálogo)
    print(f"\n🛑 Archivo '{file_path}' no encontrado. Por favor, selecciona el archivo desde tu disco duro:")

    try:
        # Abre el cuadro de diálogo de carga y espera la selección
        uploaded = files.upload()

        if not uploaded:
            print("❌ No se seleccionó ningún archivo. Cancelando la carga.")
            return pd.DataFrame()

        # El nombre del archivo subido es la clave del diccionario
        uploaded_filename = list(uploaded.keys())[0]

        # Lee el archivo subido
        df = pd.read_csv(io.BytesIO(uploaded[uploaded_filename]), low_memory=False)
        print(f"✅ Archivo subido y cargado exitosamente: {uploaded_filename}")

        return df

    except Exception as e:
        print(f"❌ Error durante la subida o lectura del archivo: {e}")
        return pd.DataFrame()


# =================================================================
# 2. FUNCIÓN DE GENERACIÓN DE CONOCIMIENTO ROBUSTO (VERSION MEJORADA)
# =================================================================
def generate_robust_knowledge_base(df):
    """
    Genera un archivo de texto con resúmenes estadísticos exhaustivos
    para ser utilizado como base de conocimiento avanzada del Agente de IA.
    """
    if df.empty:
        print("❌ El DataFrame está vacío. No se puede generar la base de conocimiento.")
        return None

    # --- Definición de Umbral Local ---
    UMBRAL_RIESGO_ALTO = 3.0

    # --- Preprocesamiento de Columnas (Asegurar existencia) ---
    df_temp = df.copy()
    # Aseguramos que todas las columnas clave existan para los cálculos
    columnas_clave = {
        'uid': np.arange(len(df_temp)),
        'prioridad_riesgo_score': 0.0,
        'completitud_score': 0.0,
        'antiguedad_datos_dias': 0.0,
        'estado_actualizacion': 'N/A',
        # ⚠️ CAMBIO: Usar 'publico' en lugar de 'common_core_public_access_level'
        'publico': 'N/A',
        # 🌟 ADICIÓN: Nueva columna de temas
        'common_core_theme': 'Sin Tema',
        'dueño': 'Desconocido',
        'categoria': 'Sin Categoría',
        'titulo': 'Sin Título',
        'riesgo_datos_incompletos': 0.0,
        'riesgo_consistencia_tipo': 0.0,
        'riesgo_duplicado': 0.0,
        # 🌟 ADICIÓN: Para resultados de ML
        'anomalia_score': 1 # 1: normal, -1: anomalia
    }
    for col, default in columnas_clave.items():
        if col not in df_temp.columns:
            df_temp[col] = default

    # --- Creación de Agrupaciones (BINNING) para análisis avanzado ---
    bins_riesgo = [0, 1, 2, 3, df_temp['prioridad_riesgo_score'].max() + 0.1] # Ajustar el bin superior
    labels_riesgo = ['Bajo (0-1)', 'Medio (1-2)', 'Alto (2-3)', 'Crítico (3+ Vea Max)']
    df_temp['grupo_riesgo'] = pd.cut(df_temp['prioridad_riesgo_score'], bins=bins_riesgo, labels=labels_riesgo, right=False)

    bins_comp = [0, 50, 80, 101]
    labels_comp = ['Bajo (0-50%)', 'Medio (50-80%)', 'Alto (80-100%)']
    df_temp['grupo_completitud'] = pd.cut(df_temp['completitud_score'], bins=bins_comp, labels=labels_comp, right=False)

    knowledge_parts = []

    # --- A. CONTEXTO Y FECHA DE GENERACIÓN ---
    fecha_generacion = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    knowledge_parts.append(f"## A. CONTEXTO Y FECHA DE GENERACIÓN")
    knowledge_parts.append(f"Este resumen se generó el: **{fecha_generacion}**.")
    knowledge_parts.append("El agente debe usar estos datos para responder preguntas, cálculos y rankings de alto nivel.")
    knowledge_parts.append("\n---")

    # --- B. ESTADÍSTICAS GLOBALES CLAVE (KPIs) ---
    knowledge_parts.append("## B. RESUMEN GLOBAL DEL INVENTARIO (KPIs CLAVE)")
    total_activos = len(df_temp)

    riesgo_promedio_total = df_temp['prioridad_riesgo_score'].mean()
    completitud_promedio_total = df_temp['completitud_score'].mean()
    antiguedad_promedio = df_temp['antiguedad_datos_dias'].mean()

    # Cálculo de la fecha promedio de actualización (asumiendo que hoy es la referencia para la antigüedad)
    fecha_hoy = pd.to_datetime(datetime.now().date())
    fecha_promedio_actualizacion = fecha_hoy - pd.Timedelta(days=antiguedad_promedio)

    activos_alto_riesgo = int((df_temp['prioridad_riesgo_score'] >= UMBRAL_RIESGO_ALTO).sum())
    activos_incumplimiento = int((df_temp['estado_actualizacion'] == '🔴 INCUMPLIMIENTO').sum())

    # ⚠️ CAMBIO: Usar columna 'publico' con valor 'public'
    activos_publicos = int((df_temp['publico'] == 'public').sum())
    # 🌟 ADICIÓN: Detección ML
    anomalias_detectadas = int((df_temp['anomalia_score'] == -1).sum())

    global_kpis = {
        "Total de Activos Registrados": int(total_activos),
        "Activos con Acceso Público ('public')": f"{activos_publicos} ({activos_publicos / total_activos * 100:.1f}%)" if total_activos > 0 else "0 (0.0%)",
        "Riesgo Promedio General (Score)": f"{riesgo_promedio_total:.2f}",
        "Completitud Promedio General (%)": f"{completitud_promedio_total:.2f}%",
        "Antigüedad Promedio (Días)": f"{antiguedad_promedio:.0f}",
        "Fecha Promedio de Última Actualización": fecha_promedio_actualizacion.strftime("%Y-%m-%d"),
        "Conteo Activos en Alto Riesgo (Score >= 3.0)": activos_alto_riesgo,
        "Conteo Activos en Incumplimiento de Actualización": activos_incumplimiento,
        # 🌟 ADICIÓN: ML Anomaly Detection
        "Anomalías Detectadas (ML - Score -1)": anomalias_detectadas
    }
    knowledge_parts.append(json.dumps(global_kpis, indent=4))
    knowledge_parts.append("\n---")

    # --- C. DESGLOSE POR ENTIDAD ('dueño') - MEJORADO ---
    knowledge_parts.append("\n## C. ANÁLISIS POR ENTIDAD RESPONSABLE ('dueño')")
    resumen_entidades = df_temp.groupby('dueño', observed=True).agg(
        Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
        Completitud_Promedio=('completitud_score', 'mean'),
        Antiguedad_Promedio_Dias=('antiguedad_datos_dias', 'mean'),
        Activos_Totales=('uid', 'count')
    ).reset_index()

    # 1. Top 5 Entidades por Mayor Conteo de Activos (Responde: ¿qué entidad tiene más activos?)
    top_activos_entidad = resumen_entidades.sort_values(by='Activos_Totales', ascending=False).head(5)
    knowledge_parts.append("### Top 5 Entidades por Mayor Conteo de Activos:")
    knowledge_parts.append(top_activos_entidad.to_markdown(index=False, floatfmt=".2f"))

    # 2. Top 10 Entidades con Mayor Riesgo Promedio (Responde: Top 10 peores entidades)
    top_riesgo = resumen_entidades.sort_values(by='Riesgo_Promedio', ascending=False).head(10)
    knowledge_parts.append("\n### Top 10 Entidades con Mayor Riesgo Promedio (Peores):")
    knowledge_parts.append(top_riesgo.to_markdown(index=False, floatfmt=".2f"))

    # 3. Top 5 Entidades con Peor Completitud Promedio (Más incompletas)
    worst_comp = resumen_entidades.sort_values(by='Completitud_Promedio', ascending=True).head(5)
    knowledge_parts.append("\n### Top 5 Entidades con Peor Completitud Promedio (Más Incompletas):")
    knowledge_parts.append(worst_comp.to_markdown(index=False, floatfmt=".2f"))

    # 4. Top 5 Entidades con Mejor Completitud Promedio (Mejores activos por completitud)
    best_comp = resumen_entidades.sort_values(by='Completitud_Promedio', ascending=False).head(5)
    knowledge_parts.append("\n### Top 5 Entidades con Mejor Completitud Promedio (Mejores):")
    knowledge_parts.append(best_comp.to_markdown(index=False, floatfmt=".2f"))

    # 5. Top 5 Entidades con Menor Riesgo Promedio (Mejores activos por riesgo)
    low_riesgo = resumen_entidades.sort_values(by='Riesgo_Promedio', ascending=True).head(5)
    knowledge_parts.append("\n### Top 5 Entidades con Menor Riesgo Promedio (Mejores):")
    knowledge_parts.append(low_riesgo.to_markdown(index=False, floatfmt=".2f"))
    knowledge_parts.append("\n---")

    # --- D. DESGLOSE POR CATEGORÍA ('categoria') - MEJORADO (Sección base) ---
    knowledge_parts.append("\n## D. ANÁLISIS POR COBERTURA TEMÁTICA ('categoria')")
    resumen_categorias = df_temp.groupby('categoria', observed=True).agg(
        Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
        Completitud_Promedio=('completitud_score', 'mean'),
        Activos_Totales=('uid', 'count')
    ).reset_index()

    # 🌟 ADICIÓN: Categorías con Más Activos
    top_categorias = resumen_categorias.sort_values(by='Activos_Totales', ascending=False).head(5)
    knowledge_parts.append("### Top 5 Categorías con Mayor Número de Activos:")
    knowledge_parts.append(top_categorias.to_markdown(index=False, floatfmt=".2f"))

    # Cruce de Categoría vs. Completitud (Responde: qué categoría tiene más activos incompletos/completos)
    resumen_categoria_comp = df_temp.groupby('categoria', observed=True)['grupo_completitud'].value_counts().unstack(fill_value=0)
    knowledge_parts.append("\n### Conteo de Activos por Categoría y Nivel de Completitud:")
    knowledge_parts.append(resumen_categoria_comp.head(10).to_markdown(floatfmt=".0f"))

    knowledge_parts.append("\n---")

    # --- E. RANKINGS INDIVIDUALES DE ACTIVOS CLAVE - MEJORADO ---
    knowledge_parts.append("\n## E. RANKINGS INDIVIDUALES DE ACTIVOS CLAVE")
    columns_to_select = ['titulo', 'dueño', 'prioridad_riesgo_score']

    # 1. Top 10 Activos con Mayor Score de Riesgo (Peores activos)
    top_riesgo_activos = df_temp[columns_to_select].sort_values(by='prioridad_riesgo_score', ascending=False).head(10)
    knowledge_parts.append("### Top 10 Activos con Mayor Score de Riesgo (Prioridad Máxima - PEOR):")
    knowledge_parts.append(top_riesgo_activos.to_markdown(index=False, floatfmt=".2f"))

    # 2. Top 10 Activos con Menor Score de Riesgo (Mejores activos)
    low_riesgo_activos = df_temp[columns_to_select].sort_values(by='prioridad_riesgo_score', ascending=True).head(10)
    knowledge_parts.append("\n### Top 10 Activos con Menor Score de Riesgo (Prioridad Mínima - MEJOR):")
    knowledge_parts.append(low_riesgo_activos.to_markdown(index=False, floatfmt=".2f"))

    # 3. Top 10 Activos con Mayor Antigüedad de Datos (Menos Actualizados)
    columns_to_select_ant = ['titulo', 'dueño', 'antiguedad_datos_dias']
    top_antiguedad = df_temp[columns_to_select_ant].sort_values(by='antiguedad_datos_dias', ascending=False).head(10)
    knowledge_parts.append("\n### Top 10 Activos con Mayor Antigüedad de Datos (Menos Actualizados):")
    top_antiguedad['antiguedad_datos_dias'] = top_antiguedad['antiguedad_datos_dias'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else 'N/A')
    knowledge_parts.append(top_antiguedad.to_markdown(index=False))
    knowledge_parts.append("\n---")

    # --- F. DESGLOSE CUANTITATIVO POR GRUPO DE RIESGO ---
    knowledge_parts.append("\n## F. DESGLOSE CUANTITATIVO POR GRUPO DE RIESGO")
    resumen_grupos_riesgo = df_temp.groupby('grupo_riesgo', observed=True).agg(
        Activos_Totales=('uid', 'count'),
        Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
        Completitud_Promedio=('completitud_score', 'mean')
    ).reset_index()
    resumen_grupos_riesgo['Porcentaje_Inventario'] = (resumen_grupos_riesgo['Activos_Totales'] / total_activos) * 100
    knowledge_parts.append("### Distribución de Activos por Nivel de Riesgo (Bajo, Medio, Alto, Crítico)")
    knowledge_parts.append(resumen_grupos_riesgo.to_markdown(index=False, floatfmt=".2f"))
    knowledge_parts.append("\n---")

    # --- G. CRUCE DE DATOS: RIESGO vs. COMPLETITUD ---
    knowledge_parts.append("\n## G. MATRIZ DE CRUCE: GRUPO DE RIESGO vs. COMPLETITUD")
    cruce_riesgo_comp = pd.crosstab(df_temp['grupo_riesgo'], df_temp['grupo_completitud'], margins=True, margins_name='TOTAL')
    knowledge_parts.append("### Conteo de Activos por Cuadrante (Filas: Riesgo / Columnas: Completitud)")
    knowledge_parts.append(cruce_riesgo_comp.to_markdown(floatfmt=".0f"))

    cuadrante_critico = df_temp[
        (df_temp['grupo_riesgo'].astype(str).str.contains('Alto|Crítico', na=False)) &
        (df_temp['grupo_completitud'] == 'Bajo (0-50%)')
    ]
    if not cuadrante_critico.empty:
        top_entidades_criticas = cuadrante_critico['dueño'].value_counts().head(3).reset_index()
        top_entidades_criticas.columns = ['Entidad', 'Activos_Criticos_Baja_Comp']
        knowledge_parts.append("\n### Top 3 Entidades en el Cuadrante CRÍTICO (Alto Riesgo y Baja Completitud):")
        knowledge_parts.append(top_entidades_criticas.to_markdown(index=False))
    knowledge_parts.append("\n---")

    # --- H. DESGLOSE POR ESTADO DE ACTUALIZACIÓN ---
    knowledge_parts.append("\n## H. DESGLOSE POR ESTADO DE ACTUALIZACIÓN")
    resumen_estado_act = df_temp.groupby('estado_actualizacion', observed=True).agg(
        Activos_Totales=('uid', 'count'),
        Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
        Antiguedad_Promedio=('antiguedad_datos_dias', 'mean')
    ).reset_index()
    knowledge_parts.append("### Métricas Clave por Estado de Actualización")
    knowledge_parts.append(resumen_estado_act.to_markdown(index=False, floatfmt=".2f"))
    knowledge_parts.append("\n---")

    # --- I. ESQUEMA DEL DATAFRAME ORIGINAL ---
    knowledge_parts.append("\n## I. ESQUEMA DEL DATAFRAME ORIGINAL (Columnas y Tipos)")
    buffer = io.StringIO()
    df_temp.info(buf=buffer)
    knowledge_parts.append(buffer.getvalue())
    knowledge_parts.append("\n---")

    # --- J. ANÁLISIS DE FALLAS DE RIESGO RECURRENTES (General) ---
    knowledge_parts.append("\n## J. ANÁLISIS DE FALLAS DE RIESGO RECURRENTES (Nivel General)")
    risk_cols = ['riesgo_datos_incompletos', 'riesgo_consistencia_tipo', 'riesgo_duplicado']
    resumen_fallas = df_temp[risk_cols].sum().reset_index()
    resumen_fallas.columns = ['Tipo_Riesgo', 'Suma_Score_Total']
    resumen_fallas_rank = resumen_fallas.sort_values(by='Suma_Score_Total', ascending=False)
    knowledge_parts.append("### Recurrencia de Fallas de Riesgo por Suma de Score:")
    knowledge_parts.append(resumen_fallas_rank.to_markdown(index=False, floatfmt=".2f"))
    knowledge_parts.append("\n---")

    # ⚠️ CAMBIO: Usar columna 'publico' con valor 'public'
    # --- K. RANKING DE ACTIVOS PÚBLICOS ('public') PARA INTERVENCIÓN ---
    knowledge_parts.append("\n## K. RANKING DE ACTIVOS PÚBLICOS ('public') PARA INTERVENCIÓN")
    df_public = df_temp[df_temp['publico'] == 'public'].copy()

    if not df_public.empty:
        cols_public = ['titulo', 'dueño', 'prioridad_riesgo_score', 'completitud_score']

        # 1. Activos Públicos con más Riesgo (Requieren más intervención)
        top_riesgo_publico = df_public[cols_public].sort_values(by='prioridad_riesgo_score', ascending=False).head(10)
        knowledge_parts.append("### Top 10 Activos Públicos con Mayor Riesgo (Requieren Intervención):")
        knowledge_parts.append(top_riesgo_publico.to_markdown(index=False, floatfmt=".2f"))

        # 2. Activos Públicos más Incompletos (Completitud Mínima)
        worst_comp_publico = df_public[cols_public].sort_values(by='completitud_score', ascending=True).head(10)
        knowledge_parts.append("\n### Top 10 Activos Públicos más Incompletos (Completitud Mínima):")
        knowledge_parts.append(worst_comp_publico.to_markdown(index=False, floatfmt=".2f"))

        # 3. Activos Públicos más Completos (Completitud Máxima)
        best_comp_publico = df_public[cols_public].sort_values(by='completitud_score', ascending=False).head(10)
        knowledge_parts.append("\n### Top 10 Activos Públicos más Completos (Completitud Máxima):")
        knowledge_parts.append(best_comp_publico.to_markdown(index=False, floatfmt=".2f"))
    else:
        knowledge_parts.append("No se encontraron activos con nivel de acceso 'public' para el análisis.")

    knowledge_parts.append("\n---")

    # 🌟 NUEVA SECCIÓN: L. ANÁLISIS POR TEMA ('common_core_theme')
    knowledge_parts.append("\n## L. ANÁLISIS POR TEMA ('common_core_theme')")
    resumen_temas = df_temp.groupby('common_core_theme', observed=True).agg(
        Riesgo_Promedio=('prioridad_riesgo_score', 'mean'),
        Completitud_Promedio=('completitud_score', 'mean'),
        Activos_Totales=('uid', 'count')
    ).reset_index()

    # 1. Temas con más Activos
    top_activos_tema = resumen_temas.sort_values(by='Activos_Totales', ascending=False).head(5)
    knowledge_parts.append("### Top 5 Temas con Mayor Número de Activos:")
    knowledge_parts.append(top_activos_tema.to_markdown(index=False, floatfmt=".2f"))

    # 2. Temas con Más Falencias (Peor Riesgo Promedio)
    peor_riesgo_tema = resumen_temas.sort_values(by='Riesgo_Promedio', ascending=False).head(5)
    knowledge_parts.append("\n### Top 5 Temas con Mayor Riesgo Promedio (Más Falencias):")
    knowledge_parts.append(peor_riesgo_tema.to_markdown(index=False, floatfmt=".2f"))

    # 3. Temas con Menor Riesgo Promedio (Mejores)
    mejor_riesgo_tema = resumen_temas.sort_values(by='Riesgo_Promedio', ascending=True).head(5)
    knowledge_parts.append("\n### Top 5 Temas con Menor Riesgo Promedio (Mejores):")
    knowledge_parts.append(mejor_riesgo_tema.to_markdown(index=False, floatfmt=".2f"))

    # 4. Temas más Incompletos
    peor_comp_tema = resumen_temas.sort_values(by='Completitud_Promedio', ascending=True).head(5)
    knowledge_parts.append("\n### Top 5 Temas con Peor Completitud Promedio:")
    knowledge_parts.append(peor_comp_tema.to_markdown(index=False, floatfmt=".2f"))

    knowledge_parts.append("\n---")

    # 🌟 NUEVA SECCIÓN: M. RANKING COMBINADO (MEJORES/PEORES POR CATEGORÍA/TEMA)
    knowledge_parts.append("\n## M. RANKING COMBINADO: CALIDAD POR CATEGORÍA Y TEMA")

    # Top 5 Peores Categorías (Alto Riesgo y Baja Completitud)
    peores_categorias = resumen_categorias.sort_values(by=['Riesgo_Promedio', 'Completitud_Promedio'], ascending=[False, True]).head(5)
    knowledge_parts.append("### Top 5 Peores Categorías (Mayor Riesgo / Menor Completitud):")
    knowledge_parts.append(peores_categorias.to_markdown(index=False, floatfmt=".2f"))

    # Top 5 Mejores Categorías (Bajo Riesgo y Alta Completitud)
    mejores_categorias = resumen_categorias.sort_values(by=['Riesgo_Promedio', 'Completitud_Promedio'], ascending=[True, False]).head(5)
    knowledge_parts.append("\n### Top 5 Mejores Categorías (Menor Riesgo / Mayor Completitud):")
    knowledge_parts.append(mejores_categorias.to_markdown(index=False, floatfmt=".2f"))

    # Top 5 Peores Temas (Alto Riesgo y Baja Completitud)
    peores_temas = resumen_temas.sort_values(by=['Riesgo_Promedio', 'Completitud_Promedio'], ascending=[False, True]).head(5)
    knowledge_parts.append("\n### Top 5 Peores Temas (Mayor Riesgo / Menor Completitud):")
    knowledge_parts.append(peores_temas.to_markdown(index=False, floatfmt=".2f"))

    # Top 5 Mejores Temas (Bajo Riesgo y Alta Completitud)
    mejores_temas = resumen_temas.sort_values(by=['Riesgo_Promedio', 'Completitud_Promedio'], ascending=[True, False]).head(5)
    knowledge_parts.append("\n### Top 5 Mejores Temas (Menor Riesgo / Mayor Completitud):")
    knowledge_parts.append(mejores_temas.to_markdown(index=False, floatfmt=".2f"))

    knowledge_parts.append("\n---")

    # 🌟 NUEVA SECCIÓN: N. ANÁLISIS DE FALENCIAS DESAGREGADAS (Riesgos por Categ/Tema)
    knowledge_parts.append("\n## N. ANÁLISIS DE FALENCIAS DESAGREGADAS (Por Categoría y Tema)")

    # 1. Falencias por Categoría
    falencias_categoria = df_temp.groupby('categoria', observed=True).agg(
        Riesgo_Incompleto_Prom=('riesgo_datos_incompletos', 'mean'),
        Riesgo_Consistencia_Prom=('riesgo_consistencia_tipo', 'mean'),
        Riesgo_Duplicado_Prom=('riesgo_duplicado', 'mean'),
        Activos_Totales=('uid', 'count')
    ).reset_index()

    knowledge_parts.append("### Falencias por Categoría (Riesgo Promedio por Dimensión):")
    knowledge_parts.append(falencias_categoria.sort_values(by='Riesgo_Incompleto_Prom', ascending=False).head(5).to_markdown(index=False, floatfmt=".2f"))

    # 2. Falencias por Tema
    falencias_tema = df_temp.groupby('common_core_theme', observed=True).agg(
        Riesgo_Incompleto_Prom=('riesgo_datos_incompletos', 'mean'),
        Riesgo_Consistencia_Prom=('riesgo_consistencia_tipo', 'mean'),
        Riesgo_Duplicado_Prom=('riesgo_duplicado', 'mean'),
        Activos_Totales=('uid', 'count')
    ).reset_index()

    knowledge_parts.append("\n### Falencias por Tema (Riesgo Promedio por Dimensión):")
    knowledge_parts.append(falencias_tema.sort_values(by='Riesgo_Incompleto_Prom', ascending=False).head(5).to_markdown(index=False, floatfmt=".2f"))

    knowledge_parts.append("\n---")

    # 🌟 NUEVA SECCIÓN: O. DETECCIÓN DE ANOMALÍAS (ML)
    knowledge_parts.append("\n## O. DETECCIÓN DE ANOMALÍAS (ML - Isolation Forest)")
    df_anomalias = df_temp[df_temp['anomalia_score'] == -1].copy()

    if not df_anomalias.empty:
        knowledge_parts.append(f"Se detectaron **{len(df_anomalias)}** activos como anomalías (outliers) por el modelo Isolation Forest.")

        # Activos individuales más riesgosos que son anomalías
        top_anomalias = df_anomalias[['titulo', 'dueño', 'prioridad_riesgo_score']].sort_values(by='prioridad_riesgo_score', ascending=False).head(5)
        knowledge_parts.append("### Top 5 Activos Anómalos con Mayor Riesgo:")
        knowledge_parts.append(top_anomalias.to_markdown(index=False, floatfmt=".2f"))

        # Distribución de anomalías por entidad
        anomalias_por_dueno = df_anomalias['dueño'].value_counts().head(5).reset_index()
        anomalias_por_dueno.columns = ['Entidad', 'Num_Anomalias']
        knowledge_parts.append("\n### Top 5 Entidades con Mayor Conteo de Activos Anómalos:")
        knowledge_parts.append(anomalias_por_dueno.to_markdown(index=False))

    else:
        knowledge_parts.append("No se detectaron activos anómalos (score -1) en el inventario.")

    knowledge_parts.append("\n---")


    # Guardar el archivo
    knowledge_content = "\n".join(knowledge_parts)
    with open(KNOWLEDGE_FILE, 'w', encoding='utf-8') as f:
        f.write(knowledge_content)

    print(f"✅ Archivo '{KNOWLEDGE_FILE}' robustecido y generado con {len(df_temp)} registros.")
    return knowledge_content


# =================================================================
# 3. EJECUCIÓN PRINCIPAL DEL SCRIPT DE GENERACIÓN (LLAMADA FINAL)
# =================================================================

print(f"Iniciando la carga del archivo procesado: '{ARCHIVO_PROCESADO}'...")
df_analisis_completo = load_processed_data_script(ARCHIVO_PROCESADO)

if not df_analisis_completo.empty:
    print("Iniciando la generación de la base de conocimiento...")
    generate_robust_knowledge_base(df_analisis_completo)
else:
    print("⚠️ Falló la carga del archivo. No se pudo generar la base de conocimiento.")