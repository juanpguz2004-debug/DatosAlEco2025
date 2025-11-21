import streamlit as st
import pandas as pd
import requests
import numpy as np
import re

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Dashboard de Calidad de Datos Abiertos (Fórmulas Oficiales)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTES Y VALORES ASUMIDOS (SEGÚN GUÍA) ---
API_URL = "https://www.datos.gov.co/resource/uzcf-b9dh.json?$limit=100000"
# Variables para cálculos complejos (Necesitan validación de negocio)
RIESGO_ALTO = 3
RIESGO_MEDIO = 2
RIESGO_BAJO = 1
MIN_FILAS_RELEVANCIA = 50


# --- FUNCIONES DE INGESTA DE DATOS (SIN CAMBIOS) ---

@st.cache_data(show_spinner="Conectando a la API y cargando datos...")
def fetch_api_data(url: str) -> pd.DataFrame:
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        st.success(f"Datos cargados exitosamente desde la API. Filas: {len(df)}")
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error al conectar con la API: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al procesar los datos de la API: {e}")
        return pd.DataFrame()

def handle_csv_upload(uploaded_file) -> pd.DataFrame:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"CSV cargado exitosamente. Filas: {len(df)}")
        return df
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")
        return pd.DataFrame()

# --- FUNCIONES AUXILIARES DE LA GUÍA (VARIABLES DE ENTRADA) ---

def calculate_df_stats(df: pd.DataFrame):
    """Calcula las variables fundamentales para las fórmulas."""
    if df.empty:
        return 0, 0, 0, 0, 0

    df_columnas = len(df.columns)
    total_filas = len(df)
    total_celdas = df.size
    total_nulos = df.isnull().sum().sum()
    
    # numColPorcNulos: Número de columnas con más de 1% de nulos.
    numColPorcNulos = (df.isnull().sum() / total_filas > 0.01).sum()
    
    return df_columnas, total_filas, total_celdas, total_nulos, numColPorcNulos

def calculate_placeholder_measures(df: pd.DataFrame):
    """
    Simula variables complejas que dependen de metadatos o reglas de negocio
    que deben ser extraídas del Asset Inventory (ej. 'metadatos_completos').
    """
    df_columnas, total_filas, total_celdas, total_nulos, numColPorcNulos = calculate_df_stats(df)
    
    # 1. Variables para Confidencialidad
    # **NOTA DE IMPLEMENTACIÓN:** Debes identificar columnas PII aquí.
    # Usaremos el proxy de la advertencia si se encontraron columnas sensibles en la ejecución anterior.
    numColConfidencial = 1 if any(col in df.columns for col in ['nombre_completo', 'identificacion']) else 0
    riesgo_total = RIESGO_ALTO if numColConfidencial > 0 else 0
    
    # 2. medidaMetadatosCompletos (Para Credibilidad y Recuperabilidad)
    # **NOTA DE IMPLEMENTACIÓN:** Se basa en cuántos campos obligatorios de metadatos (título, descripción, etc.) están llenos. Asumimos 5/10.
    medidaMetadatosCompletos = 5.0 
    
    # 3. medidaPublicadorValido (Para Credibilidad)
    # **NOTA DE IMPLEMENTACIÓN:** 10 si la columna 'department' está presente y tiene un valor válido.
    medidaPublicadorValido = 10.0 if 'department' in df.columns and df['department'].notna().sum() > 0 else 0.0
    
    # 4. medidaColDescValida (Para Credibilidad)
    # **NOTA DE IMPLEMENTACIÓN:** 10 si las columnas tienen descripciones/glosarios asociados (lo cual no está en el JSON plano, pero se asume si tiene buena Comprensibilidad).
    medidaColDescValida = 5.0
    
    # 5. medidaFilas (Para Relevancia)
    # **NOTA DE IMPLEMENTACIÓN:** Válida tamaño mínimo y pocos nulos.
    medidaFilas = 10.0 if total_filas >= MIN_FILAS_RELEVANCIA and (total_nulos / total_celdas) < 0.2 else 0.0
    
    # 6. Variables Exactitud (Sintáctica y Semántica)
    # **NOTA DE IMPLEMENTACIÓN:** Muy difícil de automatizar sin reglas. Asumimos 5/10 columnas son correctas.
    numColValoresUnicosSimilares = df_columnas * 0.5 
    numColNoSimSemantica = df_columnas * 0.5

    # 7. metadatosAuditados (Para Recuperabilidad)
    # **NOTA DE IMPLEMENTACIÓN:** Se asume que los metadatos de auditoría (fechas de creación/actualización) están presentes y son válidos.
    metadatosAuditados = 10.0 if 'updated_at' in df.columns else 0.0
    
    return {
        'df_columnas': df_columnas,
        'total_filas': total_filas,
        'total_celdas': total_celdas,
        'total_nulos': total_nulos,
        'numColPorcNulos': numColPorcNulos,
        'numColConfidencial': numColConfidencial,
        'riesgo_total': riesgo_total,
        'medidaMetadatosCompletos': medidaMetadatosCompletos,
        'medidaPublicadorValido': medidaPublicadorValido,
        'medidaColDescValida': medidaColDescValida,
        'medidaFilas': medidaFilas,
        'numColValoresUnicosSimilares': numColValoresUnicosSimilares,
        'numColNoSimSemantica': numColNoSimSemantica,
        'metadatosAuditados': metadatosAuditados
    }

# --- FUNCIONES DE CÁLCULO DE LOS 17 CRITERIOS (FÓRMULAS OFICIALES) ---

# Criterio 15. Relevancia (Depende de medidas internas)
def calculate_relevance(df: pd.DataFrame, measures: dict) -> float:
    """FÓRMULA: (medidaCategoria + medidaFilas) / 2"""
    # **NOTA DE IMPLEMENTACIÓN:** medidaCategoria (coincidencia con temas oficiales) no se puede calcular
    # desde el JSON plano. Asumimos un valor base si tiene etiquetas ('tags').
    medidaCategoria = 10.0 if 'tags' in df.columns else 5.0
    medidaFilas = measures['medidaFilas']
    return (medidaCategoria + medidaFilas) / 2

# Criterio 6. Confidencialidad (Fórmula condicional)
def calculate_confidentiality(df: pd.DataFrame, measures: dict) -> float:
    """FÓRMULA CONDICIONAL: Aplica penalización por columnas sensibles."""
    numColConfidencial = measures['numColConfidencial']
    riesgo_total = measures['riesgo_total']
    df_columnas = measures['df_columnas']
    
    if numColConfidencial == 0:
        return 10.0
    else:
        # 10 - (riesgo_total / dfColumnas * numColConfidencial * 3)
        # Se normaliza la penalización por 10 para que esté en escala de 0 a 100.
        penalty_factor = (riesgo_total / df_columnas) * numColConfidencial * 3
        # Máximo 100
        return max(0.0, 100.0 - (penalty_factor * 10))

# Criterio 4. Completitud (Fórmula compuesta)
def calculate_completeness(df: pd.DataFrame, measures: dict) -> float:
    """FÓRMULA: (medidaCompletitudDatos + medidaCompletitudCol + medidaColNoVacias) / 3"""
    total_nulos = measures['total_nulos']
    total_celdas = measures['total_celdas']
    df_columnas = measures['df_columnas']
    numColPorcNulos = measures['numColPorcNulos']

    if total_celdas == 0 or df_columnas == 0:
        return 0.0

    # a) Completitud de datos (Penalización exponencial por Nulos)
    # 10 * (1 - (totalNulos / totalCeldas)^1.5)
    medidaCompletitudDatos = 10 * (1 - (total_nulos / total_celdas)**1.5)
    
    # b) Completitud por columnas (Penalización cuadrática por Columnas con >1% Nulos)
    # 10 * (1 - (numColPorcNulos / dfColumnas)^2)
    medidaCompletitudCol = 10 * (1 - (numColPorcNulos / df_columnas)**2)
    
    # c) Columnas no vacías (Promedio proporcional de columnas sin nulos)
    # **NOTA DE IMPLEMENTACIÓN:** La guía dice "Promedio proporcional de columnas sin nulos".
    num_cols_no_null = (df.isnull().sum() == 0).sum()
    medidaColNoVacias = 10 * (num_cols_no_null / df_columnas)

    # El score final se escala de 0-100
    score_10_scale = (medidaCompletitudDatos + medidaCompletitudCol + medidaColNoVacias) / 3
    return score_10_scale * 10

# Criterio 7. Consistencia (Basada en varianza y unicidad de columnas)
def calculate_consistency(df: pd.DataFrame, measures: dict) -> float:
    """FÓRMULA: (columnas_cumplen_criterios / dfColumnas)"""
    df_columnas = measures['df_columnas']
    if df_columnas == 0: return 0.0
    
    columnas_cumplen = 0
    
    for col in df.columns:
        # Se requiere que una columna cumpla: 1) Varianza >= 0.1 y 2) Valores únicos >= 2
        cumple_var = False
        cumple_unique = False
        
        # 2) Valores únicos >= 2
        if df[col].nunique() >= 2:
            cumple_unique = True
            
        # 1) Varianza >= 0.1 (Solo aplica a tipos numéricos)
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].var() >= 0.1:
                cumple_var = True
        else:
            # Para columnas no numéricas (texto), asumimos cumplimiento de varianza
            # si los únicos son >= 2 (se consideran las dos condiciones en una)
            cumple_var = True 

        if cumple_var and cumple_unique:
            columnas_cumplen += 1
            
    # El score final se escala de 0-100
    return (columnas_cumplen / df_columnas) * 100

# Criterio 8. Credibilidad (Fórmula ponderada)
def calculate_credibility(df: pd.DataFrame, measures: dict) -> float:
    """FÓRMULA: 0.70 * medMetadatosCompletos + 0.05 * medPublicadorValido + 0.25 * medColDescValida"""
    mMC = measures['medidaMetadatosCompletos'] # Max 10.0
    mPV = measures['medidaPublicadorValido']   # Max 10.0
    mCDV = measures['medidaColDescValida']     # Max 10.0

    # Los scores internos se escalan a 100.
    cred_score_10_scale = (0.70 * mMC) + (0.05 * mPV) + (0.25 * mCDV)
    return cred_score_10_scale * 10

# Criterio 11. Exactitud Sintáctica (Penalización por valores similares)
def calculate_exactitud_sintactica(measures: dict) -> float:
    """FÓRMULA: 10 * (1 - (numColValoresUnicosSimilares / dfColumnas)^2)"""
    df_columnas = measures['df_columnas']
    nCVS = measures['numColValoresUnicosSimilares']
    
    if df_columnas == 0: return 0.0

    # 10 * (1 - (nCVS / dfColumnas)^2)
    score_10_scale = 10 * (1 - (nCVS / df_columnas)**2)
    return score_10_scale * 10

# Criterio 11. Exactitud Semántica (Penalización por no similares semánticamente)
def calculate_exactitud_semantica(measures: dict) -> float:
    """FÓRMULA: 10 - (1 - (numColNoSimSemantica / dfColumnas)^2)"""
    df_columnas = measures['df_columnas']
    nCNS = measures['numColNoSimSemantica']

    if df_columnas == 0: return 0.0

    # 10 - (1 - (nCNS / dfColumnas)^2)
    score_10_scale = 10 - (1 - (nCNS / df_columnas)**2)
    return score_10_scale * 10

# Criterio 11. Exactitud (Combinación Sintáctica y Semántica)
def calculate_accuracy(df: pd.DataFrame, measures: dict) -> float:
    """Criterio 11: Exactitud = Promedio simple de Sintáctica y Semántica."""
    sintact_score = calculate_exactitud_sintactica(measures)
    semant_score = calculate_exactitud_semantica(measures)
    return (sintact_score + semant_score) / 2

# Criterio 14. Recuperabilidad (Fórmula compuesta)
def calculate_recoverability(accessibility_score: float, measures: dict) -> float:
    """FÓRMULA: (accesibilidad + medidaMetadatosCompletos*10 + metadatosAuditados*10) / 3"""
    # Escalar medidas internas a 100 para un promedio simple
    mMC = measures['medidaMetadatosCompletos'] * 10 
    mA = measures['metadatosAuditados'] * 10
    
    return (accessibility_score + mMC + mA) / 3

# Criterios que no tienen fórmula explícita, se implementa lógica proxy:

# Criterio 1. Accesibilidad
def calculate_accessibility(df: pd.DataFrame) -> float:
    """Evalúa si: hay metadatos, archivo es descargable, está en formato abierto."""
    # Si la carga API es exitosa (descargable, formato abierto JSON/CSV), es 100%
    return 100.0 if not df.empty else 0.0

# Criterio 2. Actualidad (Lógica proxy de 90 días)
def calculate_actuality(df: pd.DataFrame) -> float:
    """Depende de Fecha de última actualización y Frecuencia declarada."""
    date_column = 'updated_at' # Asunción común en Socrata
    if df.empty or date_column not in df.columns: return 0.0

    try:
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy.get(date_column), errors='coerce', utc=True)
        df_copy.dropna(subset=[date_column], inplace=True)

        three_months_ago = pd.Timestamp.now(tz='UTC') - pd.DateOffset(months=3)
        recent_count = df_copy[df_copy[date_column] >= three_months_ago].shape[0]
        
        # Ponderación basada en la proporción reciente
        return min(100.0, (recent_count / len(df)) * 100) 
    except Exception:
        return 0.0

# Criterio 17. Unicidad (Lógica proxy de detección de duplicados)
def calculate_uniqueness(df: pd.DataFrame) -> float:
    """Menciona: se agregan niveles de riesgo y penalización por duplicados."""
    if df.empty: return 0.0
    total_rows = len(df)
    unique_rows = len(df.drop_duplicates())
    # Penalización simple: (Filas Únicas / Total de Filas) * 100
    return (unique_rows / total_rows) * 100

# Criterio 5. Conformidad (Lógica proxy de cumplimiento de estándares)
def calculate_conformity(df: pd.DataFrame) -> float:
    """Depende del cumplimiento de estándares, formatos y normativas."""
    # Asumimos que si la columna 'resource_type' existe y tiene valores, cumple el estándar de metadatos.
    col = 'resource_type'
    if df.empty or col not in df.columns: return 0.0
    
    conforming_rows = df[col].notna().sum()
    total_rows = len(df)
    return (conforming_rows / total_rows) * 100

# Criterio 4. Comprensibilidad (Lógica proxy de claridad de nombres)
def calculate_comprehensibility(df: pd.DataFrame) -> float:
    """Evalúa: nombres claros, glosarios, descripciones."""
    col_names = df.columns.tolist()
    # Puntuación: % de nombres de columna que son cortos, usan snake_case y son alfanuméricos
    understandable_count = sum(1 for col in col_names if len(col) < 30 and re.match(r'^[a-z0-9_]+$', col))
    return (understandable_count / len(col_names)) * 100

# Criterio 10. Eficiencia (Fórmula compuesta sustituta)
def calculate_efficiency(uniqueness_score: float, completeness_score: float) -> float:
    """Se calcula como combinación de: completitud de datos, columnas no duplicadas, filas no duplicadas."""
    # Usamos Unicidad y Completitud como proxies.
    # Score 100% si ambas son perfectas, promedio si no.
    return (uniqueness_score + completeness_score) / 2

# Criterio 16. Trazabilidad (Lógica proxy)
def calculate_traceability(actuality_score: float, credibility_score: float) -> float:
    """Depende de que haya historial de versiones, creación y actualización."""
    # Trazabilidad = (Actualidad + Credibilidad) / 2 (El proxy más fuerte)
    return (actuality_score + credibility_score) / 2

# Criterio 12. Portabilidad
def calculate_portability(completeness_score: float, conformity_score: float) -> float:
    """FÓRMULA: 0.50 * portabilidad_base + 0.25 * conformidad + 0.25 * completitud"""
    # Portabilidad base se asume 100% si el formato es estándar (JSON/CSV)
    portabilidad_base = 100.0
    
    return (0.50 * portabilidad_base) + (0.25 * conformity_score) + (0.25 * completeness_score)

# Criterio 13. Precisión (Lógica proxy de variabilidad)
def calculate_precision(df: pd.DataFrame, consistency_score: float) -> float:
    """El criterio fue sustituido por medidas de variabilidad y valores únicos (vinculado a Consistencia)."""
    # Usamos Consistencia como proxy principal, ya que mide la varianza y los valores únicos.
    return consistency_score

# Criterio 9. Disponibilidad
def calculate_availability(accessibility_score: float, actuality_score: float) -> float:
    """FÓRMULA: (accesibilidad + actualidad) / 2"""
    return (accessibility_score + actuality_score) / 2


# --- FUNCIÓN PRINCIPAL DE CÁLCULO Y DISPLAY ---

def calculate_and_display_metrics(df: pd.DataFrame):
    """
    Calcula y muestra las 17 métricas de calidad de datos usando las fórmulas de la Guía 2025.
    """
    if df.empty:
        st.info("No hay datos cargados para calcular las métricas.")
        return

    # 1. CÁLCULO DE VARIABLES DE ENTRADA (Métricas internas)
    measures = calculate_placeholder_measures(df)
    
    # 2. CÁLCULO DE CRITERIOS BASE E INTERMEDIOS
    accessibility_score = calculate_accessibility(df) # Criterio 15 (Texto)
    actuality_score = calculate_actuality(df)         # Criterio 2 (Texto)
    uniqueness_score = calculate_uniqueness(df)       # Criterio 17 (Proxy)
    
    # Criterios con Fórmulas de la Guía 2025
    completeness_score = calculate_completeness(df, measures)     # Criterio 4 (Fórmula)
    confidentiality_score = calculate_confidentiality(df, measures) # Criterio 6 (Fórmula condicional)
    relevance_score = calculate_relevance(df, measures)           # Criterio 15 (Fórmula compuesta)
    consistency_score = calculate_consistency(df, measures)       # Criterio 7 (Fórmula)
    credibility_score = calculate_credibility(df, measures)       # Criterio 8 (Fórmula ponderada)
    accuracy_score = calculate_accuracy(df, measures)             # Criterio 11 (Fórmula compuesta)
    
    # Criterios Proxy / No Matemáticos
    conformity_score = calculate_conformity(df)                   # Criterio 5 (Proxy)
    comprehensibility_score = calculate_comprehensibility(df)     # Criterio 4 (Proxy)

    # 3. CÁLCULO DE CRITERIOS COMPUESTOS / DEPENDIENTES
    disponibility_score = calculate_availability(accessibility_score, actuality_score) # Criterio 9 (Fórmula)
    efficiency_score = calculate_efficiency(uniqueness_score, completeness_score) # Criterio 10 (Proxy)
    recoverability_score = calculate_recoverability(accessibility_score, measures) # Criterio 14 (Fórmula)
    traceability_score = calculate_traceability(actuality_score, credibility_score) # Criterio 16 (Proxy)
    portability_score = calculate_portability(completeness_score, conformity_score) # Criterio 12 (Fórmula)
    precision_score = calculate_precision(df, consistency_score)  # Criterio 13 (Proxy)
    
    # 4. AGRUPACIÓN DE MÉTRICAS (Los 17 Criterios)
    metrics = {
        "1. Accesibilidad": accessibility_score,
        "2. Actualidad": actuality_score,
        "3. Completitud": completeness_score,
        "4. Comprensibilidad": comprehensibility_score,
        "5. Conformidad": conformity_score,
        "6. Confidencialidad": confidentiality_score,
        "7. Consistencia": consistency_score,
        "8. Credibilidad": credibility_score,
        "9. Disponibilidad": disponibility_score,
        "10. Eficiencia": efficiency_score,
        "11. Exactitud": accuracy_score,
        "12. Portabilidad": portability_score,
        "13. Precisión": precision_score,
        "14. Recuperabilidad": recoverability_score,
        "15. Relevancia": relevance_score,
        "16. Trazabilidad": traceability_score,
        "17. Unicidad": uniqueness_score,
    }

    # CÁLCULO DEL SCORE GLOBAL DE CALIDAD
    overall_score = np.mean(list(metrics.values()))
    st.markdown(f"## 🏆 Score Global de Calidad: **{overall_score:.2f}%**")

    # 5. VISUALIZACIÓN DE MÉTRICAS (KPIs)
    st.subheader("Evaluación Detallada de los 17 Criterios (%)")
    
    cols = st.columns(8) 
    
    for i, (name, value) in enumerate(metrics.items()):
        score = round(value, 2)
        with cols[i % 8]:
            st.metric(label=name, value=f"{score}%")
            
    st.markdown("---")
    
    # 6. PERFILADO DETALLADO (Completitud por Columna)
    st.subheader("Detalle de la Completitud por Atributo (No afecta el cálculo de Completitud 2025)")
    completeness_detail = pd.DataFrame({
        'Atributo': df.columns,
        'Valores No Nulos': df.count().values,
        'Total Filas': len(df),
        'Completitud (%)': (df.count().values / len(df)) * 100
    }).sort_values(by='Completitud (%)', ascending=False)
    
    st.dataframe(completeness_detail, use_container_width=True)

    st.markdown("---")

    # 7. TABLA DE DATOS (Muestra)
    st.subheader("Vista Previa del Dataset")
    st.dataframe(df.head(10), use_container_width=True)


# --- LAYOUT DE LA APLICACIÓN STREAMLIT (SIN CAMBIOS) ---

def main():
    st.title("Sistema de Monitoreo de Calidad de Datos Abiertos")
    st.caption("Implementación de la Guía de Calidad e Interoperabilidad 2025 para Asset Inventory.")

    # SIDEBAR: Opciones de Ingesta
    st.sidebar.header("Opciones de Ingesta de Datos")
    ingestion_mode = st.sidebar.radio(
        "Seleccione el origen de datos:",
        ('Asset Inventory (API)', 'Cargar CSV Local')
    )
    
    df_data = pd.DataFrame()
    
    if ingestion_mode == 'Asset Inventory (API)':
        st.sidebar.code(API_URL, language='text')
        if st.sidebar.button("Cargar Datos desde API"):
            df_data = fetch_api_data(API_URL)
            
    elif ingestion_mode == 'Cargar CSV Local':
        uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])
        if uploaded_file is not None:
            df_data = handle_csv_upload(uploaded_file)
            
    # MAIN CONTENT
    if not df_data.empty:
        calculate_and_display_metrics(df_data)
    else:
        st.info("Utilice la barra lateral para cargar el Asset Inventory desde la API o un archivo CSV local.")


if __name__ == "__main__":
    main()
