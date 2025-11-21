import streamlit as st
import pandas as pd
import requests
import numpy as np
import re
import math
from datetime import timedelta

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

# **ACTUALIZACIÓN:** Constante para Frecuencia (Asunción)
# En un sistema real, este valor se leería de un metadato declarado.
DEFAULT_UPDATE_FREQUENCY_DAYS = 365 

# **ACTUALIZACIÓN:** Constante para Conformidad (Lista de Validación Ejemplo)
# Usado para validar la columna de 'departamento' o 'entidad'.
VALID_DEPARTMENTS = ['ANTIOQUIA', 'ATLÁNTICO', 'BOGOTÁ, D.C.', 'BOLÍVAR', 'BOYACÁ', 'CALDAS', 
                     'CAQUETÁ', 'CAUCA', 'CESAR', 'CHOCÓ', 'CÓRDOBA', 'CUNDINAMARCA', 'HUILA', 
                     'LA GUAJIRA', 'MAGDALENA', 'META', 'NARIÑO', 'NORTE DE SANTANDER', 
                     'QUINDÍO', 'RISARALDA', 'SANTANDER', 'SUCRE', 'TOLIMA', 'VALLE DEL CAUCA']


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
    Simula variables complejas que dependen de metadatos o reglas de negocio.
    """
    df_columnas, total_filas, total_celdas, total_nulos, numColPorcNulos = calculate_df_stats(df)
    
    # 1. Variables para Confidencialidad
    # Si se detecta una columna con 'nombre' o 'identificacion', se considera sensible.
    numColConfidencial = 1 if any(col in df.columns for col in ['nombre_completo', 'identificacion']) else 0
    riesgo_total = RIESGO_ALTO if numColConfidencial > 0 else 0
    
    # 2. medidaMetadatosCompletos (Para Credibilidad y Recuperabilidad)
    # Proxy: Asumimos 5/10 si hay columnas clave presentes
    medidaMetadatosCompletos = 5.0
    
    # 3. medidaPublicadorValido (Para Credibilidad)
    # Proxy: 10 si la columna 'department' está presente y con datos.
    medidaPublicadorValido = 10.0 if 'department' in df.columns and df['department'].notna().sum() > 0 else 0.0
    
    # 4. medidaColDescValida (Para Credibilidad)
    medidaColDescValida = 5.0
    
    # 5. medidaFilas (Para Relevancia)
    medidaFilas = 10.0 if total_filas >= MIN_FILAS_RELEVANCIA and (total_nulos / total_celdas) < 0.2 else 0.0
    
    # 6. Variables Exactitud (Sintáctica y Semántica)
    numColValoresUnicosSimilares = df_columnas * 0.5 
    numColNoSimSemantica = df_columnas * 0.5

    # 7. metadatosAuditados (Para Recuperabilidad)
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

# ***************************************************************
# ** CRITERIOS ACTUALIZADOS CON FÓRMULAS DEL USUARIO **
# ***************************************************************

# Criterio 2. Actualidad (Actualizado con la fórmula 0 o 10)
def calculate_actuality(df: pd.DataFrame) -> float:
    """FÓRMULA: 0 o 10 (escalado a 100) basado en la Frecuencia Declarada."""
    # MODIFICAR: Inserta el nombre REAL de la columna de fecha de actualización de tu dataset.
    date_column = 'fecha_de_ultima_actualizacion' # Reemplaza con el nombre REAL (ej. 'updated_at')
    update_frequency_days = DEFAULT_UPDATE_FREQUENCY_DAYS
    
    if df.empty or date_column not in df.columns: return 0.0

    try:
        df_copy = df.copy()
        # Intentar convertir la columna de fecha, forzar a UTC para comparación
        df_copy[date_column] = pd.to_datetime(df_copy.get(date_column), errors='coerce', utc=True)
        
        # Eliminar nulos o inválidos. Si hay al menos un registro válido, se evalúa.
        df_copy.dropna(subset=[date_column], inplace=True)
        if df_copy.empty: return 0.0

        # En el Asset Inventory, evaluamos el registro MÁS RECIENTE
        fecha_actualizacion_max = df_copy[date_column].max()
        fecha_actual = pd.Timestamp.now(tz='UTC')

        # Diferencia: FechaActual - FechaActualizacion (Resultado en días)
        diff_days = (fecha_actual - fecha_actualizacion_max).days

        # Aplicar la condición:
        # A_actualidad = { 0 si (FechaActual – FechaActualizacion) > FrecuenciaActualizacion 
        #               { 10 si (FechaActual – FechaActualizacion) ≤ FrecuenciaActualizacion
        if diff_days > update_frequency_days:
            return 0.0 # 0 puntos
        else:
            return 100.0 # 10 puntos (escalado a 100)
    except Exception:
        return 0.0

# Criterio 5. Conformidad (Actualizado con la fórmula exponencial)
def calculate_conformity(df: pd.DataFrame) -> float:
    """FÓRMULA: Conformidad = e^(-5 * ProporcionErrores) * 100 (para escala)."""
    # MODIFICAR: Inserta el nombre REAL de la columna a validar (ej. 'departamento', 'tipo_recurso').
    validated_column = 'nombre_del_departamento' # Reemplaza con el nombre REAL (ej. 'department')
    
    if df.empty or validated_column not in df.columns: return 0.0
    
    # 1. Preparar la columna para la validación
    validating_series = df[validated_column].fillna('').astype(str).str.upper().str.strip()

    # 2. Total Valores Validados: Filas que no están vacías
    total_valores_validados = (validating_series.str.len() > 0).sum()
    
    if total_valores_validados == 0: return 0.0

    # 3. Num Valores Incorrectos: Filas que NO están en la lista VALID_DEPARTMENTS
    # Excluyo los valores vacíos del conteo de errores
    is_valid = validating_series.isin(VALID_DEPARTMENTS)
    is_non_empty = validating_series.str.len() > 0
    
    num_valores_incorrectos = (~is_valid & is_non_empty).sum()

    # 4. Proporción de Errores
    # ProporcionErrores = NumValoresIncorrectos / TotalValoresValidados
    proporcion_errores = num_valores_incorrectos / total_valores_validados

    # 5. Aplicar la Fórmula de Conformidad (exponencial)
    # C_conformidad = e^(-5 * ProporcionErrores) * 100 (escalado a 100)
    conformity_score = math.exp(-5 * proporcion_errores) * 100
    
    return max(0.0, min(100.0, conformity_score))

# ***************************************************************
# ** RESTO DE CRITERIOS (Sin cambios en lógica de fórmula) **
# ***************************************************************

# Criterio 15. Relevancia (Depende de medidas internas)
def calculate_relevance(df: pd.DataFrame, measures: dict) -> float:
    """FÓRMULA: (medidaCategoria + medidaFilas) / 2"""
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
        return 100.0 # Escalado de 10 a 100
    else:
        penalty_factor = (riesgo_total / df_columnas) * numColConfidencial * 3
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
    medidaCompletitudDatos = 10 * (1 - (total_nulos / total_celdas)**1.5)
    
    # b) Completitud por columnas (Penalización cuadrática por Columnas con >1% Nulos)
    medidaCompletitudCol = 10 * (1 - (numColPorcNulos / df_columnas)**2)
    
    # c) Columnas no vacías 
    num_cols_no_null = (df.isnull().sum() == 0).sum()
    medidaColNoVacias = 10 * (num_cols_no_null / df_columnas)

    score_10_scale = (medidaCompletitudDatos + medidaCompletitudCol + medidaColNoVacias) / 3
    return score_10_scale * 10

# Criterio 7. Consistencia (Basada en varianza y unicidad de columnas)
def calculate_consistency(df: pd.DataFrame, measures: dict) -> float:
    """FÓRMULA: (columnas_cumplen_criterios / dfColumnas)"""
    df_columnas = measures['df_columnas']
    if df_columnas == 0: return 0.0
    
    columnas_cumplen = 0
    for col in df.columns:
        cumple_unique = df[col].nunique() >= 2
        cumple_var = True # Asunción para columnas no numéricas
        
        if pd.api.types.is_numeric_dtype(df[col].dropna()):
            cumple_var = df[col].var() >= 0.1
        
        if cumple_var and cumple_unique:
            columnas_cumplen += 1
            
    return (columnas_cumplen / df_columnas) * 100

# Criterio 8. Credibilidad (Fórmula ponderada)
def calculate_credibility(df: pd.DataFrame, measures: dict) -> float:
    """FÓRMULA: 0.70 * medMetadatosCompletos + 0.05 * medPublicadorValido + 0.25 * medColDescValida"""
    mMC = measures['medidaMetadatosCompletos']
    mPV = measures['medidaPublicadorValido']
    mCDV = measures['medidaColDescValida']

    cred_score_10_scale = (0.70 * mMC) + (0.05 * mPV) + (0.25 * mCDV)
    return cred_score_10_scale * 10

# Criterio 11. Exactitud Sintáctica
def calculate_exactitud_sintactica(measures: dict) -> float:
    """FÓRMULA: 10 * (1 - (numColValoresUnicosSimilares / dfColumnas)^2)"""
    df_columnas = measures['df_columnas']
    nCVS = measures['numColValoresUnicosSimilares']
    if df_columnas == 0: return 0.0
    score_10_scale = 10 * (1 - (nCVS / df_columnas)**2)
    return score_10_scale * 10

# Criterio 11. Exactitud Semántica
def calculate_exactitud_semantica(measures: dict) -> float:
    """FÓRMULA: 10 - (1 - (numColNoSimSemantica / dfColumnas)^2)"""
    df_columnas = measures['df_columnas']
    nCNS = measures['numColNoSimSemantica']
    if df_columnas == 0: return 0.0
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
    mMC = measures['medidaMetadatosCompletos'] * 10 
    mA = measures['metadatosAuditados'] * 10
    return (accessibility_score + mMC + mA) / 3

# Criterio 1. Accesibilidad
def calculate_accessibility(df: pd.DataFrame) -> float:
    """Evalúa si: hay metadatos, archivo es descargable, está en formato abierto."""
    return 100.0 if not df.empty else 0.0

# Criterio 17. Unicidad
def calculate_uniqueness(df: pd.DataFrame) -> float:
    """Menciona: se agregan niveles de riesgo y penalización por duplicados."""
    if df.empty: return 0.0
    total_rows = len(df)
    unique_rows = len(df.drop_duplicates())
    return (unique_rows / total_rows) * 100

# Criterio 4. Comprensibilidad
def calculate_comprehensibility(df: pd.DataFrame) -> float:
    """Evalúa: nombres claros, glosarios, descripciones."""
    col_names = df.columns.tolist()
    understandable_count = sum(1 for col in col_names if len(col) < 30 and re.match(r'^[a-z0-9_]+$', col))
    return (understandable_count / len(col_names)) * 100

# Criterio 10. Eficiencia
def calculate_efficiency(uniqueness_score: float, completeness_score: float) -> float:
    """Se calcula como combinación de: completitud de datos, columnas no duplicadas, filas no duplicadas."""
    return (uniqueness_score + completeness_score) / 2

# Criterio 16. Trazabilidad
def calculate_traceability(actuality_score: float, credibility_score: float) -> float:
    """Depende de que haya historial de versiones, creación y actualización."""
    return (actuality_score + credibility_score) / 2

# Criterio 12. Portabilidad
def calculate_portability(completeness_score: float, conformity_score: float) -> float:
    """FÓRMULA: 0.50 * portabilidad_base + 0.25 * conformidad + 0.25 * completitud"""
    portabilidad_base = 100.0
    return (0.50 * portabilidad_base) + (0.25 * conformity_score) + (0.25 * completeness_score)

# Criterio 13. Precisión
def calculate_precision(df: pd.DataFrame, consistency_score: float) -> float:
    """El criterio fue sustituido por medidas de variabilidad y valores únicos (vinculado a Consistencia)."""
    return consistency_score

# Criterio 9. Disponibilidad
def calculate_availability(accessibility_score: float, actuality_score: float) -> float:
    """FÓRMULA: (accesibilidad + actualidad) / 2"""
    return (accessibility_score + actuality_score) / 2


# --- FUNCIÓN PRINCIPAL DE CÁLCULO Y DISPLAY (SIN CAMBIOS) ---

def calculate_and_display_metrics(df: pd.DataFrame
