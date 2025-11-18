import streamlit as st
import pandas as pd
import numpy as np
import joblib
import unicodedata
import warnings
import cloudpickle 
import os 

# Suprimir advertencias
warnings.filterwarnings("ignore")

# --- 0) CONFIGURACIÓN INICIAL Y CONSTANTES ---
st.set_page_config(layout="wide", page_title="Dashboard ALECO: Modelo de Dos Partes")

st.title("📊 Dashboard ALECO: Modelo de Dos Partes")
st.markdown("Predicción de Ganancia/Pérdida (incluyendo pérdidas reales) usando Modelado de Dos Partes. Todas las cifras se muestran en **Billones de Pesos**.")
st.markdown("---")

TARGET_COL = 'GANANCIA_PERDIDA'
COLS_TO_PROJECT = [
    'INGRESOS_OPERACIONALES', 'TOTAL_ACTIVOS', 
    'TOTAL_PASIVOS', 'TOTAL_PATRIMONIO'
]
# NOTA: 'ANO_DE_CORTE' NO está aquí porque es numérico en el nuevo modelo
OHE_COLS = ['SUPERVISOR', 'REGION', 'MACROSECTOR']
LE_COLS = ['DEPARTAMENTO_DOMICILIO', 'CIUDAD_DOMICILIO', 'CIIU']
AGR = 1.05 

# Definición de los nombres de archivo
FILE_PROCESSED = "dataset_limpio_para_streamlit.csv" 
FILE_RAW = "10.000_Empresas_mas_Grandes_del_País_20251115.csv"

def normalize_col(col):
    col = col.strip().upper().replace(" ", "_").replace("(", "").replace(")", "").replace("Ñ", "N")
    return ''.join(c for c in unicodedata.normalize('NFD', col) if unicodedata.category(c) != 'Mn')

def safe_le_transform(encoder, val):
    s = str(val)
    if pd.isna(s): s = "nan"
    if s in encoder.classes_:
        return encoder.transform([s])[0]
    return -1

# --- 1) CARGA DE DATOS Y ACTIVOS ---
@st.cache_data
def load_data(file_processed, file_raw):
    
    numeric_cols = COLS_TO_PROJECT + [TARGET_COL]
    df = pd.DataFrame()
    
    # Intentar cargar archivo procesado o raw
    file_to_load = file_processed if os.path.exists(file_processed) else (file_raw if os.path.exists(file_raw) else None)
    
    if not file_to_load:
        st.error("No se encontró ningún archivo CSV de datos.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_to_load)
        df.columns = [normalize_col(c) for c in df.columns]
        
        # CORRECCIÓN DE LIMPIEZA DE NÚMEROS (Formato $129.51)
        for col in numeric_cols:
             if col in df.columns:
                 s = df[col].astype(str).str.strip()
                 # Eliminar símbolos de moneda, comas de miles (si las hay), paréntesis
                 s = (s.str.replace('$', '', regex=False)
                       .str.replace(',', '', regex=False) # Eliminar coma (asumiendo formato US o que no hay comas decimales)
                       .str.replace('(', '', regex=False)
                       .str.replace(')', '', regex=False)
                       .str.replace(' ', '', regex=False)
                       .str.replace('−', '-', regex=False))
                 
                 # Convertir a float. NO ELIMINAR EL PUNTO (es el decimal)
                 df[col] = pd.to_numeric(s, errors='coerce').fillna(0.0)

        # Limpieza de ANO_DE_CORTE
        if 'ANO_DE_CORTE' in df.columns:
            df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].astype(str).str.replace(",", "", regex=False)
            df['ANO_DE_CORTE'] = pd.to_numeric(df['ANO_DE_CORTE'], errors='coerce').fillna(-1).astype(int)
        
        df = df[df['ANO_DE_CORTE'] > 2000].copy()
        
        # NOTA: NO DIVIDIMOS POR 1e9 AQUÍ porque los datos ya vienen escalados (según tus KPIs correctos)
        st.success(f"Datos cargados desde **{file_to_load}**.")
        return df

    except Exception as e:
        st.error(f"Error cargando el archivo: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_assets():
    try:
        models = {
            'cls': joblib.load("model_clasificacion.pkl"),
            'reg_gan': joblib.load("model_reg_ganancia.pkl"),
            'reg_per': joblib.load("model_reg_perdida.pkl"),
        }
        assets = {
            'model_features': joblib.load("model_features.pkl"),
            'AGR': joblib.load("growth_rate.pkl"),
            'base_year': joblib.load("base_year.pkl"),
        }
        try:
            assets['label_encoders'] = joblib.load("label_encoders.pkl")
        except:
            with open("label_encoders.pkl", "rb") as f:
                 assets['label_encoders'] = cloudpickle.load(f)
        return models, assets
    except Exception as e:
        st.error(f"Error cargando activos .pkl: {e}")
        return None, None

# Carga inicial
df = load_data(FILE_PROCESSED, FILE_RAW) 
models, assets = load_assets()

if df.empty or not models:
    st.stop()

# Desempaquetar
model_cls, model_reg_gan, model_reg_per = models['cls'], models['reg_gan'], models['reg_per']
label_encoders = assets['label_encoders']
MODEL_FEATURE_NAMES = assets['model_features']
AGR = assets['AGR']
ANO_CORTE_BASE_GLOBAL = assets['base_year']

# --- 2) FILTROS Y KPIS ---
st.header("1. Filtros y Datos")
col1, col2 = st.columns(2)
with col1:
    filtro_macrosector = st.selectbox("Filtrar por Macrosector", ["Todos"] + sorted(df['MACROSECTOR'].dropna().unique().tolist()))
with col2:
    filtro_region = st.selectbox("Filtrar por Región", ["Todos"] + sorted(df['REGION'].dropna().unique().tolist()))

df_filtrado = df.copy()
if filtro_macrosector != "Todos":
    df_filtrado = df_filtrado[df_filtrado['MACROSECTOR'] == filtro_macrosector]
if filtro_region != "Todos":
    df_filtrado = df_filtrado[df_filtrado['REGION'] == filtro_region]

if df_filtrado.empty:
    st.warning("No hay datos disponibles.")
    st.stop()

# KPIs
st.header("2. KPIs Agregados")
def format_currency(val):
    return f"${val:,.2f}"

ingresos_totales = df_filtrado['INGRESOS_OPERACIONALES'].sum()
patrimonio_promedio = df_filtrado['TOTAL_PATRIMONIO'].mean()

c1, c2 = st.columns(2)
c1.metric("Ingresos Operacionales Totales (Billones COP)", format_currency(ingresos_totales))
c2.metric("Patrimonio Promedio (Billones COP)", format_currency(patrimonio_promedio))

st.markdown("---")

# ----------------------------------------------------
# FUNCIÓN DE PREDICCIÓN RECURSIVA
# ----------------------------------------------------
def predict_recursive(row_base, ano_corte_empresa, ano_prediccion_final, 
                      model_cls, model_reg_gan, model_reg_per, 
                      label_encoders, MODEL_FEATURE_NAMES, AGR, 
                      COLS_TO_PROJECT, LE_COLS, OHE_COLS):
    
    # Asegurar que row_base sea una SERIE
    if isinstance(row_base, pd.DataFrame):
        row_current_base = row_base.iloc[0].copy()
    else:
        row_current_base = row_base.copy()
    
    # Asegurar tipos float
    for col in COLS_TO_PROJECT:
        row_current_base[col] = float(row_current_base[col])
    
    años_a_predecir = range(ano_corte_empresa + 1, ano_prediccion_final + 1)
    pred_real_final = 0.0 
    
    for ano_actual in años_a_predecir:
        # A. Proyección
        for col in COLS_TO_PROJECT:
            row_current_base[col] = row_current_base[col] * AGR
            
        # B. Preparar fila de predicción
        row_prediccion = row_current_base.to_frame().T 
        row_prediccion["ANO_DE_CORTE"] = int(ano_actual)
        
        # C. Codificación
        for col in LE_COLS:
            val = row_prediccion[col].iloc[0]
            row_prediccion[col] = safe_le_transform(label_encoders[col], val)
        
        for col in OHE_COLS:
            row_prediccion[col] = row_prediccion[col].astype(str)
            
        row_prediccion_ohe = pd.get_dummies(
            row_prediccion, 
            columns=ohe_cols_to_use=OHE_COLS, prefix=OHE_COLS, drop_first=True, dtype=int
        )
        
        # D. Alineación
        X_pred = pd.DataFrame(0, index=[0], columns=MODEL_FEATURE_NAMES)
        common_cols = [c for c in row_prediccion_ohe.columns if c in X_pred.columns]
        X_pred[common_cols] = row_prediccion_ohe[common_cols]
        
        cols_direct = COLS_TO_PROJECT + LE_COLS + ['ANO_DE_CORTE']
        for col in cols_direct:
            if col in X_pred.columns:
                X_pred[col] = row_prediccion[col].iloc[0]
        
        X_pred = X_pred.astype(float)
        
        # E. Predicción
        pred_cls = model_cls.predict(X_pred)[0]
        
        if pred_cls == 1:
            pred_log = model_reg_gan.predict(X_pred)[0]
            pred_val = np.expm1(pred_log)
        else:
            pred_log = model_reg_per.predict(X_pred)[0]
            pred_val = -np.expm1(pred_log)
            
        pred_real_final = pred_val
        
    return pred_real_final

# ----------------------------------------------------
# SECCIÓN 5: EJECUCIÓN
# ----------------------------------------------------
st.header("3. Predicción de Ganancia/Pérdida")

col_sel_company, col_sel_year = st.columns(2) 

# --- CORRECCIÓN DEL ERROR DE SORTED ---
# Eliminamos nulos y aseguramos que sean string antes de ordenar
lista_empresas = df_filtrado["RAZON_SOCIAL"].dropna().astype(str).unique().tolist()
empresas_disponibles = sorted(lista_empresas)

if not empresas_disponibles:
    st.warning("No hay empresas disponibles.")
    st.stop()

with col_sel_company:
    idx = empresas_disponibles.index("ECOPETROL S.A") if "ECOPETROL S.A" in empresas_disponibles else 0
    empresa_seleccionada = st.selectbox("Selecciona la Empresa", empresas_disponibles, index=idx)

with col_sel_year:
    ano_prediccion = st.selectbox("Año de Predicción", [2025, 2026, 2027, 2028, 2029, 2030])

try:
    df_empresa = df_filtrado[df_filtrado["RAZON_SOCIAL"] == empresa_seleccionada]
    ano_corte_empresa = int(df_empresa["ANO_DE_CORTE"].max())
    
    if ano_prediccion <= ano_corte_empresa:
        st.warning(f"Selecciona un año mayor al último corte ({ano_corte_empresa}).")
    else:
        # Obtener datos base
        row_data = df_empresa[df_empresa["ANO_DE_CORTE"] == ano_corte_empresa].reset_index(drop=True).iloc[[0]]
        ganancia_anterior = row_data[TARGET_COL].iloc[0]
        
        row_base = row_data.drop(columns=[TARGET_COL, 'NIT', 'RAZON_SOCIAL'], errors='ignore').iloc[0]

        st.info(f"Prediciendo para **{ano_prediccion}** (Base: {ano_corte_empresa}) | AGR: {AGR}")
        
        pred_final = predict_recursive(
            row_base, ano_corte_empresa, ano_prediccion,
            model_cls, model_reg_gan, model_reg_per,
            label_encoders, MODEL_FEATURE_NAMES, AGR,
            COLS_TO_PROJECT, LE_COLS, OHE_COLS
        )
        
        # Resultados
        diff = pred_final - ganancia_anterior
        c1, c2 = st.columns(2)
        c1.metric(f"Predicción {ano_prediccion}", format_currency(pred_final), f"{diff:,.2f}", delta_color="normal")
        c2.metric(f"Real {ano_corte_empresa}", format_currency(ganancia_anterior))
        
        if pred_final > 0:
            st.success(f"Se proyecta una GANANCIA de {format_currency(pred_final)}")
        else:
            st.error(f"Se proyecta una PÉRDIDA de {format_currency(abs(pred_final))}")

except Exception as e:
    st.error(f"Error en predicción: {e}")
