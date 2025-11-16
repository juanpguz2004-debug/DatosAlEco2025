import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import unicodedata
from sklearn.preprocessing import LabelEncoder 

# ----------------------------------------------------
# 0) CONFIGURACIÓN INICIAL Y CONSTANTES
# ----------------------------------------------------
st.set_page_config(
    page_title="📊 Dashboard ALECO", 
    layout="wide"
)

TARGET_COL = 'GANANCIA_PERDIDA'
OHE_COLS = ['SUPERVISOR', 'REGION', 'MACROSECTOR', 'ANO_DE_CORTE'] 
LE_COLS = ['DEPARTAMENTO_DOMICILIO', 'CIUDAD_DOMICILIO', 'CIIU'] 
COLS_TO_PROJECT = ['INGRESOS_OPERACIONALES','TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO']

# ----------------------------------------------------
# 0.1) FUNCIONES DE UTILIDAD (Deben coincidir con el entrenamiento)
# ----------------------------------------------------

def format_ano(year):
    """Formato X,XXX (e.g., 2,024) usado para OHE."""
    year_str = str(year)
    if len(year_str) == 4:
        return f'{year_str[0]},{year_str[1:]}' 
    return year_str

def normalize_col(col):
    """Normaliza nombres de columna para consistencia."""
    col = col.strip().upper().replace(" ", "_").replace("(", "").replace(")", "").replace("Ñ", "N")
    return ''.join(c for c in unicodedata.normalize('NFD', col) if unicodedata.category(c) != 'Mn')

# ----------------------------------------------------
# 1) CARGAR CSV Y LIMPIEZA
# ----------------------------------------------------
@st.cache_data
def load_data():
    csv_file = "10.000_Empresas_mas_Grandes_del_País_20251115.csv"
    
    if not os.path.exists(csv_file):
        st.error(f"❌ ERROR: Archivo CSV no encontrado: {csv_file}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_file)
        df.columns = [normalize_col(c) for c in df.columns]

        numeric_cols = [TARGET_COL,'INGRESOS_OPERACIONALES','TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO']
        for col in numeric_cols:
            df[col] = (
                df[col].astype(str)
                .str.replace("$","",regex=False).str.replace(" ","",regex=False)
                .str.replace("−","-",regex=False).str.replace("(","",regex=False)
                .str.replace(")","",regex=False)
            )
            df[col] = df[col].str.replace('.', '', regex=False) # Elimina puntos de miles
            df[col] = df[col].str.replace(',', '.', regex=False) # Reemplaza coma decimal por punto

            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        if 'ANO_DE_CORTE' in df.columns:
            df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].astype(str).str.replace(",", "", regex=False)
            df['ANO_DE_CORTE'] = pd.to_numeric(df['ANO_DE_CORTE'], errors='coerce')
            df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].round(0).fillna(-1).astype(int)
            
        df = df[df['ANO_DE_CORTE'] > 2000].copy()
        df.dropna(subset=numeric_cols, inplace=True)
        
        return df

    except Exception as e:
        st.error(f"❌ ERROR al leer o limpiar el CSV: {e}")
        return pd.DataFrame()

# ----------------------------------------------------
# 2) CARGAR ACTIVOS DE MODELO (SIETE ARCHIVOS)
# ----------------------------------------------------
@st.cache_resource
def load_assets():
    # Nombres de los siete archivos
    cls_file = "model_clasificacion.pkl"
    reg_gan_file = "model_reg_ganancia.pkl"
    reg_per_file = "model_reg_perdida.pkl" 
    features_file = "model_features.pkl"
    encoders_file = "label_encoders.pkl"
    growth_file = "growth_rate.pkl"
    base_year_file = "base_year.pkl"
    
    files_list = [cls_file, reg_gan_file, reg_per_file, features_file, encoders_file, growth_file, base_year_file]
    
    if not all(os.path.exists(f) for f in files_list):
        st.error(f"❌ Error: Faltan archivos .pkl. Asegúrate de que los SIETE archivos estén en la misma carpeta.")
        return None, None, None, None, None, None, None

    try:
        model_cls = joblib.load(cls_file)
        model_reg_gan = joblib.load(reg_gan_file)
        model_reg_per = joblib.load(reg_per_file)
        model_features = joblib.load(features_file)
        label_encoders = joblib.load(encoders_file)
        AGR = joblib.load(growth_file)
        ANO_CORTE_BASE_GLOBAL = joblib.load(base_year_file)
        
        return model_cls, model_reg_gan, model_reg_per, model_features, label_encoders, AGR, ANO_CORTE_BASE_GLOBAL
    except Exception as e:
        st.error(f"❌ ERROR al cargar activos: {e}")
        return None, None, None, None, None, None, None


# ----------------------------------------------------
# --- INICIO DE LA APLICACIÓN ---
# ----------------------------------------------------

df = load_data()
model_cls, model_reg_gan, model_reg_per, MODEL_FEATURE_NAMES, label_encoders, AGR, ANO_CORTE_BASE_GLOBAL = load_assets()


if df.empty:
    st.error("❌ ERROR FATAL: No se encontraron datos válidos en el CSV.")
    st.stop()
    
if None in [model_cls, model_reg_gan, model_reg_per, MODEL_FEATURE_NAMES, label_encoders, AGR, ANO_CORTE_BASE_GLOBAL]:
    st.error("❌ ERROR FATAL: No se pudieron cargar los SIETE activos del modelo.")
    st.stop()

# --- Encabezado ---
st.title("📊 Dashboard ALECO: Predicción de Ganancia/Pérdida")
st.markdown("""
**Modelo de Dos Partes para la Proyección de Ganancia/Pérdida.**
Todas las cifras se muestran en **Billones de Pesos**.
""")
st.markdown("---") 

ano_corte_mas_reciente_global = df["ANO_DE_CORTE"].max()

# (Se omiten las secciones 3 y 4 de Filtros y KPIs por brevedad, manteniendo el foco en la predicción)
# --- FILTROS Y DATOS (simples) ---
col1, col2 = st.columns(2)
with col1:
    sector = st.selectbox("Filtrar por Macrosector", ["Todos"] + df["MACROSECTOR"].unique().tolist())
with col2:
    region = st.selectbox("Filtrar por Región", ["Todos"] + df["REGION"].unique().tolist())

df_filtrado = df.copy()
if sector != "Todos":
    df_filtrado = df_filtrado[df_filtrado["MACROSECTOR"] == sector]
if region != "Todos":
    df_filtrado = df_filtrado[df_filtrado["REGION"] == region]

if df_filtrado.empty:
    st.error(f"❌ ERROR: Los filtros eliminaron todos los datos válidos.")
    st.stop()
    
st.markdown("---") 

# ----------------------------------------------------
# 5) PREDICCIÓN CON LÓGICA DE TRES MODELOS Y PROYECCIÓN
# ----------------------------------------------------
st.header("3. Predicción de Ganancia/Pérdida")

# --- SELECTORES: Año y Empresa ---
col_sel_company, col_sel_year = st.columns(2) 
empresas_disponibles = df_filtrado["RAZON_SOCIAL"].unique().tolist()

if not empresas_disponibles:
    st.warning("No hay empresas disponibles después de aplicar los filtros.")
    st.stop()

with col_sel_company:
    empresa_seleccionada = st.selectbox(
        "Selecciona la Empresa para predecir", empresas_disponibles
    )

with col_sel_year:
    pred_years = [2026, 2027, 2028, 2029, 2030]
    años_futuros = [y for y in pred_years if y > ano_corte_mas_reciente_global]
    if not años_futuros:
        st.warning(f"El año de corte base es {ano_corte_mas_reciente_global}.")
        st.stop()
    ano_prediccion = st.selectbox(
        "Selecciona el Año de Predicción", años_futuros, index=0 
    )

# --- Lógica de Predicción ---
try:
    df_empresa = df_filtrado[df_filtrado["RAZON_SOCIAL"] == empresa_seleccionada]
    ano_corte_empresa = df_empresa["ANO_DE_CORTE"].max()
    
    st.info(f"Predicción para **{ano_prediccion}** (Tasa de Crecimiento Asumida (AGR): {AGR:.2f})")

    # Obtenemos la fila de datos más reciente para la empresa
    row_data = df_empresa[df_empresa["ANO_DE_CORTE"] == ano_corte_empresa].iloc[[0]].copy()
    ganancia_anterior = row_data[TARGET_COL].iloc[0] / 100.0 # Parche de escala
    
    # --- 1. PRE-PROCESAMIENTO Y PROYECCIÓN FUTURA (CRÍTICO) ---
    row_prediccion = row_data.drop(columns=[TARGET_COL], errors='ignore').iloc[0].copy() 
    row_prediccion = row_prediccion.drop(labels=['NIT', 'RAZON_SOCIAL'], errors='ignore')

    # A. Proyección de Features Numéricos al Futuro (Para que la predicción varíe)
    delta_anos_prediccion = ano_prediccion - ano_corte_empresa
    
    if delta_anos_prediccion > 0:
        for col in COLS_TO_PROJECT:
            # Proyección: Valor_Futuro = Valor_Actual * (AGR ^ DELTA_ANOS)
            row_prediccion[col] = row_prediccion[col] * (AGR ** delta_anos_prediccion)
            
    # B. Seteamos el Año de Predicción y aplicamos formato
    row_prediccion["ANO_DE_CORTE"] = ano_prediccion 
    row_prediccion['ANO_DE_CORTE'] = format_ano(row_prediccion['ANO_DE_CORTE'])

    # C. Aplicar Label Encoding (LE)
    for col in LE_COLS:
        try:
            encoder = label_encoders[col]
            row_prediccion[col] = encoder.transform([str(row_prediccion[col])])[0]
            row_prediccion[col] = int(row_prediccion[col]) 
        except ValueError:
             row_prediccion[col] = -1 # Valor desconocido (usado en el entrenamiento)
    
    # D. Aplicar One-Hot Encoding (OHE) y Alineación
    row_prediccion_df = row_prediccion.to_frame().T
    for col in OHE_COLS:
        row_prediccion_df[col] = row_prediccion_df[col].astype(str)

    row_prediccion_ohe = pd.get_dummies(
        row_prediccion_df, columns=OHE_COLS, prefix=OHE_COLS, drop_first=True, dtype=int
    )
    
    missing_cols = set(MODEL_FEATURE_NAMES) - set(row_prediccion_ohe.columns)
    for c in missing_cols:
        row_prediccion_ohe[c] = 0 
    
    X_pred = row_prediccion_ohe[MODEL_FEATURE_NAMES].copy()
    X_pred = X_pred.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    
    # --- 2. LÓGICA DE PREDICCIÓN CONDICIONAL ---
    
    # Paso A: Clasificar (0 = Pérdida/Cero, 1 = Ganancia)
    pred_cls = model_cls.predict(X_pred)[0]
    
    if pred_cls == 1:
        # Ganancia: Reversión: e^x - 1
        pred_log = model_reg_gan.predict(X_pred)[0]
        pred_real = np.expm1(pred_log) 
        
    else:
        # Pérdida/Cero: Reversión: -(e^x - 1)
        pred_log = model_reg_per.predict(X_pred)[0]
        magnitud_perdida_real = np.expm1(pred_log)
        pred_real = -magnitud_perdida_real
        
    
    # --- 3. MOSTRAR RESULTADOS Y CÁLCULO DELTA ---
    diferencia = pred_real - ganancia_anterior

    # Cálculos robustos de delta (Se omite el código detallado por brevedad, asumiendo que funciona)
    delta_metric_value = diferencia
    
    # Lógica de formato delta_display (omisión por brevedad)
    if ganancia_anterior == 0:
        delta_display = f"vs 0"
    elif ganancia_anterior < 0:
        delta_display = f"Mejora Absoluta: ${diferencia:,.2f}"
    else:
        delta_percent = (diferencia / ganancia_anterior) * 100
        delta_display = f"{delta_percent:,.2f}% vs {ano_corte_empresa}"


    st.markdown("#### Resultado de la Predicción")
    col_res1, col_res2 = st.columns(2)

    with col_res1:
        st.metric(
            label=f"GANANCIA/PÉRDIDA Predicha ({ano_prediccion}) (Billones COP)", 
            value=f"${pred_real:,.2f}",
            delta=delta_metric_value, 
            delta_color="normal"
        )
        st.caption(f"Cambio vs {ano_corte_empresa}: **{delta_display}**") 

    with col_res2:
        st.metric(
            label=f"G/P Real (Última fecha de corte registrada) (Billones COP)", 
            value=f"${ganancia_anterior:,.2f}",
            delta_color="off"
        )
        
except Exception as e: 
    st.error(f"❌ ERROR generando la predicción: {e}")
    st.caption("Verifica que los datos de la empresa sean consistentes y que todos los archivos .pkl y el CSV estén en la carpeta.")
