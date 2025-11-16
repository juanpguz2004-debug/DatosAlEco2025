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
    page_title="Dashboard ALECO", 
    layout="wide"
)

TARGET_COL = 'GANANCIA_PERDIDA'
OHE_COLS = ['SUPERVISOR', 'REGION', 'MACROSECTOR', 'ANO_DE_CORTE'] 
LE_COLS = ['DEPARTAMENTO_DOMICILIO', 'CIUDAD_DOMICILIO', 'CIIU'] 

# Función de formato de año (CRÍTICO: debe ser la misma que en el entrenamiento)
def format_ano(year):
    year_str = str(year)
    if len(year_str) == 4:
        return f'{year_str[0]},{year_str[1:]}' 
    return year_str

# Función de normalización de columna
def normalize_col(col):
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
            df[col] = df[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'ANO_DE_CORTE' in df.columns:
            df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].astype(str).str.replace(",", "", regex=False)
            df['ANO_DE_CORTE'] = pd.to_numeric(df['ANO_DE_CORTE'], errors='coerce')
            df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].fillna(-1).astype(int)
        
        df = df[df['ANO_DE_CORTE'] > 2000].copy()
        df.dropna(subset=numeric_cols, inplace=True)
        
        return df

    except Exception as e:
        st.error(f"❌ ERROR al leer o limpiar el CSV: {e}")
        return pd.DataFrame()

# ----------------------------------------------------
# 2) CARGAR MODELO Y REFERENCIAS (CRÍTICO)
# ----------------------------------------------------
@st.cache_resource
def load_assets():
    model_file = "model_corregido.pkl" 
    features_file = "model_features.pkl"
    encoders_file = "label_encoders.pkl"
    
    if not (os.path.exists(model_file) and os.path.exists(features_file) and os.path.exists(encoders_file)):
        return None, None, None

    try:
        model = joblib.load(model_file)
        model_features = joblib.load(features_file)
        label_encoders = joblib.load(encoders_file)
        return model, model_features, label_encoders
    except Exception as e:
        st.error(f"❌ ERROR al cargar activos: {e}")
        return None, None, None


# ----------------------------------------------------
# --- INICIO DE LA APLICACIÓN ---
# ----------------------------------------------------

df = load_data()
model, MODEL_FEATURE_NAMES, label_encoders = load_assets()


if df.empty:
    st.error("❌ ERROR FATAL: No se encontraron datos válidos en el CSV.")
    st.stop()
    
if model is None or MODEL_FEATURE_NAMES is None or label_encoders is None:
    st.error("❌ ERROR FATAL: No se pudieron cargar el modelo o las referencias. Verifica los archivos .pkl.")
    st.stop()

# --- Encabezado ---
st.title("📊 Dashboard ALECO: Final")
st.markdown("""
**Reporte de las diez mil empresas más grandes del país.**
Todas las cifras en este reporte se muestran en **Billones de Pesos**.
""")
st.markdown("---") 

ano_corte_mas_reciente_global = df["ANO_DE_CORTE"].max()

# ----------------------------------------------------
# 3) DASHBOARD PRINCIPAL Y FILTROS
# ----------------------------------------------------
st.header("1. Filtros y Datos")
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

st.info(f"✅ Año de corte máximo global: **{ano_corte_mas_reciente_global}**")
st.dataframe(df_filtrado.head(5))

# ----------------------------------------------------
# 4) KPIs AGREGADOS
# ----------------------------------------------------
st.header("2. KPIs Agregados")

ingresos_total = df_filtrado["INGRESOS_OPERACIONALES"].sum()
patrimonio_prom = df_filtrado["TOTAL_PATRIMONIO"].mean()

col_kpi1, col_kpi2 = st.columns(2)
with col_kpi1:
    st.metric(label="Ingresos Operacionales Totales (Billones COP)", value=f"${ingresos_total:,.2f}")
with col_kpi2:
    st.metric(label="Patrimonio Promedio (Billones COP)", value=f"${patrimonio_prom:,.2f}")


# ----------------------------------------------------
# 5) PREDICCIÓN CON COMPARACIÓN (LÓGICA FINAL Y ROBUSTA)
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
        "Selecciona el Año de Predicción (2026 por defecto)", años_futuros, index=0 
    )


# --- Lógica de Predicción ---
try:
    df_empresa = df_filtrado[df_filtrado["RAZON_SOCIAL"] == empresa_seleccionada]
    ano_corte_empresa = df_empresa["ANO_DE_CORTE"].max()
    
    if ano_corte_empresa <= 2000:
        st.error(f"Error: La empresa '{empresa_seleccionada}' no tiene un año de corte válido.")
        st.stop()

    st.info(f"Predicción para **{ano_prediccion}**, comparando contra la última fecha de corte registrada de la empresa: **{ano_corte_empresa}**.")

    row_data = df_empresa[df_empresa["ANO_DE_CORTE"] == ano_corte_empresa].iloc[[0]].copy()
    ganancia_anterior = row_data[TARGET_COL].iloc[0]
    
    # --- PRE-PROCESAMIENTO IDÉNTICO AL ENTRENAMIENTO ---
    row_prediccion = row_data.drop(columns=[TARGET_COL], errors='ignore').copy()
    row_prediccion = row_prediccion.drop(columns=['NIT', 'RAZON_SOCIAL'], errors='ignore')
    row_prediccion["ANO_DE_CORTE"] = ano_prediccion
    
    # 1. Aplicar Label Encoding (Usando los encoders cargados)
    for col in LE_COLS:
        try:
            encoder = label_encoders[col]
            row_prediccion[col] = encoder.transform(row_prediccion[col].astype(str))[0]
            row_prediccion[col] = int(row_prediccion[col]) 
        except ValueError:
             # Valor no visto: se asigna 0 o un valor por defecto
             row_prediccion[col] = 0 
    
    # 2. FIX CRÍTICO: Formato de Año para OHE (Asegura la coincidencia de nombre de columna)
    row_prediccion['ANO_DE_CORTE'] = row_prediccion['ANO_DE_CORTE'].apply(format_ano)

    # 3. Aplicar One-Hot Encoding
    row_prediccion = pd.get_dummies(
        row_prediccion, columns=OHE_COLS, prefix=OHE_COLS, drop_first=True, dtype=int
    )
    
    # 4. Alinear y ordenar las columnas (CRÍTICO)
    missing_cols = set(MODEL_FEATURE_NAMES) - set(row_prediccion.columns)
    for c in missing_cols:
        row_prediccion[c] = 0 
    
    row_prediccion = row_prediccion[MODEL_FEATURE_NAMES].copy()
    
    # 5. Conversión final a numérico (prevención de errores de dtype)
    row_prediccion = row_prediccion.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # --- PREDICCIÓN Y REVERSIÓN ---
    pred_log = model.predict(row_prediccion)[0]
    pred_real = np.expm1(pred_log) # Reversión np.log1p(x) -> e^x - 1
    
    # --- MOSTRAR RESULTADOS ---
    diferencia = pred_real - ganancia_anterior
    
    delta_percent = 0.0
    if ganancia_anterior != 0:
        delta_percent = (diferencia / ganancia_anterior) * 100
    
    delta_display = f"{delta_percent:,.2f}% vs {ano_corte_empresa}"

    st.markdown("#### Resultado de la Predicción")
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.metric(
            label=f"GANANCIA/PÉRDIDA Predicha ({ano_prediccion}) (Billones COP)", 
            value=f"${pred_real:,.2f}",
            delta=delta_display
        )
        
    with col_res2:
        st.metric(
            label=f"G/P Real (Última fecha de corte registrada) (Billones COP)", 
            value=f"${ganancia_anterior:,.2f}",
            delta_color="off"
        )
        
    # Mensaje condicional
    st.markdown("---") 
    if pred_real >= 0:
        if diferencia >= 0:
            st.success(f"📈 Se predice un **aumento** de la ganancia del {delta_percent:,.2f}% respecto al año {ano_corte_empresa} (Ganancia total: ${pred_real:,.2f} Billones COP).")
        else:
            st.warning(f"⚠️ Se predice una **reducción** en la ganancia del {abs(delta_percent):,.2f}% respecto al año {ano_corte_empresa} (Ganancia total: ${pred_real:,.2f} Billones COP).")
    else:
        st.error(f"📉 Se predice una **pérdida** neta para {ano_prediccion} (Pérdida total: ${pred_real:,.2f} Billones COP).")

    st.markdown("---")
    st.markdown("Lo invitamos a participar en la **siguiente encuesta**.")


except Exception as e:
    st.error(f"❌ ERROR generando la predicción: {e}")
    st.caption("Asegúrate de que la empresa seleccionada tiene datos completos y que el modelo es compatible con la estructura de la fila.")
