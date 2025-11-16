import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import unicodedata

# ----------------------------------------------------
# 0) CONFIGURACIÓN INICIAL
# ----------------------------------------------------
st.set_page_config(
    page_title="Dashboard ALECO Base", 
    layout="wide"
)

# ----------------------------------------------------
# 1) FUNCIÓN DE NORMALIZACIÓN (Necesaria para cargar datos)
# ----------------------------------------------------
def normalize_col(col):
    col = col.strip()
    col = col.upper()
    col = col.replace(" ", "_")
    col = col.replace("(", "")
    col = col.replace(")", "")
    col = col.replace("Ñ", "N")
    col = ''.join(
        c for c in unicodedata.normalize('NFD', col)
        if unicodedata.category(c) != 'Mn'
    )
    return col


# ----------------------------------------------------
# 2) CARGAR CSV Y LIMPIEZA DE COLUMNAS (CRÍTICA)
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

        # Aplicar limpieza básica a las columnas necesarias
        numeric_cols = ['INGRESOS_OPERACIONALES','GANANCIA_PERDIDA','TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO']
        for col in numeric_cols:
            if col in df.columns:
                 df[col] = (
                    df[col].astype(str)
                    .str.replace("$","",regex=False)
                    .str.replace(",","",regex=False)
                    .str.replace(".","",regex=False)
                    .str.replace(" ","",regex=False)
                    .str.replace("−","-",regex=False)
                    .str.replace("(","",regex=False)
                    .str.replace(")","",regex=False)
                    .astype(float)
                )

        # Asegurar que ANO_DE_CORTE sea int (para la lógica del dashboard)
        if 'ANO_DE_CORTE' in df.columns:
            df['ANO_DE_CORTE'] = pd.to_numeric(df['ANO_DE_CORTE'], errors='coerce').fillna(-1).astype(int)
        
        return df

    except Exception as e:
        st.error(f"❌ ERROR al leer o limpiar el CSV: {e}")
        return pd.DataFrame()


# ----------------------------------------------------
# 3) CARGAR MODELO
# ----------------------------------------------------
@st.cache_resource
def load_model():
    model_file = "model.pkl" 
    
    if not os.path.exists(model_file):
        st.error(f"❌ ERROR: Archivo del modelo no encontrado: {model_file}")
        return None

    try:
        model = joblib.load(model_file)
        return model
    except Exception as e:
        st.error(f"❌ ERROR al cargar el modelo: {e}. Revisa las versiones de joblib/XGBoost.")
        return None


# ----------------------------------------------------
# --- INICIO DE LA APLICACIÓN ---
# ----------------------------------------------------

df = load_data()
model = load_model()

if df.empty or model is None:
    st.error("⚠️ La aplicación no puede continuar debido a errores de carga.")
    st.stop()


# ----------------------------------------------------
# 4) DASHBOARD PRINCIPAL Y FILTROS
# ----------------------------------------------------
st.title("📊 Dashboard ALECO: Paso 2 (Filtros y KPIs)")

st.header("1. Filtros y Datos")
col1, col2 = st.columns(2)
with col1:
    # Obtener opciones únicas del DataFrame cargado
    sector_options = ["Todos"] + df["MACROSECTOR"].unique().tolist()
    sector = st.selectbox("Filtrar por Macrosector", sector_options)
with col2:
    region_options = ["Todos"] + df["REGION"].unique().tolist()
    region = st.selectbox("Filtrar por Región", region_options)

# Aplicar filtros
df_filtrado = df.copy()
if sector != "Todos":
    df_filtrado = df_filtrado[df_filtrado["MACROSECTOR"] == sector]
if region != "Todos":
    df_filtrado = df_filtrado[df_filtrado["REGION"] == region]


# Diagnóstico del año de corte
ano_corte_mas_reciente = df_filtrado["ANO_DE_CORTE"].max()

if ano_corte_mas_reciente <= 2000:
    st.error(f"❌ ERROR CRÍTICO: El año de corte más reciente es inválido ({ano_corte_mas_reciente}).")
    st.dataframe(df_filtrado["ANO_DE_CORTE"].value_counts()) # Muestra los valores defectuosos
    st.warning("⚠️ Debes corregir los datos de 'ANO_DE_CORTE' en el CSV original o el error de limpieza de datos.")
    st.stop()

st.info(f"✅ Año de corte más reciente encontrado: **{ano_corte_mas_reciente}**")
st.dataframe(df_filtrado.head(5))


# ----------------------------------------------------
# 5) KPIs AGREGADOS
# ----------------------------------------------------
st.header("2. KPIs Agregados")

if df_filtrado.empty:
    st.warning("No hay datos para los filtros seleccionados.")
else:
    ingresos_total = df_filtrado["INGRESOS_OPERACIONALES"].sum()
    patrimonio_prom = df_filtrado["TOTAL_PATRIMONIO"].mean()

    col_kpi1, col_kpi2 = st.columns(2)
    with col_kpi1:
        st.metric(label="Ingresos Operacionales Totales", value=f"${ingresos_total:,.2f}")
    with col_kpi2:
        st.metric(label="Patrimonio Promedio", value=f"${patrimonio_prom:,.2f}")
        
st.success("🎉 Paso 2 completado. Pasemos a la Predicción.")
