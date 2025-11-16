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
st.title("Paso 1: Diagnóstico de Carga de Archivos")

# ----------------------------------------------------
# 1) FUNCIÓN DE NORMALIZACIÓN (Necesaria para cargar datos)
# ----------------------------------------------------
# Esta función debe coincidir con la usada en tu Colab
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
        
        # Normalizar columnas
        df.columns = [normalize_col(c) for c in df.columns]

        # Aplicar limpieza básica a las columnas necesarias (para evitar errores de tipo de dato)
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

        # Asegurar que ANO_DE_CORTE sea int (para evitar errores en la predicción)
        if 'ANO_DE_CORTE' in df.columns:
            df['ANO_DE_CORTE'] = pd.to_numeric(df['ANO_DE_CORTE'], errors='coerce').fillna(-1).astype(int)
        
        st.success("✅ CSV cargado y columnas principales limpiadas.")
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
        st.success("✅ Modelo cargado correctamente.")
        return model
    except Exception as e:
        st.error(f"❌ ERROR al cargar el modelo: {e}. Revisa las versiones de joblib/XGBoost.")
        return None


# ----------------------------------------------------
# --- INICIO DE LA EJECUCIÓN ---
# ----------------------------------------------------

df = load_data()
model = load_model()

st.header("1. Resultado de Carga de Datos")
if not df.empty:
    st.dataframe(df.head())
    st.write(f"Filas cargadas: **{len(df)}**")
else:
    st.warning("⚠️ No se pudo cargar el DataFrame.")

st.header("2. Resultado de Carga del Modelo")
if model is not None:
    st.info("El modelo está listo. Podemos pasar al Paso 2: Filtrado y KPIs.")
else:
    st.warning("⚠️ El modelo no está cargado. No se podrá predecir.")
