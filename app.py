import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import unicodedata

# ----------------------------------------------------
# 1) FUNCIONES DE NORMALIZACIÓN (IGUALES A COLAB)
# ----------------------------------------------------

def normalize_col(col):
    col = col.strip()
    col = col.upper()
    col = col.replace(" ", "_")
    col = col.replace("(", "")
    col = col.replace(")", "")
    col = col.replace("Ñ", "N")
    # Eliminar acentos
    col = ''.join(
        c for c in unicodedata.normalize('NFD', col)
        if unicodedata.category(c) != 'Mn'
    )
    return col


# ----------------------------------------------------
# 2) CARGAR CSV
# ----------------------------------------------------
@st.cache_data
def load_data():
    csv_file = "10.000_Empresas_mas_Grandes_del_País_20251115.csv"

    if not os.path.exists(csv_file):
        st.error(f"No se encontró el archivo: {csv_file}")
        return pd.DataFrame()

    df = pd.read_csv(csv_file)

    # Normalizar columnas igual que en el entrenamiento
    df.columns = [normalize_col(c) for c in df.columns]

    # Columnas requeridas para la predicción y vista
    required_cols = [
        'NIT','RAZON_SOCIAL','SUPERVISOR','REGION','DEPARTAMENTO_DOMICILIO',
        'CIUDAD_DOMICILIO','CIIU','MACROSECTOR',
        'INGRESOS_OPERACIONALES','GANANCIA_PERDIDA',
        'TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO','ANO_DE_CORTE'
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas necesarias: {missing}")
        return pd.DataFrame()

    # LIMPIAR columnas numéricas (igual que en Colab)
    numeric_cols = [
        'INGRESOS_OPERACIONALES','GANANCIA_PERDIDA',
        'TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO'
    ]

    for col in numeric_cols:
        df[col] = (
            df[col].astype(str)
            .str.replace("$","",regex=False)
            .str.replace(",","",regex=False)
            .str.replace(".","",regex=False)
            .str.replace(" ","",regex=False)
            .str.replace("−","-",regex=False)
            .str.replace("(","-",regex=False)
            .str.replace(")","",regex=False)
            .astype(float)
        )
    
    # Asegurar que ANO_DE_CORTE sea int (crucial para la comparación)
    df['ANO_DE_CORTE'] = pd.to_numeric(df['ANO_DE_CORTE'], errors='coerce').fillna(-1).astype(int)

    return df


df = load_data()

if df.empty:
    st.stop()


# ----------------------------------------------------
# 3) CARGAR MODELO (CORREG
