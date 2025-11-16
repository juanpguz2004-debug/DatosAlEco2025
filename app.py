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
            # FIX DE LIMPIEZA DE FORMATO (Asume formato de miles sin punto y decimal con coma)
            df[col] = df[col].str.replace('.', '', regex=False) # Elimina puntos de miles
            df[col] = df[col].str.replace(',', '.', regex=False) # Reemplaza coma decimal por punto

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
    features_file = "model_features.
