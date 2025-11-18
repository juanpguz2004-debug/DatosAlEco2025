import streamlit as st
import pandas as pd
import numpy as np
import joblib
import unicodedata

# ============================
# 1. FUNCIONES DE NORMALIZACIÓN
# ============================

def normalize_col(col):
    col = col.strip().upper().replace(" ", "_").replace("(", "").replace(")", "").replace("Ñ", "N")
    return ''.join(c for c in unicodedata.normalize('NFD', col) if unicodedata.category(c) != 'Mn')


def safe_le_transform(encoder, val):
    """Transforma un valor con LabelEncoder devolviendo -1 si es desconocido."""
    s = str(val)
    return encoder.transform([s])[0] if s in encoder.classes_ else -1


# ============================
# 2. CARGA DE ACTIVOS
# ============================

xgb_cls = joblib.load("model_clasificacion.pkl")
xgb_reg_gan = joblib.load("model_reg_ganancia.pkl")
xgb_reg_per = joblib.load("model_reg_perdida.pkl")

encoders = joblib.load("label_encoders.pkl")
FEATURES = joblib.load("model_features.pkl")

AGR = joblib.load("growth_rate.pkl")
ANO_BASE = joblib.load("base_year.pkl")

# ============================
# 3. CARGA DEL CSV ORIGINAL
# ============================

@st.cache_data
def load_data():
    df = pd.read_csv("10.000_Empresas_mas_Grandes_del_País_20251115.csv")
    df.columns = [normalize_col(c) for c in df.columns]

    # Limpieza numérica
    numeric_cols = [
        'INGRESOS_OPERACIONALES', 'GANANCIA_PERDIDA', 'TOTAL_ACTIVOS',
        'TOTAL_PASIVOS', 'TOTAL_PATRIMONIO'
    ]

    for col in numeric_cols:
        df[col] = (
            df[col].astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.replace(' ', '', regex=False)
            .str.replace('−', '-', regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Manejo año corte
    df["ANO_DE_CORTE"] = (
        df["ANO_DE_CORTE"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    df["ANO_DE_CORTE"] = pd.to_numeric(df["ANO_DE_CORTE"], errors="coerce").fillna(-1).astype(int)

    return df[df["ANO_DE_CORTE"] > 2000].copy()

df = load_data()

# ============================
# 4. UI STREAMLIT
# ============================

st.title("📊 Dashboard ALECO — Modelo de Dos Partes")

empresa = st.selectbox("Selecciona Empresa", sorted(df["RAZON_SOCIAL"].unique()))
anio_target = st.number_input("Selecciona Año
