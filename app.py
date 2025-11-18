import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="ALECO — Modelo de Dos Partes", layout="wide")

st.title("📊 ALECO — Modelo de Dos Partes (Ganancia / Pérdida)")
st.markdown("Predicción multianual usando modelos XGBoost entrenados en Colab.")

# ============================
# 1. CARGA DE ARCHIVOS
# ============================

@st.cache_data
def load_raw_data():
    return pd.read_csv("10.000_Empresas_mas_Grandes_del_País_20251115.csv")

@st.cache_data
def load_processed_features():
    return pd.read_csv("dataset_procesado.csv")

@st.cache_resource
def load_models():
    cls = joblib.load("model_clasificacion.pkl")
    reg_g = joblib.load("model_reg_ganancia.pkl")
    reg_p = joblib.load("model_reg_perdida.pkl")
    enc = joblib.load("label_encoders.pkl")
    feats = joblib.load("model_features.pkl")
    agr = joblib.load("growth_rate.pkl")
    base_year = joblib.load("base_year.pkl")
    return cls, reg_g, reg_p, enc, feats, agr, base_year

df_raw = load_raw_data()
df_features = load_processed_features()
cls, reg_g, reg_p, encoders, MODEL_FEATURE_NAMES, AGR, BASE_YEAR = load_models()

st.success("✓ Modelos y dataset procesado cargados correctamente.")

# ============================
# 2. SELECCIÓN DE EMPRESA
# ============================

if "RAZON_SOCIAL" not in df_raw.columns:
    st.error("El CSV original debe contener la columna 'RAZON_SOCIAL'.")
    st.stop()

empresa = st.selectbox("Selecciona una empresa:", df_raw["RAZON_SOCIAL"].unique())

df_empresa = df_raw[df_raw["RAZON_SOCIAL"] == empresa]

if df_empresa.empty:
    st.error("No se encontró información para la empresa seleccionada.")
    st.stop()

ultimo_ano = int(df_empresa["ANO_DE_CORTE"].max())

anio_target = st.number_input(
    "Selecciona Año objetivo:",
    min_value=ultimo_ano + 1,
    max_value=2035,
    value=ultimo_ano + 1,
    step=1
)

# ============================
# 3. OBTENER EL VECTOR DE FEATURES
# ============================

# Tomamos SOLO la última fila real de esa empresa
idx = df_empresa.index[-1]

try:
    x0 = df_features.loc[idx].copy()
except:
    st.error("No existe fila en dataset_procesado.csv que coincida con la empresa seleccionada.")
    st.stop()

# ============================
# 4. PREDICCIÓN RECURSIVA
# ============================

def predecir_anio(x, ano_nuevo):
    """Genera predicción de GAN/PER para un año específico."""
    x_mod = x.copy()
    x_mod["ANO_DE_CORTE"] = ano_nuevo

    proba_gan = cls.predict_proba([x_mod])[0][1]
    es_gan = 1 if proba_gan >= 0.5 else 0

    if es_gan:
        pred_log = reg_g.predict([x_mod])[0]
        return np.expm1(pred_log)
    else:
        pred_log = reg_p.predict([x_mod])[0]
        return -np.expm1(pred_log)

pred = None
x_running = x0.copy()

for year in range(ultimo_ano + 1, anio_target + 1):
    pred = predecir_anio(x_running, year)
    # aplicar crecimiento AGR
    x_running["INGRESOS_OPERACIONALES"] *= AGR
    x_running["TOTAL_ACTIVOS"] *= AGR
    x_running["TOTAL_PASIVOS"] *= AGR
    x_running["TOTAL_PATRIMONIO"] *= AGR

# ============================
# 5. SALIDA FINAL
# ============================

if pred is not None:
    st.subheader("📈 Predicción final")
    st.metric(
        f"Ganancia/Pérdida esperada en {anio_target}",
        f"{pred:,.2f} millones COP"
    )
else:
    st.error("No se logró generar predicción.")
