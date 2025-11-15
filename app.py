# app.py

import streamlit as st
import pandas as pd
import joblib
import os
import altair as alt

# Para GPT4All
from gpt4all import GPT4All

# ==========================================
# 1️⃣ Cargar datos
# ==========================================
@st.cache_data
def load_data():
    csv_file = "10.000_Empresas_mas_Grandes_del_País_20251115.csv"
    if not os.path.exists(csv_file):
        st.error(f"No se encontró el archivo CSV: {csv_file}")
        return pd.DataFrame()
    df = pd.read_csv(csv_file)
    return df

df = load_data()

if df.empty:
    st.stop()  # Si no hay datos, detener la app

# ==========================================
# 2️⃣ Cargar modelo
# ==========================================
@st.cache_resource
def load_model():
    model_file = "model.pkl"
    if not os.path.exists(model_file):
        st.error(f"No se encontró el archivo del modelo: {model_file}")
        return None
    model = joblib.load(model_file)
    return model

model = load_model()
if model is None:
    st.stop()

# ==========================================
# 3️⃣ Configurar GPT4All
# ==========================================
@st.cache_resource
def init_gpt():
    # Cambia "ggml-model.bin" por el modelo que descargaste
    gpt_model = GPT4All("ggml-model.bin")
    return gpt_model

gpt = init_gpt()

# ==========================================
# 4️⃣ Interfaz Streamlit
# ==========================================
st.title("Dashboard de Empresas")
st.markdown("Explora los datos de las 10.000 empresas más grandes del país y genera predicciones de ganancias usando XGBoost.")

# Filtros interactivos
sector = st.selectbox("Selecciona un sector (MACROSECTOR)", ["Todos"] + df["MACROSECTOR"].dropna().unique().tolist())
region = st.selectbox("Selecciona una región (REGIÓN)", ["Todos"] + df["REGIÓN"].dropna().unique().tolist())

# Filtrar datos
df_filtered = df.copy()
if sector != "Todos":
    df_filtered = df_filtered[df_filtered["MACROSECTOR"] == sector]
if region != "Todos":
    df_filtered = df_filtered[df_filtered["REGIÓN"] == region]

st.dataframe(df_filtered.head(50))

# ==========================================
# 5️⃣ Visualizaciones
# ==========================================
st.subheader("Visualizaciones")
# Ejemplo: Ingresos por macrosector
chart = alt.Chart(df_filtered).mark_bar().encode(
    x='MACROSECTOR',
    y='INGRESOS_OPERACIONALES',
    tooltip=['RAZÓN_SOCIAL', 'INGRESOS_OPERACIONALES']
)
st.altair_chart(chart, use_container_width=True)

# ==========================================
# 6️⃣ Predicciones
# ==========================================
st.subheader("Predicciones de Ganancias")
empresa_idx = st.number_input("Selecciona índice de empresa para predecir ganancia", min_value=0, max_value=len(df_filtered)-1, value=0)

X_pred = df_filtered.drop(columns=["GANANCIA_(PÉRDIDA)", "RAZÓN_SOCIAL", "NIT", "SUPERVISOR", "REGIÓN", "DEPARTAMENTO DOMICILIO", "CIUDAD DOMICILIO", "CIIU", "MACROSECTOR", "Año de Corte"], errors='ignore')
pred_value = model.predict(X_pred.iloc[[empresa_idx]])[0]
st.write(f"Predicción de ganancia para la empresa {df_filtered.iloc[empresa_idx]['RAZÓN_SOCIAL']}: **{pred_value:.2f}**")

# ==========================================
# 7️⃣ Preguntas con GPT4All
# ==========================================
st.subheader("Consulta en lenguaje natural")
user_question = st.text_input("Hazle una pregunta a GPT4All sobre los datos o las predicciones:")

if user_question:
    # Podemos pasar un resumen de los datos al prompt si quieres
    context = f"Datos de empresas:\n{df_filtered.head(10).to_string()}"
    prompt = f"{context}\nPregunta: {user_question}\nRespuesta:"
    response = gpt.chat(prompt)
    st.markdown(f"**GPT4All:** {response}")
