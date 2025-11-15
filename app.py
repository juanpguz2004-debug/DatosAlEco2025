import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# Título y descripción
# ==========================================
st.title("Dashboard de Empresas")
st.write("""
Este dashboard permite explorar los datos de las 10.000 empresas más grandes del país,
filtrar por sector, ubicación y tamaño, y generar predicciones de ganancias usando XGBoost.
""")

# ==========================================
# Cargar datos
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("datos.csv")  # tu CSV de datos
    return df

df = load_data()

# Mostrar dataframe completo (opcional)
if st.checkbox("Mostrar datos completos"):
    st.dataframe(df)

# ==========================================
# Filtros
# ==========================================
regiones = df['REGIÓN'].unique()
sectores = df['MACROSECTOR'].unique()

sel_region = st.sidebar.multiselect("Selecciona región", regiones, default=regiones)
sel_sector = st.sidebar.multiselect("Selecciona sector", sectores, default=sectores)

df_filtrado = df[(df['REGIÓN'].isin(sel_region)) & (df['MACROSECTOR'].isin(sel_sector))]

st.write(f"Empresas filtradas: {df_filtrado.shape[0]}")
st.dataframe(df_filtrado)

# ==========================================
# Cargar modelo XGBoost
# ==========================================
model = joblib.load("model.pkl")  # tu modelo entrenado

# Selección de inputs para predicción
st.sidebar.header("Predicción de Ganancia")
ingresos = st.sidebar.number_input("Ingresos Operacionales", min_value=0.0, value=10.0)
total_activos = st.sidebar.number_input("Total Activos", min_value=0.0, value=10.0)
total_pasivos = st.sidebar.number_input("Total Pasivos", min_value=0.0, value=5.0)
total_patrimonio = st.sidebar.number_input("Total Patrimonio", min_value=0.0, value=5.0)

X_pred = pd.DataFrame({
    'INGRESOS_OPERACIONALES':[ingresos],
    'TOTAL_ACTIVOS':[total_activos],
    'TOTAL_PASIVOS':[total_pasivos],
    'TOTAL_PATRIMONIO':[total_patrimonio]
})

pred = model.predict(X_pred)[0]
st.subheader(f"Predicción de Ganancia: {pred:.2f}")

# ==========================================
# Interpretabilidad SHAP
# ==========================================
if st.checkbox("Mostrar explicación SHAP"):
    explainer = shap.Explainer(model)
    shap_values = explainer(df_filtrado[['INGRESOS_OPERACIONALES','TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO']])
    
    st.subheader("SHAP Summary Plot")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values.values, df_filtrado[['INGRESOS_OPERACIONALES','TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO']], show=False)
    st.pyplot(fig)
