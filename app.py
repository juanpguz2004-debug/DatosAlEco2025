import streamlit as st
import pandas as pd
import joblib

st.title("Dashboard de Empresas")

# ==============================
# 1️⃣ Cargar datos
# ==============================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("10.000_Empresas_mas_Grandes_del_País_20251115.csv")
        return df
    except FileNotFoundError:
        st.error("No se pudo encontrar el archivo CSV. Asegúrate de que está en el repositorio.")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error("El archivo CSV está vacío o no se pudo leer correctamente.")
        st.stop()

df = load_data()
st.write("Datos cargados:", df.shape[0], "filas y", df.shape[1], "columnas")

# ==============================
# 2️⃣ Cargar modelo
# ==============================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        return model
    except FileNotFoundError:
        st.error("No se pudo encontrar el archivo model.pkl en el repositorio.")
        st.stop()
