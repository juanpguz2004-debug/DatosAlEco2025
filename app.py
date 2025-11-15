import streamlit as st
import pandas as pd
import joblib

# ----------------------
# Función para cargar datos
# ----------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("10.000_Empresas_mas_Grandes_del_País_20251115.csv")
        return df
    except Exception as e:
        st.error(f"No se pudo cargar el archivo CSV: {e}")
        return pd.DataFrame()  # devuelve DataFrame vacío si hay error

# ----------------------
# Cargar modelo
# ----------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        return model
    except Exception as e:
        st.error(f"No se pudo cargar el modelo: {e}")
        return None

# ----------------------
# App Streamlit
# ----------------------
st.title("Dashboard de Empresas")

# Cargar datos
df = load_data()
if df.empty:
    st.stop()  # detiene la app si no hay datos

st.dataframe(df.head())

# Filtros de usuario
sector = st.selectbox("Seleccione un sector", df["MACROSECTOR"].unique())
df_filtrado = df[df["MACROSECTOR"] == sector]

st.write(f"Empresas en el sector {sector}:")
st.dataframe(df_filtrado)

# Predicciones
model = load_model()
if model:
    # Ejemplo: predecir la primera fila del sector seleccionado
    X_pred = df_filtrado.drop(columns=["GANANCIA (PÉRDIDA)", "RAZÓN SOCIAL", "NIT"], errors="ignore")
    pred = model.predict(X_pred)[0]
    st.write(f"Predicción de ganancia para la primera empresa del sector: {pred:.2f}")
