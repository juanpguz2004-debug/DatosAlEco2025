import streamlit as st
import pandas as pd
import joblib

# =========================
# Configuración de la app
# =========================
st.set_page_config(page_title="Dashboard de Empresas", layout="wide")
st.title("Dashboard de Empresas")
st.markdown("""
Este dashboard permite explorar los datos de las 10.000 empresas más grandes del país, 
filtrar por sector, ubicación y tamaño, y generar predicciones de ganancias usando XGBoost.
""")

# =========================
# Función para cargar datos
# =========================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("datos.csv")  # Asegúrate de subir tu CSV a GitHub junto con app.py
        return df
    except Exception as e:
        st.error(f"No se pudo cargar el archivo datos.csv: {e}")
        return pd.DataFrame()

# =========================
# Cargar modelo
# =========================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")  # Asegúrate de subir tu model.pkl
        return model
    except Exception as e:
        st.error(f"No se pudo cargar el modelo model.pkl: {e}")
        return None

# =========================
# Cargar datos y modelo
# =========================
df = load_data()
model = load_model()

if df.empty or model is None:
    st.stop()

# =========================
# Sidebar de filtros
# =========================
st.sidebar.header("Filtros")
macrosectores = df['MACROSECTOR'].unique()
regiones = df['REGIÓN'].unique()

selected_macro = st.sidebar.multiselect("Macrosector", macrosectores, default=macrosectores)
selected_region = st.sidebar.multiselect("Región", regiones, default=regiones)
min_ingresos = st.sidebar.number_input("Ingresos mínimos", value=float(df['INGRESOS_OPERACIONALES'].min()))
max_ingresos = st.sidebar.number_input("Ingresos máximos", value=float(df['INGRESOS_OPERACIONALES'].max()))

# =========================
# Aplicar filtros
# =========================
df_filtrado = df[
    (df['MACROSECTOR'].isin(selected_macro)) &
    (df['REGIÓN'].isin(selected_region)) &
    (df['INGRESOS_OPERACIONALES'] >= min_ingresos) &
    (df['INGRESOS_OPERACIONALES'] <= max_ingresos)
]

st.subheader(f"Datos filtrados: {df_filtrado.shape[0]} empresas")
st.dataframe(df_filtrado.head(50))

# =========================
# Predicciones
# =========================
st.subheader("Predicciones de Ganancias")

# Asegúrate de que model_features sean las columnas que el modelo espera
model_features = ['INGRESOS_OPERACIONALES', 'TOTAL_ACTIVOS', 'TOTAL_PASIVOS', 'TOTAL_PATRIMONIO']
missing_cols = [c for c in model_features if c not in df_filtrado.columns]

if missing_cols:
    st.warning(f"Faltan columnas requeridas para el modelo: {missing_cols}")
else:
    if st.button("Generar predicciones"):
        X_pred = df_filtrado[model_features]
        preds = model.predict(X_pred)
        df_filtrado = df_filtrado.copy()
        df_filtrado['Predicción_GANANCIA'] = preds
        st.dataframe(df_filtrado[['RAZÓN SOCIAL', 'MACROSECTOR', 'REGIÓN', 'Predicción_GANANCIA']].head(50))
