import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import os
import joblib

st.set_page_config(page_title="Analytics ALECO 2025", layout="wide")

# =========================================================
#  FUNCIÓN ROBUSTA PARA CARGAR CSV
# =========================================================

@st.cache_data
def load_data():
    csv_file = "10.000_Empresas_mas_Grandes_del_País_20251115.csv"

    if not os.path.exists(csv_file):
        st.error(f"No se encontró el archivo CSV en el repositorio: {csv_file}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Error al leer el CSV: {e}")
        return pd.DataFrame()

    # --------------------------
    # Normalizar nombres de columnas
    # --------------------------
    def normalize_col(col):
        if not isinstance(col, str):
            col = str(col)
        col = col.strip()
        col = col.upper()
        col = ''.join(c for c in unicodedata.normalize('NFD', col)
                      if unicodedata.category(c) != 'Mn')
        col = col.replace("Ñ", "N")
        col = col.replace(" ", "_").replace("(", "").replace(")", "")
        col = col.replace(".", "").replace("-", "_").replace("/", "_")
        while "__" in col:
            col = col.replace("__", "_")
        return col

    df.columns = [normalize_col(c) for c in df.columns]

    # ======================================================
    #     DETECCIÓN DE LA COLUMNA OBJETIVO
    # ======================================================
    posibles_objetivo = [
        "GANANCIA_PERDIDA",
        "GANANCIA",
        "GANANCIA_PERDIDAS",
        "GANANCIA_(PERDIDA)",
        "UTILIDAD",
        "UTILIDAD_NETA"
    ]

    encontrado = None
    for opc in posibles_objetivo:
        opc_norm = normalize_col(opc)
        if opc_norm in df.columns:
            encontrado = opc_norm
            break

    if encontrado:
        if encontrado != "GANANCIA_PERDIDA":
            df = df.rename(columns={encontrado: "GANANCIA_PERDIDA"})
    else:
        st.warning("⚠ No se encontró columna de GANANCIA_PERDIDA en el CSV. Se creó vacía.")
        df["GANANCIA_PERDIDA"] = np.nan

    # ======================================================
    #     VERIFICAR COLUMNAS CLAVE PARA LA APP
    # ======================================================
    columnas_necesarias = [
        "MACROSECTOR", "REGION", "INGRESOS_OPERACIONALES",
        "TOTAL_ACTIVOS", "TOTAL_PASIVOS", "TOTAL_PATRIMONIO"
    ]

    faltantes = [c for c in columnas_necesarias if c not in df.columns]

    if faltantes:
        st.warning(f"Columnas faltantes detectadas en el CSV: {faltantes}")
        st.info(f"Columnas encontradas: {list(df.columns)}")

    return df


# =========================================================
#   CARGA DEL MODELO
# =========================================================

def load_model():
    try:
        model = joblib.load("model.pkl")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None


# =========================================================
#   INTERFAZ PRINCIPAL
# =========================================================

st.title("📊 Dashboard ALECO 2025")
st.subheader("Predicciones de Ganancias y Análisis de Datos")

df = load_data()
model = load_model()

if df.empty:
    st.stop()

st.write("### Vista previa de datos cargados")
st.dataframe(df.head())

# =========================================================
#   SECCIÓN DE FILTROS
# =========================================================

st.sidebar.title("Filtros")
region_sel = st.sidebar.selectbox("Seleccione región", df["REGION"].dropna().unique())
macro_sel = st.sidebar.selectbox("Seleccione macrosector", df["MACROSECTOR"].dropna().unique())

df_filtrado = df[(df["REGION"] == region_sel) & (df["MACROSECTOR"] == macro_sel)]

st.write(f"### Datos filtrados por {region_sel} - {macro_sel}")
st.dataframe(df_filtrado.head())

# =========================================================
#   PREDICCIÓN
# =========================================================

st.subheader("🔮 Predicción de Ganancia/Pérdida")

if model is not None:
    with st.form("pred_form"):
        ingresos = st.number_input("Ingresos operacionales", min_value=0.0)
        activos = st.number_input("Total activos", min_value=0.0)
        pasivos = st.number_input("Total pasivos", min_value=0.0)
        patrimonio = st.number_input("Total patrimonio", min_value=0.0)

        submit = st.form_submit_button("Predecir")

        if submit:
            try:
                X = pd.DataFrame([{
                    "INGRESOS_OPERACIONALES": ingresos,
                    "TOTAL_ACTIVOS": activos,
                    "TOTAL_PASIVOS": pasivos,
                    "TOTAL_PATRIMONIO": patrimonio,
                    "REGION": region_sel,
                    "MACROSECTOR": macro_sel
                }])

                # Codificación dummy automática si el modelo lo requiere
                X = pd.get_dummies(X)
                pred = model.predict(X)[0]

                st.success(f"Predicción estimada: {pred:,.2f} millones")
            except Exception as e:
                st.error(f"Error generando predicción: {e}")

else:
    st.error("No se pudo cargar el modelo. Verifica model.pkl")


# =========================================================
#     FIN DEL ARCHIVO app.py
# =========================================================
