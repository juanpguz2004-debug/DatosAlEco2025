import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# -------------------------------------------------
#                 CONFIGURACIÓN UI
# -------------------------------------------------
st.set_page_config(
    page_title="ALECO — Predicción Multianual",
    layout="wide"
)

st.title("📊 ALECO — Modelo de Dos Partes (Ganancia / Pérdida)")
st.write("Predicción multianual usando modelos XGBoost entrenados en Colab.")


# -------------------------------------------------
#          ARCHIVOS NECESARIOS EN GITHUB
# -------------------------------------------------
REQUIRED_FILES = [
    "dataset_procesado.csv",
    "label_encoders.pkl",
    "model_features.pkl",
    "base_year.pkl",
    "growth_rate.pkl",
    "model_clasificacion.pkl",
    "model_reg_ganancia.pkl",
    "model_reg_perdida.pkl"
]

missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]

if missing:
    st.error("❌ Faltan archivos necesarios:\n" + "\n".join(missing))
    st.stop()


# -------------------------------------------------
#           CARGAR MODELOS Y ARCHIVOS
# -------------------------------------------------
try:
    df = pd.read_csv("dataset_procesado.csv")

    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    with open("model_features.pkl", "rb") as f:
        MODEL_FEATURE_NAMES = pickle.load(f)

    with open("base_year.pkl", "rb") as f:
        base_year_model = pickle.load(f)

    with open("growth_rate.pkl", "rb") as f:
        growth_model = pickle.load(f)

    with open("model_clasificacion.pkl", "rb") as f:
        model_clasificacion = pickle.load(f)

    with open("model_reg_ganancia.pkl", "rb") as f:
        model_reg_ganancia = pickle.load(f)

    with open("model_reg_perdida.pkl", "rb") as f:
        model_reg_perdida = pickle.load(f)

    st.success("✓ Modelos y dataset procesado cargados correctamente.")

except Exception as e:
    st.error(f"❌ Error cargando los modelos: {e}")
    st.stop()


# -------------------------------------------------
#           VERIFICAR QUE EXISTA RAZON_SOCIAL
# -------------------------------------------------
if "RAZON_SOCIAL" not in df.columns:
    st.error("❌ El CSV procesado debe contener la columna 'RAZON_SOCIAL'.")
    st.stop()


# -------------------------------------------------
#           SELECCIÓN DE EMPRESA
# -------------------------------------------------
empresa = st.selectbox(
    "Selecciona una empresa para predecir:",
    df["RAZON_SOCIAL"].unique()
)

df_emp = df[df["RAZON_SOCIAL"] == empresa]

if df_emp.empty:
    st.error("❌ No se encontró la empresa en el dataset.")
    st.stop()

# Remover columna identificadora
X = df_emp.drop(columns=["RAZON_SOCIAL"]).copy()

# Asegurar orden exacto de columnas
X = X.reindex(columns=MODEL_FEATURE_NAMES, fill_value=0)


# -------------------------------------------------
#             BOTÓN DE PREDICCIÓN
# -------------------------------------------------
if st.button("🔮 Predecir Escenario Multianual"):

    try:
        # 1️⃣ Clasificación ganancia/pérdida
        clase = model_clasificacion.predict(X)[0]

        # 2️⃣ Predicción regresiva según clase
        if clase == 1:
            pred_base = model_reg_ganancia.predict(X)[0]
        else:
            pred_base = model_reg_perdida.predict(X)[0]

        pred_base = float(pred_base)

        # 3️⃣ Predicción multianual
        base_year = int(base_year_model)
        growth_rate = float(growth_model)

        years = list(range(base_year, base_year + 6))
        preds = [pred_base * ((1 + growth_rate) ** i) for i in range(6)]

        # Mostrar resultados
        st.subheader(f"📈 Proyección para {empresa}")

        df_pred = pd.DataFrame({
            "Año": years,
            "Predicción ($)": preds
        })

        st.dataframe(df_pred, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error generando predicción: {e}")
