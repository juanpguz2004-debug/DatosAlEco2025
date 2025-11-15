# ============================================================
#                       APP STREAMLIT
#   10.000 empresas – Predicción de Ganancia (XGBoost REAL)
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import os
import altair as alt

# ------------------------------------------------------------
# 1️⃣ Cargar CSV (robusto)
# ------------------------------------------------------------

@st.cache_data
def load_data():
    csv_name = "10.000_Empresas_mas_Grandes_del_País_20251115.csv"

    if not os.path.exists(csv_name):
        st.error(f"No se encontró el archivo CSV: {csv_name}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_name)
    except Exception as e:
        st.error(f"Error leyendo CSV: {e}")
        return pd.DataFrame()

    # Limpieza uniforme de columnas
    df.columns = (
        df.columns.str.strip()
                  .str.upper()
                  .str.replace(" ", "_")
                  .str.replace("Á", "A")
                  .str.replace("É", "E")
                  .str.replace("Í", "I")
                  .str.replace("Ó", "O")
                  .str.replace("Ú", "U")
                  .str.replace("Ñ", "N")
    )

    required = [
        "NIT", "RAZON_SOCIAL", "SUPERVISOR", "REGION",
        "DEPARTAMENTO_DOMICILIO", "CIUDAD_DOMICILIO", "CIIU", "MACROSECTOR",
        "INGRESOS_OPERACIONALES", "TOTAL_ACTIVOS",
        "TOTAL_PASIVOS", "TOTAL_PATRIMONIO",
        "ANO_DE_CORTE", "GANANCIA_PERDIDA"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        st.error(f"Faltan columnas necesarias: {missing}")
        return pd.DataFrame()

    return df


df = load_data()
if df.empty:
    st.stop()

# ------------------------------------------------------------
# 2️⃣ Cargar modelo
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    model_file = "model.pkl"

    if not os.path.exists(model_file):
        st.error("No se encontró model.pkl")
        return None

    try:
        return joblib.load(model_file)
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None


model = load_model()
if model is None:
    st.stop()

# ------------------------------------------------------------
# 3️⃣ UI
# ------------------------------------------------------------
st.title("📊 Dashboard – 10.000 Empresas más Grandes del País")
st.write("Predicción de ganancia usando tu modelo XGBoost entrenado en Colab.")

# ------------------------------------------------------------
# 4️⃣ Filtros
# ------------------------------------------------------------

sector = st.selectbox(
    "Selecciona Macrosector",
    ["Todos"] + sorted(df["MACROSECTOR"].dropna().unique().tolist())
)

region = st.selectbox(
    "Selecciona Región",
    ["Todos"] + sorted(df["REGION"].dropna().unique().tolist())
)

# Aplicar filtros
df_filtered = df.copy()
if sector != "Todos":
    df_filtered = df_filtered[df_filtered["MACROSECTOR"] == sector]
if region != "Todos":
    df_filtered = df_filtered[df_filtered["REGION"] == region]

st.dataframe(df_filtered.head(50))

# ------------------------------------------------------------
# 5️⃣ Visualización
# ------------------------------------------------------------

st.subheader("📈 Ingresos por Macrosector")

chart_data = df_filtered.groupby("MACROSECTOR")["INGRESOS_OPERACIONALES"].sum().reset_index()

chart = (
    alt.Chart(chart_data)
       .mark_bar()
       .encode(
            x="MACROSECTOR",
            y="INGRESOS_OPERACIONALES",
            tooltip=["MACROSECTOR", "INGRESOS_OPERACIONALES"]
       )
)

st.altair_chart(chart, use_container_width=True)

# ------------------------------------------------------------
# 6️⃣ PREDICCIÓN (AQUÍ DEBE IR EL SUBHEADER)
# ------------------------------------------------------------

st.subheader("🔮 Predicción de Ganancia/Pérdida")

# Columnas EXACTAS del modelo
columnas_modelo = [
    "NIT", "RAZON_SOCIAL", "SUPERVISOR", "REGION",
    "DEPARTAMENTO_DOMICILIO", "CIUDAD_DOMICILIO",
    "CIIU", "MACROSECTOR",
    "INGRESOS_OPERACIONALES", "TOTAL_ACTIVOS",
    "TOTAL_PASIVOS", "TOTAL_PATRIMONIO",
    "ANO_DE_CORTE"
]

with st.form("form_pred"):
    st.write("Completa los datos para predecir:")

    r_sel = st.selectbox("Región", sorted(df["REGION"].unique().tolist()))
    m_sel = st.selectbox("Macrosector", sorted(df["MACROSECTOR"].unique().tolist()))

    ingresos = st.number_input("Ingresos operacionales", min_value=0.0)
    activos = st.number_input("Activos", min_value=0.0)
    pasivos = st.number_input("Pasivos", min_value=0.0)
    patrimonio = st.number_input("Patrimonio", min_value=0.0)

    ok = st.form_submit_button("Predecir")

if ok:
    try:
        X = pd.DataFrame([{
            "NIT": "0",
            "RAZON_SOCIAL": "EMPRESA_GENERICA",
            "SUPERVISOR": "NO_APLICA",
            "REGION": r_sel,
            "DEPARTAMENTO_DOMICILIO": "NO_APLICA",
            "CIUDAD_DOMICILIO": "NO_APLICA",
            "CIIU": "0000",
            "MACROSECTOR": m_sel,
            "INGRESOS_OPERACIONALES": ingresos,
            "TOTAL_ACTIVOS": activos,
            "TOTAL_PASIVOS": pasivos,
            "TOTAL_PATRIMONIO": patrimonio,
            "ANO_DE_CORTE": 2025
        }])[columnas_modelo]

        pred = model.predict(X)[0]
        st.success(f"Ganancia / Pérdida estimada: **${pred:,.2f}**")

    except Exception as e:
        st.error(f"Error generando predicción: {e}")
