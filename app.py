import streamlit as st
import pandas as pd
import joblib
import os
import altair as alt

# ==========================================
# 1️⃣ Intentar cargar GPT4All
# ==========================================
try:
    from gpt4all import GPT4All
except ImportError:
    GPT4All = None


# ==========================================
# 2️⃣ Cargar datos
# ==========================================
@st.cache_data
def load_data():
    csv_file = "10.000_Empresas_mas_Grandes_del_País_20251115.csv"
    if not os.path.exists(csv_file):
        st.error(f"No se encontró el archivo CSV: {csv_file}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Error al leer el CSV: {e}")
        return pd.DataFrame()

    # Normalizar nombres de columnas (importante)
    df.columns = (
        df.columns
        .str.strip()
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
        "MACROSECTOR", "REGION", "INGRESOS_OPERACIONALES",
        "TOTAL_PATRIMONIO", "TOTAL_ACTIVOS", "TOTAL_PASIVOS",
        "GANANCIA_PERDIDA"
    ]

    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        st.error(f"Faltan columnas necesarias: {missing_cols}")
        st.stop()

    return df


df = load_data()
if df.empty:
    st.stop()


# ==========================================
# 3️⃣ Cargar modelo
# ==========================================
@st.cache_resource
def load_model():
    model_file = "model.pkl"
    if not os.path.exists(model_file):
        st.error("No se encontró el archivo model.pkl")
        return None
    try:
        return joblib.load(model_file)
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None


model = load_model()
if model is None:
    st.stop()


# ==========================================
# 4️⃣ Inicializar GPT4All
# ==========================================
@st.cache_resource
def init_gpt():
    if GPT4All is None:
        return None
    model_path = "ggml-model.bin"   # Debes subirlo al repo
    if not os.path.exists(model_path):
        st.warning("No se encontró el modelo ggml-model.bin. GPT4All no estará disponible.")
        return None
    try:
        return GPT4All(model_path)
    except Exception as e:
        st.warning(f"Error inicializando GPT4All: {e}")
        return None


gpt = init_gpt()



# ==========================================
# 5️⃣ INTERFAZ PRINCIPAL
# ==========================================
st.title("Dashboard de Empresas – ALECO 2025")
st.markdown("Explora las **10.000 empresas más grandes de Colombia** y predice su ganancia usando un modelo **XGBoost** entrenado.")



# ==========================================
# 6️⃣ Filtros
# ==========================================
sector = st.selectbox("Selecciona Macrosector", ["Todos"] + sorted(df["MACROSECTOR"].unique().tolist()))
region = st.selectbox("Selecciona Región", ["Todos"] + sorted(df["REGION"].unique().tolist()))

df_filtered = df.copy()

if sector != "Todos":
    df_filtered = df_filtered[df_filtered["MACROSECTOR"] == sector]

if region != "Todos":
    df_filtered = df_filtered[df_filtered["REGION"] == region]

st.dataframe(df_filtered.head(50))

if df_filtered.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()



# ==========================================
# 7️⃣ Visualización
# ==========================================
st.subheader("Ingresos operacionales por macrosector")

df_chart = df_filtered.groupby("MACROSECTOR")["INGRESOS_OPERACIONALES"].sum().reset_index()

chart = alt.Chart(df_chart).mark_bar().encode(
    x="MACROSECTOR",
    y="INGRESOS_OPERACIONALES",
    tooltip=["MACROSECTOR", "INGRESOS_OPERACIONALES"]
).properties(title="Ingresos por Macrosector")

st.altair_chart(chart, use_container_width=True)



# ==========================================
# 8️⃣ PREDICCIONES
# ==========================================
st.subheader("Predicción de Ganancia (Modelo XGBoost)")

max_index = len(df_filtered) - 1
empresa_idx = st.number_input(
    f"Selecciona empresa por índice (0 a {max_index})",
    min_value=0, max_value=max_index, value=0
)

# Columnas reales que usa tu modelo (extraídas de feature_importances)
FEATURE_COLUMNS = [
    "TOTAL_PATRIMONIO",
    "TOTAL_ACTIVOS",
    "TOTAL_PASIVOS",
    "INGRESOS_OPERACIONALES",
    "RAZON_SOCIAL",
    "NIT",
    "CIIU",
    "ANO_DE_CORTE",
    "MACROSECTOR",
    "SUPERVISOR",
    "CIUDAD_DOMICILIO",
    "REGION",
    "DEPARTAMENTO_DOMICILIO"
]

missing = [c for c in FEATURE_COLUMNS if c not in df_filtered.columns]
if missing:
    st.error(f"Faltan columnas requeridas para el modelo: {missing}")
    st.stop()

X_pred = df_filtered[FEATURE_COLUMNS]

try:
    pred = model.predict(X_pred.iloc[[empresa_idx]])[0]
    nombre = df_filtered.iloc[empresa_idx]["RAZON_SOCIAL"]
    st.success(f"**Ganancia estimada para {nombre}:  ${pred:,.2f}**")
except Exception as e:
    st.error(f"Error en predicción: {e}")



# ==========================================
# 9️⃣ GPT4All – Consultas en lenguaje natural
# ==========================================
st.subheader("Pregúntale a la IA sobre los datos")

user_q = st.text_input("¿Qué deseas consultar?")

if user_q:
    if gpt:
        context = df_filtered.head(5).to_string()
        prompt = f"Estos son datos de empresas:\n{context}\nPregunta: {user_q}\nRespuesta:"
        try:
            response = gpt.chat(prompt)
            st.markdown("**GPT4All:** " + response)
        except Exception as e:
            st.error(f"Error consultando GPT4All: {e}")
    else:
        st.warning("GPT4All no está disponible (no se encontró ggml-model.bin).")
