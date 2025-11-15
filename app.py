import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import unicodedata

# ----------------------------------------------------
# 0) FUNCIÓN ROBUSTA PARA VALIDAR COLUMNAS
# ----------------------------------------------------
def validate_required_columns(df, required_cols):
    """
    Verifica columnas, y si falta alguna la agrega con NaN.
    Esto evita que el modelo falle por columnas faltantes.
    """
    df_cols = list(df.columns)
    missing = [c for c in required_cols if c not in df_cols]

    if missing:
        st.warning(f"Columnas faltantes detectadas, se agregan automáticamente: {missing}")
        for col in missing:
            df[col] = np.nan

    return df


# ----------------------------------------------------
# 1) FUNCIONES DE NORMALIZACIÓN (IGUALES A COLAB)
# ----------------------------------------------------
def normalize_col(col):
    col = col.strip()
    col = col.upper()
    col = col.replace(" ", "_")
    col = col.replace("(", "")
    col = col.replace(")", "")
    col = col.replace("Ñ", "N")
    col = ''.join(
        c for c in unicodedata.normalize('NFD', col)
        if unicodedata.category(c) != 'Mn'
    )
    return col


# ----------------------------------------------------
# 2) CARGAR CSV
# ----------------------------------------------------
@st.cache_data
def load_data():
    csv_file = "10.000_Empresas_mas_Grandes_del_País_20251115.csv"

    if not os.path.exists(csv_file):
        st.error(f"No se encontró el archivo: {csv_file}")
        return pd.DataFrame()

    df = pd.read_csv(csv_file)

    # Normalizar columnas
    df.columns = [normalize_col(c) for c in df.columns]

    # Columnas requeridas para este dashboard y el modelo
    required_cols = [
        'NIT','RAZON_SOCIAL','SUPERVISOR','REGION','DEPARTAMENTO_DOMICILIO',
        'CIUDAD_DOMICILIO','CIIU','MACROSECTOR',
        'INGRESOS_OPERACIONALES','GANANCIA_PERDIDA',
        'TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO','ANO_DE_CORTE'
    ]

    # Validación robusta
    df = validate_required_columns(df, required_cols)

    # LIMPIAR columnas numéricas
    numeric_cols = [
        'INGRESOS_OPERACIONALES','GANANCIA_PERDIDA',
        'TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO'
    ]

    for col in numeric_cols:
        df[col] = (
            df[col].astype(str)
            .replace(["$", ",", ".", " ", "−", "(", ")"], "", regex=True)
            .astype(float, errors="ignore")
        )

    return df


df = load_data()
if df.empty:
    st.stop()


# ----------------------------------------------------
# 3) CARGAR MODELO
# ----------------------------------------------------
@st.cache_resource
def load_model():
    model_file = "model.pkl"

    if not os.path.exists(model_file):
        st.error("No se encontró model.pkl en el repositorio.")
        return None

    try:
        return joblib.load(model_file)
    except Exception as e:
        st.error(f"Error al cargar model.pkl: {e}")
        return None


model = load_model()
if model is None:
    st.stop()


# ----------------------------------------------------
# 4) DASHBOARD
# ----------------------------------------------------
st.title("📊 Dashboard ALECO 2025")
st.markdown("Explora las empresas y predice GANANCIA_PERDIDA usando el modelo XGBoost entrenado.")

sector = st.selectbox("Filtrar por Macrosector", ["Todos"] + df["MACROSECTOR"].unique().tolist())
region = st.selectbox("Filtrar por Región", ["Todos"] + df["REGION"].unique().tolist())

df_filtrado = df.copy()
if sector != "Todos":
    df_filtrado = df_filtrado[df_filtrado["MACROSECTOR"] == sector]
if region != "Todos":
    df_filtrado = df_filtrado[df_filtrado["REGION"] == region]

st.subheader("Vista del conjunto filtrado")
st.dataframe(df_filtrado.head(30))


# ----------------------------------------------------
# 5) KPIs
# ----------------------------------------------------
st.subheader("KPIs agregados")

def safe_float(x):
    try: return float(x)
    except: return np.nan

for col in ["INGRESOS_OPERACIONALES","TOTAL_ACTIVOS","TOTAL_PASIVOS","TOTAL_PATRIMONIO"]:
    df_filtrado[col] = df_filtrado[col].apply(safe_float)

ingresos_total = df_filtrado["INGRESOS_OPERACIONALES"].sum()
patrimonio_prom = df_filtrado["TOTAL_PATRIMONIO"].mean()

st.write(f"**Ingresos totales:** ${ingresos_total:,.2f}")
st.write(f"**Patrimonio promedio:** ${patrimonio_prom:,.2f}")


# ----------------------------------------------------
# 6) PREDICCIÓN ROBUSTA
# ----------------------------------------------------
st.subheader("🔮 Predicción GANANCIA_PERDIDA")

if len(df_filtrado) == 0:
    st.warning("No hay empresas con ese filtro.")
    st.stop()

idx = st.number_input(
    "Selecciona índice para predecir",
    min_value=0,
    max_value=len(df_filtrado)-1,
    value=0
)

FEATURE_ORDER = [
    'NIT','RAZON_SOCIAL','SUPERVISOR','REGION','DEPARTAMENTO_DOMICILIO',
    'CIUDAD_DOMICILIO','CIIU','MACROSECTOR',
    'INGRESOS_OPERACIONALES','TOTAL_ACTIVOS','TOTAL_PASIVOS',
    'TOTAL_PATRIMONIO','ANO_DE_CORTE'
]

row = df_filtrado.iloc[[idx]].copy()

# VALIDAR QUE TODAS LAS FEATURES EXISTAN
row = validate_required_columns(row, FEATURE_ORDER)

# Extraer solo features del modelo
row = row[FEATURE_ORDER]

# Convertir categóricas
for col in row.columns:
    try:
        row[col] = pd.to_numeric(row[col])
    except:
        row[col] = row[col].astype("category").cat.codes

try:
    pred = model.predict(row)[0]
    st.success(f"Predicción de GANANCIA_PERDIDA: **${pred:,.2f}**")
except Exception as e:
    st.error(f"Error generando predicción: {e}")
