import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import unicodedata

# ----------------------------------------------------
# 1) NORMALIZAR COLUMNAS
# ----------------------------------------------------
def normalize_col(col):
    col = col.strip()
    col = col.upper()
    col = col.replace(" ", "_")
    col = col.replace("(", "").replace(")", "")
    col = col.replace("Ñ", "N")
    col = ''.join(
        c for c in unicodedata.normalize('NFD', col)
        if unicodedata.category(c) != 'Mn'
    )
    return col

# ----------------------------------------------------
# 2) CARGAR DATASET
# ----------------------------------------------------
@st.cache_data
def load_data():
    csv_file = "10.000_Empresas_mas_Grandes_del_País_20251115.csv"

    if not os.path.exists(csv_file):
        st.error(f"No se encontró el archivo: {csv_file}")
        return pd.DataFrame()

    df = pd.read_csv(csv_file)
    df.columns = [normalize_col(c) for c in df.columns]

    required_cols = [
        'NIT','RAZON_SOCIAL','SUPERVISOR','REGION','DEPARTAMENTO_DOMICILIO',
        'CIUDAD_DOMICILIO','CIIU','MACROSECTOR',
        'INGRESOS_OPERACIONALES','GANANCIA_PERDIDA',
        'TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO','ANO_DE_CORTE'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas necesarias: {missing}")
        return pd.DataFrame()

    num_cols = [
        'INGRESOS_OPERACIONALES','GANANCIA_PERDIDA',
        'TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO'
    ]

    for col in num_cols:
        df[col] = (
            df[col].astype(str)
            .str.replace("$","",regex=False)
            .str.replace(",","",regex=False)
            .str.replace(".","",regex=False)
            .str.replace(" ","",regex=False)
            .str.replace("−","-",regex=False)
            .str.replace("(","-",regex=False)
            .str.replace(")","",regex=False)
            .astype(float)
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
    model_file = "model.pkl"  # <--- CORREGIDO

    if not os.path.exists(model_file):
        st.error(f"No se encontró {model_file} en el repositorio.")
        return None

    try:
        return joblib.load(model_file)
    except Exception as e:
        st.error(f"Error cargando {model_file}: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# ----------------------------------------------------
# 4) UI PRINCIPAL
# ----------------------------------------------------
st.title("📊 Dashboard ALECO 2025")
st.markdown("Explora empresas, predice GANANCIA_PERDIDA y compara contra años anteriores.")

# Filtros
sector = st.selectbox("Filtrar por Macrosector", ["Todos"] + df["MACROSECTOR"].unique().tolist())
region = st.selectbox("Filtrar por Región", ["Todos"] + df["REGION"].unique().tolist())

df_filtrado = df.copy()
if sector != "Todos":
    df_filtrado = df_filtrado[df_filtrado["MACROSECTOR"] == sector]
if region != "Todos":
    df_filtrado = df_filtrado[df_filtrado["REGION"] == region]

st.subheader("Vista de empresas")
st.dataframe(df_filtrado.head(20))

# ----------------------------------------------------
# 5) KPIs
# ----------------------------------------------------
st.subheader("KPIs")

def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

for col in ["INGRESOS_OPERACIONALES","TOTAL_ACTIVOS","TOTAL_PASIVOS","TOTAL_PATRIMONIO"]:
    df_filtrado[col] = df_filtrado[col].apply(safe_float)

st.write(f"**Ingresos totales:** ${df_filtrado['INGRESOS_OPERACIONALES'].sum():,.0f}")
st.write(f"**Patrimonio promedio:** ${df_filtrado['TOTAL_PATRIMONIO'].mean():,.0f}")

# ----------------------------------------------------
# 6) PREDICCIÓN MEJORADA
# ----------------------------------------------------
st.subheader("🔮 Predicción de Ganancia/Pérdida por Empresa")

# Selección por nombre
empresa = st.selectbox(
    "Selecciona empresa",
    df_filtrado["RAZON_SOCIAL"].unique().tolist()
)

df_emp = df_filtrado[df_filtrado["RAZON_SOCIAL"] == empresa]

st.write("Datos de la empresa seleccionada:")
st.dataframe(df_emp)

# Año a predecir
year_to_predict = st.number_input(
    "Año que deseas predecir",
    min_value=int(df["ANO_DE_CORTE"].min()),
    max_value=int(df["ANO_DE_CORTE"].max()) + 5,
    value=int(df["ANO_DE_CORTE"].max()) + 1
)

last_year = year_to_predict - 1
prev_data = df_emp[df_emp["ANO_DE_CORTE"] == last_year]

if prev_data.empty:
    st.warning("⚠ La empresa no tiene datos del año anterior.")
else:
    st.info(f"Comparación contra año: {last_year}")

# ----------------------------------------------------
# Construir fila para predicción
# ----------------------------------------------------
row = df_emp.iloc[0].copy()
row["ANO_DE_CORTE"] = year_to_predict

FEATURE_ORDER = [
    'NIT','RAZON_SOCIAL','SUPERVISOR','REGION','DEPARTAMENTO_DOMICILIO',
    'CIUDAD_DOMICILIO','CIIU','MACROSECTOR',
    'INGRESOS_OPERACIONALES','TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO','ANO_DE_CORTE'
]

row = row[FEATURE_ORDER].to_frame().T

# Convertir categorías a códigos (igual que antes)
for col in row.columns:
    try:
        row[col] = pd.to_numeric(row[col])
    except:
        row[col] = row[col].astype("category").cat.codes

# ----------------------------------------------------
# PREDICCIÓN
# ----------------------------------------------------
pred = model.predict(row)[0]

st.success(f"🔮 **Predicción para {empresa} en {year_to_predict}:**  
`${pred:,.0f}`")

# ----------------------------------------------------
# DIFERENCIA CON EL AÑO ANTERIOR
# ----------------------------------------------------
if not prev_data.empty:
    prev_value = float(prev_data["GANANCIA_PERDIDA"].iloc[0])
    diff = pred - prev_value

    if diff >= 0:
        st.success(f"📈 Cambio respecto a {last_year}: **+${diff:,.0f}**")
    else:
        st.error(f"📉 Cambio respecto a {last_year}: **${diff:,.0f}**")
