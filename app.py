import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import unicodedata

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
    # Eliminar acentos
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

    # Normalizar columnas igual que en el entrenamiento
    df.columns = [normalize_col(c) for c in df.columns]

    # Columnas requeridas para la predicción
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

    # LIMPIAR columnas numéricas (igual que en Colab)
    numeric_cols = [
        'INGRESOS_OPERACIONALES','GANANCIA_PERDIDA',
        'TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO'
    ]

    for col in numeric_cols:
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
    
    # Asegurar que ANO_DE_CORTE sea int (crucial para la comparación)
    df['ANO_DE_CORTE'] = pd.to_numeric(df['ANO_DE_CORTE'], errors='coerce').fillna(-1).astype(int)

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
# 4) DASHBOARD PRINCIPAL
# ----------------------------------------------------
st.title("📊 Dashboard ALECO 2025")
st.markdown("Explora las empresas y predice **GANANCIA_PERDIDA** usando el modelo XGBoost entrenado.")

# --- Filtros ---
col1, col2 = st.columns(2)
with col1:
    sector = st.selectbox("Filtrar por Macrosector", ["Todos"] + df["MACROSECTOR"].unique().tolist())
with col2:
    region = st.selectbox("Filtrar por Región", ["Todos"] + df["REGION"].unique().tolist())

df_filtrado = df.copy()
if sector != "Todos":
    df_filtrado = df_filtrado[df_filtrado["MACROSECTOR"] == sector]

if region != "Todos":
    df_filtrado = df_filtrado[df_filtrado["REGION"] == region]

st.subheader("Vista (primeras filas) del conjunto filtrado")
st.dataframe(df_filtrado.head(30))


# ----------------------------------------------------
# 5) KPIs SEGUROS
# ----------------------------------------------------
st.subheader("📊 KPIs agregados")

def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

for col in ["INGRESOS_OPERACIONALES","TOTAL_ACTIVOS","TOTAL_PASIVOS","TOTAL_PATRIMONIO"]:
    df_filtrado[col] = df_filtrado[col].apply(safe_float)

ingresos_total = df_filtrado["INGRESOS_OPERACIONALES"].sum()
patrimonio_prom = df_filtrado["TOTAL_PATRIMONIO"].mean()

col_kpi1, col_kpi2 = st.columns(2)
with col_kpi1:
    st.metric(label="Ingresos Operacionales Totales", value=f"${ingresos_total:,.2f}")
with col_kpi2:
    st.metric(label="Patrimonio Promedio", value=f"${patrimonio_prom:,.2f}")


# ----------------------------------------------------
# 6) PREDICCIÓN CON COMPARACIÓN
# ----------------------------------------------------
st.subheader("🔮 Predicción de Ganancia/Pérdida")

if len(df_filtrado) == 0:
    st.warning("No hay empresas con ese filtro.")
    st.stop()

# --- MEJORA 1: Selector de empresa por nombre ---
empresa_seleccionada = st.selectbox(
    "Selecciona la Empresa para predecir",
    df_filtrado["RAZON_SOCIAL"].unique().tolist()
)

# Columna para determinar el año de predicción (el más reciente + 1)
ano_corte_mas_reciente = df_filtrado["ANO_DE_CORTE"].max()
ano_prediccion = ano_corte_mas_reciente + 1
st.info(f"Se realizará la predicción para el **Año de Corte: {ano_prediccion}**")

# columnas EXACTAS que usa XGBoost
FEATURE_ORDER = [
    'NIT','RAZON_SOCIAL','SUPERVISOR','REGION','DEPARTAMENTO_DOMICILIO',
    'CIUDAD_DOMICILIO','CIIU','MACROSECTOR',
    'INGRESOS_OPERACIONALES','TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO','ANO_DE_CORTE'
]

# Copiar la fila más reciente de la empresa seleccionada
row = df_filtrado[
    (df_filtrado["RAZON_SOCIAL"] == empresa_seleccionada) &
    (df_filtrado["ANO_DE_CORTE"] == ano_corte_mas_reciente)
].iloc[[0]].copy()

# ------------------------------------------------------------------
# Preparar la fila para la predicción (Ajustar al año siguiente)
# ------------------------------------------------------------------

# El modelo predice el valor para el año siguiente, por lo que actualizamos la característica 'ANO_DE_CORTE'
row["ANO_DE_CORTE"] = ano_prediccion

# Quitar columna objetivo del set de predicción
if "GANANCIA_PERDIDA" in row.columns:
    row = row.drop(columns=["GANANCIA_PERDIDA"])

# Asegurar orden correcto
row = row[FEATURE_ORDER]

# Convertir todo a numérico o categorías codificadas
# IMPORTANTE: Esto debe replicar EXACTAMENTE la codificación del entrenamiento.
# Si el entrenamiento usó One-Hot Encoding o Target Encoding, este paso es insuficiente.
# Asumiendo que el modelo XGBoost fue entrenado con la codificación de `cat.codes`
# para las features categóricas.
row_prediccion = row.copy()
for col in row_prediccion.columns:
    try:
        row_prediccion[col] = pd.to_numeric(row_prediccion[col])
    except:
        # Si no es numérico, convertir string en hash estable (cat.codes)
        # NOTA: Esto solo funciona si el modelo fue entrenado con TODAS
        # las categorías disponibles. En producción, se usaría un `LabelEncoder`
        # guardado con `joblib`. Para fines de demo, se usa el enfoque simple.
        row_prediccion[col] = row_prediccion[col].astype("category").cat.codes

try:
    pred = model.predict(row_prediccion)[0]
    
    # ------------------------------------------------------------------
    # MEJORA 2: Comparar con el año de corte anterior
    # ------------------------------------------------------------------
    
    # Buscar el valor real de GANANCIA_PERDIDA del año anterior (el año de corte más reciente)
    ganancia_anterior = df_filtrado[
        (df_filtrado["RAZON_SOCIAL"] == empresa_seleccionada) &
        (df_filtrado["ANO_DE_CORTE"] == ano_corte_mas_reciente)
    ]["GANANCIA_PERDIDA"].iloc[0]

    # Calcular la variación
    diferencia = pred - ganancia_anterior

    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.metric(
            label=f"GANANCIA/PÉRDIDA Predicha ({ano_prediccion})",
            value=f"${pred:,.2f}",
            delta=f"${diferencia:,.2f} vs {ano_corte_mas_reciente}"
        )
        
    with col_res2:
        st.metric(
            label=f"GANANCIA/PÉRDIDA Real ({ano_corte_mas_reciente})",
            value=f"${ganancia_anterior:,.2f}",
            delta_color="off"
        )
        
    st.success(f"Predicción generada con éxito para **{empresa_seleccionada}**.")
    st.caption(f"La comparación muestra la variación de la predicción de **{ano_prediccion}** respecto al valor real de **{ano_corte_mas_reciente}**.")

except Exception as e:
    st.error(f"Error generando predicción o comparación: {e}")
    st.caption("Asegúrate de que la empresa seleccionada tiene datos disponibles para el año anterior.")
