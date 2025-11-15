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

    # Columnas requeridas para la predicción y vista
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
# 3) CARGAR MODELO (CORREGIDO: usa "model.pkl")
# ----------------------------------------------------
@st.cache_resource
def load_model():
    # 🚨 Nombre de archivo corregido a "model.pkl"
    model_file = "model.pkl" 
    
    if not os.path.exists(model_file):
        st.error(f"No se encontró el archivo del modelo: {model_file}")
        return None

    try:
        return joblib.load(model_file)
    except Exception as e:
        st.error(f"Error al cargar {model_file}: {e}")
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

# Asegurar la conversión a float para suma/media
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
# 6) PREDICCIÓN CON COMPARACIÓN (MEJORADO)
# ----------------------------------------------------
st.subheader("🔮 Predicción de Ganancia/Pérdida")

# Determinar el año base para la predicción
if df_filtrado.empty:
    st.warning("No hay empresas con ese filtro.")
    st.stop()

ano_corte_mas_reciente = df_filtrado["ANO_DE_CORTE"].max()
ano_prediccion = ano_corte_mas_reciente + 1
st.info(f"Se realizará la predicción para el **Año de Corte: {ano_prediccion}**")

# --- MEJORA: Selector de empresa por nombre ---
empresas_disponibles = df_filtrado[df_filtrado["ANO_DE_CORTE"] == ano_corte_mas_reciente]["RAZON_SOCIAL"].unique().tolist()

if not empresas_disponibles:
    st.warning(f"No hay datos de empresas disponibles para el año {ano_corte_mas_reciente} en este filtro.")
    st.stop()

empresa_seleccionada = st.selectbox(
    "Selecciona la Empresa para predecir",
    empresas_disponibles
)

# columnas EXACTAS que usa XGBoost
FEATURE_ORDER = [
    'NIT','RAZON_SOCIAL','SUPERVISOR','REGION','DEPARTAMENTO_DOMICILIO',
    'CIUDAD_DOMICILIO','CIIU','MACROSECTOR',
    'INGRESOS_OPERACIONALES','TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO','ANO_DE_CORTE'
]

# Copiar la fila más reciente (base) de la empresa seleccionada
row = df_filtrado[
    (df_filtrado["RAZON_SOCIAL"] == empresa_seleccionada) &
    (df_filtrado["ANO_DE_CORTE"] == ano_corte_mas_reciente)
].iloc[[0]].copy()

# Preparar la fila para la predicción
# El modelo predice el valor para el año siguiente
row["ANO_DE_CORTE"] = ano_prediccion

# Quitar columna objetivo del set de predicción
if "GANANCIA_PERDIDA" in row.columns:
    ganancia_anterior = row["GANANCIA_PERDIDA"].iloc[0]
    row = row.drop(columns=["GANANCIA_PERDIDA"])
else:
    ganancia_anterior = np.nan # En caso de que la columna se haya eliminado antes

# Asegurar orden correcto
row = row[FEATURE_ORDER]

# Convertir a códigos categóricos/numéricos (replicando el entrenamiento)
row_prediccion = row.copy()
for col in row_prediccion.columns:
    try:
        row_prediccion[col] = pd.to_numeric(row_prediccion[col], errors='raise')
    except:
        # Se asume que el modelo fue entrenado usando cat.codes
        # En un entorno real, usarías el LabelEncoder/Pipeline guardado.
        row_prediccion[col] = row_prediccion[col].astype("category").cat.codes


try:
    pred = model.predict(row_prediccion)[0]
    
    # --- MEJORA: Comparar con el año de corte anterior ---
    
    # Calcular la variación
    if not pd.isna(ganancia_anterior):
        diferencia = pred - ganancia_anterior
    else:
        diferencia = 0

    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.metric(
            label=f"GANANCIA/PÉRDIDA Predicha ({ano_prediccion})",
            value=f"${pred:,.2f}",
            delta=f"${diferencia:,.2f} vs {ano_corte_mas_reciente}" if not pd.isna(ganancia_anterior) else "Sin datos para comparar"
        )
        
    with col_res2:
        st.metric(
            label=f"GANANCIA/PÉRDIDA Real ({ano_corte_mas_reciente})",
            value=f"${ganancia_anterior:,.2f}" if not pd.isna(ganancia_anterior) else "N/A",
            delta_color="off"
        )
        
    st.success(f"Predicción generada con éxito para **{empresa_seleccionada}**.")
    st.caption(f"La comparación muestra la variación de la predicción de **{ano_prediccion}** respecto al valor real de **{ano_corte_mas_reciente}**.")

except Exception as e:
    st.error(f"Error generando predicción o comparación: {e}")
    st.caption("Asegúrate de que el modelo y la estructura de datos sean compatibles.")
