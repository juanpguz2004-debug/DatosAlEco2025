import streamlit as st
import pandas as pd
import numpy as np
import joblib
import unicodedata

# ------------------------------------------------------
# UTILIDADES
# ------------------------------------------------------

def normalize_col(col):
    col = col.strip().upper().replace(" ", "_").replace("(", "").replace(")", "").replace("Ñ", "N")
    return ''.join(c for c in unicodedata.normalize('NFD', col) if unicodedata.category(c) != 'Mn')


def safe_le_transform(encoder, val):
    """Transforma un valor usando LabelEncoder, devolviendo -1 si el valor no existe."""
    val = str(val)
    return encoder.transform([val])[0] if val in encoder.classes_ else -1


def ensure_features(input_df, feature_list):
    """Alinea las columnas del DataFrame a las features esperadas por el modelo."""
    df = input_df.copy()

    for col in feature_list:
        if col not in df.columns:
            df[col] = 0  # OHE faltante
    return df[feature_list]


# ------------------------------------------------------
# CARGA DE MODELOS Y PARAMETROS
# ------------------------------------------------------

model_cls = joblib.load("model_clasificacion.pkl")
model_gan = joblib.load("model_reg_ganancia.pkl")
model_per = joblib.load("model_reg_perdida.pkl")
encoders = joblib.load("label_encoders.pkl")
FEATURES = joblib.load("model_features.pkl")
AGR = joblib.load("growth_rate.pkl")
ANO_BASE = joblib.load("base_year.pkl")


# ------------------------------------------------------
# INTERFAZ
# ------------------------------------------------------

st.title("📊 ALECO — Modelo de Dos Partes (Ganancia / Pérdida)")

st.write("Predicción recursiva multianual con modelos XGBoost entreados en Colab.")

uploaded = st.file_uploader("Sube el archivo procesado (el mismo del entrenamiento)", type=["csv"])

if uploaded is None:
    st.stop()


# ------------------------------------------------------
# CARGA Y PREPARACIÓN DEL CSV
# ------------------------------------------------------

df = pd.read_csv(uploaded)
df.columns = [normalize_col(c) for c in df.columns]

numeric_cols = [
    'INGRESOS_OPERACIONALES', 'GANANCIA_PERDIDA', 'TOTAL_ACTIVOS',
    'TOTAL_PASIVOS', 'TOTAL_PATRIMONIO'
]

for col in numeric_cols:
    df[col] = (
        df[col].astype(str)
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.replace(' ', '', regex=False)
        .str.replace('−', '-', regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['ANO_DE_CORTE'] = pd.to_numeric(df['ANO_DE_CORTE'], errors='coerce').fillna(-1).astype(int)

# Filtros usuario
macro = st.selectbox("Filtrar por Macrosector", ["Todos"] + sorted(df["MACROSECTOR"].dropna().unique().tolist()))
reg = st.selectbox("Filtrar por Región", ["Todos"] + sorted(df["REGION"].dropna().unique().tolist()))

df_view = df.copy()
if macro != "Todos":
    df_view = df_view[df_view["MACROSECTOR"] == macro]
if reg != "Todos":
    df_view = df_view[df_view["REGION"] == reg]


# ------------------------------------------------------
# KPIs
# ------------------------------------------------------

col1, col2 = st.columns(2)

col1.metric("Ingresos Operacionales Totales (Miles de millones COP)",
            f"${df_view['INGRESOS_OPERACIONALES'].sum() / 1e3:,.2f}")

col2.metric("Patrimonio Promedio (Miles de millones COP)",
            f"${df_view['TOTAL_PATRIMONIO'].mean() / 1e3:,.2f}")

st.divider()


# ------------------------------------------------------
# SELECCION EMPRESA Y AÑO
# ------------------------------------------------------

empresa = st.selectbox("Selecciona la Empresa para predecir", sorted(df["RAZON_SOCIAL"].unique().tolist()))
ano_target = st.number_input("Selecciona el año objetivo", min_value=ANO_BASE, max_value=2100, value=2026, step=1)

df_emp = df[df["RAZON_SOCIAL"] == empresa].copy()

if df_emp.empty:
    st.error("No hay datos para esta empresa.")
    st.stop()

ano_actual = df_emp["ANO_DE_CORTE"].max()

if ano_actual < ANO_BASE or np.isnan(ano_actual):
    st.error("La empresa no tiene datos consistentes.")
    st.stop()


# ------------------------------------------------------
# FUNCIÓN DE PREDICCIÓN RECURSIVA
# ------------------------------------------------------

def predict_next_year(row_input):
    """Aplica el pipeline completo a un año específico."""

    row = row_input.copy()

    # Label Encoding
    for col in encoders.keys():
        row[col] = safe_le_transform(encoders[col], row[col])

    # One Hot Encoding
    ohe_cols = ['SUPERVISOR', 'REGION', 'MACROSECTOR']
    row = pd.get_dummies(row.to_frame().T, columns=ohe_cols, prefix=ohe_cols, drop_first=True)

    # Restaurar ANO_DE_CORTE numérico
    row["ANO_DE_CORTE"] = pd.to_numeric(row["ANO_DE_CORTE"], errors="coerce")

    # Alinear columnas
    row = ensure_features(row, FEATURES)

    # Parte 1 — Clasificador
    es_gan = model_cls.predict(row)[0]

    # Parte 2 — Regresores
    if es_gan == 1:
        pred_log = model_gan.predict(row)[0]
        return np.expm1(pred_log)  # ganancia positiva
    else:
        pred_log = model_per.predict(row)[0]
        return -np.expm1(pred_log)  # pérdida negativa


# ------------------------------------------------------
# EJECUCIÓN RECURSIVA
# ------------------------------------------------------

if st.button("Generar Predicción"):
    try:
        df_pred = df_emp[df_emp["ANO_DE_CORTE"] == ano_actual].iloc[-1].copy()

        pred_year = ano_actual
        pred_value = None

        while pred_year < ano_target:
            pred_year += 1
            df_pred["ANO_DE_CORTE"] = pred_year
            pred_value = predict_next_year(df_pred)
            df_pred["GANANCIA_PERDIDA"] = pred_value

            # aplicar AGR
            df_pred["INGRESOS_OPERACIONALES"] *= AGR
            df_pred["TOTAL_ACTIVOS"] *= AGR
            df_pred["TOTAL_PASIVOS"] *= AGR
            df_pred["TOTAL_PATRIMONIO"] *= AGR

        st.success(f"Predicción final para {empresa} en {ano_target}:")
        st.metric("GANANCIA / PÉRDIDA (COP)", f"${pred_value:,.0f}")

    except Exception as e:
        st.error(f"❌ ERROR generando la predicción: {e}")
