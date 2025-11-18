import streamlit as st
import pandas as pd
import numpy as np
import joblib
import unicodedata

# ================================================
# 1. Normalización y codificación segura
# ================================================

def normalize_col(col):
    col = col.strip().upper().replace(" ", "_").replace("(", "").replace(")", "").replace("Ñ", "N")
    return ''.join(c for c in unicodedata.normalize('NFD', col) if unicodedata.category(c) != 'Mn')

def safe_le_transform(encoder, val):
    s = str(val)
    return encoder.transform([s])[0] if s in encoder.classes_ else -1


# ================================================
# 2. Cargar activos del modelo
# ================================================

xgb_cls = joblib.load("model_clasificacion.pkl")
xgb_reg_gan = joblib.load("model_reg_ganancia.pkl")
xgb_reg_per = joblib.load("model_reg_perdida.pkl")

encoders = joblib.load("label_encoders.pkl")
FEATURES = joblib.load("model_features.pkl")

AGR = joblib.load("growth_rate.pkl")
ANO_BASE = joblib.load("base_year.pkl")


# ================================================
# 3. Interfaz Streamlit
# ================================================

st.title("📊 ALECO — Modelo de Dos Partes (Ganancia / Pérdida)")
st.write("Predicción recursiva multianual usando modelos XGBoost entrenados en Colab.")
st.write("Sube el **archivo procesado**, el mismo usado durante el entrenamiento.")

uploaded = st.file_uploader("Sube el archivo procesado (CSV)", type=["csv"])


# ================================================
# 4. Procesamiento del archivo subido
# ================================================
def procesar_df(df):
    df.columns = [normalize_col(c) for c in df.columns]

    numeric_cols = [
        'INGRESOS_OPERACIONALES', 'GANANCIA_PERDIDA', 'TOTAL_ACTIVOS',
        'TOTAL_PASIVOS', 'TOTAL_PATRIMONIO'
    ]

    # Limpieza numérica
    for col in numeric_cols:
        df[col] = (
            df[col].astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.replace(' ', '', regex=False)
            .str.replace('−', '-', regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Año de corte
    df["ANO_DE_CORTE"] = (
        df["ANO_DE_CORTE"].astype(str).str.replace(",", "", regex=False)
    )
    df["ANO_DE_CORTE"] = pd.to_numeric(df["ANO_DE_CORTE"], errors="coerce").fillna(-1).astype(int)

    df = df[df["ANO_DE_CORTE"] > 2000].copy()

    return df


# ================================================
# 5. Función para preparar fila
# ================================================
def preparar_fila_prediccion(fila):
    fila_copia = fila.copy()

    # Label encoding seguro
    for col in ["DEPARTAMENTO_DOMICILIO", "CIUDAD_DOMICILIO", "CIIU"]:
        fila_copia[col] = safe_le_transform(encoders[col], fila[col])

    # OHE manual
    fila_ohe = pd.get_dummies(
        fila_copia,
        columns=["SUPERVISOR", "REGION", "MACROSECTOR"],
        prefix=["SUPERVISOR", "REGION", "MACROSECTOR"],
        drop_first=True,
        dtype=int
    )

    # Alinear a features del modelo
    for col in FEATURES:
        if col not in fila_ohe.columns:
            fila_ohe[col] = 0

    return fila_ohe[FEATURES].iloc[[0]]


# ================================================
# 6. Predicción recursiva año a año
# ================================================
def proyectar_recursivo(fila_base, anio_objetivo):
    fila = fila_base.copy()
    anio_actual = fila["ANO_DE_CORTE"]

    while anio_actual < anio_objetivo:
        fila["ANO_DE_CORTE"] = anio_actual

        X = preparar_fila_prediccion(fila)

        # 1. Clasificación
        prob_gan = xgb_cls.predict_proba(X)[0][1]
        es_gan = prob_gan > 0.5

        # 2. Regresión
        if es_gan:
            pred_log = xgb_reg_gan.predict(X)[0]
            pred_valor = np.expm1(pred_log)
        else:
            pred_log = xgb_reg_per.predict(X)[0]
            pred_valor = -np.expm1(pred_log)

        # Aplicar crecimiento
        fila["GANANCIA_PERDIDA"] = pred_valor
        fila["INGRESOS_OPERACIONALES"] *= AGR
        fila["TOTAL_ACTIVOS"] *= AGR
        fila["TOTAL_PASIVOS"] *= AGR
        fila["TOTAL_PATRIMONIO"] *= AGR

        anio_actual += 1

    return pred_valor


# ================================================
# 7. Ejecutar si el usuario subió archivo
# ================================================
if uploaded is not None:

    df = pd.read_csv(uploaded)
    df = procesar_df(df)

    st.success("✅ Archivo cargado correctamente.")

    empresa = st.selectbox("Selecciona Empresa", sorted(df["RAZON_SOCIAL"].unique()))

    anio_target = st.number_input(
        "Selecciona Año de Predicción",
        min_value=2025,
        max_value=2035,
        value=2026
    )

    if st.button("Calcular Predicción"):

        fila_empresa = df[df["RAZON_SOCIAL"] == empresa].sort_values("ANO_DE_CORTE").iloc[-1]
        pred = proyectar_recursivo(fila_empresa, anio_target)

        st.subheader(f"Predicción para {empresa} en {anio_target}:")
        st.metric("Ganancia / Pérdida Estimada (COP)", f"${pred:,.0f}")
