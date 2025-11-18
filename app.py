import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------------------------------------
# 1. Cargar archivos del repositorio
# -------------------------------------------------------------
@st.cache_resource
def load_models():
    cls = joblib.load("model_clasificacion.pkl")
    reg_g = joblib.load("model_reg_ganancia.pkl")
    reg_p = joblib.load("model_reg_perdida.pkl")
    encoders = joblib.load("label_encoders.pkl")
    features = joblib.load("model_features.pkl")
    AGR = joblib.load("growth_rate.pkl")
    base_year = joblib.load("base_year.pkl")
    return cls, reg_g, reg_p, encoders, features, AGR, base_year

@st.cache_data
def load_dataset():
    return pd.read_csv("dataset_procesado.csv")


# -------------------------------------------------------------
# 2. APP UI
# -------------------------------------------------------------
st.title("📊 ALECO — Modelo de Dos Partes (Ganancia / Pérdida)")
st.subheader("Predicción multianual usando modelos XGBoost entrenados en Colab.")

st.info("📁 El sistema utiliza automáticamente el archivo procesado incluido en GitHub.")


# -------------------------------------------------------------
# 3. Cargar modelos y dataset procesado
# -------------------------------------------------------------
try:
    df = load_dataset()
    cls, reg_g, reg_p, encoders, feature_names, AGR, BASE_YEAR = load_models()
    st.success("✓ Modelos y dataset procesado cargados correctamente.")
except Exception as e:
    st.error("❌ Error cargando los modelos o el dataset. Verifica los archivos en GitHub.")
    st.exception(e)
    st.stop()


# -------------------------------------------------------------
# 4. Selección de empresa y año
# -------------------------------------------------------------
empresa = st.selectbox("Selecciona una empresa:", df["RAZON_SOCIAL"].unique())

anio_target = st.number_input(
    "Selecciona el año a proyectar:",
    min_value=BASE_YEAR,
    max_value=2100,
    value=BASE_YEAR + 1,
    step=1
)


# -------------------------------------------------------------
# 5. Extraer la fila base
# -------------------------------------------------------------
row = df[df["RAZON_SOCIAL"] == empresa].iloc[0].copy()
anio_actual = row["ANO_DE_CORTE"]

# -------------------------------------------------------------
# 6. Proyección recursiva
# -------------------------------------------------------------
def proyectar_empresa(row, anio_inicio, anio_fin):

    row = row.copy()
    resultados = []

    for year in range(anio_inicio + 1, anio_fin + 1):

        # aplicar crecimiento a ingresos operacionales
        row["INGRESOS_OPERACIONALES"] *= AGR
        row["ANO_DE_CORTE"] = year

        # preparar features
        X = pd.DataFrame([row])[feature_names]

        # clasificador
        es_ganancia = cls.predict(X)[0]

        # regresor adecuado
        if es_ganancia == 1:
            pred_log = reg_g.predict(X)[0]
            pred = np.expm1(pred_log)
        else:
            pred_log = reg_p.predict(X)[0]
            pred = -np.expm1(pred_log)

        # actualizar fila
        row["GANANCIA_PERDIDA"] = pred

        # guardar resultado
        resultados.append({
            "AÑO": year,
            "INGRESOS_PROYECTADOS": row["INGRESOS_OPERACIONALES"],
            "GANANCIA/PÉRDIDA_PROYECTADA": pred
        })

    return pd.DataFrame(resultados)


# -------------------------------------------------------------
# 7. Ejecutar proyección
# -------------------------------------------------------------
if st.button("🔮 Ejecutar Proyección"):

    resultados = proyectar_empresa(row, anio_actual, anio_target)

    st.subheader(f"📈 Proyección {anio_actual+1} → {anio_target}")
    st.dataframe(resultados, use_container_width=True)


    # Mostrar último valor
    final_val = resultados.iloc[-1]["GANANCIA/PÉRDIDA_PROYECTADA"]
    if final_val >= 0:
        st.success(f"🏆 Resultado final proyectado ({anio_target}): **Ganancia de ${final_val:,.0f}**")
    else:
        st.error(f"📉 Resultado final proyectado ({anio_target}): **Pérdida de ${final_val:,.0f}**")
