import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle as pickle

st.set_page_config(page_title="ALECO — Modelo de Dos Partes", layout="wide")

# ============================================
#          CARGA DE MODELOS Y DATASET
# ============================================

@st.cache_resource
def load_all():
    try:
        with open("model_clasificacion.pkl", "rb") as f:
            model_clf = pickle.load(f)

        with open("model_reg_ganancia.pkl", "rb") as f:
            model_gain = pickle.load(f)

        with open("model_reg_perdida.pkl", "rb") as f:
            model_loss = pickle.load(f)

        with open("label_encoders.pkl", "rb") as f:
            encoders = pickle.load(f)

        with open("model_features.pkl", "rb") as f:
            model_features = pickle.load(f)

        with open("base_year.pkl", "rb") as f:
            base_year = pickle.load(f)

        with open("growth_rate.pkl", "rb") as f:
            growth_rate = pickle.load(f)

        df = pd.read_csv("dataset_procesado.csv")

        return model_clf, model_gain, model_loss, encoders, model_features, base_year, growth_rate, df

    except Exception as e:
        st.error(f"❌ Error cargando los modelos: {e}")
        raise

(
    model_clf,
    model_gain,
    model_loss,
    encoders,
    MODEL_FEATURE_NAMES,
    BASE_YEAR,
    GROWTH_RATE,
    df
) = load_all()

st.success("✓ Modelos y dataset procesado cargados correctamente.")

# ============================================
#          FUNCIONES DE PREDICCIÓN
# ============================================

def predict_next_year(row_input):
    """Predice si es ganancia o pérdida, y calcula el valor correspondiente."""
    X = row_input[MODEL_FEATURE_NAMES].values.reshape(1, -1)

    clase = model_clf.predict(X)[0]

    if clase == 1:
        pred = model_gain.predict(X)[0]
    else:
        pred = -abs(model_loss.predict(X)[0])

    return pred


def recursive_forecast(df, empresa, target_year):
    df_emp = df[df["RAZON_SOCIAL"] == empresa].copy()

    if df_emp.empty:
        raise ValueError("La empresa seleccionada no existe en el dataset procesado.")

    last_year = df_emp["ANO_DE_CORTE"].max()

    if target_year <= last_year:
        raise ValueError(f"El año objetivo ({target_year}) debe ser > último año disponible ({last_year})")

    df_emp = df_emp.sort_values("ANO_DE_CORTE")

    current_row = df_emp.iloc[-1].copy()
    current_year = int(last_year)

    history = []

    while current_year < target_year:
        next_year = current_year + 1

        # Aplicar crecimiento recursivo
        current_row["INGRESOS_OPERACIONALES"] *= GROWTH_RATE
        current_row["TOTAL_ACTIVOS"] *= GROWTH_RATE
        current_row["TOTAL_PASIVOS"] *= GROWTH_RATE
        current_row["TOTAL_PATRIMONIO"] *= GROWTH_RATE

        pred_value = predict_next_year(current_row)

        history.append((next_year, pred_value))

        current_row["ANO_DE_CORTE"] = next_year
        current_year = next_year

    return history

# ============================================
#               INTERFAZ STREAMLIT
# ============================================

st.title("📊 ALECO — Modelo de Dos Partes (Ganancia / Pérdida)")
st.write("Predicción multianual usando modelos XGBoost entrenados en Colab.")

# Selección de empresa
empresa = st.selectbox(
    "Selecciona una empresa:",
    options=sorted(df["RAZON_SOCIAL"].unique())
)

# Año target
anio_target = st.number_input(
    "Selecciona Año de Predicción",
    min_value=int(df["ANO_DE_CORTE"].min()) + 1,
    max_value=2050,
    value=2026,
    step=1
)

# Botón predecir
if st.button("🔮 Generar Predicción"):
    try:
        forecast = recursive_forecast(df, empresa, anio_target)

        st.subheader(f"📈 Predicción para {empresa}")

        for year, value in forecast:
            st.write(f"**{year}:** {value:,.2f} millones COP")

    except Exception as e:
        st.error(f"❌ ERROR generando la predicción: {e}")
