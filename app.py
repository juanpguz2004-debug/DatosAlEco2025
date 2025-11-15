import streamlit as st
import pandas as pd
import joblib
import os
import altair as alt

# =========================================================
# 1️⃣ CARGAR DATOS
# =========================================================
@st.cache_data
def load_data():
    csv_file = "10.000_Empresas_mas_Grandes_del_País_20251115.csv"
    if not os.path.exists(csv_file):
        st.error(f"No se encontró el archivo CSV: **{csv_file}**. Asegúrate de que esté en el mismo directorio.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")
        return pd.DataFrame()

    # Limpiar y estandarizar nombres de columnas
    df.columns = [c.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("Ñ","N") for c in df.columns]
    
    # Columnas críticas
    required_cols = ['MACROSECTOR', 'REGION', 'INGRESOS_OPERACIONALES', 'GANANCIA_PERDIDA']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Faltan columnas necesarias: {missing_cols}")
        return pd.DataFrame()
    
    return df

df = load_data()
if df.empty:
    st.stop()

# =========================================================
# 2️⃣ CARGAR MODELO
# =========================================================
@st.cache_resource
def load_model():
    model_file = "model.pkl"
    if not os.path.exists(model_file):
        st.error(f"No se encontró el archivo del modelo: {model_file}")
        return None
    try:
        model = joblib.load(model_file)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# =========================================================
# 3️⃣ INTERFAZ STREAMLIT
# =========================================================
st.title("Dashboard de Empresas")
st.markdown("Explora los datos de las 10.000 empresas más grandes del país y genera predicciones de ganancias.")

# Filtros
sector = st.selectbox("Selecciona un sector (MACROSECTOR)", ["Todos"] + df["MACROSECTOR"].dropna().unique().tolist())
region = st.selectbox("Selecciona una región (REGION)", ["Todos"] + df["REGION"].dropna().unique().tolist())

df_filtered = df.copy()
if sector != "Todos":
    df_filtered = df_filtered[df_filtered["MACROSECTOR"] == sector]
if region != "Todos":
    df_filtered = df_filtered[df_filtered["REGION"] == region]

if df_filtered.empty:
    st.warning("No hay empresas que coincidan con los filtros.")
st.dataframe(df_filtered.head(50))

# =========================================================
# 4️⃣ VISUALIZACIONES
# =========================================================
st.subheader("Visualizaciones")
df_chart = df_filtered.groupby('MACROSECTOR')['INGRESOS_OPERACIONALES'].sum().reset_index()
chart = alt.Chart(df_chart).mark_bar().encode(
    x=alt.X('MACROSECTOR', title='Macrosector'),
    y=alt.Y('INGRESOS_OPERACIONALES', title='Ingresos Operacionales (Suma)'),
    tooltip=['MACROSECTOR', alt.Tooltip('INGRESOS_OPERACIONALES', format=',.0f')]
).properties(title='Suma de Ingresos Operacionales por Macrosector')
st.altair_chart(chart, use_container_width=True)

# =========================================================
# 5️⃣ PREDICCIONES
# =========================================================
st.subheader("Predicciones de Ganancias")

# Lista de columnas de features usadas para entrenar el modelo
feature_cols = [c for c in df.columns if c not in ['GANANCIA_PERDIDA','RAZON_SOCIAL','NIT','SUPERVISOR','REGION','DEPARTAMENTO_DOMICILIO','CIUDAD_DOMICILIO','CIIU','MACROSECTOR','Año_de_Corte']]

if df_filtered.empty:
    st.stop()

max_index = len(df_filtered) - 1
empresa_idx = st.number_input(f"Selecciona índice de empresa para predecir (0 a {max_index})",
                              min_value=0, max_value=max_index, value=0)

try:
    X_pred = df_filtered[feature_cols]
    pred_value = model.predict(X_pred.iloc[[empresa_idx]])[0]
    st.write(f"Predicción de ganancia para **{df_filtered.iloc[empresa_idx]['RAZON_SOCIAL']}**: **${pred_value:,.2f}**")
except Exception as e:
    st.error(f"Error al realizar la predicción. {e}")

# =========================================================
# 6️⃣ GPT4All (opcional, comentado)
# =========================================================
"""
try:
    from gpt4all import GPT4All
    model_path = "ggml-model.bin"  # Debes subir este archivo al repo
    gpt = GPT4All(model_path)
    st.subheader("Consulta en lenguaje natural")
    user_question = st.text_input("Pregunta sobre los datos o predicciones:")
    if user_question:
        context = f"Top 5 empresas:\n{df_filtered.head(5).to_string()}"
        prompt = f"{context}\nPregunta: {user_question}\nRespuesta:"
        response = gpt.chat(prompt)
        st.markdown(f"**GPT4All:** {response}")
except Exception as e:
    st.warning("GPT4All no disponible: {e}")
"""
