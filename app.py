import streamlit as st
import pandas as pd
import joblib
import os
import altair as alt

# Para GPT4All
try:
    from gpt4all import GPT4All
except ImportError:
    st.error("La librería 'gpt4all' no está instalada. Ejecuta 'pip install gpt4all'.")
    GPT4All = None 

# ==========================================
# 1️⃣ Cargar datos
# ==========================================
@st.cache_data
def load_data():
    csv_file = "10.000_Empresas_mas_Grandes_del_País_20251115.csv"
    if not os.path.exists(csv_file):
        st.error(f"No se encontró el archivo CSV: **{csv_file}**. Asegúrate de que esté en el mismo directorio.")
        return pd.DataFrame()
    df = pd.read_csv(csv_file)
    return df

df = load_data()

if df.empty:
    st.stop()  # Si no hay datos, detener la app

# ==========================================
# 2️⃣ Cargar modelo
# ==========================================
@st.cache_resource
def load_model():
    model_file = "model.pkl"
    if not os.path.exists(model_file):
        st.error(f"No se encontró el archivo del modelo: **{model_file}**. Asegúrate de que esté en el mismo directorio.")
        return None
    try:
        model = joblib.load(model_file)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo PKL: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# ==========================================
# 3️⃣ Configurar GPT4All
# ==========================================
@st.cache_resource
def init_gpt():
    if GPT4All is None:
        return None
    # Cambia "ggml-model.bin" por el modelo que descargaste
    model_path = "ggml-model.bin"
    try:
        gpt_model = GPT4All(model_path)
        return gpt_model
    except Exception as e:
        st.warning(f"Error al inicializar GPT4All. La funcionalidad de consulta en lenguaje natural no estará disponible. Asegúrate de que el modelo **{model_path}** esté disponible. Error: {e}")
        return None

gpt = init_gpt()

# ==========================================
# 4️⃣ Interfaz Streamlit
# ==========================================
st.title("Dashboard de Empresas")
st.markdown("Explora los datos de las 10.000 empresas más grandes del país y genera predicciones de ganancias usando XGBoost.")

# Filtros interactivos
sector = st.selectbox("Selecciona un sector (MACROSECTOR)", ["Todos"] + df["MACROSECTOR"].dropna().unique().tolist())
region = st.selectbox("Selecciona una región (REGIÓN)", ["Todos"] + df["REGIÓN"].dropna().unique().tolist())

# Filtrar datos
df_filtered = df.copy()
if sector != "Todos":
    df_filtered = df_filtered[df_filtered["MACROSECTOR"] == sector]
if region != "Todos":
    df_filtered = df_filtered[df_filtered["REGIÓN"] == region]

st.dataframe(df_filtered.head(50))

if df_filtered.empty:
    st.error("No hay empresas que coincidan con los filtros seleccionados.")
    st.stop()

# ==========================================
# 5️⃣ Visualizaciones
# ==========================================
st.subheader("Visualizaciones")

# Agrupar por MACROSECTOR y sumar ingresos
df_chart = df_filtered.groupby('MACROSECTOR')['INGRESOS_OPERACIONALES'].sum().reset_index()

# Ejemplo: Ingresos por macrosector
chart = alt.Chart(df_chart).mark_bar().encode(
    x=alt.X('MACROSECTOR', title='Macrosector'),
    y=alt.Y('INGRESOS_OPERACIONALES', title='Ingresos Operacionales (Suma)'),
    tooltip=['MACROSECTOR', alt.Tooltip('INGRESOS_OPERACIONALES', format=',.0f', title='Total Ingresos')]
).properties(
    title='Suma de Ingresos Operacionales por Macrosector'
)
st.altair_chart(chart, use_container_width=True)

# ==========================================
# 6️⃣ Predicciones
# ==========================================
st.subheader("Predicciones de Ganancias")
max_index = len(df_filtered) - 1
empresa_idx = st.number_input(
    f"Selecciona índice de empresa para predecir ganancia (0 a {max_index})", 
    min_value=0, 
    max_value=max_index, 
    value=0
)

# Columnas que se deben eliminar del DataFrame para obtener solo las features del modelo
X_pred = df_filtered.drop(columns=["GANANCIA_(PÉRDIDA)", "RAZÓN_SOCIAL", "NIT", "SUPERVISOR", "REGIÓN", "DEPARTAMENTO DOMICILIO", "CIUDAD DOMICILIO", "CIIU", "MACROSECTOR", "Año de Corte"], errors='ignore')

# Realizar la predicción
try:
    pred_value = model.predict(X_pred.iloc[[empresa_idx]])[0]
    st.write(f"Predicción de ganancia para la empresa **{df_filtered.iloc[empresa_idx]['RAZÓN_SOCIAL']}**: **${pred_value:,.2f}**")
except Exception as e:
    st.error(f"Error al realizar la predicción. Asegúrate de que las columnas de features de la empresa coincidan con las que espera el modelo. Error: {e}")

# ==========================================
# 7️⃣ Preguntas con GPT4All
# ==========================================
st.subheader("Consulta en lenguaje natural")
user_question = st.text_input("Hazle una pregunta a GPT4All sobre los datos o las predicciones:")

if user_question:
    if gpt:
        # Pasamos un resumen de los datos al prompt para contexto
        context = f"Datos de empresas (top 5):\n{df_filtered.head(5).to_string()}"
        prompt = f"{context}\nPregunta: {user_question}\nRespuesta:"
        
        try:
            response = gpt.chat(prompt)
            st.markdown(f"**GPT4All:** {response}")
        except Exception as e:
            st.error(f"Error al generar respuesta con GPT4All: {e}")
    else:
        st.warning("GPT4All no se pudo inicializar. Consulta en lenguaje natural no disponible.")
