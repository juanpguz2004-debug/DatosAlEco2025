# app.py - Versión corregida: crea GANANCIA_PERDIDA si falta y evita que la app se caiga
import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import os
import joblib
import altair as alt

st.set_page_config(page_title="Dashboard ALECO 2025", layout="wide")

# -------------------------
# Función robusta para cargar CSV
# -------------------------
@st.cache_data
def load_data():
    csv_file = "10.000_Empresas_mas_Grandes_del_País_20251115.csv"
    if not os.path.exists(csv_file):
        st.error(f"No se encontró el archivo CSV en el repositorio: {csv_file}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Error al leer el CSV: {e}")
        return pd.DataFrame()

    # Normalizar nombres de columnas
    def normalize_col(col):
        if not isinstance(col, str):
            col = str(col)
        col = col.strip()
        col = col.upper()
        col = ''.join(c for c in unicodedata.normalize('NFD', col) if unicodedata.category(c) != 'Mn')
        col = col.replace("Ñ", "N")
        # limpiar caracteres
        col = col.replace(" ", "_").replace("(", "").replace(")", "")
        col = col.replace(".", "").replace("-", "_").replace("/", "_")
        while "__" in col:
            col = col.replace("__", "_")
        return col

    df.columns = [normalize_col(c) for c in df.columns]

    # Detectar columna objetivo en varias formas posibles
    posibles_objetivo = [
        "GANANCIA_PERDIDA", "GANANCIA", "GANANCIA_PERCIDA", "GANANCIA_PERDIDAS",
        "GANANCIA_PERDIDAS", "GANANCIA_PÉRDIDA", "UTILIDAD", "UTILIDAD_NETA"
    ]
    encontrado = None
    for opc in posibles_objetivo:
        opc_norm = normalize_col(opc)
        if opc_norm in df.columns:
            encontrado = opc_norm
            break

    if encontrado:
        # renombrar a la forma estándar interna
        if encontrado != "GANANCIA_PERDIDA":
            df = df.rename(columns={encontrado: "GANANCIA_PERDIDA"})
    else:
        # crear columna objetivo vacía para que la app no falle
        df["GANANCIA_PERDIDA"] = np.nan
        st.warning("No se encontró columna de ganancia en el CSV. Se creó 'GANANCIA_PERDIDA' con valores NaN.")

    # Comprobar columnas clave (las advertimos pero NO detenemos la app)
    columnas_clave = ["MACROSECTOR", "REGION", "INGRESOS_OPERACIONALES", "TOTAL_ACTIVOS", "TOTAL_PASIVOS", "TOTAL_PATRIMONIO"]
    faltantes = [c for c in columnas_clave if c not in df.columns]
    if faltantes:
        st.warning(f"Faltan columnas clave en el CSV: {faltantes}")
        st.info(f"Columnas detectadas: {list(df.columns)}")

    return df

# -------------------------
# Cargar datos y mostrar debug
# -------------------------
df = load_data()
if df.empty:
    st.stop()

st.sidebar.title("Información del dataset")
st.sidebar.write(f"Filas cargadas: {len(df)}")
st.sidebar.write("Columnas detectadas:")
st.sidebar.write(list(df.columns))

# -------------------------
# Cargar modelo
# -------------------------
@st.cache_resource
def load_model():
    model_file = "model.pkl"
    if not os.path.exists(model_file):
        st.error("No se encontró 'model.pkl' en el repo.")
        return None
    try:
        model = joblib.load(model_file)
        return model
    except Exception as e:
        st.error(f"Error cargando model.pkl: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# -------------------------
# Interfaz principal
# -------------------------
st.title("📊 Dashboard ALECO 2025")
st.markdown("Explora las 10.000 empresas y predice ganancias. Si faltan columnas, la app intenta continuar y te avisa.")

# -------------------------
# Filtros para explorar por sector/region/tamaño
# -------------------------
# Aseguramos que las columnas existen antes de usarlas en selectboxes
sectores = ["Todos"] + sorted(df["MACROSECTOR"].dropna().unique().tolist()) if "MACROSECTOR" in df.columns else ["Todos"]
regiones = ["Todos"] + sorted(df["REGION"].dropna().unique().tolist()) if "REGION" in df.columns else ["Todos"]

f_sector = st.selectbox("Filtrar por Macrosector", sectores)
f_region = st.selectbox("Filtrar por Región", regiones)

df_filtrado = df.copy()
if f_sector != "Todos" and "MACROSECTOR" in df.columns:
    df_filtrado = df_filtrado[df_filtrado["MACROSECTOR"] == f_sector]
if f_region != "Todos" and "REGION" in df.columns:
    df_filtrado = df_filtrado[df_filtrado["REGION"] == f_region]

st.subheader("Vista (primeras filas) del conjunto filtrado")
st.dataframe(df_filtrado.head(20))

# -------------------------
# KPI / Resúmenes por sector/region/tamaño
# -------------------------
st.subheader("KPIs agregados")

if not df_filtrado.empty:
    # Agregados simples (manejar si columnas faltan)
    ingresos_sum = df_filtrado["INGRESOS_OPERACIONALES"].sum() if "INGRESOS_OPERACIONALES" in df_filtrado.columns else np.nan
    patrimonio_mean = df_filtrado["TOTAL_PATRIMONIO"].mean() if "TOTAL_PATRIMONIO" in df_filtrado.columns else np.nan
    count_emp = len(df_filtrado)
    c1, c2, c3 = st.columns(3)
    c1.metric("Empresas en el filtro", count_emp)
    c2.metric("Suma ingresos (filtrado)", f"{ingresos_sum:,.2f}" if not np.isnan(ingresos_sum) else "N/A")
    c3.metric("Promedio patrimonio", f"{patrimonio_mean:,.2f}" if not np.isnan(patrimonio_mean) else "N/A")
else:
    st.info("No hay filas para los filtros seleccionados.")

# -------------------------
# Visualización: ingresos por macrosector (si la columna existe)
# -------------------------
if "MACROSECTOR" in df_filtrado.columns and "INGRESOS_OPERACIONALES" in df_filtrado.columns:
    st.subheader("Ingresos operacionales por Macrosector (filtrado)")
    df_chart = df_filtrado.groupby("MACROSECTOR")["INGRESOS_OPERACIONALES"].sum().reset_index()
    chart = alt.Chart(df_chart).mark_bar().encode(
        x="MACROSECTOR",
        y="INGRESOS_OPERACIONALES",
        tooltip=["MACROSECTOR", "INGRESOS_OPERACIONALES"]
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No hay columnas necesarias para graficar ingresos por macrosector.")

# -------------------------
# PREDICCIÓN: construir input exactamente como el modelo espera
# -------------------------
st.subheader("🔮 Predicción de Ganancia/Pérdida (entrada manual)")

# columnas exactas desde el entrenamiento (según tu modelo)
COLUMNAS_MODELO = [
    "NIT",
    "RAZON_SOCIAL",
    "SUPERVISOR",
    "REGION",
    "DEPARTAMENTO_DOMICILIO",
    "CIUDAD_DOMICILIO",
    "CIIU",
    "MACROSECTOR",
    "INGRESOS_OPERACIONALES",
    "TOTAL_ACTIVOS",
    "TOTAL_PASIVOS",
    "TOTAL_PATRIMONIO",
    "ANO_DE_CORTE"
]

# Selección de región/macrosector por defecto si existen
regiones_disp = df["REGION"].dropna().unique().tolist() if "REGION" in df.columns else ["NO_APLICA"]
macros_disp = df["MACROSECTOR"].dropna().unique().tolist() if "MACROSECTOR" in df.columns else ["NO_APLICA"]

with st.form("pred_form"):
    r_sel = st.selectbox("Región", sorted(regiones_disp))
    m_sel = st.selectbox("Macrosector", sorted(macros_disp))
    ing = st.number_input("Ingresos operacionales", min_value=0.0, value=0.0)
    act = st.number_input("Total activos", min_value=0.0, value=0.0)
    pas = st.number_input("Total pasivos", min_value=0.0, value=0.0)
    pat = st.number_input("Total patrimonio", min_value=0.0, value=0.0)
    ano = st.number_input("Año de corte", min_value=1900, max_value=2100, value=2025)
    enviar = st.form_submit_button("Calcular predicción")

if enviar:
    try:
        X_in = pd.DataFrame([{
            "NIT": "0",
            "RAZON_SOCIAL": "EMPRESA_MANUAL",
            "SUPERVISOR": "NO_APLICA",
            "REGION": r_sel,
            "DEPARTAMENTO_DOMICILIO": "NO_APLICA",
            "CIUDAD_DOMICILIO": "NO_APLICA",
            "CIIU": "0000",
            "MACROSECTOR": m_sel,
            "INGRESOS_OPERACIONALES": ing,
            "TOTAL_ACTIVOS": act,
            "TOTAL_PASIVOS": pas,
            "TOTAL_PATRIMONIO": pat,
            "ANO_DE_CORTE": ano
        }])
        # Asegurar orden y presencia de columnas exactamente como el modelo espera
        for col in COLUMNAS_MODELO:
            if col not in X_in.columns:
                X_in[col] = "NO_APLICA"  # rellenar vacío si falta
        X_in = X_in[COLUMNAS_MODELO]

        pred = model.predict(X_in)[0]
        st.success(f"Predicción estimada: ${pred:,.2f}")
    except Exception as e:
        st.error(f"Error generando predicción: {e}")

# -------------------------
# Predicción masiva por sector/region/tamaño (opcional)
# -------------------------
st.subheader("📋 Predicción masiva para el conjunto filtrado (opcional)")

if st.button("Generar predicciones para el conjunto filtrado"):
    try:
        # Preparar X masivo: debemos asegurar las mismas columnas (COLUMNAS_MODELO)
        # Tomamos df_filtrado y garantizamos columnas
        X_mass = df_filtrado.copy()
        # Añadir columnas faltantes con valores por defecto
        for col in COLUMNAS_MODELO:
            if col not in X_mass.columns:
                # Si es numérica conocida, poner 0; sino 'NO_APLICA'
                if col in ["INGRESOS_OPERACIONALES", "TOTAL_ACTIVOS", "TOTAL_PASIVOS", "TOTAL_PATRIMONIO", "ANO_DE_CORTE"]:
                    X_mass[col] = 0
                else:
                    X_mass[col] = "NO_APLICA"
        X_mass = X_mass[COLUMNAS_MODELO]
        preds = model.predict(X_mass)
        df_result = df_filtrado.copy().reset_index(drop=True)
        df_result["PREDICCION_GANANCIA"] = preds
        st.dataframe(df_result.head(50))
        # Agregar resumen por sector/region/tamaño
        st.write("Resumen por macrosector (media de predicción):")
        if "MACROSECTOR" in df_result.columns:
            resumen = df_result.groupby("MACROSECTOR")["PREDICCION_GANANCIA"].mean().reset_index().sort_values("PREDICCION_GANANCIA", ascending=False)
            st.dataframe(resumen)
    except Exception as e:
        st.error(f"Error generando predicciones masivas: {e}")

# -------------------------
# FIN
# -------------------------
