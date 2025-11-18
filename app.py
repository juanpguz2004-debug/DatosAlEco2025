import streamlit as st
import pandas as pd
import plotly.express as px

# ================================
# CONFIGURACIÓN DEL DASHBOARD
# ================================
st.set_page_config(
    page_title="Asset Inventory Dashboard",
    layout="wide"
)

st.title("📊 Asset Inventory Dashboard")
st.write("Visualización interactiva del inventario público de activos.")

# ================================
# CARGA DEL CSV
# ================================
CSV_PATH = "Asset_Inventory_-_Public_20251118.csv"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(CSV_PATH)
        return df, None
    except Exception as e:
        return None, str(e)

df, error = load_data()

if error:
    st.error(f"Error al cargar el CSV: {error}")
    st.stop()

if df is None or df.empty:
    st.warning("No se pudo cargar información del archivo.")
    st.stop()

# ================================
# SIDEBAR – FILTROS
# ================================
st.sidebar.header("Filtros")

columnas_categoricas = [c for c in df.columns if df[c].dtype == object]

filtros = {}
for col in columnas_categoricas:
    valores = sorted(df[col].dropna().unique().tolist())
    seleccion = st.sidebar.multiselect(f"Filtrar por {col}:", valores)
    filtros[col] = seleccion

# Aplicar filtros
df_filtrado = df.copy()
for col, val in filtros.items():
    if val:
        df_filtrado = df_filtrado[df_filtrado[col].isin(val)]

# ================================
# MOSTRAR TABLA FILTRADA
# ================================
st.subheader("📁 Datos Filtrados")
st.dataframe(df_filtrado, use_container_width=True)

# ================================
# GRÁFICOS (si existen columnas numéricas)
# ================================
columnas_numericas = df_filtrado.select_dtypes(include=["int64", "float64"]).columns.tolist()

if columnas_numericas:
    col_1, col_2 = st.columns(2)

    with col_1:
        colnum = st.selectbox("Seleccionar columna numérica para gráfico de barras", columnas_numericas)
        fig1 = px.histogram(df_filtrado, x=colnum)
        st.plotly_chart(fig1, use_container_width=True)

    with col_2:
        colnum2 = st.selectbox("Seleccionar columna numérica para gráfico de dispersión", columnas_numericas)
        fig2 = px.scatter(df_filtrado, x=colnum2, y=df_filtrado.index)
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("No se encontraron columnas numéricas para generar gráficos.")

# ================================
# DESCARGA DEL CSV FILTRADO
# ================================
st.download_button(
    label="📥 Descargar CSV filtrado",
    data=df_filtrado.to_csv(index=False),
    file_name="Asset_Inventory_filtrado.csv",
    mime="text/csv"
)
