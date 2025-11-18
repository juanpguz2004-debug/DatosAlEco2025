# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

# ========================================
# Título del dashboard
# ========================================
st.title("📊 Dashboard Inventario de Datos Abiertos - MinTIC")

# ========================================
# Cargar CSV desde el repositorio (ya en main)
# ========================================
CSV_PATH = "Asset_Inventory_-_Public_20251118.csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    
    # Normalizar nombres de columnas
    df.columns = df.columns.str.strip()  # quitar espacios
    df.columns = df.columns.str.lower()  # minúsculas
    df.columns = df.columns.str.replace(r"[\s\(\):]", "_", regex=True)
    df.columns = df.columns.str.replace(r"__+", "_", regex=True)
    df.columns = df.columns.str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
    
    return df

df = load_data(CSV_PATH)
st.success(f"✅ Archivo '{CSV_PATH}' cargado correctamente. Total de registros: {len(df)}")

# ========================================
# Filtrar solo activos públicos
# ========================================
# Normalizar columna de público
if "publico" in df.columns:
    df['publico'] = df['publico'].astype(str).str.lower().str.strip()
    public_values = ['public', 'público', 'si', 'sí', 'yes']
    df_publico = df[df['publico'].isin(public_values)]
    st.write(f"Total de activos públicos: {len(df_publico)} / {len(df)}")
else:
    st.warning("⚠️ No se encontró la columna de público en el CSV")
    df_publico = df.copy()  # para no romper el resto de la app

# ========================================
# Métricas básicas de completitud
# ========================================
st.header("📈 Métricas básicas de completitud")
# Columnas mínimas consideradas importantes
required_cols = [
    "titulo", "descripcion", "dueño", "dominio", "categoria",
    "informacindedatos_frecuenciadeactualizacin", "informacindedatos_coberturageogrfica",
    "license"
]

# Verificar si existen en df
required_cols = [c for c in required_cols if c in df_publico.columns]

if required_cols:
    completeness = df_publico[required_cols].notna().mean(axis=1) * 100
    st.write(f"Completitud promedio: {completeness.mean():.2f}%")
else:
    st.warning("⚠️ No se encontraron columnas mínimas para calcular completitud.")

# ========================================
# Gráfico de distribución por dominio
# ========================================
st.header("📊 Distribución de datasets por dominio")
if "dominio" in df_publico.columns:
    domain_counts = df_publico['dominio'].value_counts().reset_index()
    domain_counts.columns = ['dominio', 'cantidad']
    fig_domain = px.bar(domain_counts, x='dominio', y='cantidad', text='cantidad', height=400)
    st.plotly_chart(fig_domain)
else:
    st.warning("⚠️ No se encontró la columna 'dominio' para graficar.")

# ========================================
# Gráfico de distribución territorial
# ========================================
st.header("🌎 Cobertura territorial")
if "informacindelaentidad_departamento" in df_publico.columns:
    dept_counts = df_publico['informacindelaentidad_departamento'].value_counts().reset_index()
    dept_counts.columns = ['departamento', 'cantidad']
    fig_dept = px.bar(dept_counts, x='departamento', y='cantidad', text='cantidad', height=400)
    st.plotly_chart(fig_dept)
else:
    st.warning("⚠️ No se encontró la columna 'informacindelaentidad_departamento' para graficar.")

# ========================================
# Detección de anomalías simple
# ========================================
st.header("🚨 Detección de datasets atípicos")
num_features = ["row_count", "column_count", "visits", "downloads"]
num_features = [c for c in num_features if c in df_publico.columns]

if num_features:
    df_model = df_publico[num_features].fillna(0)
    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(df_model)
    df_publico['anomaly'] = iso.predict(df_model)
    st.write(f"Total de datasets detectados como atípicos: {(df_publico['anomaly'] == -1).sum()}")
else:
    st.warning("⚠️ No se encontraron columnas numéricas para detección de anomalías.")

# ========================================
# Mostrar tabla de datasets públicos
# ========================================
st.header("📋 Vista de datasets públicos")
st.dataframe(df_publico.head(50))
