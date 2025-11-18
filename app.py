# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Dashboard Inventario de Datos Abiertos - MinTIC", layout="wide")
st.title("📊 Dashboard Inventario de Datos Abiertos - MinTIC")

# --------------------------
# 1️⃣ CARGA DEL CSV LOCAL
# --------------------------
CSV_PATH = "Asset_Inventory_-_Public_20251118.csv"  # <-- archivo en main de github
try:
    df = pd.read_csv(CSV_PATH)
    st.success(f"✅ Archivo '{CSV_PATH}' cargado correctamente.")
    st.write(f"Total de registros: {len(df)}")
except FileNotFoundError:
    st.error(f"❌ No se encontró el archivo '{CSV_PATH}'. Verifica que exista en tu repositorio.")
    st.stop()

# --------------------------
# 2️⃣ FILTRO DE ACTIVOS PÚBLICOS
# --------------------------
df['audience'] = df['audience'].astype(str).str.lower().str.strip()
public_values = ['public', 'público', 'si', 'sí']
df_publico = df[df['audience'].isin(public_values)]
st.write(f"Total de activos públicos: {len(df_publico)} / {len(df)}")

# --------------------------
# 3️⃣ COMPLETITUD DE METADATOS
# --------------------------
campos_minimos = [
    'Titulo','description','owner','contact_email','license','domain','category',
    'informacindedatos_frecuenciadeactualizacin','informacindedatos_coberturageogrfica',
    'commoncore_publicaccesslevel'
]

df_publico['campos_diligenciados'] = df_publico[campos_minimos].notna().sum(axis=1)
df_publico['completitud_score'] = (df_publico['campos_diligenciados'] / len(campos_minimos)) * 100
completitud_promedio = df_publico['completitud_score'].mean()
st.write(f"Completitud promedio: {completitud_promedio:.2f}%")

# --------------------------
# 4️⃣ ANOMALÍAS CON ISOLATION FOREST
# --------------------------
features_cols = ['visits','downloads','completitud_score']
df_modelo = df_publico[features_cols].dropna()

if not df_modelo.empty:
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(df_modelo)
    df_publico.loc[df_modelo.index,'anomalia_score'] = model.predict(df_modelo)
else:
    df_publico['anomalia_score'] = 0

# --------------------------
# 5️⃣ VISUALIZACIONES
# --------------------------
st.subheader("📈 Gráficos de KPIs")
col1, col2, col3 = st.columns(3)
col1.metric("Total de datasets públicos", len(df_publico))
col2.metric("Completitud promedio (%)", f"{completitud_promedio:.2f}")
col3.metric("Datasets con anomalías", (df_publico['anomalia_score'] == -1).sum())

# Distribución de completitud
fig1 = px.histogram(df_publico, x='completitud_score', nbins=20, title="Distribución de Completitud")
st.plotly_chart(fig1, use_container_width=True)

# Top 10 dominios
top_domains = df_publico['domain'].value_counts().head(10).reset_index()
fig2 = px.bar(top_domains, x='index', y='domain', title="Top 10 Dominios")
st.plotly_chart(fig2, use_container_width=True)

# --------------------------
# 6️⃣ TABLA FILTRABLE
# --------------------------
st.subheader("📋 Tabla de Activos Públicos")
st.dataframe(df_publico.sort_values(by='completitud_score', ascending=False))
