import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Dashboard Inventario de Datos Abiertos - MinTIC")

# Carga de CSV
df = pd.read_csv("Asset_Inventory_-_Public_20251118.csv")

# Limpieza de columnas
df.columns = [c.lower().replace(" ", "_") for c in df.columns]

# Filtro público
df_publico = df[df['publico'].str.lower() == 'public']

# Métrica de completitud
campos_minimos = ['titulo', 'descripcion', 'dueño', 'correo_electronico_de_contacto', 'licencia', 'dominio']
df_publico['campos_diligenciados'] = df_publico[campos_minimos].notna().sum(axis=1)
df_publico['completitud_score'] = df_publico['campos_diligenciados'] / len(campos_minimos) * 100

# Scatter Plot Prioridad vs Antigüedad
fig = px.scatter(df_publico, x='antiguedad_datos_dias', y='completitud_score',
                 color='estado_actualizacion', size='popularidad_score',
                 hover_data=['titulo', 'dueño'])
st.plotly_chart(fig)

# Top 10 entidades incumplimiento
resumen_entidad = df_publico.groupby('dueño').agg(
    Total_Activos=('uid','count'),
    Activos_Incumplimiento=('estado_actualizacion', lambda x: (x=='🔴 INCUMPLIMIENTO').sum())
).reset_index()
resumen_entidad['Porcentaje_Incumplimiento'] = 100 * resumen_entidad['Activos_Incumplimiento'] / resumen_entidad['Total_Activos']
st.dataframe(resumen_entidad.sort_values(by='Porcentaje_Incumplimiento', ascending=False).head(10))
