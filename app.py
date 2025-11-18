# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Dashboard Inventario de Datos Abiertos - MinTIC", layout="wide")

st.title("📊 Dashboard Inventario de Datos Abiertos - MinTIC")

# 1️⃣ Cargar CSV
CSV_PATH = "Asset_Inventory_-_Public_20251118.csv"

try:
    df = pd.read_csv(CSV_PATH, low_memory=False)
    st.success(f"✅ Archivo '{CSV_PATH}' cargado correctamente.")
except Exception as e:
    st.error(f"❌ Error al cargar CSV: {e}")
    st.stop()

# 2️⃣ Limpiar nombres de columnas
def clean_col_name(col):
    name = col.lower().strip()
    name = name.replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u')
    name = name.replace(' ', '_').replace('.', '').replace('/', '_').replace(':','').replace('(', '').replace(')', '')
    return name

df.columns = [clean_col_name(c) for c in df.columns]
st.write("✅ Columnas procesadas:", df.columns.tolist())

# 3️⃣ Filtrar activos públicos de forma segura
public_cols = [c for c in df.columns if 'public' in c.lower()]
if not public_cols:
    st.warning("No se encontró columna para filtrar por público. Mostrando todos los datasets.")
    df_publico = df.copy()
else:
    col_publico = public_cols[0]
    df_publico = df[df[col_publico].astype(str).str.lower().str.strip() == 'public']
    st.write(f"Total de activos públicos: {len(df_publico)} / {len(df)}")

# 4️⃣ Métrica de completitud
campos_minimos = [
    'titulo', 'descripcion', 'dueño', 'correo_electronico_de_contacto',
    'licencia', 'dominio', 'categoria', 'informacion_de_datos_frecuencia_de_actualizacion',
    'common_core_public_access_level', 'informacion_de_datos_cobertura_geografica'
]
campos_existentes = [c for c in campos_minimos if c in df_publico.columns]
num_campos_totales = len(campos_existentes)

if num_campos_totales > 0:
    df_publico['campos_diligenciados'] = df_publico[campos_existentes].notna().sum(axis=1)
    df_publico['completitud_score'] = (df_publico['campos_diligenciados'] / num_campos_totales) * 100
    st.write(f"Completitud promedio: {df_publico['completitud_score'].mean():.2f}%")
else:
    df_publico['completitud_score'] = 0
    st.warning("❌ No se encontraron campos mínimos para calcular completitud")

# 5️⃣ Detección de anomalías con Isolation Forest
features_cols = [c for c in ['antiguedad_datos_dias','popularidad_score','completitud_score'] if c in df_publico.columns]
if features_cols:
    df_modelo = df_publico.copy()
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(df_modelo[features_cols])
    df_publico['anomalia_score'] = model.predict(df_modelo[features_cols])
    df_publico['anomalia_distancia'] = model.decision_function(df_modelo[features_cols])
    st.write(f"Anomalías detectadas: {(df_publico['anomalia_score']==-1).sum()}")
else:
    df_publico['anomalia_score'] = 0
    st.warning("❌ No se encontraron columnas para detección de anomalías")

# 6️⃣ Score de riesgo/prioridad
df_publico['riesgo_incumplimiento'] = np.where(df_publico.get('estado_actualizacion','')=='🔴 INCUMPLIMIENTO', 3.0, 0.0)
df_publico['riesgo_completitud'] = np.where(df_publico['completitud_score']<50,1.5,0.0)
max_pop = df_publico.get('popularidad_score', pd.Series([0])).max()
df_publico['riesgo_demanda'] = (df_publico.get('popularidad_score',0)/max_pop) if max_pop>0 else 0
df_publico['riesgo_anomalia'] = np.where(df_publico['anomalia_score']==-1,2.0,0.0)
df_publico['prioridad_riesgo_score'] = df_publico['riesgo_incumplimiento'] + df_publico['riesgo_completitud'] + df_publico['riesgo_demanda'] + df_publico['riesgo_anomalia']

# 7️⃣ Visualizaciones
st.header("📈 Visualizaciones de Inventario")

# 7a) Score de prioridad vs antigüedad
if 'antiguedad_datos_dias' in df_publico.columns:
    fig, ax = plt.subplots(figsize=(12,6))
    sns.scatterplot(
        data=df_publico,
        x='antiguedad_datos_dias',
        y='prioridad_riesgo_score',
        hue=df_publico.get('estado_actualizacion',None),
        size=df_publico.get('popularidad_score', None),
        sizes=(20,400),
        alpha=0.7
    )
    plt.xlabel("Antigüedad de Datos (días)")
    plt.ylabel("Score de Prioridad/Riesgo")
    plt.title("Prioridad de Intervención vs Antigüedad")
    st.pyplot(fig)

# 7b) Top entidades por incumplimiento
if 'dueño' in df_publico.columns and 'estado_actualizacion' in df_publico.columns:
    resumen_entidad = df_publico.groupby('dueño').agg(
        Total_Activos=('uid','count'),
        Activos_Incumplimiento=('estado_actualizacion', lambda x: (x=='🔴 INCUMPLIMIENTO').sum())
    ).reset_index()
    resumen_entidad['Porcentaje_Incumplimiento'] = (resumen_entidad['Activos_Incumplimiento']/resumen_entidad['Total_Activos'])*100
    top_entidades = resumen_entidad.sort_values('Porcentaje_Incumplimiento', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(data=top_entidades, y='dueño', x='Porcentaje_Incumplimiento', palette='Reds_d')
    plt.xlabel("% Activos en INCUMPLIMIENTO")
    plt.ylabel("Entidad Responsable")
    plt.title("Top 10 Entidades con Mayor Incumplimiento")
    st.pyplot(fig)

# 7c) Top categorías
if 'categoria' in df_publico.columns:
    top_categorias = df_publico['categoria'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=top_categorias.values, y=top_categorias.index, palette='viridis')
    plt.xlabel("Número de Activos")
    plt.ylabel("Categoría")
    plt.title("Top 10 Categorías de Activos")
    st.pyplot(fig)

st.success("✅ Dashboard generado con éxito.")
