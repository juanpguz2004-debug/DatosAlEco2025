# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

st.title("📊 Dashboard Inventario de Datos Abiertos - MinTIC")

# ========================================
# Cargar CSV desde repo (ya en main)
# ========================================
CSV_PATH = "Asset_Inventory_-_Public_20251118.csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    # Limpiar nombres de columnas (snake_case)
    def clean_col_name(col):
        name = col.lower().strip()
        name = (name.replace('á','a').replace('é','e').replace('í','i')
                    .replace('ó','o').replace('ú','u'))
        name = name.replace(' ','_').replace('.','').replace('/','_').replace(':','')
        name = name.replace('(','').replace(')','')
        return name
    df.columns = [clean_col_name(col) for col in df.columns]
    return df

df = load_data(CSV_PATH)
st.success(f"✅ Archivo '{CSV_PATH}' cargado. Total registros: {len(df)}")

# ========================================
# Métrica de completitud
# ========================================
campos_minimos = [
    'titulo', 'descripcion', 'dueño', 'correo_electronico_de_contacto',
    'licencia', 'dominio', 'categoria', 'informacion_de_datos_frecuencia_de_actualizacion',
    'common_core_public_access_level', 'informacion_de_datos_cobertura_geografica'
]
campos_existentes = [c for c in campos_minimos if c in df.columns]
num_campos_totales = len(campos_existentes)

if num_campos_totales > 0:
    df['campos_diligenciados'] = df[campos_existentes].notna().sum(axis=1)
    df['completitud_score'] = (df['campos_diligenciados'] / num_campos_totales) * 100
    st.write(f"Completitud promedio: {df['completitud_score'].mean():.2f}%")
else:
    st.warning("⚠️ Ninguno de los campos mínimos existe en el DataFrame.")

# ========================================
# Detección de anomalías
# ========================================
df_modelo = df[(df['antiguedad_datos_dias'] < 9999) & (df['popularidad_score'] > 0)].copy()
if not df_modelo.empty:
    features = df_modelo[['antiguedad_datos_dias','popularidad_score','completitud_score']]
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(features)
    anomalias = model.predict(features)
    df['anomalia_score'] = 0
    df.loc[df_modelo.index,'anomalia_score'] = anomalias
    df.loc[df_modelo.index,'anomalia_distancia'] = model.decision_function(features)
    st.write(f"Total de anomalías detectadas: {(df['anomalia_score']==-1).sum()}")
else:
    df['anomalia_score'] = 0

# ========================================
# Score de prioridad/riesgo
# ========================================
df['riesgo_incumplimiento'] = np.where(df['estado_actualizacion']=='🔴 INCUMPLIMIENTO', 3.0, 0.0)
df['riesgo_completitud'] = np.where(df['completitud_score']<50, 1.5, 0.0)
max_pop = df['popularidad_score'].max()
df['riesgo_demanda'] = (df['popularidad_score']/max_pop) if max_pop>0 else 0.0
df['riesgo_anomalia'] = np.where(df['anomalia_score']==-1, 2.0, 0.0)
df['prioridad_riesgo_score'] = df['riesgo_incumplimiento'] + df['riesgo_completitud'] + df['riesgo_demanda'] + df['riesgo_anomalia']

# ========================================
# Filtrado público
# ========================================
COLUMNA_ACCESO = 'publico'
VALOR_PUBLICO_REAL = 'public'
df_publico = df[df[COLUMNA_ACCESO].astype(str).str.lower().str.strip()==VALOR_PUBLICO_REAL.lower().strip()].copy()
st.write(f"Total de activos públicos: {len(df_publico)}")

# ========================================
# Visualización 1: Score de riesgo vs antigüedad
# ========================================
st.subheader("OE3: Prioridad de Intervención vs. Antigüedad")
fig, ax = plt.subplots(figsize=(12,6))
sns.scatterplot(
    data=df_publico,
    x='antiguedad_datos_dias',
    y='prioridad_riesgo_score',
    hue='estado_actualizacion',
    palette={'🔴 INCUMPLIMIENTO':'red','🟢 CUMPLE':'green'},
    size='popularidad_score',
    sizes=(20,400),
    alpha=0.7,
    ax=ax
)
ax.axhline(y=df_publico['prioridad_riesgo_score'].quantile(0.75), color='red', linestyle='--', label='Prioridad Alta (Q3)')
ax.set_xlabel("Antigüedad de Datos (días)")
ax.set_ylabel("Score de Prioridad/Riesgo")
plt.legend(title='Estado de Actualización')
st.pyplot(fig)

# ========================================
# Visualización 2: Entidades con mayor incumplimiento
# ========================================
st.subheader("Top 10 Entidades con Mayor Incumplimiento")
COLUMNA_ENTIDAD = 'dueño'
resumen_entidad = df_publico.groupby(COLUMNA_ENTIDAD).agg(
    Total_Activos=('uid','count'),
    Activos_Incumplimiento=('estado_actualizacion', lambda x: (x=='🔴 INCUMPLIMIENTO').sum()),
    Popularidad_Media=('popularidad_score','mean')
).reset_index()
resumen_entidad['Porcentaje_Incumplimiento'] = (resumen_entidad['Activos_Incumplimiento']/resumen_entidad['Total_Activos'])*100
resumen_entidad_top = resumen_entidad[resumen_entidad['Total_Activos']>=5].sort_values('Porcentaje_Incumplimiento',ascending=False).head(10)

fig2, ax2 = plt.subplots(figsize=(12,6))
sns.barplot(
    x='Porcentaje_Incumplimiento',
    y=COLUMNA_ENTIDAD,
    data=resumen_entidad_top,
    palette='Reds_d',
    ax=ax2
)
ax2.set_xlabel("Porcentaje de Activos en INCUMPLIMIENTO (%)")
ax2.set_ylabel("Entidad Responsable")
st.pyplot(fig2)

# ========================================
# Visualización 3: Top 10 categorías
# ========================================
st.subheader("Top 10 Categorías con Mayor Cobertura Temática")
COLUMNA_CATEGORIA = 'categoria'
conteo_categoria = df_publico[COLUMNA_CATEGORIA].value_counts().head(10)
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.barplot(x=conteo_categoria.values, y=conteo_categoria.index, palette='viridis', ax=ax3)
ax3.set_xlabel("Número de Activos")
ax3.set_ylabel("Categoría")
st.pyplot(fig3)

# ========================================
# Tabla interactiva de datasets públicos
# ========================================
st.subheader("Vista de datasets públicos")
st.dataframe(df_publico.head(50))
