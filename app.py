# ==============================================================================
# 1. IMPORTS Y CONFIGURACIÓN INICIAL
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

st.set_page_config(layout="wide") # Configuración de página amplia

st.title("📊 Dashboard Inventario de Datos Abiertos - MinTIC")
st.markdown("---")

# ========================================
# 2. Cargar CSV y Limpieza
# ========================================
# Asegúrate de que esta ruta sea correcta para tu entorno (e.g., si está en la raíz, usa "Asset_Inventory_-_Public_20251118.csv")
CSV_PATH = "data/Asset_Inventory_-_Public_20251118.csv" 

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    # Limpiar nombres de columnas (snake_case)
    def clean_col_name(col):
        name = col.lower().strip()
        # Limpieza de tildes
        name = name.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
        # Limpieza de caracteres especiales y reemplazo por guión bajo
        name = name.replace(' ', '_').replace('.', '').replace('/', '_').replace(':', '').replace('(', '').replace(')', '')
        return name
    df.columns = [clean_col_name(col) for col in df.columns]
    return df

try:
    df = load_data(CSV_PATH)
    st.success(f"✅ Archivo '{CSV_PATH}' cargado. Total registros: {len(df)}")
except FileNotFoundError:
    st.error(f"Error: No se encontró el archivo '{CSV_PATH}'. Verifica la ruta y que el archivo exista.")
    st.stop()
    
# ========================================
# 3. Métrica de completitud (OE1)
# ========================================
st.subheader("Métricas de Calidad de Datos")

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
    st.info(f"Completitud promedio: **{df['completitud_score'].mean():.2f}%**")
else:
    df['completitud_score'] = 0
    st.warning("⚠️ Ninguno de los campos mínimos existe en el DataFrame.")

# =======================================================
# 4. Métricas de Tiempo y Uso (Antigüedad, Frecuencia, Popularidad)
# =======================================================
COLUMNA_FECHA_ACTUALIZACION = 'fecha_de_ultima_actualizacion_de_datos_utc' 
COLUMNA_FRECUENCIA = 'informacion_de_datos_frecuencia_de_actualizacion'

# A. Antigüedad
if COLUMNA_FECHA_ACTUALIZACION in df.columns:
    df[COLUMNA_FECHA_ACTUALIZACION] = pd.to_datetime(df[COLUMNA_FECHA_ACTUALIZACION], errors='coerce', utc=True)
    HOY = pd.Timestamp.now(tz='utc') 
    df['antiguedad_datos_dias'] = (HOY - df[COLUMNA_FECHA_ACTUALIZACION]).dt.days.fillna(9999) 
else:
    df['antiguedad_datos_dias'] = 9999

# B. Frecuencia y Cumplimiento
mapa_frecuencia = {'diaria': 1, 'diario': 1, 'continuamente': 1, 'semanal': 7, 'quincenal': 15, 'mensual': 30, 'mensualmente': 30, 'bimestral': 60, 'trimestral': 90, 'semestral': 182, 'anual': 365, 'anualmente': 365, 'no aplica': 365 * 10, 'null': 365 * 10 }
if COLUMNA_FRECUENCIA in df.columns:
    df['frecuencia_esperada_dias'] = df[COLUMNA_FRECUENCIA].astype(str).str.lower().str.strip().map(mapa_frecuencia).fillna(365 * 10) 
else:
    df['frecuencia_esperada_dias'] = 365 * 10

UMBRAL_GRACIA_DIAS = 15 
df['estado_actualizacion'] = np.where(
    df['antiguedad_datos_dias'] > (df['frecuencia_esperada_dias'] + UMBRAL_GRACIA_DIAS),
    '🔴 INCUMPLIMIENTO',
    '🟢 CUMPLE'
)

# C. Popularidad
if 'vistas' in df.columns and 'descargas' in df.columns:
    df['vistas'] = pd.to_numeric(df['vistas'], errors='coerce')
    df['descargas'] = pd.to_numeric(df['descargas'], errors='coerce')
    vistas_norm = df['vistas'].fillna(0)
    descargas_norm = df['descargas'].fillna(0)
    max_vistas = vistas_norm.max()
    max_descargas = descargas_norm.max()
    
    if max_vistas > 0:
        vistas_norm = vistas_norm / max_vistas
    if max_descargas > 0:
        descargas_norm = descargas_norm / max_descargas
        
    df['popularidad_score'] = (vistas_norm * 0.6) + (descargas_norm * 0.4)
    df['popularidad_score'] = df['popularidad_score'] * 10 # Re-escalar a 0-10
else:
    df['popularidad_score'] = 0

# ========================================
# 5. Detección de anomalías (OE1 - ML)
# ========================================
df['anomalia_score'] = 0 # Inicializar
df_modelo = df[(df['antiguedad_datos_dias'] < 9999) & (df['popularidad_score'] > 0)].copy()

if not df_modelo.empty:
    features = df_modelo[['antiguedad_datos_dias','popularidad_score','completitud_score']]
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(features)
    anomalias = model.predict(features)
    
    df.loc[df_modelo.index,'anomalia_score'] = anomalias
    df.loc[df_modelo.index,'anomalia_distancia'] = model.decision_function(features)
    st.write(f"Total de anomalías detectadas: **{(df['anomalia_score']==-1).sum()}**")
else:
    st.warning("No hay suficientes datos válidos para ejecutar el modelo Isolation Forest.")

# ========================================
# 6. Score de prioridad/riesgo (OE2)
# ========================================
# Ponderación de riesgo: Incumplimiento(3.0) + BajaCompletitud(1.5) + Demanda(1.0) + Anomalía(2.0) = 7.5
df['riesgo_incumplimiento'] = np.where(df['estado_actualizacion']=='🔴 INCUMPLIMIENTO', 3.0, 0.0)
df['riesgo_completitud'] = np.where(df['completitud_score']<50, 1.5, 0.0)
max_pop = df['popularidad_score'].max()
df['riesgo_demanda'] = (df['popularidad_score']/max_pop) if max_pop>0 else 0.0
df['riesgo_anomalia'] = np.where(df['anomalia_score']==-1, 2.0, 0.0)
df['prioridad_riesgo_score'] = df['riesgo_incumplimiento'] + df['riesgo_completitud'] + df['riesgo_demanda'] + df['riesgo_anomalia']

# ========================================
# 7. Filtrado público
# ========================================
COLUMNA_ACCESO = 'common_core_public_access_level'
VALOR_PUBLICO_REAL = 'public'

if COLUMNA_ACCESO in df.columns:
    df_publico = df[df[COLUMNA_ACCESO].astype(str).str.lower().str.strip()==VALOR_PUBLICO_REAL.lower().strip()].copy()
else:
    df_publico = df.copy() 

st.write(f"Total de activos públicos (filtrados por '{VALOR_PUBLICO_REAL}'): **{len(df_publico)}**")
st.markdown("---")

# ========================================
# 8. Visualización 1: Score de riesgo vs antigüedad
# ========================================
st.header("🚨 OE2: Prioridad de Intervención (Riesgo)")
st.subheader("Riesgo por Antigüedad vs. Score de Intervención")

if df_publico.empty:
    st.warning("No hay activos públicos para visualizar después de aplicar los filtros.")
    st.stop()

# Lógica para cuartil
if df_publico['prioridad_riesgo_score'].nunique() > 1:
    prioridad_q3 = df_publico['prioridad_riesgo_score'].quantile(0.75)
else:
    prioridad_q3 = df_publico['prioridad_riesgo_score'].max()

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
ax.axhline(y=prioridad_q3, color='red', linestyle='--', label=f'Prioridad Alta (Q3: {prioridad_q3:.2f})')
ax.set_xlabel("Antigüedad de Datos (días)")
ax.set_ylabel("Score de Prioridad/Riesgo")
plt.legend(title='Estado de Actualización')
st.pyplot(fig)

# ========================================
# 9. Visualización 2: Entidades con mayor incumplimiento
# ========================================
st.header("🔍 OE1: Diagnóstico de Coherencia y Cobertura")
st.subheader("Top 10 Entidades con Mayor Incumplimiento")
COLUMNA_ENTIDAD = 'dueño'

resumen_entidad = df_publico.groupby(COLUMNA_ENTIDAD).agg(
    Total_Activos=('titulo','count'),
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
# 10. Visualización 3: Top 10 categorías
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
# 11. Tabla interactiva de datasets públicos
# ========================================
st.subheader("Vista Detallada de Activos Públicos (Top 50)")
st.dataframe(df_publico.head(50), use_container_width=True)
