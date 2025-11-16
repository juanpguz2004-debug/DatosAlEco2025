import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import unicodedata

# ----------------------------------------------------
# 0) CONFIGURACIÓN DE PÁGINA
# ----------------------------------------------------
# Esto ayuda a evitar algunos problemas de renderizado inicial
st.set_page_config(
    page_title="Dashboard ALECO", 
    layout="wide"
)

# ----------------------------------------------------
# 1) FUNCIONES DE NORMALIZACIÓN
# ----------------------------------------------------

def normalize_col(col):
    col = col.strip()
    col = col.upper()
    col = col.replace(" ", "_")
    col = col.replace("(", "")
    col = col.replace(")", "")
    col = col.replace("Ñ", "N")
    # Eliminar acentos
    col = ''.join(
        c for c in unicodedata.normalize('NFD', col)
        if unicodedata.category(c) != 'Mn'
    )
    return col


# ----------------------------------------------------
# 2) CARGAR CSV (CON CORRECCIÓN CRÍTICA)
# ----------------------------------------------------
@st.cache_data
def load_data():
    csv_file = "10.000_Empresas_mas_Grandes_del_País_20251115.csv"

    if not os.path.exists(csv_file):
        st.error(f"No se encontró el archivo: {csv_file}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Error al leer el CSV: {e}")
        return pd.DataFrame()


    # Normalizar columnas
    df.columns = [normalize_col(c) for c in df.columns]

    # Columnas requeridas
    required_cols = [
        'NIT','RAZON_SOCIAL','SUPERVISOR','REGION','DEPARTAMENTO_DOMICILIO',
        'CIUDAD_DOMICILIO','CIIU','MACROSECTOR',
        'INGRESOS_OPERACIONALES','GANANCIA_PERDIDA',
        'TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO','ANO_DE_CORTE'
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas necesarias: {missing}")
        return pd.DataFrame()

    # LIMPIAR columnas numéricas
    numeric_cols = [
        'INGRESOS_OPERACIONALES','GANANCIA_PERDIDA',
        'TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO'
    ]

    for col in numeric_cols:
        df[col] = (
            df[col].astype(str)
            .str.replace("$","",regex=False)
            .str.replace(",","",regex=False)
            .str.replace(".","",regex=False)
            .str.replace(" ","",regex=False)
            .str.replace("−","-",regex=False)
            .str.replace("(","",regex=False)
            .str.replace(")","",regex=False)
            .astype(float)
        )
    
    # -----------------------------------------------------------------
    # FIX: CORRECCIÓN CRÍTICA PARA EL AÑO DE CORTE
    # -----------------------------------------------------------------
    # Convertir a numérico, los errores (texto) se vuelven NaN
    df['ANO_DE_CORTE'] = pd.to_numeric(df['ANO_DE_CORTE'], errors='coerce')
    # Rellenar NaNs con -1 y convertir a entero
    df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].fillna(-1).astype(int) 
    # Filtrar todas las filas con años no válidos INMEDIATAMENTE
    df = df[df['ANO_DE_CORTE'] > 2000].copy()
    # -----------------------------------------------------------------

    return df


# ----------------------------------------------------
# 3) CARGAR MODELO
# ----------------------------------------------------
@st.cache_resource
def load_model():
    model_file = "model.pkl" 
    
    if not os.path.exists(model_file):
        st.error(f"No se encontró el archivo del modelo: {model_file}")
        return None

    try:
        return joblib.load(model_file)
    except Exception as e:
        st.error(f"Error al cargar {model_file}: {e}")
        return None

# ----------------------------------------------------
# --- INICIO DE LA APLICACIÓN ---
# ----------------------------------------------------

st.title("📊 Dashboard ALECO 2025")

# --- Carga segura de datos y modelo ---
try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"Error fatal al iniciar la app: {e}")
    st.stop()

# Si los datos o el modelo no se cargaron, detener la app.
if df.empty or model is None:
    st.warning("La aplicación no puede iniciar. Revisa los archivos (CSV y model.pkl) y los errores anteriores.")
    st.stop()


# ----------------------------------------------------
# 4) DASHBOARD PRINCIPAL (FILTROS)
# ----------------------------------------------------
st.markdown("Explora las empresas y predice **GANANCIA_PERDIDA** usando el modelo XGBoost entrenado.")

col1, col2 = st.columns(2)
with col1:
    sector = st.selectbox("Filtrar por Macrosector", ["Todos"] + df["MACROSECTOR"].unique().tolist())
with col2:
    region = st.selectbox("Filtrar por Región", ["Todos"] + df["REGION"].unique().tolist())

df_filtrado = df.copy()
if sector != "Todos":
    df_filtrado = df_filtrado[df_filtrado["MACROSECTOR"] == sector]
if region != "Todos":
    df_filtrado = df_filtrado[df_filtrado["REGION"] == region]

st.subheader("Vista (primeras filas) del conjunto filtrado")
st.dataframe(df_filtrado.head(30))


# ----------------------------------------------------
# 5) KPIs SEGUROS
# ----------------------------------------------------
st.subheader("📊 KPIs agregados")

if df_filtrado.empty:
    st.warning("No hay datos para los filtros seleccionados.")
else:
    def safe_float(x):
        try:
            return float(x)
        except:
            return np.nan

    for col in ["INGRESOS_OPERACIONALES","TOTAL_ACTIVOS","TOTAL_PASIVOS","TOTAL_PATRIMONIO"]:
        df_filtrado[col] = df_filtrado[col].apply(safe_float)

    ingresos_total = df_filtrado["INGRESOS_OPERACIONALES"].sum()
    patrimonio_prom = df_filtrado["TOTAL_PATRIMONIO"].mean()

    col_kpi1, col_kpi2 = st.columns(2)
    with col_kpi1:
        st.metric(label="Ingresos Operacionales Totales", value=f"${ingresos_total:,.2f}")
    with col_kpi2:
        st.metric(label="Patrimonio Promedio", value=f"${patrimonio_prom:,.2f}")


# ----------------------------------------------------
# 6) PREDICCIÓN CON COMPARACIÓN (CON DOS SELECTORES)
# ----------------------------------------------------
st.subheader("🔮 Predicción de Ganancia/Pérdida")

if df_filtrado.empty:
    st.warning("No hay empresas con ese filtro para realizar predicciones.")
    st.stop()

# 1. Determinar el año base (¡Esto ahora SÍ funcionará gracias al FIX en load_data!)
ano_corte_mas_reciente = df_filtrado["ANO_DE_CORTE"].max()

# --- SELECTORES: Año y Empresa ---
col_sel_year, col_sel_company = st.columns(2)

with col_sel_year:
    # MEJORA: Lista desplegable de Años
    pred_years = [2026, 2027, 2028, 2029, 2030]
    # Solo mostrar años futuros al año más reciente
    años_futuros = [y for y in pred_years if y > ano_corte_mas_reciente]
    
    if not años_futuros:
        st.warning(f"El año de corte más reciente es {ano_corte_mas_reciente}. Ajusta la lista de años futuros.")
        st.stop()

    ano_prediccion = st.selectbox(
        "Selecciona el Año de Predicción",
        años_futuros,
        index=0 
    )

# 2. Filtrar empresas disponibles (basado en el año más reciente)
empresas_disponibles = df_filtrado[df_filtrado["ANO_DE_CORTE"] == ano_corte_mas_reciente]["RAZON_SOCIAL"].unique().tolist()

if not empresas_disponibles:
    st.warning(f"No hay datos de empresas disponibles para el año {ano_corte_mas_reciente} en este filtro.")
    st.stop()

with col_sel_company:
    # MEJORA: Lista desplegable de Empresa
    empresa_seleccionada = st.selectbox(
        "Selecciona la Empresa para predecir",
        empresas_disponibles
    )

st.info(f"Predicción para **{ano_prediccion}**, comparando contra el año de corte más reciente disponible: **{ano_corte_mas_reciente}**.")

# 3. Preparar datos para la predicción
try:
    # Obtener la fila BASE (del año más reciente)
    row_data = df_filtrado[
        (df_filtrado["RAZON_SOCIAL"] == empresa_seleccionada) &
        (df_filtrado["ANO_DE_CORTE"] == ano_corte_mas_reciente)
    ]
        
    row = row_data.iloc[[0]].copy()

    # Guardar ganancia anterior y preparar fila para predicción
    if "GANANCIA_PERDIDA" in row.columns:
        ganancia_anterior = row["GANANCIA_PERDIDA"].iloc[0] 
        row = row.drop(columns=["GANANCIA_PERDIDA"])
    else:
        ganancia_anterior = np.nan

    # Modificar la fila para el año futuro
    row["ANO_DE_CORTE"] = ano_prediccion

    # Ordenar columnas EXACTAMENTE como en el entrenamiento
    FEATURE_ORDER = [
        'NIT','RAZON_SOCIAL','SUPERVISOR','REGION','DEPARTAMENTO_DOMICILIO',
        'CIUDAD_DOMICILIO','CIIU','MACROSECTOR',
        'INGRESOS_OPERACIONALES','TOTAL_ACTIVOS','TOTAL_PASIVOS',
        'TOTAL_PATRIMONIO','ANO_DE_CORTE'
    ]
    row = row[FEATURE_ORDER]

    # Convertir a códigos categóricos/numéricos
    row_prediccion = row.copy()
    for col in row_prediccion.columns:
        if row_prediccion[col].dtype == 'object':
            row_prediccion[col] = row_prediccion[col].astype("category").cat.codes
        else:
            try:
                row_prediccion[col] = pd.to_numeric(row_prediccion[col], errors='raise')
            except:
                row_prediccion[col] = 0 # Fallback

    # 4. Realizar Predicción
    pred = model.predict(row_prediccion)[0]
    
    # 5. Calcular y mostrar la comparación
    if not pd.isna(ganancia_anterior):
        diferencia = pred - ganancia_anterior
    else:
        diferencia = 0

    st.markdown("#### Resultado de la Predicción")
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.metric(
            label=f"GANANCIA/PÉRDIDA Predicha ({ano_prediccion})",
            value=f"${pred:,.2f}",
            delta=f"${diferencia:,.2f} vs {ano_corte_mas_reciente}" if not pd.isna(ganancia_anterior) else "Sin datos para comparar"
        )
        
    with col_res2:
        st.metric(
            label=f"GANANCIA/PÉRDIDA Real ({ano_corte_mas_reciente})",
            value=f"${ganancia_anterior:,.2f}" if not pd.isna(ganancia_anterior) else "N/A",
            delta_color="off"
        )
        
    st.success(f"Predicción generada con éxito para **{empresa_seleccionada}**.")

except Exception as e:
    st.error(f"Error generando la predicción: {e}")
    st.caption("Esto puede ocurrir si la empresa seleccionada tiene datos incompatibles con el modelo.")
