import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import unicodedata

# ----------------------------------------------------
# 0) CONFIGURACIÓN INICIAL
# ----------------------------------------------------
st.set_page_config(
    page_title="Dashboard ALECO", 
    layout="wide"
)

# ----------------------------------------------------
# 1) FUNCIÓN DE NORMALIZACIÓN
# ----------------------------------------------------
def normalize_col(col):
    col = col.strip()
    col = col.upper()
    col = col.replace(" ", "_")
    col = col.replace("(", "")
    col = col.replace(")", "")
    col = col.replace("Ñ", "N")
    col = ''.join(
        c for c in unicodedata.normalize('NFD', col)
        if unicodedata.category(c) != 'Mn'
    )
    return col


# ----------------------------------------------------
# 2) CARGAR CSV Y LIMPIEZA (CON LOS FIXES FINALES)
# ----------------------------------------------------
@st.cache_data
def load_data():
    csv_file = "10.000_Empresas_mas_Grandes_del_País_20251115.csv"

    if not os.path.exists(csv_file):
        st.error(f"❌ ERROR: Archivo CSV no encontrado: {csv_file}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_file)
        df.columns = [normalize_col(c) for c in df.columns]

        required_cols = [
            'NIT','RAZON_SOCIAL','SUPERVISOR','REGION','DEPARTAMENTO_DOMICILIO',
            'CIUDAD_DOMICILIO','CIIU','MACROSECTOR','INGRESOS_OPERACIONALES',
            'GANANCIA_PERDIDA','TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO','ANO_DE_CORTE'
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"❌ ERROR: Faltan columnas necesarias: {missing}")
            return pd.DataFrame()

        # Limpieza de columnas numéricas (Ingresos, Activos, etc.)
        numeric_cols = ['INGRESOS_OPERACIONALES','GANANCIA_PERDIDA','TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO']
        for col in numeric_cols:
            df[col] = (
                df[col].astype(str)
                .str.replace("$","",regex=False).str.replace(",","",regex=False)
                .str.replace(".","",regex=False).str.replace(" ","",regex=False)
                .str.replace("−","-",regex=False).str.replace("(","",regex=False)
                .str.replace(")","",regex=False).astype(float)
            )

        # 🟢 FIX FINAL PARA ANO_DE_CORTE (Eliminar la coma y convertir)
        if 'ANO_DE_CORTE' in df.columns:
            df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].astype(str).str.replace(",", "", regex=False)
            df['ANO_DE_CORTE'] = pd.to_numeric(df['ANO_DE_CORTE'], errors='coerce')
            df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].fillna(-1).astype(int)
        
        # 🟢 FIX CRÍTICO: Descartar filas con años de corte inválidos o faltantes.
        df = df[df['ANO_DE_CORTE'] > 2000].copy()
        
        return df

    except Exception as e:
        st.error(f"❌ ERROR al leer o limpiar el CSV: {e}")
        return pd.DataFrame()


# ----------------------------------------------------
# 3) CARGAR MODELO
# ----------------------------------------------------
@st.cache_resource
def load_model():
    model_file = "model.pkl" 
    
    if not os.path.exists(model_file):
        st.error(f"❌ ERROR: Archivo del modelo no encontrado: {model_file}")
        return None

    try:
        model = joblib.load(model_file)
        return model
    except Exception as e:
        st.error(f"❌ ERROR al cargar el modelo: {e}. Revisa las versiones de joblib/XGBoost.")
        return None


# ----------------------------------------------------
# --- INICIO DE LA APLICACIÓN ---
# ----------------------------------------------------

df = load_data()
model = load_model()

if df.empty:
    st.error("❌ ERROR FATAL: No se encontraron datos válidos (con año > 2000) en el CSV.")
    st.stop()
    
if model is None:
    st.error("❌ ERROR FATAL: El modelo no está cargado.")
    st.stop()


# ----------------------------------------------------
# 4) DASHBOARD PRINCIPAL Y FILTROS
# ----------------------------------------------------
st.title("📊 Dashboard ALECO: Final")

# Determinar el año máximo global (para establecer los límites de predicción)
ano_corte_mas_reciente_global = df["ANO_DE_CORTE"].max()

st.header("1. Filtros y Datos")
col1, col2 = st.columns(2)
with col1:
    sector = st.selectbox("Filtrar por Macrosector", ["Todos"] + df["MACROSECTOR"].unique().tolist())
with col2:
    region = st.selectbox("Filtrar por Región", ["Todos"] + df["REGION"].unique().tolist())

# Aplicar filtros
df_filtrado = df.copy()
if sector != "Todos":
    df_filtrado = df_filtrado[df_filtrado["MACROSECTOR"] == sector]
if region != "Todos":
    df_filtrado = df_filtrado[df_filtrado["REGION"] == region]

if df_filtrado.empty:
    st.error(f"❌ ERROR: Los filtros eliminaron todos los datos válidos.")
    st.stop()

st.info(f"✅ Año de corte máximo global: **{ano_corte_mas_reciente_global}**")
st.dataframe(df_filtrado.head(5))


# ----------------------------------------------------
# 5) KPIs AGREGADOS
# ----------------------------------------------------
st.header("2. KPIs Agregados")

ingresos_total = df_filtrado["INGRESOS_OPERACIONALES"].sum()
patrimonio_prom = df_filtrado["TOTAL_PATRIMONIO"].mean()

col_kpi1, col_kpi2 = st.columns(2)
with col_kpi1:
    st.metric(label="Ingresos Operacionales Totales", value=f"${ingresos_total:,.2f}")
with col_kpi2:
    st.metric(label="Patrimonio Promedio", value=f"${patrimonio_prom:,.2f}")
# ----------------------------------------------------
# 6) PREDICCIÓN CON COMPARACIÓN (LÓGICA FINAL Y ROBUSTA)
# ----------------------------------------------------
st.header("3. Predicción de Ganancia/Pérdida")

# 🟢 FIX: Aplicar la codificación categórica al DataFrame filtrado COMPLETO
# Esto genera el DataFrame que el modelo espera.
df_codificado = df_filtrado.copy()
for col in df_codificado.columns:
    if df_codificado[col].dtype == 'object':
        df_codificado[col] = df_codificado[col].astype("category").cat.codes


# --- SELECTORES: Empresa y Año ---
col_sel_company, col_sel_year = st.columns(2) 

# 2. Lista de empresas disponibles (TODAS las que pasaron el filtro)
empresas_disponibles = df_filtrado["RAZON_SOCIAL"].unique().tolist()

if not empresas_disponibles:
    st.warning("No hay empresas disponibles después de aplicar los filtros. Ajusta tus selecciones.")
    st.stop()

with col_sel_company:
    empresa_seleccionada = st.selectbox(
        "Selecciona la Empresa para predecir",
        empresas_disponibles
    )

# 1. Selector de año de predicción
with col_sel_year:
    pred_years = [2026, 2027, 2028, 2029, 2030]
    años_futuros = [y for y in pred_years if y > ano_corte_mas_reciente_global]
    
    if not años_futuros:
        st.warning(f"El año de corte base es {ano_corte_mas_reciente_global}. Ajusta la lista de años futuros en el código.")
        st.stop()

    ano_prediccion = st.selectbox(
        "Selecciona el Año de Predicción (2026 por defecto)",
        años_futuros,
        index=0 
    )


# 3. Preparar datos para la predicción
try:
    # 🟢 LÓGICA CLAVE: Usar el NIT como identificador
    df_empresa_original = df_filtrado[df_filtrado["RAZON_SOCIAL"] == empresa_seleccionada]
    
    # 1. Encontrar el NIT y el año de corte de la empresa seleccionada
    nit_empresa = df_empresa_original["NIT"].iloc[0]
    ano_corte_empresa = df_empresa_original["ANO_DE_CORTE"].max()
    
    if ano_corte_empresa <= 2000:
        st.error(f"Error: La empresa '{empresa_seleccionada}' no tiene un año de corte válido.")
        st.stop()

    st.info(f"Predicción para **{ano_prediccion}**, comparando contra la última fecha de corte registrada de la empresa: **{ano_corte_empresa}**.")

    FEATURE_ORDER = [
        'NIT','RAZON_SOCIAL','SUPERVISOR','REGION','DEPARTAMENTO_DOMICILIO',
        'CIUDAD_DOMICILIO','CIIU','MACROSECTOR',
        'INGRESOS_OPERACIONALES','TOTAL_ACTIVOS','TOTAL_PASIVOS',
        'TOTAL_PATRIMONIO','ANO_DE_CORTE'
    ]
    
    # 2. Extraer la fila de datos ya CODIFICADA usando el NIT y el año más reciente de la empresa
    row_data = df_codificado[
        (df_codificado["NIT"] == nit_empresa) &
        (df_codificado["ANO_DE_CORTE"] == ano_corte_empresa)
    ].iloc[[0]].copy()

    # 3. Guardar ganancia anterior (usando el DF original)
    ganancia_anterior = df_empresa_original[
        df_empresa_original["ANO_DE_CORTE"] == ano_corte_empresa
    ]["GANANCIA_PERDIDA"].iloc[0]

    # Preparamos la fila para la predicción, eliminando la G/P
    row_prediccion = row_data.drop(columns=["GANANCIA_PERDIDA"])

    # Modificar la fila para el año futuro
    row_prediccion["ANO_DE_CORTE"] = ano_prediccion
    row_prediccion = row_prediccion[FEATURE_ORDER]
    
    # 4. Realizar Predicción
    pred = model.predict(row_prediccion)[0]
    
    # 5. Mostrar la comparación
    diferencia = pred - ganancia_anterior

    st.markdown("#### Resultado de la Predicción")
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.metric(
            label=f"GANANCIA/PÉRDIDA Predicha ({ano_prediccion})",
            value=f"${pred:,.2f}",
            delta=f"${diferencia:,.2f} vs {ano_corte_empresa}" 
        )
        
    with col_res2:
        st.metric(
            label=f"G/P Real (Última fecha de corte registrada)", 
            value=f"${ganancia_anterior:,.2f}",
            delta_color="off"
        )
        
    # LÓGICA DE MENSAJE MEJORADA
    st.markdown("---")
    if pred >= 0:
        if diferencia >= 0:
            st.success(f"📈 Se predice un **aumento** de la ganancia (Ganancia total: ${pred:,.2f}).")
        else:
            st.warning(f"⚠️ Se predice una **reducción** en la ganancia respecto al año {ano_corte_empresa} (Ganancia total: ${pred:,.2f}).")
    else:
        st.error(f"📉 Se predice una **pérdida** neta para {ano_prediccion} (Pérdida total: ${pred:,.2f}).")

except Exception as e:
    st.error(f"❌ ERROR generando la predicción: {e}. Revisa la codificación y la alineación de las características.")

