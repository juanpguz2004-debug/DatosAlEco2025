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

    # ... (omitiendo la carga y verificación de columnas) ...

    try:
        df = pd.read_csv(csv_file)
        df.columns = [normalize_col(c) for c in df.columns]

        # ... (omitiendo la verificación de columnas requeridas) ...

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
        
        # 🟢 NUEVO FIX: Filtrar valores de G/P que son exageradamente altos o nulos.
        # Esto previene que datos mal formateados que resultaron en valores extremos
        # (ej. 1,493.00 en lugar de 33.41) afecten el resultado.
        # Definimos un umbral: si el 99% de los datos es menor a X, descartamos valores muy por encima.
        
        # Primero, rellenamos cualquier NaN en Ganancia/Pérdida con 0 para calcular el umbral
        gp_temp = df['GANANCIA_PERDIDA'].fillna(0)
        
        # Calculamos el percentil 99 para identificar outliers extremos
        # Si la mayoría de tus valores están en billones, este umbral debe ser ajustado.
        umbral_outlier = gp_temp.quantile(0.999) 
        
        if umbral_outlier > 0:
             # Solo filtramos valores que están MUY por encima de casi todos los demás
             df = df[df['GANANCIA_PERDIDA'] < (umbral_outlier * 2)].copy()
        
        # Finalmente, eliminamos cualquier NaN que pueda haber quedado en las columnas numéricas clave
        df.dropna(subset=numeric_cols, inplace=True)
        
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

# 🟢 CAMBIO 1: Encabezado y unidades
st.title("📊 Dashboard ALECO: Final")
st.markdown("""
**Reporte de las diez mil empresas más grandes del país.**
Todas las cifras en este reporte se muestran en **Billones de Pesos**.
""")
st.markdown("---") # Separador para el encabezado

# ----------------------------------------------------
# 4) DASHBOARD PRINCIPAL Y FILTROS
# ----------------------------------------------------

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
    # 🟢 Etiqueta KPI con unidades
    st.metric(label="Ingresos Operacionales Totales (Billones COP)", value=f"${ingresos_total:,.2f}")
with col_kpi2:
    # 🟢 Etiqueta KPI con unidades
    st.metric(label="Patrimonio Promedio (Billones COP)", value=f"${patrimonio_prom:,.2f}")


# ----------------------------------------------------
# 6) PREDICCIÓN CON COMPARACIÓN (LÓGICA POR EMPRESA)
# ----------------------------------------------------
st.header("3. Predicción de Ganancia/Pérdida")

# --- SELECTORES: Año y Empresa ---
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

# 1. Selector de año de predicción (utiliza el máximo global como base)
with col_sel_year:
    pred_years = [2026, 2027, 2028, 2029, 2030]
    # Filtramos la lista de años futuros respecto al max año que vimos en el dataset
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
    # 🟢 LÓGICA CLAVE: Encontrar el año más reciente registrado para *ESTA EMPRESA*
    df_empresa = df_filtrado[df_filtrado["RAZON_SOCIAL"] == empresa_seleccionada]
    ano_corte_empresa = df_empresa["ANO_DE_CORTE"].max()
    
    # Detener si no hay datos de corte para la empresa seleccionada (debería ser imposible)
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
    
    # 🟢 Extraer la fila de datos usando el año más reciente de la empresa
    row_data = df_empresa[
        df_empresa["ANO_DE_CORTE"] == ano_corte_empresa
    ].iloc[[0]].copy()

    # Guardar ganancia anterior y preparar fila para predicción
    ganancia_anterior = row_data["GANANCIA_PERDIDA"].iloc[0]
    row = row_data.drop(columns=["GANANCIA_PERDIDA"])

    # Modificar la fila para el año futuro
    row["ANO_DE_CORTE"] = ano_prediccion
    row = row[FEATURE_ORDER]

    # Convertir a códigos categóricos/numéricos (simulando el entrenamiento)
    row_prediccion = row.copy()
    for col in row_prediccion.columns:
        if row_prediccion[col].dtype == 'object':
            row_prediccion[col] = row_prediccion[col].astype("category").cat.codes
        else:
            row_prediccion[col] = pd.to_numeric(row_prediccion[col], errors='coerce').fillna(0) 

    # 4. Realizar Predicción
    pred = model.predict(row_prediccion)[0]
    
    # 5. Mostrar la comparación
    diferencia = pred - ganancia_anterior
    
    # 🟢 CAMBIO 2: Cálculo del porcentaje de cambio (delta_percent)
    delta_percent = 0.0
    if ganancia_anterior != 0:
        # Usamos abs(ganancia_anterior) para evitar divisiones por cero si es muy cercano, 
        # aunque si es 0 el resultado no es un porcentaje significativo.
        # El cálculo de cambio porcentual se basa en el valor real: (Predicción - Real) / Real
        delta_percent = (diferencia / ganancia_anterior) * 100
    
    # Formatear el delta como porcentaje
    delta_display = f"{delta_percent:,.2f}% vs {ano_corte_empresa}"


    st.markdown("#### Resultado de la Predicción")
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.metric(
            label=f"GANANCIA/PÉRDIDA Predicha ({ano_prediccion}) (Billones COP)", # 🟢 Unidades
            value=f"${pred:,.2f}",
            delta=delta_display # 🟢 Mostrar el porcentaje
        )
        
    with col_res2:
        st.metric(
            label=f"G/P Real (Última fecha de corte registrada) (Billones COP)", # 🟢 Unidades
            value=f"${ganancia_anterior:,.2f}",
            delta_color="off"
        )
        
    # 🟢 CAMBIO 3: Mensaje condicional más claro y enlace a encuesta
    st.markdown("---") 
    if pred >= 0:
        if diferencia >= 0:
            st.success(f"📈 Se predice un **aumento** de la ganancia del {delta_percent:,.2f}% respecto al año {ano_corte_empresa} (Ganancia total: ${pred:,.2f} Billones COP).")
        else:
            st.warning(f"⚠️ Se predice una **reducción** en la ganancia del {abs(delta_percent):,.2f}% respecto al año {ano_corte_empresa} (Ganancia total: ${pred:,.2f} Billones COP).")
    else:
        st.error(f"📉 Se predice una **pérdida** neta para {ano_prediccion} (Pérdida total: ${pred:,.2f} Billones COP).")

    # 🟢 CAMBIO 4: Invitación a la encuesta
    st.markdown("---")
    st.markdown("Lo invitamos a participar en la **siguiente encuesta**.")


except Exception as e:
    st.error(f"❌ ERROR generando la predicción: {e}")
    st.caption("Asegúrate de que la empresa seleccionada tiene datos completos y que el modelo es compatible con la estructura de la fila.")

