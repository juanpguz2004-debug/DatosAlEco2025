import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import unicodedata
from sklearn.preprocessing import LabelEncoder 

# ----------------------------------------------------
# 0) CONFIGURACIÓN INICIAL Y CONSTANTES
# ----------------------------------------------------
st.set_page_config(
    page_title="📊 Dashboard ALECO", 
    layout="wide"
)

TARGET_COL = 'GANANCIA_PERDIDA'
OHE_COLS = ['SUPERVISOR', 'REGION', 'MACROSECTOR', 'ANO_DE_CORTE'] 
LE_COLS = ['DEPARTAMENTO_DOMICILIO', 'CIUDAD_DOMICILIO', 'CIIU'] 

# Función de formato de año (CRÍTICO: debe ser la misma que en el entrenamiento)
def format_ano(year):
    year_str = str(year)
    if len(year_str) == 4:
        return f'{year_str[0]},{year_str[1:]}' 
    return year_str

# Función de normalización de columna
def normalize_col(col):
    col = col.strip().upper().replace(" ", "_").replace("(", "").replace(")", "").replace("Ñ", "N")
    return ''.join(c for c in unicodedata.normalize('NFD', col) if unicodedata.category(c) != 'Mn')


# ----------------------------------------------------
# 1) CARGAR CSV Y LIMPIEZA
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

        numeric_cols = [TARGET_COL,'INGRESOS_OPERACIONALES','TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO']
        for col in numeric_cols:
            df[col] = (
                df[col].astype(str)
                .str.replace("$","",regex=False).str.replace(" ","",regex=False)
                .str.replace("−","-",regex=False).str.replace("(","",regex=False)
                .str.replace(")","",regex=False)
            )
            # FIX DE LIMPIEZA DE FORMATO: Asume formato de miles sin punto y decimal con coma
            df[col] = df[col].str.replace('.', '', regex=False) # Elimina puntos de miles
            df[col] = df[col].str.replace(',', '.', regex=False) # Reemplaza coma decimal por punto

            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'ANO_DE_CORTE' in df.columns:
            df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].astype(str).str.replace(",", "", regex=False)
            df['ANO_DE_CORTE'] = pd.to_numeric(df['ANO_DE_CORTE'], errors='coerce')
            df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].fillna(-1).astype(int)
        
        df = df[df['ANO_DE_CORTE'] > 2000].copy()
        df.dropna(subset=numeric_cols, inplace=True)
        
        return df

    except Exception as e:
        st.error(f"❌ ERROR al leer o limpiar el CSV: {e}")
        return pd.DataFrame()

# ----------------------------------------------------
# 2) CARGAR TRES MODELOS Y REFERENCIAS (CRÍTICO)
# ----------------------------------------------------
@st.cache_resource
def load_assets():
    # Nombres de los archivos
    cls_file = "model_clasificacion.pkl"
    reg_gan_file = "model_reg_ganancia.pkl"
    reg_per_file = "model_reg_perdida.pkl" 
    features_file = "model_features.pkl"
    encoders_file = "label_encoders.pkl"
    
    files_exist = (os.path.exists(cls_file) and os.path.exists(reg_gan_file) and 
                   os.path.exists(reg_per_file) and os.path.exists(features_file) and 
                   os.path.exists(encoders_file))

    if not files_exist:
        return None, None, None, None, None

    try:
        model_cls = joblib.load(cls_file)
        model_reg_gan = joblib.load(reg_gan_file)
        model_reg_per = joblib.load(reg_per_file)
        model_features = joblib.load(features_file)
        label_encoders = joblib.load(encoders_file)
        
        return model_cls, model_reg_gan, model_reg_per, model_features, label_encoders
    except Exception as e:
        st.error(f"❌ ERROR al cargar activos: {e}")
        return None, None, None, None, None


# ----------------------------------------------------
# --- INICIO DE LA APLICACIÓN ---
# ----------------------------------------------------

df = load_data()
model_cls, model_reg_gan, model_reg_per, MODEL_FEATURE_NAMES, label_encoders = load_assets()


if df.empty:
    st.error("❌ ERROR FATAL: No se encontraron datos válidos en el CSV.")
    st.stop()
    
if None in [model_cls, model_reg_gan, model_reg_per, MODEL_FEATURE_NAMES, label_encoders]:
    st.error("❌ ERROR FATAL: No se pudieron cargar los TRES modelos o las referencias. Verifica los archivos .pkl.")
    st.stop()

# --- Encabezado ---
st.title("📊 Dashboard ALECO: Modelo de Dos Partes")
st.markdown("""
**Predicción de Ganancia/Pérdida (incluyendo pérdidas reales) usando Modelado de Dos Partes.**
Todas las cifras se muestran en **Billones de Pesos**.
""")
st.markdown("---") 

ano_corte_mas_reciente_global = df["ANO_DE_CORTE"].max()

# ----------------------------------------------------
# 3) DASHBOARD PRINCIPAL Y FILTROS
# ----------------------------------------------------
st.header("1. Filtros y Datos")
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

if df_filtrado.empty:
    st.error(f"❌ ERROR: Los filtros eliminaron todos los datos válidos.")
    st.stop()

st.info(f"✅ Año de corte máximo global: **{ano_corte_mas_reciente_global}**")
st.dataframe(df_filtrado.head(5))

# ----------------------------------------------------
# 4) KPIs AGREGADOS
# ----------------------------------------------------
st.header("2. KPIs Agregados")

ingresos_total = df_filtrado["INGRESOS_OPERACIONALES"].sum()
patrimonio_prom = df_filtrado["TOTAL_PATRIMONIO"].mean()

col_kpi1, col_kpi2 = st.columns(2)
with col_kpi1:
    st.metric(label="Ingresos Operacionales Totales (Billones COP)", value=f"${ingresos_total:,.2f}")
with col_kpi2:
    st.metric(label="Patrimonio Promedio (Billones COP)", value=f"${patrimonio_prom:,.2f}")


# ----------------------------------------------------
# 5) PREDICCIÓN CON LÓGICA DE TRES MODELOS
# ----------------------------------------------------
st.header("3. Predicción de Ganancia/Pérdida")

# --- SELECTORES: Año y Empresa ---
col_sel_company, col_sel_year = st.columns(2) 
empresas_disponibles = df_filtrado["RAZON_SOCIAL"].unique().tolist()

if not empresas_disponibles:
    st.warning("No hay empresas disponibles después de aplicar los filtros.")
    st.stop()

with col_sel_company:
    empresa_seleccionada = st.selectbox(
        "Selecciona la Empresa para predecir", empresas_disponibles
    )

with col_sel_year:
    pred_years = [2026, 2027, 2028, 2029, 2030]
    años_futuros = [y for y in pred_years if y > ano_corte_mas_reciente_global]
    if not años_futuros:
        st.warning(f"El año de corte base es {ano_corte_mas_reciente_global}.")
        st.stop()
    ano_prediccion = st.selectbox(
        "Selecciona el Año de Predicción", años_futuros, index=0 
    )

# ... (Código anterior: LÓGICA DE PREDICCIÓN CONDICIONAL) ...

# --- 3. MOSTRAR RESULTADOS (Cálculo Delta Robusto y Formato) ---
diferencia = pred_real - ganancia_anterior

# --- CÁLCULO ROBUSTO DEL DELTA PORCENTUAL ---
delta_display = ""
delta_metric_value = diferencia # Usaremos la diferencia absoluta para el delta de la métrica

# 1. Base Cero (Anterior fue CERO)
if ganancia_anterior == 0:
    if pred_real > 0:
        delta_display = f"Ganó ${pred_real:,.2f} vs 0"
    elif pred_real < 0:
        delta_display = f"Perdió ${abs(pred_real):,.2f} vs 0"
    else:
        delta_display = "Sin cambio vs 0"

# 2. Base Negativa (Anterior fue una PÉRDIDA)
elif ganancia_anterior < 0:
    if pred_real >= 0:
        # Pasó de Negativo a Positivo/Cero
        delta_abs = pred_real - ganancia_anterior
        delta_display = f"Mejoró ${delta_abs:,.2f} (Cambio Absoluto)"
        
    else:
        # Siguió en Pérdida
        delta_percent_mag = (diferencia / abs(ganancia_anterior)) * 100
        
        if delta_percent_mag > 0:
            delta_display = f"Pérdida REDUCIDA {delta_percent_mag:,.2f}%"
        else:
            delta_display = f"Pérdida PROFUNDIZADA {abs(delta_percent_mag):,.2f}%"
            
# 3. Base Positiva (Anterior fue una GANANCIA)
else:
    # Caso de Ganancia a Ganancia (o a Pérdida)
    delta_percent = (diferencia / ganancia_anterior) * 100
    delta_display = f"{delta_percent:,.2f}% vs {ano_corte_empresa}"


st.markdown("#### Resultado de la Predicción")
col_res1, col_res2 = st.columns(2)

with col_res1:
    st.metric(
        label=f"GANANCIA/PÉRDIDA Predicha ({ano_prediccion}) (Billones COP)", 
        value=f"${pred_real:,.2f}",
        delta=delta_metric_value, 
        delta_color="normal"
    )
    # 🎯 FIX: Mostramos el texto formateado en el caption
    st.caption(f"Cambio vs {ano_corte_empresa}: **{delta_display}**") 

with col_res2:
    st.metric(
        label=f"G/P Real (Última fecha de corte registrada) (Billones COP)", 
        value=f"${ganancia_anterior:,.2f}",
        delta_color="off"
    )
    
# Mensaje condicional final (Reintroducimos la lógica de Ganancia Aumento/Reducción)
st.markdown("---") 

if pred_real >= 0.01: 
    # Predicción: GANANCIA
    if ganancia_anterior > 0 and diferencia >= 0:
        st.success(f"📈 El modelo clasifica la operación como **GANANCIA** y predice un **AUMENTO** en la magnitud de la ganancia (Resultado: ${pred_real:,.2f} Billones COP).")
    elif ganancia_anterior < 0:
        st.success(f"🚀 El modelo predice una **RECUPERACIÓN TOTAL** al pasar de pérdida a **GANANCIA** (Resultado: ${pred_real:,.2f} Billones COP).")
    elif ganancia_anterior == 0:
        st.success(f"📈 El modelo predice que la empresa pasa a **GANANCIA** desde equilibrio (Resultado: ${pred_real:,.2f} Billones COP).")
    else: # pred_real >= 0.01 and diferencia < 0 (Ganancia a Ganancia, pero menor)
        st.warning(f"⚠️ El modelo clasifica la operación como **GANANCIA**, pero predice una **REDUCCIÓN** en su magnitud (Resultado: ${pred_real:,.2f} Billones COP).")

elif pred_real < -0.01: 
    # Predicción: PÉRDIDA
    st.error(f"📉 El modelo clasifica la operación como **PÉRDIDA** neta (Resultado: **${abs(pred_real):,.2f} Billones COP**).")
else:
    # Predicción: CERO (Equilibrio)
    st.info("ℹ️ El modelo predice que el resultado será **cercano a cero** (equilibrio financiero).")

st.markdown("---")
st.markdown("Lo invitamos a participar en la **siguiente encuesta**.")


except Exception as e: 
    st.error(f"❌ ERROR generando la predicción: {e}")
    st.caption("Asegúrate de que la empresa seleccionada tiene datos completos y que los CINCO archivos .pkl son correctos.")

