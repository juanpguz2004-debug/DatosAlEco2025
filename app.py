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
COLS_TO_PROJECT = ['INGRESOS_OPERACIONALES','TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO']

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
            df[col] = df[col].str.replace('.', '', regex=False) 
            df[col] = df[col].str.replace(',', '.', regex=False) 

            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        if 'ANO_DE_CORTE' in df.columns:
            df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].astype(str).str.replace(",", "", regex=False)
            df['ANO_DE_CORTE'] = pd.to_numeric(df['ANO_DE_CORTE'], errors='coerce')
            df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].round(0).fillna(-1).astype(int)
            
        df = df[df['ANO_DE_CORTE'] > 2000].copy()
        df.dropna(subset=numeric_cols, inplace=True)
        
        return df

    except Exception as e:
        st.error(f"❌ ERROR al leer o limpiar el CSV: {e}")
        return pd.DataFrame()

# ----------------------------------------------------
# 2) CARGAR SIETE ACTIVOS DEL MODELO
# ----------------------------------------------------
@st.cache_resource
def load_assets():
    cls_file = "model_clasificacion.pkl"
    reg_gan_file = "model_reg_ganancia.pkl"
    reg_per_file = "model_reg_perdida.pkl" 
    features_file = "model_features.pkl"
    encoders_file = "label_encoders.pkl"
    growth_file = "growth_rate.pkl"      
    base_year_file = "base_year.pkl"     
    
    files_list = [cls_file, reg_gan_file, reg_per_file, features_file, encoders_file, growth_file, base_year_file]
    
    if not all(os.path.exists(f) for f in files_list):
        st.error(f"❌ Error: Faltan archivos .pkl. Asegúrate de que los SIETE archivos estén en la misma carpeta.")
        return None, None, None, None, None, None, None

    try:
        model_cls = joblib.load(cls_file)
        model_reg_gan = joblib.load(reg_gan_file)
        model_reg_per = joblib.load(reg_per_file)
        model_features = joblib.load(features_file)
        label_encoders = joblib.load(encoders_file)
        AGR = joblib.load(growth_file)              
        ANO_CORTE_BASE_GLOBAL = joblib.load(base_year_file) 
        
        return model_cls, model_reg_gan, model_reg_per, model_features, label_encoders, AGR, ANO_CORTE_BASE_GLOBAL
    except Exception as e:
        st.error(f"❌ ERROR al cargar activos: {e}")
        return None, None, None, None, None, None, None


# ----------------------------------------------------
# --- INICIO DE LA APLICACIÓN ---
# ----------------------------------------------------

df = load_data()
model_cls, model_reg_gan, model_reg_per, MODEL_FEATURE_NAMES, label_encoders, AGR, ANO_CORTE_BASE_GLOBAL = load_assets()


if df.empty:
    st.error("❌ ERROR FATAL: No se encontraron datos válidos en el CSV.")
    st.stop()
    
if None in [model_cls, model_reg_gan, model_reg_per, MODEL_FEATURE_NAMES, label_encoders, AGR, ANO_CORTE_BASE_GLOBAL]:
    st.error("❌ ERROR FATAL: No se pudieron cargar los SIETE activos del modelo.")
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
# 5) PREDICCIÓN CON LÓGICA DE TRES MODELOS Y PROYECCIÓN (FIXED RECURSIVO FINAL)
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
    ano_prediccion_final = st.selectbox(
        "Selecciona el Año de Predicción (Target)", años_futuros, index=0 
    )

# --- Lógica de Predicción ---
try:
    df_empresa = df_filtrado[df_filtrado["RAZON_SOCIAL"] == empresa_seleccionada]
    ano_corte_empresa = df_empresa["ANO_DE_CORTE"].max()
    
    if ano_corte_empresa <= 2000:
        st.error(f"Error: La empresa '{empresa_seleccionada}' no tiene un año de corte válido.")
        st.stop()

    st.info(f"Predicción recursiva hasta **{ano_prediccion_final}**, iniciando desde la última fecha de corte ({ano_corte_empresa}). Tasa de Crecimiento Asumida (AGR): **{AGR:.2f}**")

    # 1. PREPARACIÓN INICIAL
    row_data = df_empresa[df_empresa["ANO_DE_CORTE"] == ano_corte_empresa].iloc[[0]].copy()
    ganancia_anterior = row_data[TARGET_COL].iloc[0]
    ganancia_anterior = ganancia_anterior / 100.0 # Parche de escala
    
    # Inicializar la base para la recursividad
    # Aseguramos que los features a proyectar sean float desde el inicio.
    row_current_base = row_data.drop(columns=[TARGET_COL, 'NIT', 'RAZON_SOCIAL'], errors='ignore').iloc[0].copy() 
    for col in COLS_TO_PROJECT:
         row_current_base[col] = pd.to_numeric(row_current_base[col], errors='coerce').astype(float)
    
    
    # 2. BUCLE DE PREDICCIÓN RECURSIVA
    años_a_predecir = range(ano_corte_empresa + 1, ano_prediccion_final + 1)
    pred_real_final = 0.0 # Almacena el resultado del último paso
    
    for ano_actual in años_a_predecir:
        
        # A. Proyección de Features Numéricos (Recursiva: Valor Anterior * AGR)
        for col in COLS_TO_PROJECT:
            # Multiplicamos el valor del año anterior (almacenado en row_current_base) por AGR
            row_current_base[col] = row_current_base[col] * AGR
            
        # B. Codificación (LE y OHE) para el año actual
        row_prediccion = row_current_base.copy() # Copia de los features proyectados
        
        # CRÍTICO: Actualizar el feature del año para el OHE
        row_prediccion["ANO_DE_CORTE"] = ano_actual
        row_prediccion['ANO_DE_CORTE'] = format_ano(row_prediccion['ANO_DE_CORTE'])
        
        # Aplicar Label Encoding (LE)
        for col in LE_COLS:
            try:
                encoder = label_encoders[col]
                row_prediccion[col] = encoder.transform([str(row_prediccion[col])])[0]
            except ValueError:
                 row_prediccion[col] = -1 
        
        # Aplicar One-Hot Encoding (OHE) y Alineación
        row_prediccion_df = row_prediccion.to_frame().T
        for col in OHE_COLS:
            row_prediccion_df[col] = row_prediccion_df[col].astype(str)

        row_prediccion_ohe = pd.get_dummies(
            row_prediccion_df, columns=OHE_COLS, prefix=OHE_COLS, drop_first=True, dtype=int
        )
        
        # Alinear y ordenar las columnas (CRÍTICO)
        missing_cols = set(MODEL_FEATURE_NAMES) - set(row_prediccion_ohe.columns)
        for c in missing_cols:
            row_prediccion_ohe[c] = 0 
        
        # Preparación final del DataFrame de predicción para el modelo
        X_pred = row_prediccion_ohe[MODEL_FEATURE_NAMES].copy()
        X_pred = X_pred.astype(float).fillna(0)
        
        
        # C. Lógica de Predicción Condicional
        pred_cls = model_cls.predict(X_pred)[0]
        
        if pred_cls == 1:
            pred_log = model_reg_gan.predict(X_pred)[0]
            pred_g_p_actual = np.expm1(pred_log) 
        else:
            pred_log = model_reg_per.predict(X_pred)[0]
            magnitud_perdida_real = np.expm1(pred_log)
            pred_g_p_actual = -magnitud_perdida_real
            
        # D. ALMACENAMIENTO: El resultado final es el que se muestra al usuario
        if ano_actual == ano_prediccion_final:
            pred_real_final = pred_g_p_actual
            
        # Nota: row_current_base ya contiene los valores proyectados para el año actual y es la base para la siguiente iteración.
        
    
    # --- 3. MOSTRAR RESULTADOS (Usando pred_real_final) ---
    diferencia = pred_real_final - ganancia_anterior

    delta_metric_value = diferencia 

    # --- CÁLCULO ROBUSTO DEL DELTA PORCENTUAL ---
    if ganancia_anterior == 0:
        if pred_real_final > 0:
            delta_display = f"Ganó ${pred_real_final:,.2f} vs 0"
        elif pred_real_final < 0:
            delta_display = f"Perdió ${abs(pred_real_final):,.2f} vs 0"
        else:
            delta_display = "Sin cambio vs 0"
    elif ganancia_anterior < 0:
        if pred_real_final >= 0:
            delta_abs = pred_real_final - ganancia_anterior
            delta_display = f"Mejoró ${delta_abs:,.2f} (Cambio Absoluto)"
        else:
            delta_percent_mag = (diferencia / abs(ganancia_anterior)) * 100
            if diferencia > 0:
                delta_display = f"Pérdida REDUCIDA {abs(delta_percent_mag):,.2f}%"
            else:
                delta_display = f"Pérdida PROFUNDIZADA {abs(delta_percent_mag):,.2f}%"
    else:
        delta_percent = (diferencia / ganancia_anterior) * 100
        delta_display = f"{delta_percent:,.2f}% vs {ano_corte_empresa}"


    st.markdown("#### Resultado de la Predicción")
    col_res1, col_res2 = st.columns(2)

    with col_res1:
        st.metric(
            label=f"GANANCIA/PÉRDIDA Predicha ({ano_prediccion_final}) (Billones COP)", 
            value=f"${pred_real_final:,.2f}",
            delta=delta_metric_value, 
            delta_color="normal"
        )
        st.caption(f"Cambio vs {ano_corte_empresa}: **{delta_display}**") 

    with col_res2:
        st.metric(
            label=f"G/P Real (Última fecha de corte registrada) (Billones COP)", 
            value=f"${ganancia_anterior:,.2f}",
            delta_color="off"
        )
        
    # Mensaje condicional final (Lógica detallada)
    st.markdown("---") 

    if pred_real_final >= 0.01: 
        if ganancia_anterior > 0 and diferencia >= 0:
            st.success(f"📈 El modelo clasifica la operación como **GANANCIA** y predice un **AUMENTO** en la magnitud de la ganancia (Resultado: ${pred_real_final:,.2f} Billones COP).")
        elif ganancia_anterior < 0:
            st.success(f"🚀 El modelo predice una **RECUPERACIÓN TOTAL** al pasar de pérdida a **GANANCIA** (Resultado: ${pred_real_final:,.2f} Billones COP).")
        elif ganancia_anterior == 0:
            st.success(f"📈 El modelo predice que la empresa pasa a **GANANCIA** desde equilibrio (Resultado: ${pred_real_final:,.2f} Billones COP).")
        else: 
            st.warning(f"⚠️ El modelo clasifica la operación como **GANANCIA**, pero predice una **REDUCCIÓN** en su magnitud (Resultado: ${pred_real_final:,.2f} Billones COP).")

    elif pred_real_final < -0.01: 
        st.error(f"📉 El modelo clasifica la operación como **PÉRDIDA** neta (Resultado: **${abs(pred_real_final):,.2f} Billones COP**).")
    else:
        st.info("ℹ️ El modelo predice que el resultado será **cercano a cero** (equilibrio financiero).")

    st.markdown("---")
    st.markdown("Lo invitamos a participar en la **siguiente encuesta**.")


except Exception as e: 
    st.error(f"❌ ERROR generando la predicción: {e}")
    st.caption("Asegúrate de que la empresa seleccionada tiene datos completos y que todos los SIETE archivos .pkl son correctos.")
