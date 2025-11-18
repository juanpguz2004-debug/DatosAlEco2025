import streamlit as st
import pandas as pd
import numpy as np
import joblib
import unicodedata
import warnings
import cloudpickle # Necesario para cargar encoders si fueron guardados con cloudpickle

# Suprimir advertencias de Streamlit 
warnings.filterwarnings("ignore")

# --- 0) CONFIGURACIÓN INICIAL Y CONSTANTES (Sincronizadas con el Notebook) ---

# Configuración de página
st.set_page_config(layout="wide", page_title="Dashboard ALECO: Modelo de Dos Partes")
st.title("📊 Dashboard ALECO: Modelo de Dos Partes")
st.markdown("Predicción de Ganancia/Pérdida (incluyendo pérdidas reales) usando Modelado de Dos Partes. Todas las cifras se muestran en **Billones de Pesos**.")
st.markdown("---")

# Nombres de las columnas clave (deben coincidir con el entrenamiento)
TARGET_COL = 'GANANCIA_PERDIDA'
COLS_TO_PROJECT = [
    'INGRESOS_OPERACIONALES', 'TOTAL_ACTIVOS', 
    'TOTAL_PASIVOS', 'TOTAL_PATRIMONIO'
]
OHE_COLS = ['SUPERVISOR', 'REGION', 'MACROSECTOR']
LE_COLS = ['DEPARTAMENTO_DOMICILIO', 'CIUDAD_DOMICILIO', 'CIIU']
AGR = 1.05 # Tasa de crecimiento asumida (de tu notebook)

# Función de normalización
def normalize_col(col):
    col = col.strip().upper().replace(" ", "_").replace("(", "").replace(")", "").replace("Ñ", "N")
    return ''.join(c for c in unicodedata.normalize('NFD', col) if unicodedata.category(c) != 'Mn')

# Función de Label Encoder segura (Para manejar valores no vistos)
def safe_le_transform(encoder, val):
    """Transforma un valor usando LabelEncoder, devolviendo -1 si es desconocido."""
    s = str(val)
    if s in encoder.classes_:
        # Usamos .transform([s])[0] porque la función .transform espera una lista/array de valores
        return encoder.transform([s])[0] 
    return -1

# --- 1) CARGA DE DATOS Y ACTIVOS (LIMPIEZA SINCRONIZADA) ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = [normalize_col(c) for c in df.columns]
        
        numeric_cols = COLS_TO_PROJECT + [TARGET_COL]
        
        # Lógica de limpieza sincronizada con el Notebook (Celda <KMjfoiIVDQRe>):
        for col in numeric_cols:
             s = df[col].astype(str).str.strip()
             
             # Eliminamos $, coma (miles/decimal), espacio y convertimos el signo menos especial (−) a guión (-)
             s = (
                 s.str.replace('$', '', regex=False)
                 .str.replace(',', '', regex=False) # CRÍTICO: Remueve la coma (miles o decimal), sincronizado
                 .str.replace('(', '', regex=False).str.replace(')', '', regex=False)
                 .str.replace(' ', '', regex=False).str.replace('−', '-', regex=False)
             )
             
             # Conversión final a numérico. Coerce errors to NaN.
             df[col] = pd.to_numeric(s, errors='coerce').astype(float)


        # Limpieza de ANO_DE_CORTE
        if 'ANO_DE_CORTE' in df.columns:
            df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].astype(str).str.replace(",", "", regex=False)
            df['ANO_DE_CORTE'] = pd.to_numeric(df['ANO_DE_CORTE'], errors='coerce')
            df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].fillna(-1).astype(int)
        
        # Filtrar años inválidos y llenar NaNs de numéricos con 0
        df = df[df['ANO_DE_CORTE'] > 2000].copy()
        df[numeric_cols] = df[numeric_cols].fillna(0.0)

        return df
    
    except Exception as e:
        st.error(f"Error cargando o limpiando el archivo CSV: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_assets():
    try:
        # Carga de los 7 activos (pkl)
        models = {
            'cls': joblib.load("model_clasificacion.pkl"),
            'reg_gan': joblib.load("model_reg_ganancia.pkl"),
            'reg_per': joblib.load("model_reg_perdida.pkl"),
        }
        assets = {
            'model_features': joblib.load("model_features.pkl"),
            'AGR': joblib.load("growth_rate.pkl"),
            'base_year': joblib.load("base_year.pkl"),
        }
        
        # Lógica para cargar Label Encoders (compatible con joblib y cloudpickle)
        try:
            # 1. Intentar cargar con joblib (método original)
            assets['label_encoders'] = joblib.load("label_encoders.pkl")
        except:
            # 2. Intentar cargar con cloudpickle (método usado en tu celda final)
            with open("label_encoders.pkl", "rb") as f:
                 assets['label_encoders'] = cloudpickle.load(f)
        
        return models, assets
    except FileNotFoundError as e:
        st.error(f"Error: No se encontró el archivo de activo '{e.filename}'. Asegúrate de que los SIETE archivos .pkl estén en el mismo directorio que app.py y el CSV.")
        return None, None
    except Exception as e:
        st.error(f"Error cargando los activos (pkls): {e}")
        return None, None


# Cargar el dataset y los activos
# Asegúrate que el CSV tenga el nombre '10.000_Empresas_mas_Grandes_del_País_20251115.csv'
df = load_data("10.000_Empresas_mas_Grandes_del_País_20251115.csv") 
models, assets = load_assets()

if df.empty or not models:
    st.stop()

# Desempaquetar activos
model_cls, model_reg_gan, model_reg_per = models['cls'], models['reg_gan'], models['reg_per']
label_encoders = assets['label_encoders']
MODEL_FEATURE_NAMES = assets['model_features']
AGR = assets['AGR']
ANO_CORTE_BASE_GLOBAL = assets['base_year']

# --- 2) LÓGICA DE FILTROS Y KPIS ---
st.header("1. Filtros y Datos")

# Obtener macrosectores y regiones únicas
macrosectores = ["Todos"] + df['MACROSECTOR'].dropna().unique().tolist()
regiones = ["Todos"] + df['REGION'].dropna().unique().tolist()

col_m, col_r, col_c = st.columns([1, 1, 0.5])

with col_m:
    filtro_macrosector = st.selectbox("Filtrar por Macrosector", macrosectores)

with col_r:
    filtro_region = st.selectbox("Filtrar por Región", regiones)

with col_c:
    ano_corte_mas_reciente_global = df['ANO_DE_CORTE'].max()
    st.markdown(f"✅ Año de corte máximo global: **{ano_corte_mas_reciente_global}**")

# Aplicar filtros
df_filtrado = df.copy()
if filtro_macrosector != "Todos":
    df_filtrado = df_filtrado[df_filtrado['MACROSECTOR'] == filtro_macrosector]
if filtro_region != "Todos":
    df_filtrado = df_filtrado[df_filtrado['REGION'] == filtro_region]

# KPIs
st.header("2. KPIs Agregados")
kpis_df = df_filtrado[df_filtrado['ANO_DE_CORTE'] == ano_corte_mas_reciente_global]

# Conversión a billones COP (1 Billón = 1,000,000,000)
def format_billones(value):
    return f"${value / 1e9:,.2f}"

total_ingresos = kpis_df['INGRESOS_OPERACIONALES'].sum()
promedio_patrimonio = kpis_df['TOTAL_PATRIMONIO'].mean()

col_kpi1, col_kpi2 = st.columns(2)

with col_kpi1:
    st.metric(
        label="Ingresos Operacionales Totales (Billones COP)",
        value=format_billones(total_ingresos)
    )

with col_kpi2:
    st.metric(
        label="Patrimonio Promedio (Billones COP)",
        value=format_billones(promedio_patrimonio)
    )

st.markdown("---")

# ----------------------------------------------------
# FUNCIÓN DE PREDICCIÓN RECURSIVA (Lógica de Inferencia)
# ----------------------------------------------------
def predict_recursive(row_base, ano_corte_empresa, ano_prediccion_final, 
                      model_cls, model_reg_gan, model_reg_per, 
                      label_encoders, MODEL_FEATURE_NAMES, AGR, 
                      COLS_TO_PROJECT, LE_COLS, OHE_COLS):
    
    row_current_base = row_base.copy() 
    
    años_a_predecir = range(ano_corte_empresa + 1, ano_prediccion_final + 1)
    pred_real_final = 0.0 
    
    for ano_actual in años_a_predecir:
        
        # A. Proyección de Features Numéricos
        for col in COLS_TO_PROJECT:
            row_current_base[col] = row_current_base[col] * AGR
            
        # B. Creación de la fila de predicción y codificación
        row_prediccion = row_current_base.to_frame().T.copy() 
        
        # 1. Seteamos el año 
        row_prediccion["ANO_DE_CORTE"] = ano_actual 
        
        # 2. Aplicar Label Encoding (LE) usando función segura
        for col in LE_COLS:
            val_to_encode = row_prediccion[col].iloc[0]
            # Usamos la función segura que maneja valores no vistos
            row_prediccion[col] = safe_le_transform(label_encoders[col], val_to_encode)
        
        # 3. Aplicar One-Hot Encoding (OHE) a las categóricas
        ohe_cols_to_use = [c for c in OHE_COLS]
        
        for col in ohe_cols_to_use:
            row_prediccion[col] = row_prediccion[col].astype(str)

        row_prediccion_ohe = pd.get_dummies(
            row_prediccion, 
            columns=ohe_cols_to_use, prefix=ohe_cols_to_use, drop_first=True, dtype=int
        )
        
        # C. ALINEACIÓN FINAL DE X_PRED
        
        # 1. Crear DataFrame base con ceros
        X_pred = pd.DataFrame(0, index=[0], columns=MODEL_FEATURE_NAMES)
        
        # 2. Inyectar OHE/calculados con reindexación robusta
        common = [c for c in row_prediccion_ohe.columns if c in X_pred.columns]
        X_pred.loc[0, common] = row_prediccion_ohe.loc[0, common].values
        
        # 3. Inyectar numéricos/LE/Año (variables continuas)
        cols_to_inject = COLS_TO_PROJECT + LE_COLS + ['ANO_DE_CORTE'] 

        for col in cols_to_inject:
            if col in X_pred.columns and col in row_prediccion.columns:
                val = row_prediccion[col].iloc[0]
                try:
                    val = float(val)
                except ValueError:
                    val = 0.0 
                X_pred.at[0, col] = val
        
        # 4. Asegurar tipos finales
        # CORRECCIÓN DE BUG: fillna se aplica sobre el DataFrame, no sobre un escalar.
        X_pred = X_pred.astype(float).fillna(0) 
        
        
        # D. Lógica de Predicción Condicional
        
        if X_pred.shape[1] == 0:
            raise ValueError("El DataFrame de predicción (X_pred) está vacío.")

        # Predicción de Clasificación
        pred_cls = model_cls.predict(X_pred)[0]
        
        # Predicción de Regresión (Ganancia o Pérdida)
        if pred_cls == 1:
            pred_log = model_reg_gan.predict(X_pred)[0]
            pred_g_p_actual = np.expm1(pred_log) # Inversa de np.log1p
        else:
            pred_log = model_reg_per.predict(X_pred)[0]
            magnitud_perdida_real = np.expm1(pred_log)
            pred_g_p_actual = -magnitud_perdida_real
            
        # E. ALMACENAMIENTO (Solo el resultado final)
        if ano_actual == ano_prediccion_final:
            pred_real_final = pred_g_p_actual
            
    return pred_real_final

# ----------------------------------------------------
# SECCIÓN 5: EJECUCIÓN DEL DASHBOARD Y RESULTADOS
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
    
    if ano_corte_mas_reciente_global:
        años_futuros = [y for y in pred_years if y > ano_corte_mas_reciente_global]
    else:
        años_futuros = pred_years 
        
    if not años_futuros:
        st.warning(f"El año de corte base es {ano_corte_mas_reciente_global}. No hay años futuros disponibles.")
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

    # Obtener la fila base y la ganancia anterior
    row_data = df_empresa[df_empresa["ANO_DE_CORTE"] == ano_corte_empresa].iloc[[0]].copy()
    ganancia_anterior = row_data[TARGET_COL].iloc[0]
    
    row_base = row_data.drop(columns=[TARGET_COL, 'NIT', 'RAZON_SOCIAL'], errors='ignore').iloc[0]


    # LLAMADA A LA FUNCIÓN RECURSIVA
    pred_real_final = predict_recursive(
        row_base, ano_corte_empresa, ano_prediccion_final, 
        model_cls, model_reg_gan, model_reg_per, 
        label_encoders, MODEL_FEATURE_NAMES, AGR, 
        COLS_TO_PROJECT, LE_COLS, OHE_COLS
    )
    
    # --- 3. MOSTRAR RESULTADOS (Usando pred_real_final) ---
    diferencia = pred_real_final - ganancia_anterior

    delta_metric_value = diferencia 

    # --- CÁLCULO ROBUSTO DEL DELTA PORCENTUAL ---
    if ganancia_anterior == 0 or np.isnan(ganancia_anterior) or abs(ganancia_anterior) < 0.0001:
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
        elif ganancia_anterior == 0 or np.isnan(ganancia_anterior):
            st.success(f"📈 El modelo predice que la empresa pasa a **GANANCIA** desde equilibrio (Resultado: ${pred_real_final:,.2f} Billones COP).")
        else: 
            st.warning(f"⚠️ El modelo clasifica la operación como **GANANCIA**, pero predice una **REDUCCIÓN** en su magnitud (Resultado: ${pred_real_final:,.2f} Billones COP).")

    elif pred_real_final < -0.01: 
        st.error(f"📉 El modelo clasifica la operación como **PÉRDIDA** neta (Resultado: **${abs(pred_real_final):,.2f} Billones COP**).")
    else:
        st.info("ℹ️ El modelo predice que el resultado será **cercano a cero** (equilibrio financiero).")

    st.markdown("---")


except Exception as e: 
    st.error(f"❌ ERROR generando la predicción: {e}")
    st.caption(f"Detalle del error: {e}. Asegúrate de que los 7 archivos .pkl y el CSV estén correctamente cargados y sean consistentes.")
