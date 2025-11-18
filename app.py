import streamlit as st
import pandas as pd
import numpy as np
import joblib
import unicodedata
import warnings
import cloudpickle 
import os 
from sklearn.preprocessing import LabelEncoder # Necesario si se usa en load_assets o si se necesita instanciar, aunque solo se usará para tipos.

# Suprimir advertencias de Streamlit 
warnings.filterwarnings("ignore")

# --- 0) CONFIGURACIÓN INICIAL Y CONSTANTES ---
st.set_page_config(layout="wide", page_title="Dashboard ALECO: Modelo de Dos Partes")

TARGET_COL = 'GANANCIA_PERDIDA'
COLS_TO_PROJECT = [
    'INGRESOS_OPERACIONALES', 'TOTAL_ACTIVOS', 
    'TOTAL_PASIVOS', 'TOTAL_PATRIMONIO'
]
# CORRECCIÓN CRÍTICA: Se añade de nuevo 'ANO_DE_CORTE' a las columnas OHE
OHE_COLS = ['SUPERVISOR', 'REGION', 'MACROSECTOR', 'ANO_DE_CORTE'] 
LE_COLS = ['DEPARTAMENTO_DOMICILIO', 'CIUDAD_DOMICILIO', 'CIIU']
AGR = 1.05 # Tasa de crecimiento asumida (se sobrescribe si existe 'growth_rate.pkl')

# Definición de los nombres de archivo con prioridad
FILE_PROCESSED = "dataset_limpio_para_streamlit.csv" 
FILE_RAW = "10.000_Empresas_mas_Grandes_del_País_20251115.csv"

# FUNCIÓN CRÍTICA: Debe ser exactamente la misma que en el entrenamiento
def format_ano(year):
    year_str = str(year)
    year_str = str(int(float(year_str))) # Asegurar que es un entero antes de formatear
    if len(year_str) == 4:
        return f'{year_str[0]},{year_str[1:]}'  
    return year_str

def normalize_col(col):
    col = col.strip().upper().replace(" ", "_").replace("(", "").replace(")", "").replace("Ñ", "N")
    return ''.join(c for c in unicodedata.normalize('NFD', col) if unicodedata.category(c) != 'Mn')

def safe_le_transform(encoder, val):
    s = str(val)
    if pd.isna(s) or s.lower() == 'nan':
        s = "nan"
    # El valor debe ser una lista para .transform()
    if s in encoder.classes_:
        return encoder.transform([s])[0]
    
    # Si la clase no está en el encoder, devolver -1 (o el valor que el modelo use para 'desconocido')
    return -1


# --- 1) CARGA DE DATOS Y ACTIVOS (LIMPIEZA Y ESCALADO REFORZADO) ---
@st.cache_data
def load_data(file_processed, file_raw):
    
    numeric_cols = COLS_TO_PROJECT + [TARGET_COL]
    df = pd.DataFrame()
    
    # 1. INTENTAR CARGAR EL ARCHIVO PROCESADO (PRIORIDAD AL LIMPIO)
    if os.path.exists(file_processed):
        try:
            df = pd.read_csv(file_processed)
            df.columns = [normalize_col(c) for c in df.columns]
            
            # Forzar tipo flotante y manejar NaNs
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float).fillna(0.0)
            
            #st.success(f"Datos cargados exitosamente desde **{file_processed}**.")

        except Exception as e:
            st.warning(f"Error al cargar {file_processed} ({e}). Intentando con el archivo RAW...")
            df = pd.DataFrame() 
            
    # 2. CARGAR EL ARCHIVO RAW CON LIMPIEZA REFORZADA SI EL PROCESADO FALLÓ
    if df.empty and os.path.exists(file_raw):
        try:
            df = pd.read_csv(file_raw)
            df.columns = [normalize_col(c) for c in df.columns]
            
            # Lógica de limpieza sincronizada y REFORZADA para columnas numéricas
            for col in numeric_cols:
                 if col not in df.columns: continue

                 s = df[col].astype(str).str.strip()
                 s = (
                     s.str.replace('$', '', regex=False)
                     .str.replace('(', '', regex=False).str.replace(')', '', regex=False)
                     .str.replace(' ', '', regex=False).str.replace('−', '-', regex=False)
                 )
                 
                 # Lógica para manejar el formato de números
                 def clean_number_string(x):
                    if pd.isna(x): return x
                    x = str(x)
                    if x.count('.') > 1 and x.count(',') == 0:
                        # Asume que el punto es separador de miles y no decimal
                        return x.replace('.', '') 
                    elif x.count('.') == 1 and x.count(',') == 1:
                        # Asume punto de miles, coma decimal
                        return x.replace('.', '').replace(',', '.')
                    elif x.count(',') == 1:
                        # Asume coma decimal
                        return x.replace(',', '.')
                    return x.replace('.', '') # Asume punto de miles

                 s_clean = s.apply(clean_number_string)
                 df[col] = pd.to_numeric(s_clean, errors='coerce').astype(float)
            
            #st.success(f"Datos cargados exitosamente desde **{file_raw}** con limpieza reforzada.")

        except Exception as e:
            st.error(f"Error final cargando o limpiando el archivo CSV: {e}")
            return pd.DataFrame()

    elif df.empty:
        st.error(f"Error: No se encontró ninguno de los archivos de datos requeridos: **{file_processed}** o **{file_raw}**.")
        return pd.DataFrame()

    # --- 3. PASOS FINALES DE LIMPIEZA Y ESCALADO (APLICAR EN AMBOS CASOS) ---
    
    # Limpieza de ANO_DE_CORTE
    if 'ANO_DE_CORTE' in df.columns:
        df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].astype(str).str.replace(",", "", regex=False)
        df['ANO_DE_CORTE'] = pd.to_numeric(df['ANO_DE_CORTE'], errors='coerce')
        df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].fillna(-1).astype(int)
    
    # Filtrar años inválidos y llenar NaNs de numéricos con 0
    df = df[df['ANO_DE_CORTE'] > 2000].copy()
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    # *** ESCALADO CRUCIAL: Convertir a Billones de COP (dividir por 1e9) ***
    # Se asume que los modelos fueron entrenados con data escalada a Billones COP.
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col] / 1e9 

    return df

@st.cache_resource
def load_assets():
    try:
        models = {
            'cls': joblib.load("model_clasificacion.pkl"),
            'reg_gan': joblib.load("model_reg_ganancia.pkl"),
            'reg_per': joblib.load("model_reg_perdida.pkl"),
        }
        
        # Carga de archivos de referencia, AGR y base_year
        model_features = joblib.load("model_features.pkl")
        AGR_loaded = joblib.load("growth_rate.pkl")
        base_year_loaded = joblib.load("base_year.pkl")
        
        # Carga robusta de Label Encoders (cloudpickle vs joblib)
        try:
            label_encoders = joblib.load("label_encoders.pkl")
        except:
            with open("label_encoders.pkl", "rb") as f:
                 label_encoders = cloudpickle.load(f)

        assets = {
            'model_features': model_features,
            'AGR': AGR_loaded,
            'base_year': base_year_loaded,
            'label_encoders': label_encoders
        }
        
        # Verificar que todos los encoders sean instancias válidas
        for col in LE_COLS:
            if col not in assets['label_encoders']:
                 st.error(f"Error: Falta el Label Encoder para la columna '{col}' en 'label_encoders.pkl'.")
                 return None, None
            # Intento de corrección si el encoder se cargó como un objeto no deseado
            # if not hasattr(assets['label_encoders'][col], 'transform'):
            #     st.error(f"Error: El objeto cargado para '{col}' no parece ser un LabelEncoder válido.")
            #     return None, None
            
        return models, assets
    except FileNotFoundError as e:
        st.error(f"Error: No se encontró el archivo de activo '{e.filename}'. Asegúrate de que los archivos PKL requeridos estén presentes.")
        return None, None
    except Exception as e:
        st.error(f"Error cargando los activos (pkls): {e}")
        return None, None


# Cargar el dataset y los activos
df = load_data(FILE_PROCESSED, FILE_RAW) 
models, assets = load_assets()

if df.empty or not models:
    st.stop()

# Desempaquetar activos y manejar el caso donde AGR no se cargó
model_cls, model_reg_gan, model_reg_per = models['cls'], models['reg_gan'], models['reg_per']
label_encoders = assets['label_encoders']
MODEL_FEATURE_NAMES = assets['model_features']
AGR = assets.get('AGR', 1.05) # Usar 1.05 como fallback
ANO_CORTE_BASE_GLOBAL = assets.get('base_year', 2024)


# --- 2) DISPLAY INICIAL Y FILTROS ---
st.title("📊 Dashboard ALECO: Modelo de Dos Partes")
st.markdown("Predicción de Ganancia/Pérdida (incluyendo pérdidas reales) usando Modelado de Dos Partes. Todas las cifras se muestran en **Billones de Pesos**.")
st.markdown("---")

st.header("1. Filtros y Datos")

macrosectores = ["Todos"] + df['MACROSECTOR'].dropna().unique().tolist()
regiones = ["Todos"] + df['REGION'].dropna().unique().tolist()

col_m, col_r, col_c = st.columns([1, 1, 0.5])

with col_m:
    filtro_macrosector = st.selectbox("Filtrar por Macrosector", macrosectores)

with col_r:
    filtro_region = st.selectbox("Filtrar por Región", regiones)

ano_corte_mas_reciente_global = df['ANO_DE_CORTE'].max()
with col_c:
    st.markdown(f"✅ Año de corte máximo global: **{ano_corte_mas_reciente_global}**")

df_filtrado = df.copy()
if filtro_macrosector != "Todos":
    df_filtrado = df_filtrado[df_filtrado['MACROSECTOR'] == filtro_macrosector]
if filtro_region != "Todos":
    df_filtrado = df_filtrado[df_filtrado['REGION'] == filtro_region]
    
# Verificar si hay datos después del filtro
if df_filtrado.empty:
    st.warning("No hay datos disponibles para los filtros seleccionados.")
    st.stop()

# KPIs
st.header("2. KPIs Agregados")
kpis_df = df_filtrado[df_filtrado['ANO_DE_CORTE'] == ano_corte_mas_reciente_global]

def format_billones(value):
    if pd.isna(value) or value is None:
        return "$0.00"
    return f"${value:,.2f}"

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
st.dataframe(df_filtrado.head(5))

# ----------------------------------------------------
# FUNCIÓN DE PREDICCIÓN RECURSIVA (CORREGIDA)
# ----------------------------------------------------
def predict_recursive(row_base, ano_corte_empresa, ano_prediccion_final, 
                      model_cls, model_reg_gan, model_reg_per, 
                      label_encoders, MODEL_FEATURE_NAMES, AGR, 
                      COLS_TO_PROJECT, LE_COLS, OHE_COLS_ALL):
    
    row_current_base = row_base.copy() 
    
    años_a_predecir = range(ano_corte_empresa + 1, ano_prediccion_final + 1)
    pred_real_final = 0.0 
    
    for ano_actual in años_a_predecir:
        
        # A. Proyección de Features Numéricos (ya están en Billones COP)
        for col in COLS_TO_PROJECT:
            # Asegurar que la proyección se haga sobre el valor numérico, no sobre el Series
            row_current_base[col] = row_current_base[col].iloc[0] * AGR
            
        # B. Creación de la fila de predicción y codificación
        # Convertir la serie base (con un solo elemento) a un DataFrame de 1 fila
        row_prediccion = row_current_base.to_frame().T.copy() 
        row_prediccion["ANO_DE_CORTE"] = ano_actual 
        
        # 1. Aplicar Label Encoding (LE) usando función segura
        for col in LE_COLS:
            val_to_encode = row_prediccion[col].iloc[0]
            # La salida de safe_le_transform es un int/float
            row_prediccion[col] = safe_le_transform(label_encoders[col], val_to_encode)
        
        # CRITICAL FIX: Formato de Año para OHE (Reintroducido)
        # Esto genera la clave OHE esperada, ej. 'ANO_DE_CORTE_2,026'
        row_prediccion['ANO_DE_CORTE'] = row_prediccion['ANO_DE_CORTE'].apply(format_ano)
        
        # 2. Aplicar One-Hot Encoding (OHE)
        for col in OHE_COLS_ALL:
            # Asegurar que las columnas OHE son tipo string ANTES de get_dummies
            row_prediccion[col] = row_prediccion[col].astype(str)

        row_prediccion_ohe = pd.get_dummies(
            row_prediccion, 
            columns=OHE_COLS_ALL, prefix=OHE_COLS_ALL, drop_first=True, dtype=int
        )
        
        # C. ALINEACIÓN FINAL DE X_PRED: Usar el método robusto de alineación
        
        # 1. Llenar columnas OHE faltantes con 0 (columnas que existen en el modelo pero no en esta fila)
        missing_cols = set(MODEL_FEATURE_NAMES) - set(row_prediccion_ohe.columns)
        for c in missing_cols:
            row_prediccion_ohe[c] = 0  
        
        # 2. Seleccionar solo las columnas que el modelo espera, en el orden correcto (CRÍTICO)
        X_pred = row_prediccion_ohe[MODEL_FEATURE_NAMES].copy()
        
        # 3. Conversión final a numérico
        X_pred = X_pred.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # D. Lógica de Predicción Condicional
        
        # Asegurarse de que las filas numéricas tengan un tipo de dato consistente
        X_pred = X_pred.astype(float) 

        pred_cls = model_cls.predict(X_pred)[0]
        
        if pred_cls == 1:
            pred_log = model_reg_gan.predict(X_pred)[0]
            pred_g_p_actual = np.expm1(pred_log) 
        else:
            pred_log = model_reg_per.predict(X_pred)[0]
            magnitud_perdida_real = np.expm1(pred_log) 
            pred_g_p_actual = -magnitud_perdida_real
            
        # E. Actualizar la base para la siguiente iteración (si es necesario)
        if ano_actual < ano_prediccion_final:
            # Reemplazamos la ganancia/pérdida proyectada para la siguiente iteración
            row_current_base[TARGET_COL] = pred_g_p_actual
            # Las otras columnas ya se proyectaron con AGR
        
        # F. ALMACENAMIENTO (Solo el resultado final)
        if ano_actual == ano_prediccion_final:
            pred_real_final = pred_g_p_actual
            
    return pred_real_final


# ----------------------------------------------------
# SECCIÓN 5: EJECUCIÓN DEL DASHBOARD Y RESULTADOS
# ----------------------------------------------------

st.header("3. Predicción de Ganancia/Pérdida")

col_sel_company, col_sel_year = st.columns(2) 
empresas_disponibles = df_filtrado["RAZON_SOCIAL"].unique().tolist()

if not empresas_disponibles:
    st.warning("No hay empresas disponibles después de aplicar los filtros.")
    st.stop()

with col_sel_company:
    default_index = 0
    if "ECOPETROL S.A" in empresas_disponibles:
        default_index = empresas_disponibles.index("ECOPETROL S.A")
        
    empresa_seleccionada = st.selectbox(
        "Selecciona la Empresa para predecir", empresas_disponibles, index=default_index
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
        
    default_year_index = años_futuros.index(2026) if 2026 in años_futuros else 0
    
    ano_prediccion_final = st.selectbox(
        "Selecciona el Año de Predicción (Target)", años_futuros, index=default_year_index 
    )

try:
    df_empresa = df_filtrado[df_filtrado["RAZON_SOCIAL"] == empresa_seleccionada]
    ano_corte_empresa = df_empresa["ANO_DE_CORTE"].max()
    
    if ano_corte_empresa <= 2000:
        st.error(f"Error: La empresa '{empresa_seleccionada}' no tiene un año de corte válido.")
        st.stop()

    st.info(f"Predicción recursiva hasta **{ano_prediccion_final}**, iniciando desde la última fecha de corte ({ano_corte_empresa}). Tasa de Crecimiento Asumida (AGR): **{AGR:.2f}**")

    # Obtener los datos para el año de corte
    df_data_year = df_empresa[df_empresa["ANO_DE_CORTE"] == ano_corte_empresa].copy()

    if df_data_year.empty:
        st.warning(f"Advertencia: No se encontraron datos para {empresa_seleccionada} en el año {ano_corte_empresa}. Imposible predecir.")
        st.stop()
        
    # CORRECCIÓN CLAVE: Forzar el reset de índice para evitar el error '0' al usar iloc[0]
    row_data = df_data_year.reset_index(drop=True).iloc[[0]].copy() 
    
    # Valores en Billones COP
    ganancia_anterior = row_data[TARGET_COL].iloc[0]
    
    # Preparamos la base de datos para la proyección: 
    # Mantiene columnas numéricas, LE y OHE (antes de OHE/LE)
    row_base = row_data.drop(columns=['NIT', 'RAZON_SOCIAL', 'ANO_DE_CORTE', TARGET_COL], errors='ignore').iloc[0]

    # Convertir a una serie con la etiqueta del índice correcto para el bucle
    row_base = row_base.to_frame(name=0).T 

    # Asegurar que las columnas a proyectar sean numéricas y las categóricas sean strings
    for col in COLS_TO_PROJECT:
         row_base[col] = pd.to_numeric(row_base[col], errors='coerce').fillna(0.0)

    for col in LE_COLS + [c for c in OHE_COLS if c != 'ANO_DE_CORTE']:
         row_base[col] = row_base[col].astype(str)
    
    # Se añade la ganancia anterior para que el bucle pueda proyectarla si es necesario
    row_base[TARGET_COL] = ganancia_anterior

    pred_real_final = predict_recursive(
        row_base, ano_corte_empresa, ano_prediccion_final, 
        model_cls, model_reg_gan, model_reg_per, 
        label_encoders, MODEL_FEATURE_NAMES, AGR, 
        COLS_TO_PROJECT, LE_COLS, OHE_COLS
    )
    
    # --- 3. MOSTRAR RESULTADOS ---
    diferencia = pred_real_final - ganancia_anterior
    delta_metric_value = diferencia 

    # Manejo de casos de delta (mejora vs pérdida/ganancia anterior)
    if ganancia_anterior == 0 or np.isnan(ganancia_anterior) or abs(ganancia_anterior) < 0.0001:
        if pred_real_final > 0:
            delta_display = f"Ganó {format_billones(pred_real_final)} vs 0"
        elif pred_real_final < 0:
            delta_display = f"Perdió {format_billones(abs(pred_real_final))} vs 0"
        else:
            delta_display = "Sin cambio vs 0"
    elif ganancia_anterior < 0:
        if pred_real_final >= 0:
            delta_abs = pred_real_final - ganancia_anterior
            delta_display = f"Mejoró {format_billones(delta_abs)} (Cambio Absoluto)"
        else:
            delta_percent_mag = (diferencia / abs(ganancia_anterior)) * 100
            if diferencia > 0:
                delta_display = f"Pérdida REDUCIDA {abs(delta_percent_mag):,.2f}%"
            else:
                delta_display = f"Pérdida PROFUNDIZADA {abs(delta_percent_mag):,.2f}%"
    else: # ganancia_anterior > 0
        delta_percent = (diferencia / ganancia_anterior) * 100
        delta_display = f"{delta_percent:,.2f}% vs {ano_corte_empresa}"


    st.markdown("#### Resultado de la Predicción")
    col_res1, col_res2 = st.columns(2)

    with col_res1:
        st.metric(
            label=f"GANANCIA/PÉRDIDA Predicha ({ano_prediccion_final}) (Billones COP)", 
            value=format_billones(pred_real_final), 
            delta=delta_metric_value, 
            delta_color="normal"
        )
        st.caption(f"Cambio vs {ano_corte_empresa}: **{delta_display}**") 

    with col_res2:
        st.metric(
            label=f"G/P Real (Última fecha de corte registrada) (Billones COP)", 
            value=format_billones(ganancia_anterior),
            delta_color="off"
        )
        
    st.markdown("---") 

    if pred_real_final >= 0.01: 
        if ganancia_anterior > 0 and diferencia >= 0:
            st.success(f"📈 El modelo clasifica la operación como **GANANCIA** y predice un **AUMENTO** en la magnitud de la ganancia (Resultado: {format_billones(pred_real_final)} Billones COP).")
        elif ganancia_anterior < 0:
            st.success(f"🚀 El modelo predice una **RECUPERACIÓN TOTAL** al pasar de pérdida a **GANANCIA** (Resultado: {format_billones(pred_real_final)} Billones COP).")
        elif ganancia_anterior == 0 or np.isnan(ganancia_anterior):
            st.success(f"📈 El modelo predice que la empresa pasa a **GANANCIA** desde equilibrio (Resultado: {format_billones(pred_real_final)} Billones COP).")
        else: 
            st.warning(f"⚠️ El modelo clasifica la operación como **GANANCIA**, pero predice una **REDUCCIÓN** en su magnitud (Resultado: {format_billones(pred_real_final)} Billones COP).")

    elif pred_real_final < -0.01: 
        st.error(f"📉 El modelo clasifica la operación como **PÉRDIDA** neta (Resultado: **{format_billones(abs(pred_real_final))} Billones COP**).")
    else:
        st.info("ℹ️ El modelo predice que el resultado será **cercano a cero** (equilibrio financiero).")

    st.markdown("---")


except Exception as e: 
    st.error(f"❌ ERROR generando la predicción: {e}")
    st.caption(f"Detalle del error: {e}. **VERIFICAR:** La causa más probable de fallas persistentes es que la estructura de los archivos `.pkl` no coincide con la esperada (especialmente 'model_features.pkl' y 'label_encoders.pkl').")
