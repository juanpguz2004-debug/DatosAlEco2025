import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import unicodedata
from sklearn.preprocessing import LabelEncoder # Necesario para replicar Label Encoding

# ----------------------------------------------------
# 0) CONFIGURACIÓN INICIAL
# ----------------------------------------------------
st.set_page_config(
    page_title="Dashboard ALECO", 
    layout="wide"
)

# ----------------------------------------------------
# 0.1) CONSTANTES DE ENTRENAMIENTO (CRÍTICO)
# ----------------------------------------------------
# Columnas usadas para el OHE en el entrenamiento
OHE_COLS = ['SUPERVISOR', 'REGION', 'MACROSECTOR', 'ANO_DE_CORTE'] 
# Columnas que quedaron con Label Encoding en el entrenamiento (alta cardinalidad)
LE_COLS = ['DEPARTAMENTO_DOMICILIO', 'CIUDAD_DOMICILIO', 'CIIU'] 
TARGET_COL = 'GANANCIA_PERDIDA'

# ----------------------------------------------------
# 1) FUNCIÓN DE NORMALIZACIÓN
# ----------------------------------------------------
def normalize_col(col):
    col = col.strip()
    col = col.upper()
    col = col.replace(" ", "_").replace("(", "").replace(")", "").replace("Ñ", "N")
    col = ''.join(
        c for c in unicodedata.normalize('NFD', col)
        if unicodedata.category(c) != 'Mn'
    )
    return col


# ----------------------------------------------------
# 2) CARGAR CSV Y LIMPIEZA (Corregida)
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
        
        # Validación de columnas
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"❌ ERROR: Faltan columnas necesarias: {missing}")
            return pd.DataFrame()

        # Limpieza de columnas numéricas
        numeric_cols = ['INGRESOS_OPERACIONALES',TARGET_COL,'TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO']
        for col in numeric_cols:
            df[col] = (
                df[col].astype(str)
                .str.replace("$","",regex=False).str.replace(" ","",regex=False)
                .str.replace("−","-",regex=False).str.replace("(","",regex=False)
                .str.replace(")","",regex=False)
            )
            # Asume: punto = separador de miles, coma = separador decimal.
            df[col] = df[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 🚨 CORRECCIÓN CRÍTICA: Eliminamos la división por 100 que no se usó en el modelo final.
        # df[TARGET_COL] = df[TARGET_COL].apply(lambda x: x / 100 if pd.notna(x) and x > 10 else x)

        # FIX FINAL PARA ANO_DE_CORTE
        if 'ANO_DE_CORTE' in df.columns:
            df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].astype(str).str.replace(",", "", regex=False)
            df['ANO_DE_CORTE'] = pd.to_numeric(df['ANO_DE_CORTE'], errors='coerce')
            df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].fillna(-1).astype(int)
        
        # Eliminar filas con años de corte inválidos o faltantes.
        df = df[df['ANO_DE_CORTE'] > 2000].copy()
        
        # Eliminar cualquier NaN que haya quedado en las columnas numéricas clave
        df.dropna(subset=numeric_cols, inplace=True)
        
        # 🚨 NUEVO: Pre-procesar el DF completo para obtener la estructura OHE y LE de referencia
        df_processed = df.copy()
        df_processed = df_processed.drop(columns=['NIT', 'RAZON_SOCIAL'], errors='ignore')
        
        # Aplicar Label Encoding al DF completo para obtener los códigos de referencia
        for col in LE_COLS:
            label = LabelEncoder()
            df_processed[col] = label.fit_transform(df_processed[col].astype(str))
            
        # Aplicar One-Hot Encoding al DF completo para obtener la estructura de columnas final
        df_processed = pd.get_dummies(df_processed, columns=OHE_COLS, prefix=OHE_COLS, drop_first=True, dtype=int)
        
        # Guardamos las columnas de referencia del modelo entrenado
        st.session_state['MODEL_COLS'] = df_processed.drop(columns=TARGET_COL, errors='ignore').columns.tolist()
        
        # Guardamos los LabelEncoders para su uso en la predicción
        st.session_state['LABEL_ENCODERS'] = {col: LabelEncoder().fit(df[col].astype(str)) for col in LE_COLS}


        return df

    except Exception as e:
        st.error(f"❌ ERROR al leer o limpiar el CSV: {e}")
        return pd.DataFrame()

# ----------------------------------------------------
# 3) CARGAR MODELO
# ----------------------------------------------------
@st.cache_resource
def load_model():
    # 🚨 CRÍTICO: Usamos el nombre del modelo corregido
    model_file = "model_corregido.pkl" 
    
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

# Inicializar sesión si es necesario
if 'MODEL_COLS' not in st.session_state:
    st.session_state['MODEL_COLS'] = []
if 'LABEL_ENCODERS' not in st.session_state:
    st.session_state['LABEL_ENCODERS'] = {}


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
st.markdown("---") 

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
    st.metric(label="Ingresos Operacionales Totales (Billones COP)", value=f"${ingresos_total:,.2f}")
with col_kpi2:
    st.metric(label="Patrimonio Promedio (Billones COP)", value=f"${patrimonio_prom:,.2f}")


# ----------------------------------------------------
# 6) PREDICCIÓN CON COMPARACIÓN (LÓGICA POR EMPRESA - CORREGIDA FINAL)
# ----------------------------------------------------
# ... [Código anterior de Selectores y Headers] ...

# 3. Preparar datos para la predicción
try:
    # LÓGICA CLAVE: Encontrar el año más reciente registrado para *ESTA EMPRESA*
    df_empresa = df_filtrado[df_filtrado["RAZON_SOCIAL"] == empresa_seleccionada]
    ano_corte_empresa = df_empresa["ANO_DE_CORTE"].max()
    
    # ... [Validación de año de corte] ...

    st.info(f"Predicción para **{ano_prediccion}**, comparando contra la última fecha de corte registrada de la empresa: **{ano_corte_empresa}**.")

    # Extraer la fila de datos
    row_data = df_empresa[
        df_empresa["ANO_DE_CORTE"] == ano_corte_empresa
    ].iloc[[0]].copy()

    ganancia_anterior = row_data[TARGET_COL].iloc[0]
    
    # -----------------------------------------------------------------
    # 🚨 PASO CRÍTICO: REPLICAR EL PRE-PROCESAMIENTO DEL ENTRENAMIENTO
    # -----------------------------------------------------------------
    
    row_prediccion = row_data.drop(columns=[TARGET_COL], errors='ignore').copy()
    
    # 1. Eliminar columnas de Leakage
    row_prediccion = row_prediccion.drop(columns=['NIT', 'RAZON_SOCIAL'], errors='ignore')
    
    # 2. Modificar el año de corte a predecir
    row_prediccion["ANO_DE_CORTE"] = ano_prediccion
    
    # 3. Aplicar Label Encoding (Usando los encoders de referencia)
    for col in LE_COLS:
        try:
            encoder = st.session_state['LABEL_ENCODERS'][col]
            # Aplica transform y CONVIERTE EXPLÍCITAMENTE A ENTERO (FIX CRÍTICO)
            row_prediccion[col] = encoder.transform(row_prediccion[col].astype(str))[0]
            row_prediccion[col] = int(row_prediccion[col]) # Asegura que sea un entero nativo
        except ValueError:
             # Valor no visto, se asigna 0 o -1
             row_prediccion[col] = 0 
    
    # 4. Aplicar One-Hot Encoding (OHE) - FIX DE FORMATO DE COLUMNA
    
    # 🚨 FIX CRÍTICO 1: Reintroducir la coma en el año de corte antes de OHE
    def format_ano(year):
        year_str = str(year)
        # Formato '2,02X'
        return f'{year_str[0]},{year_str[1:]}' 

    row_prediccion['ANO_DE_CORTE'] = row_prediccion['ANO_DE_CORTE'].apply(format_ano)

    # Ahora sí, aplicar OHE
    row_prediccion = pd.get_dummies(
        row_prediccion, 
        columns=OHE_COLS, 
        prefix=OHE_COLS, 
        drop_first=True, 
        dtype=int
    )
    
    # 5. Alinear las columnas (CRÍTICO para XGBoost)
    # Rellenar con 0 las columnas OHE que faltan
    missing_cols = set(st.session_state['MODEL_COLS']) - set(row_prediccion.columns)
    for c in missing_cols:
        row_prediccion[c] = 0 
    
    # Eliminar columnas extra y ordenar la fila de predicción según el modelo
    row_prediccion = row_prediccion[st.session_state['MODEL_COLS']].copy()
    
    # 🚨 FIX CRÍTICO 2: Convertir TODAS las columnas a tipo numérico antes de predecir
    # Esto asegura que no quede ningún 'object' residual.
    row_prediccion = row_prediccion.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # -----------------------------------------------------------------
    
    # 6. Realizar Predicción (Predice el log(1+x))
    pred_log = model.predict(row_prediccion)[0]
    
    # 7. Revertir la transformación logarítmica (e^x - 1)
    pred_real = np.expm1(pred_log)
    
    # 8. Mostrar la comparación
    diferencia = pred_real - ganancia_anterior
    
    # ... [Cálculo de delta y métricas] ...
    
    # ... [Mostrar resultados] ...

except Exception as e:
    st.error(f"❌ ERROR generando la predicción: {e}")
    st.caption("Asegúrate de que la empresa seleccionada tiene datos completos y que el modelo es compatible con la estructura de la fila.")
