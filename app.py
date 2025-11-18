import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import unicodedata
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple

# ----------------------------------------------------
# 0) CONFIGURACIÓN INICIAL Y CONSTANTES
# ----------------------------------------------------
st.set_page_config(
    page_title="📊 Dashboard ALECO",
    layout="wide"
)

# Nombres de columnas clave
TARGET_COL = 'GANANCIA_PERDIDA'
# OHE_COLS corregidas para excluir ANO_DE_CORTE
OHE_COLS = ['SUPERVISOR', 'REGION', 'MACROSECTOR']
LE_COLS = ['DEPARTAMENTO_DOMICILIO', 'CIUDAD_DOMICILIO', 'CIIU']
COLS_TO_PROJECT = ['INGRESOS_OPERACIONALES', 'TOTAL_ACTIVOS', 'TOTAL_PASIVOS', 'TOTAL_PATRIMONIO']

# Función de normalización de columna
def normalize_col(col: str) -> str:
    col = col.strip().upper().replace(" ", "_").replace("(", "").replace(")", "").replace("Ñ", "N")
    return ''.join(c for c in unicodedata.normalize('NFD', col) if unicodedata.category(c) != 'Mn')


# Función de Label Encoding segura
def safe_le_transform(encoder: LabelEncoder, value: str) -> int:
    """Aplica la transformación de LabelEncoder, devolviendo -1 para valores no vistos."""
    try:
        # Aseguramos que el valor de entrada se maneje como string
        return int(encoder.transform([str(value)])[0])
    except ValueError:
        # Valor no visto o NaN. Devuelve el valor por defecto (-1)
        return -1 


# ----------------------------------------------------
# 1) CARGAR DATOS Y LIMPIEZA
# ----------------------------------------------------
@st.cache_data
def load_data():
    """Carga el CSV y aplica la limpieza y filtrado inicial."""
    # **IMPORTANTE:** Cambia "nombre_de_tu_archivo.csv" por el nombre de tu archivo cargado
    csv_file = "10.000_Empresas_mas_Grandes_del_País_20251115.csv"
    if not os.path.exists(csv_file):
        st.error(f"❌ ERROR: Archivo CSV no encontrado: {csv_file}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_file)
        df.columns = [normalize_col(c) for c in df.columns]

        numeric_cols = COLS_TO_PROJECT + [TARGET_COL]
        for col in numeric_cols:
            df[col] = (
                df[col].astype(str)
                .str.replace("$","",regex=False).str.replace(" ","",regex=False)
                .str.replace("−","-",regex=False).str.replace("(","",regex=False)
                .str.replace(")","",regex=False)
            )
            # Elimina puntos de miles y reemplaza coma decimal por punto (Asunción de formato COP)
            df[col] = df[col].str.replace('.', '', regex=False) 
            df[col] = df[col].str.replace(',', '.', regex=False)

            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Manejo del año de corte (Numérico, como se corrigió)
        if 'ANO_DE_CORTE' in df.columns:
            df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].astype(str).str.replace(",", "", regex=False)
            df['ANO_DE_CORTE'] = pd.to_numeric(df['ANO_DE_CORTE'], errors='coerce')
            df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].fillna(-1).astype(int)
        
        # Filtros y limpieza final
        df = df[df['ANO_DE_CORTE'] > 2000].copy()
        df.dropna(subset=numeric_cols, inplace=True)
        
        return df

    except Exception as e:
        st.error(f"❌ ERROR al leer o limpiar el CSV: {e}")
        return pd.DataFrame()


# ----------------------------------------------------
# 2) CARGAR TRES MODELOS Y REFERENCIAS
# ----------------------------------------------------
@st.cache_resource
def load_assets():
    """Carga los modelos, la lista de features, los encoders y la tasa de crecimiento (AGR)."""
    assets = {}
    files_to_load = [
        "model_clasificacion.pkl", "model_reg_ganancia.pkl", "model_reg_perdida.pkl",
        "model_features.pkl", "label_encoders.pkl", "growth_rate.pkl"
    ]
    
    if not all(os.path.exists(f) for f in files_to_load):
        st.error("❌ ERROR FATAL: Faltan archivos .pkl. Asegúrate de tener los 6 archivos.")
        return None, None, None, None, None, None
    
    try:
        assets['cls'] = joblib.load(files_to_load[0])
        assets['reg_gan'] = joblib.load(files_to_load[1])
        assets['reg_per'] = joblib.load(files_to_load[2])
        assets['features'] = joblib.load(files_to_load[3])
        assets['encoders'] = joblib.load(files_to_load[4])
        assets['agr'] = joblib.load(files_to_load[5]) # growth_rate.pkl
        
        return (assets['cls'], assets['reg_gan'], assets['reg_per'], 
                assets['features'], assets['encoders'], assets['agr'])
    except Exception as e:
        st.error(f"❌ ERROR al cargar activos: {e}")
        return None, None, None, None, None, None


# ----------------------------------------------------
# 3) FUNCIÓN DE FORECASTING RECURSIVO (LA LÓGICA CLAVE)
# ----------------------------------------------------

def run_forecasting(
    initial_row: pd.DataFrame, 
    target_year: int, 
    agr: float,
    model_cls, model_reg_gan, model_reg_per,
    model_feature_names: List[str], label_encoders: Dict[str, LabelEncoder]
) -> pd.DataFrame:
    """
    Realiza la predicción recursiva aplicando el AGR a las variables financieras
    y ejecutando el Hurdle Model para cada año hasta target_year.
    """
    # Usamos la primera fila como base
    current_data = initial_row.iloc[0].copy()
    start_year = current_data['ANO_DE_CORTE']
    
    # DataFrame para almacenar el resultado final
    df_forecast = pd.DataFrame()
    
    # Itera desde el año base + 1 hasta el año objetivo
    for year in range(start_year + 1, target_year + 1):
        
        # --- Paso 1: Proyección (Recursiva) ---
        # Aplica AGR a las variables financieras proyectables
        for col in COLS_TO_PROJECT:
            # Multiplica el valor del año anterior (current_data) por AGR
            current_data[col] *= agr
            
        # Actualiza el año
        current_data['ANO_DE_CORTE'] = year
        
        # --- Paso 2: Preprocesamiento para el Modelo ---
        # Crea la fila de predicción (temporal)
        row_prediccion = pd.DataFrame([current_data.to_dict()])
        
        # Copia de la fila para trabajar
        X_pred_temp = row_prediccion.copy()
        
        # Aplicar Label Encoding (con manejo seguro de valores no vistos)
        for col in LE_COLS:
            encoder = label_encoders[col]
            # Usar safe_le_transform, aplicado a una columna entera simulada con .apply()
            X_pred_temp[col] = X_pred_temp[col].apply(lambda x: safe_le_transform(encoder, x))

        # Aplicar One-Hot Encoding (OHE)
        X_pred_temp = pd.get_dummies(
            X_pred_temp, columns=OHE_COLS, prefix=OHE_COLS, drop_first=True, dtype=int
        )
        
        # Alinear y ordenar las columnas (CRÍTICO)
        missing_cols = set(model_feature_names) - set(X_pred_temp.columns)
        for c in missing_cols:
            X_pred_temp[c] = 0 
        
        X_pred = X_pred_temp[model_feature_names].copy()
        X_pred = X_pred.apply(pd.to_numeric, errors='coerce').fillna(0) # Limpieza final

        # --- Paso 3: Predicción con Hurdle Model ---
        
        # 3A: Clasificar (0 = Pérdida/Cero, 1 = Ganancia)
        pred_cls = model_cls.predict(X_pred)[0]
        
        pred_real = 0.0
        
        if pred_cls == 1:
            # 3B-Ganancia: Usar Regresión de Ganancias
            pred_log = model_reg_gan.predict(X_pred)[0]
            pred_real = np.expm1(pred_log) # Reversión: e^x - 1
        else:
            # 3B-Pérdida: Usar Regresión de Pérdidas
            pred_log = model_reg_per.predict(X_pred)[0]
            magnitud_perdida_real = np.expm1(pred_log)
            pred_real = -magnitud_perdida_real # CRÍTICO: Convertir a valor negativo

        # --- Paso 4: Almacenar Resultado y Preparar para el Siguiente Bucle ---
        
        # Almacena G/P en el DataFrame de resultados
        row_prediccion['GANANCIA_PERDIDA_PRED'] = pred_real 
        row_prediccion['ANO_DE_CORTE'] = year
        
        df_forecast = pd.concat([df_forecast, row_prediccion], ignore_index=True)
        
        # La G/P predicha no es un input para la siguiente proyección, 
        # solo los COLS_TO_PROJECT son inputs
        
    return df_forecast


# ----------------------------------------------------
# --- INICIO DE LA APLICACIÓN ---
# ----------------------------------------------------

# Cargar activos
df = load_data()
(model_cls, model_reg_gan, model_reg_per, 
 MODEL_FEATURE_NAMES, label_encoders, AGR) = load_assets()

# Validaciones de carga
if df.empty:
    st.error("❌ ERROR FATAL: No se encontraron datos válidos en el CSV.")
    st.stop()
    
if None in [model_cls, model_reg_gan, model_reg_per, MODEL_FEATURE_NAMES, label_encoders, AGR]:
    st.error("❌ ERROR FATAL: Faltan activos (modelos, features, encoders, AGR). Verifica los archivos .pkl.")
    st.stop()

# --- Encabezado ---
st.title("📊 Dashboard ALECO: Modelo de Dos Partes")
st.markdown(f"""
**Predicción de Ganancia/Pérdida (incluyendo pérdidas reales) usando Modelado de Dos Partes.**
Todas las cifras se muestran en **Billones de Pesos**. Tasa de Crecimiento Anual (AGR) aplicada: **{AGR*100-100:.1f}%**
""")
st.markdown("---")

ano_corte_mas_reciente_global = df["ANO_DE_CORTE"].max()

# ----------------------------------------------------
# 4) DASHBOARD PRINCIPAL Y FILTROS
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

# Aseguramos que solo trabajamos con el año más reciente de datos reales en el DF filtrado
df_filtrado_latest = df_filtrado[df_filtrado["ANO_DE_CORTE"] == df_filtrado["ANO_DE_CORTE"].max()].copy()


if df_filtrado_latest.empty:
    st.error("❌ ERROR: Los filtros eliminaron todos los datos válidos.")
    st.stop()

st.info(f"✅ Año de corte máximo global: **{ano_corte_mas_reciente_global}**. Empresas cargadas: **{len(df_filtrado_latest)}**")
st.dataframe(df_filtrado_latest[["RAZON_SOCIAL", "ANO_DE_CORTE", "INGRESOS_OPERACIONALES", TARGET_COL]].head())

# ----------------------------------------------------
# 5) PREDICCIÓN CON LÓGICA DE FORECASTING
# ----------------------------------------------------
st.header("2. Proyección de Ganancia/Pérdida")

# --- SELECTORES: Año y Empresa ---
col_sel_company, col_sel_year = st.columns(2)

# Usamos empresas disponibles en el año más reciente filtrado
empresas_disponibles = df_filtrado_latest["RAZON_SOCIAL"].unique().tolist() 

if not empresas_disponibles:
    st.warning("No hay empresas disponibles después de aplicar los filtros.")
    st.stop()

with col_sel_company:
    empresa_seleccionada = st.selectbox(
        "Selecciona la Empresa para proyectar", empresas_disponibles
    )

with col_sel_year:
    max_forecast_year = 2030
    años_futuros = list(range(ano_corte_mas_reciente_global + 1, max_forecast_year + 1))
    
    if not años_futuros:
        st.warning(f"El año de corte base ({ano_corte_mas_reciente_global}) ya es el máximo permitido para proyectar.")
        st.stop()
    
    ano_prediccion = st.slider(
        "Selecciona el Año Final de Proyección", 
        min_value=años_futuros[0], 
        max_value=años_futuros[-1], 
        value=min(2028, años_futuros[-1])
    )

# --- Lógica de Proyección ---
try:
    # Fila base de la empresa (último dato real)
    row_data_base = df_filtrado_latest[
        df_filtrado_latest["RAZON_SOCIAL"] == empresa_seleccionada
    ].iloc[[0]].copy()
    
    ano_corte_empresa = row_data_base["ANO_DE_CORTE"].iloc[0]

    # Ejecutar la función de forecasting recursivo
    df_prediccion_forecast = run_forecasting(
        initial_row=row_data_base, 
        target_year=ano_prediccion, 
        agr=AGR,
        model_cls=model_cls, model_reg_gan=model_reg_gan, model_reg_per=model_reg_per,
        model_feature_names=MODEL_FEATURE_NAMES, label_encoders=label_encoders
    )

    # --- 3. MOSTRAR RESULTADOS (Tabla y Gráfico) ---
    
    st.markdown("#### Detalle de la Proyección Anual")
    
    # Prepara los datos para la tabla y gráfico
    df_resultados = df_prediccion_forecast[['ANO_DE_CORTE', 'GANANCIA_PERDIDA_PRED', 'INGRESOS_OPERACIONALES']].copy()
    
    # Obtener el último dato real para comparación
    last_real_data = row_data_base[['ANO_DE_CORTE', TARGET_COL]].rename(
        columns={TARGET_COL: 'GANANCIA_PERDIDA_PRED'}
    )
    
    # Combina datos reales y proyectados
    df_plot_data = pd.concat([last_real_data, df_resultados[['ANO_DE_CORTE', 'GANANCIA_PERDIDA_PRED']]], ignore_index=True)
    df_plot_data['Tipo'] = df_plot_data['ANO_DE_CORTE'].apply(lambda y: 'Real' if y == ano_corte_empresa else 'Proyectado')

    # KPI final (Año de Predicción)
    pred_real = df_resultados['GANANCIA_PERDIDA_PRED'].iloc[-1]
    ganancia_anterior = last_real_data['GANANCIA_PERDIDA_PRED'].iloc[0]
    
    diferencia = pred_real - ganancia_anterior

    # Lógica simplificada de Delta para el KPI (mejorado)
    delta_display = f"vs {ano_corte_empresa}"
    
    st.metric(
        label=f"GANANCIA/PÉRDIDA Predicha Final ({ano_prediccion}) (Billones COP)",
        value=f"${pred_real:,.2f}",
        delta=diferencia,
        delta_color="normal"
    )
    st.caption(f"Cambio absoluto en la G/P: ${diferencia:,.2f} {delta_display}")


    # Gráfico de Tendencia
    st.line_chart(
        df_plot_data.set_index('ANO_DE_CORTE'),
        y='GANANCIA_PERDIDA_PRED',
        color='Tipo',
        use_container_width=True
    )
    

    # Mensaje condicional (basado en el último año proyectado)
    if pred_real >= 0.01:
        st.success(f"📈 La proyección final indica una **GANANCIA** de ${pred_real:,.2f} Billones COP en {ano_prediccion}.")
    elif pred_real < -0.01:
        st.error(f"📉 La proyección final indica una **PÉRDIDA** de ${abs(pred_real):,.2f} Billones COP en {ano_prediccion}.")
    else:
        st.info("ℹ️ La proyección final indica un resultado **cercano a cero** (equilibrio financiero).")

    st.markdown("---")
    st.markdown("Lo invitamos a participar en la **siguiente encuesta**.")


except Exception as e:
    st.error(f"❌ ERROR generando la proyección: {e}")
    st.caption(f"Detalle del error: {e}")
# ----------------------------------------------------
