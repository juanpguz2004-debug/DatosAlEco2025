import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io

# --- 0. CONFIGURACIÓN INICIAL ---
st.set_page_config(layout="wide", page_title="Dashboard ALECO: Modelo de Dos Partes")

# --- CONSTANTES ---
# Nombres de columnas clave (deben coincidir con el entrenamiento en Colab)
TARGET_COL = 'GANANCIA_PERDIDA'
OHE_COLS = ['SUPERVISOR', 'REGION', 'MACROSECTOR', 'ANO_DE_CORTE'] 
LE_COLS = ['DEPARTAMENTO_DOMICILIO', 'CIUDAD_DOMICILIO', 'CIIU']

# Columnas financieras que se proyectarán
COLS_TO_PROJECT = ['INGRESOS_OPERACIONALES','TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO']

# --- FUNCIONES AUXILIARES ---

@st.cache_resource
def load_data():
    """Simula la carga de los datos históricos."""
    # ------------------------------------------------------------------------------------------------
    # --- IMPORTANTE: Reemplaza esto con la carga real de tu DataFrame de datos si está disponible ---
    # ------------------------------------------------------------------------------------------------
    
    # Datos de ejemplo para la funcionalidad (simulando 2023 y 2024 para una empresa)
    data = {
        'NIT': ['900888000', '900888000', '900111000'],
        'RAZON_SOCIAL': ['CARBONES DEL CERREJON LIMITED', 'CARBONES DEL CERREJON LIMITED', 'ECOPETROL S.A.'],
        'ANO_DE_CORTE': [2023, 2024, 2024],
        'MACROSECTOR': ['MINERO', 'MINERO', 'PETROLERO'],
        'REGION': ['COSTA ATLANTICA', 'COSTA ATLANTICA', 'CENTRO'],
        'SUPERVISOR': ['SUP_A', 'SUP_A', 'SUP_B'],
        'DEPARTAMENTO_DOMICILIO': ['LA GUAJIRA', 'LA GUAJIRA', 'BOGOTA'],
        'CIUDAD_DOMICILIO': ['RIOHACHA', 'RIOHACHA', 'BOGOTA D.C.'],
        'CIIU': ['MINERIA DE CARBON', 'MINERIA DE CARBON', 'EXPLORACION Y PRODUCCION DE PETROLEO'],
        'INGRESOS_OPERACIONALES': [60000.0, 65000.0, 500000.0],
        'TOTAL_ACTIVOS': [10000.0, 10500.0, 70000.0],
        'TOTAL_PASIVOS': [5000.0, 6000.0, 40000.0],
        'TOTAL_PATRIMONIO': [5000.0, 4500.0, 30000.0],
        'GANANCIA_PERDIDA': [-77.0, -77.0, 150.0] 
    }
    df = pd.DataFrame(data)
    
    return df

@st.cache_resource
def load_assets():
    """Carga los 7 activos del modelo (3 modelos, 2 features, 2 constantes)."""
    cls_file = "model_clasificacion.pkl"
    reg_gan_file = "model_reg_ganancia.pkl"
    reg_per_file = "model_reg_perdida.pkl"
    features_file = "model_features.pkl"
    encoders_file = "label_encoders.pkl"
    growth_file = "growth_rate.pkl"   
    base_year_file = "base_year.pkl" 

    # Verificar que todos los 7 archivos existan
    files_exist = (os.path.exists(cls_file) and os.path.exists(reg_gan_file) and 
                   os.path.exists(reg_per_file) and os.path.exists(features_file) and 
                   os.path.exists(encoders_file) and os.path.exists(growth_file) and
                   os.path.exists(base_year_file)) 

    if not files_exist:
        st.error("Error: Faltan archivos .pkl del modelo. Asegúrate de que los 7 archivos estén en la misma carpeta.")
        return None, None, None, None, None, None, None

    try:
        model_cls = joblib.load(cls_file)
        model_reg_gan = joblib.load(reg_gan_file)
        model_reg_per = joblib.load(reg_per_file)
        model_features = joblib.load(features_file)
        label_encoders = joblib.load(encoders_file)
        
        # Cargar las nuevas constantes de proyección
        AGR = joblib.load(growth_file)
        BASE_YEAR = joblib.load(base_year_file)
        
        return model_cls, model_reg_gan, model_reg_per, model_features, label_encoders, AGR, BASE_YEAR
    except Exception as e:
        st.error(f"Error al cargar los modelos o constantes: {e}")
        return None, None, None, None, None, None, None

def format_ano(year):
    """Función para formatear el año para el OHE (igual que en Colab)."""
    year_str = str(year)
    if len(year_str) == 4:
        return f'{year_str[0]},{year_str[1:]}' 
    return year_str

# --- CARGA DE DATOS Y ASSETS ---
df = load_data()
loaded_assets = load_assets()

if loaded_assets:
    model_cls, model_reg_gan, model_reg_per, MODEL_FEATURE_NAMES, label_encoders, AGR, BASE_YEAR = loaded_assets
else:
    st.stop()


# --- 1. FILTROS Y DATOS ---

st.sidebar.markdown("### 1. Filtros y Datos")

# Definir filtros
macrosectores = ["Todos"] + list(df["MACROSECTOR"].unique())
region_options = ["Todos"] + list(df["REGION"].unique())

filtro_macrosector = st.sidebar.selectbox("Filtrar por Macrosector", macrosectores)
filtro_region = st.sidebar.selectbox("Filtrar por Región", region_options)

# Aplicar filtros (solo para la visualización de KPIs)
df_filtrado = df.copy()
if filtro_macrosector != "Todos":
    df_filtrado = df_filtrado[df_filtrado["MACROSECTOR"] == filtro_macrosector]
if filtro_region != "Todos":
    df_filtrado = df_filtrado[df_filtrado["REGION"] == filtro_region]

ano_corte_max = df_filtrado["ANO_DE_CORTE"].max()
st.sidebar.markdown(f"✅ Año de corte máximo global: **{ano_corte_max}**")


# --- TÍTULO ---
st.title("📊 Dashboard ALECO: Modelo de Dos Partes")
st.markdown("Predicción de Ganancia/Pérdida (incluyendo pérdidas reales) usando Modelado de Dos Partes. Todas las cifras se muestran en **Billones de Pesos**.")
st.markdown("---")

# --- 2. KPIs AGREGADOS ---
st.markdown("### 2. KPIs Agregados")
col1, col2 = st.columns(2)

ingresos_totales = df_filtrado["INGRESOS_OPERACIONALES"].sum()
patrimonio_promedio = df_filtrado["TOTAL_PATRIMONIO"].mean()

with col1:
    st.metric("Ingresos Operacionales Totales (Billones COP)", f"${ingresos_totales:,.2f}")
with col2:
    st.metric("Patrimonio Promedio (Billones COP)", f"${patrimonio_promedio:,.2f}")

st.markdown("---")

# --- 3. PREDICCIÓN DE GANANCIA/PÉRDIDA (INTERACTIVO) ---
st.markdown("### 3. Predicción de Ganancia/Pérdida")

# 3A. Selecciones
empresas_filtradas = sorted(df_filtrado["RAZON_SOCIAL"].unique().tolist())
if not empresas_filtradas:
    st.warning("No hay empresas disponibles con los filtros seleccionados.")
    st.stop()

col_pred1, col_pred2 = st.columns(2)

with col_pred1:
    empresa_seleccionada = st.selectbox(
        "Selecciona la Empresa para predecir", 
        empresas_filtradas,
        index=0
    )

with col_pred2:
    # Años disponibles para predicción (desde el año siguiente al máximo hasta 2030)
    anos_prediccion = list(range(ano_corte_max + 1, 2031))
    ano_prediccion = st.selectbox(
        "Selecciona el Año de Predicción", 
        anos_prediccion,
        index=anos_prediccion.index(2026) if 2026 in anos_prediccion else 0
    )


# --- Lógica de Predicción ---
if empresa_seleccionada:
    try:
        df_empresa = df_filtrado[df_filtrado["RAZON_SOCIAL"] == empresa_seleccionada]
        ano_corte_empresa = df_empresa["ANO_DE_CORTE"].max()
        
        if ano_corte_empresa <= 2000:
            st.error(f"Error: La empresa '{empresa_seleccionada}' no tiene un año de corte válido.")
            st.stop()

        st.info(f"Predicción para **{ano_prediccion}**, comparando contra la última fecha de corte registrada de la empresa: **{ano_corte_empresa}**.")

        row_data = df_empresa[df_empresa["ANO_DE_CORTE"] == ano_corte_empresa].iloc[[0]].copy()
        ganancia_anterior = row_data[TARGET_COL].iloc[0]
        
        # --- PARCHE DE CORRECCIÓN DE ESCALA (ACTIVO) ---
        ganancia_anterior = ganancia_anterior / 100.0 
        
        # --- 1. PRE-PROCESAMIENTO PARA LOS TRES MODELOS (PROYECCIÓN FUTURA) ---
        
        row_prediccion = row_data.drop(columns=[TARGET_COL], errors='ignore').copy()
        row_prediccion = row_prediccion.drop(columns=['NIT', 'RAZON_SOCIAL'], errors='ignore')

        # 1. Proyectar variables financieras al Año Base Global (igual que en el entrenamiento)
        # Llevamos los valores del año de corte (ej: 2024) al año base (ej: 2018) usando la AGR inversa.
        delta_anos_actual = ano_corte_empresa - BASE_YEAR
        if delta_anos_actual > 0:
            for col in COLS_TO_PROJECT:
                row_prediccion[col] = row_prediccion[col] / (AGR ** delta_anos_actual)

        # 2. Proyectar *desde* el Año Base Global *hasta* el Año de Predicción
        # Proyectamos los valores base (2018) hacia el año de predicción (ej: 2026) usando la AGR.
        delta_anos_prediccion = ano_prediccion - BASE_YEAR
        if delta_anos_prediccion > 0:
            for col in COLS_TO_PROJECT:
                row_prediccion[col] = row_prediccion[col] * (AGR ** delta_anos_prediccion)

        # Establecer el año de predicción (CRÍTICO para el OHE de ANO_DE_CORTE)
        row_prediccion["ANO_DE_CORTE"] = ano_prediccion

        # Aplicar Label Encoding (Usando los encoders cargados)
        for col in LE_COLS:
            try:
                encoder = label_encoders[col]
                row_prediccion[col] = encoder.transform(row_prediccion[col].astype(str))[0]
                row_prediccion[col] = int(row_prediccion[col]) 
            except ValueError:
                 row_prediccion[col] = 0 
        
        # FIX CRÍTICO: Formato de Año para OHE
        row_prediccion['ANO_DE_CORTE'] = row_prediccion['ANO_DE_CORTE'].apply(format_ano)

        # Aplicar One-Hot Encoding
        row_prediccion = pd.get_dummies(
            row_prediccion, columns=OHE_COLS, prefix=OHE_COLS, drop_first=True, dtype=int
        )
        
        # Alinear y ordenar las columnas (CRÍTICO)
        missing_cols = set(MODEL_FEATURE_NAMES) - set(row_prediccion.columns)
        for c in missing_cols:
            row_prediccion[c] = 0 
        
        X_pred = row_prediccion[MODEL_FEATURE_NAMES].copy()
        
        # Conversión final a numérico
        X_pred = X_pred.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        
        # --- 2. LÓGICA DE PREDICCIÓN CONDICIONAL ---
        
        # Paso A: Clasificar (0 = Pérdida/Cero, 1 = Ganancia)
        pred_cls = model_cls.predict(X_pred)[0]
        
        pred_log = 0.0
        
        if pred_cls == 1:
            # Ganancia: Usar Modelo de Regresión de Ganancias
            pred_log = model_reg_gan.predict(X_pred)[0]
            # Reversión: e^x - 1
            pred_real = np.expm1(pred_log) 
            
        else:
            # Pérdida/Cero: Usar Modelo de Regresión de Pérdidas
            pred_log = model_reg_per.predict(X_pred)[0]
            # Reversión: e^x - 1 (nos da la magnitud positiva de la pérdida)
            magnitud_perdida_real = np.expm1(pred_log)
            # CRÍTICO: Convertir la magnitud a valor negativo (pérdida)
            pred_real = -magnitud_perdida_real
            
        
        # --- 3. MOSTRAR RESULTADOS (Cálculo Delta Robusto y Formato) ---
        diferencia = pred_real - ganancia_anterior

        # --- CÁLCULO ROBUSTO DEL DELTA PORCENTUAL ---
        delta_display = ""
        delta_metric_value = diferencia # Usaremos la diferencia para el indicador de Streamlit (flecha)

        if ganancia_anterior == 0:
            # Caso 1: Anterior fue CERO (0).
            if pred_real > 0:
                delta_display = f"Ganó ${pred_real:,.2f} vs 0"
            elif pred_real < 0:
                delta_display = f"Perdió ${abs(pred_real):,.2f} vs 0"
            else:
                delta_display = "Sin cambio vs 0"

        elif ganancia_anterior < 0:
            # Caso 2: Anterior fue una PÉRDIDA (Negativo).
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
                    
        else:
            # Caso 3: Anterior fue una GANANCIA (Positivo).
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
            # Mostramos el texto formateado en el caption
            st.caption(f"Cambio vs {ano_corte_empresa}: **{delta_display}**") 

        with col_res2:
            st.metric(
                label=f"G/P Real (Última fecha de corte registrada) (Billones COP)", 
                value=f"${ganancia_anterior:,.2f}",
                delta_color="off"
            )
            
        # Mensaje condicional final (Lógica detallada)
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
        st.caption("Asegúrate de que la empresa seleccionada tiene datos completos y que los SIETE archivos .pkl son correctos y que la lógica de proyección es consistente.")
