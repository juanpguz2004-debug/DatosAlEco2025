import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import unicodedata
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Any
import sys # Importado para debugging en la consola

# ----------------------------------------------------
# 0) CONFIGURACIÓN INICIAL Y CONSTANTES
# ----------------------------------------------------
st.set_page_config(
    page_title="📊 Dashboard ALECO",
    layout="wide"
)

# Nombres de columnas clave
TARGET_COL = 'GANANCIA_PERDIDA'
OHE_COLS = ['SUPERVISOR', 'REGION', 'MACROSECTOR']
LE_COLS = ['DEPARTAMENTO_DOMICILIO', 'CIUDAD_DOMICILIO', 'CIIU']
COLS_TO_PROJECT = ['INGRESOS_OPERACIONALES', 'TOTAL_ACTIVOS', 'TOTAL_PASIVOS', 'TOTAL_PATRIMONIO']

# Función de normalización de columna
def normalize_col(col: str) -> str:
    """Normaliza y limpia los nombres de columna."""
    col = col.strip().upper().replace(" ", "_").replace("(", "").replace(")", "").replace("Ñ", "N")
    return ''.join(c for c in unicodedata.normalize('NFD', col) if unicodedata.category(c) != 'Mn')


# Función de Label Encoding segura
def safe_le_transform(encoder: LabelEncoder, value: Any) -> int:
    """Aplica la transformación de LabelEncoder, devolviendo -1 para valores no vistos."""
    try:
        # El valor debe ser una lista/array para transform
        return int(encoder.transform([str(value).strip().upper()])[0])
    except ValueError:
        # Devuelve un valor que será manejado como missing o 0 por el modelo
        return -1 


# ----------------------------------------------------
# 1) CARGAR DATOS Y LIMPIEZA
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

        numeric_cols = COLS_TO_PROJECT + [TARGET_COL]
        for col in numeric_cols:
            df[col] = (
                df[col].astype(str)
                .str.replace("$","",regex=False).str.replace(" ","",regex=False)
                .str.replace("−","-",regex=False).str.replace("(","",regex=False)
                .str.replace(")","",regex=False).str.replace('.', '', regex=False) 
                .str.replace(',', '.', regex=False) # Convierte coma a punto decimal
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'ANO_DE_CORTE' in df.columns:
            df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].astype(str).str.replace(",", "", regex=False)
            df['ANO_DE_CORTE'] = pd.to_numeric(df['ANO_DE_CORTE'], errors='coerce').fillna(-1).astype(int)
        
        df = df[df['ANO_DE_CORTE'] > 2000].copy()
        df.dropna(subset=numeric_cols, inplace=True)
        
        # Rellenar NaNs en categóricas con un valor seguro
        for col in LE_COLS + OHE_COLS:
            if col in df.columns:
                 df[col] = df[col].astype(str).fillna('MISSING')
        
        return df

    except Exception as e:
        st.error(f"❌ ERROR al leer o limpiar el CSV: {e}")
        return pd.DataFrame()


# ----------------------------------------------------
# 2) CARGAR SIETE MODELOS Y REFERENCIAS (Corregida)
# ----------------------------------------------------
@st.cache_resource
def load_assets():
    """Carga los 7 activos (3 modelos, 4 referencias) necesarios."""
    assets = {}
    files_to_load = [
        "model_clasificacion.pkl", "model_reg_ganancia.pkl", "model_reg_perdida.pkl",
        "model_features.pkl", "label_encoders.pkl", "growth_rate.pkl", "base_year.pkl" # <--- Ahora son 7
    ]
    
    if not all(os.path.exists(f) for f in files_to_load):
        st.error(f"❌ ERROR FATAL: Faltan archivos .pkl. Asegúrate de tener los {len(files_to_load)} archivos.")
        return [None] * len(files_to_load)
    
    try:
        assets['cls'] = joblib.load(files_to_load[0])
        assets['reg_gan'] = joblib.load(files_to_load[1])
        assets['reg_per'] = joblib.load(files_to_load[2])
        assets['features'] = joblib.load(files_to_load[3])
        assets['encoders'] = joblib.load(files_to_load[4])
        assets['agr'] = joblib.load(files_to_load[5]) 
        assets['base_year'] = joblib.load(files_to_load[6]) # <--- Cargando base_year
        
        return (assets['cls'], assets['reg_gan'], assets['reg_per'], 
                assets['features'], assets['encoders'], assets['agr'], assets['base_year'])
    except Exception as e:
        st.error(f"❌ ERROR al cargar activos: {e}")
        return [None] * len(files_to_load)


# ----------------------------------------------------
# 3) FUNCIÓN DE FORECASTING RECURSIVO (Corregida)
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
    # Usamos .iloc[0] para obtener la serie de la fila inicial
    current_data = initial_row.iloc[0].copy() 
    start_year = current_data['ANO_DE_CORTE']
    df_forecast = pd.DataFrame()
    
    #st.warning("🚨 INICIO DEBUG: Revisa la consola/terminal de Streamlit para la traza.")

    for year in range(int(start_year) + 1, target_year + 1):
        
        # --- Paso 1: Proyección Acumulativa ---
        for col in COLS_TO_PROJECT:
            try:
                # La proyección solo se aplica a los valores numéricos
                current_data[col] = float(current_data[col] * agr) 
            except Exception:
                # Si falla, mantiene el valor o lo resetea si es NaN
                current_data[col] = current_data[col] if not pd.isna(current_data[col]) else 0.0
                
        current_data['ANO_DE_CORTE'] = year
        
        # --- Paso 1.5: Feature Engineering (CRUCIAL CORRECCIÓN) ---
        
        # 1. Transformación no lineal para INGRESOS
        ingresos = current_data['INGRESOS_OPERACIONALES']
        current_data['INGRESOS_SQRT'] = np.sqrt(ingresos) if ingresos > 0 else 0
        
        # 2. Ratios Financieros
        activos = current_data['TOTAL_ACTIVOS']
        pasivos = current_data['TOTAL_PASIVOS']
        patrimonio = current_data['TOTAL_PATRIMONIO']

        # Ratio Deuda/Activo
        current_data['DEUDA_ACTIVO_RATIO'] = pasivos / activos if activos != 0 else 0
        
        # Ratio Patrimonio/Activo
        current_data['PATRIMONIO_ACTIVO_RATIO'] = patrimonio / activos if activos != 0 else 0

        # --- Paso 2: Preprocesamiento para el Modelo (Alineación) ---
        try:
            # Crear DataFrame de una sola fila para la predicción
            X_pred_temp = pd.DataFrame([current_data.to_dict()])
            
            # Aplicar Label Encoding seguro
            for col in LE_COLS:
                encoder = label_encoders[col]
                # str(X_pred_temp[col].iloc[0]) asegura que el valor sea un string
                X_pred_temp[col] = safe_le_transform(encoder, X_pred_temp[col].iloc[0])
            
            # Aplicar One-Hot Encoding (OHE) a las categóricas restantes
            X_ohe_temp = X_pred_temp[OHE_COLS]
            X_ohe_encoded = pd.get_dummies(
                X_ohe_temp, columns=OHE_COLS, drop_first=True, dtype=int
            )
            
            # Combinar features numéricas, LE y OHE
            X_numeric_le = X_pred_temp.drop(columns=OHE_COLS)
            X_combined = pd.concat([X_numeric_le, X_ohe_encoded], axis=1)

            # Alinear y Ordenar las columnas para el input del modelo
            final_input_data = {}
            for feature in model_feature_names:
                # Si la feature existe en el DF combinado, la toma. Si no (OHE no visto), es 0.
                if feature in X_combined.columns:
                    final_input_data[feature] = X_combined[feature].iloc[0]
                else:
                    final_input_data[feature] = 0.0 # Valor seguro para OHE no vistas
            
            X_pred = pd.DataFrame([final_input_data])
            # Asegurar orden y llenar NaNs de seguridad (aunque ya deberían estar limpios)
            X_pred = X_pred[model_feature_names].fillna(0.0) 
            
        except Exception as e:
            st.error(f"DEBUG ERROR: Fallo en el preprocesamiento en {year}. Error: {e}")
            X_pred = pd.DataFrame(0.0, index=[0], columns=model_feature_names)
            
        # --- Paso 3: Predicción con Hurdle Model ---
        try:
            # Predicción de clasificación (Ganancia=1 / Pérdida=0)
            pred_cls = model_cls.predict(X_pred)[0]
            
            if pred_cls == 1:
                # Si clasifica como Ganancia, usar regresor de Ganancia
                pred_log = model_reg_gan.predict(X_pred)[0]
                pred_real = np.expm1(pred_log) # Inversa de log1p: exp(x) - 1
            else:
                # Si clasifica como Pérdida/Cero, usar regresor de Pérdida (Magnitud)
                pred_log = model_reg_per.predict(X_pred)[0]
                pred_real = -np.expm1(pred_log) # Aplicar el negativo a la magnitud
            
            # Asegurar que el resultado no sea NaN/Inf
            if not np.isfinite(pred_real):
                 pred_real = 0.0
            
            # 📣 PUNTO DE DEBUG 3: Mostrar el resultado de la predicción en la consola
            print(f"DEBUG AÑO {year}: G/P Proyectada: {pred_real:,.2f} (Clasificación: {pred_cls})", file=sys.stderr)
            
        except Exception as e:
            st.error(f"DEBUG ERROR: Fallo en la predicción del modelo en {year}. Error: {e}")
            pred_real = 0.0 # Valor seguro
            
        # --- Paso 4: Almacenar Resultado ---
        current_data_for_output = current_data.to_dict()
        current_data_for_output['GANANCIA_PERDIDA_PRED'] = pred_real 
        df_forecast = pd.concat([df_forecast, pd.DataFrame([current_data_for_output])], ignore_index=True)
    
    #st.warning("🚨 FIN DEBUG. Revisar si la G/P Predicha tiene sentido en la consola.")
    return df_forecast


# ----------------------------------------------------
# --- INICIO DE LA APLICACIÓN ---
# ----------------------------------------------------

# Cargar activos
df = load_data()
(model_cls, model_reg_gan, model_reg_per, 
 MODEL_FEATURE_NAMES, label_encoders, AGR, BASE_YEAR) = load_assets()

# Validaciones de carga
if df.empty:
    st.error("❌ ERROR FATAL: No se encontraron datos válidos en el CSV.")
    st.stop()
    
if None in [model_cls, model_reg_gan, model_reg_per, MODEL_FEATURE_NAMES, label_encoders, AGR, BASE_YEAR]:
    st.error("❌ ERROR FATAL: Faltan activos. Verifica que los 7 archivos .pkl estén en la carpeta.")
    st.stop()

st.title("📊 Dashboard ALECO: Modelo de Dos Partes")
st.markdown(f"""
**Predicción de Ganancia/Pérdida (incluyendo pérdidas reales) usando Modelado de Dos Partes.**
Todas las cifras se muestran en **Billones de Pesos**. Tasa de Crecimiento Anual (AGR) aplicada: **{AGR*100-100:.1f}%**
""")
st.markdown("---")

ano_corte_mas_reciente_global = df["ANO_DE_CORTE"].max()

# ----------------------------------------------------
# 5) PREDICCIÓN CON LÓGICA DE FORECASTING
# ----------------------------------------------------
st.header("2. Proyección de Ganancia/Pérdida")

# --- SELECTORES: Año y Empresa ---
col_sel_company, col_sel_year = st.columns(2)

empresas_disponibles = df[df["ANO_DE_CORTE"] == ano_corte_mas_reciente_global]["RAZON_SOCIAL"].unique().tolist()

if not empresas_disponibles:
    st.warning("No hay empresas disponibles con datos recientes.")
    st.stop()

with col_sel_company:
    empresa_seleccionada = st.selectbox("Selecciona la Empresa para proyectar", empresas_disponibles)

with col_sel_year:
    max_forecast_year = 2030
    años_futuros = list(range(int(ano_corte_mas_reciente_global) + 1, max_forecast_year + 1))
    
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
    # Obtener la fila base del año más reciente
    row_data_base = df[(df["RAZON_SOCIAL"] == empresa_seleccionada) & 
                       (df["ANO_DE_CORTE"] == ano_corte_mas_reciente_global)].iloc[[0]].copy()
    
    ano_corte_empresa = row_data_base["ANO_DE_CORTE"].iloc[0]

    df_prediccion_forecast = run_forecasting(
        initial_row=row_data_base, 
        target_year=ano_prediccion, 
        agr=AGR,
        model_cls=model_cls, model_reg_gan=model_reg_gan, model_reg_per=model_reg_per,
        model_feature_names=MODEL_FEATURE_NAMES, label_encoders=label_encoders
    )

    # --- 3. MOSTRAR RESULTADOS (Gráfico y KPIs) ---
    
    df_resultados = df_prediccion_forecast[['ANO_DE_CORTE', 'GANANCIA_PERDIDA_PRED', 'INGRESOS_OPERACIONALES']].copy()
    
    # Añadir el punto real (base) al gráfico
    last_real_data = row_data_base[['ANO_DE_CORTE', TARGET_COL]].rename(
        columns={TARGET_COL: 'GANANCIA_PERDIDA_PRED'}
    )
    
    df_plot_data = pd.concat([last_real_data, df_resultados[['ANO_DE_CORTE', 'GANANCIA_PERDIDA_PRED']]], ignore_index=True)
    df_plot_data['Tipo'] = df_plot_data['ANO_DE_CORTE'].apply(lambda y: 'Real' if y == ano_corte_empresa else 'Proyectado')

    # Cálculos para KPIs
    pred_real = df_resultados['GANANCIA_PERDIDA_PRED'].iloc[-1]
    ganancia_anterior = last_real_data['GANANCIA_PERDIDA_PRED'].iloc[0]
    diferencia = pred_real - ganancia_anterior

    st.markdown("#### Resultado Final de la Proyección")
    col_kpi1, col_kpi2 = st.columns(2)
    
    with col_kpi1:
        st.metric(
            label=f"GANANCIA/PÉRDIDA Predicha Final ({ano_prediccion}) (Billones COP)",
            value=f"${pred_real:,.2f}",
            delta=f"{diferencia:,.2f}",
            delta_color="normal"
        )
        st.caption(f"Cambio absoluto en la G/P vs {ano_corte_empresa}")
    
    with col_kpi2:
        st.metric(
            label=f"Ingresos Proyectados Final ({ano_prediccion}) (Billones COP)",
            value=f"${df_resultados['INGRESOS_OPERACIONALES'].iloc[-1]:,.2f}",
            delta_color="off"
        )
        
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


except Exception as e:
    st.error(f"❌ ERROR generando la proyección: {e}")
    st.caption(f"Detalle del error: {e}")
