# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import unicodedata
import warnings

warnings.filterwarnings("ignore")

# --- 0) CONFIGURACIÓN INICIAL Y CONSTANTES ---
st.set_page_config(layout="wide", page_title="Dashboard ALECO: Modelo de Dos Partes")
st.title("📊 Dashboard ALECO: Modelo de Dos Partes")
st.markdown("Predicción de Ganancia/Pérdida (incluyendo pérdidas reales) usando Modelado de Dos Partes. Todas las cifras se muestran en **Billones de Pesos**.")
st.markdown("---")

TARGET_COL = 'GANANCIA_PERDIDA'
COLS_TO_PROJECT = [
    'INGRESOS_OPERACIONALES', 'TOTAL_ACTIVOS',
    'TOTAL_PASIVOS', 'TOTAL_PATRIMONIO'
]
# 'ANO_DE_CORTE' NO debe estar en OHE_COLS
OHE_COLS = ['SUPERVISOR', 'REGION', 'MACROSECTOR']
LE_COLS = ['DEPARTAMENTO_DOMICILIO', 'CIUDAD_DOMICILIO', 'CIIU']

def normalize_col(col):
    col = str(col).strip().upper().replace(" ", "_").replace("(", "").replace(")", "").replace("Ñ", "N")
    return ''.join(c for c in unicodedata.normalize('NFD', col) if unicodedata.category(c) != 'Mn')

def safe_le_transform(encoder, val):
    """Transforma un valor usando LabelEncoder, devolviendo -1 si es desconocido."""
    try:
        s = str(val)
        # encoder.classes_ suele ser array de strings
        if hasattr(encoder, "classes_") and s in encoder.classes_:
            return int(np.where(encoder.classes_ == s)[0][0])
    except Exception:
        pass
    return -1

# --- 1) CARGA DE DATOS Y ACTIVOS ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # normalizar nombres de columnas
        df.columns = [normalize_col(c) for c in df.columns]

        # Sólo procesar columnas numéricas si existen
        numeric_cols = [c for c in COLS_TO_PROJECT + [TARGET_COL] if c in df.columns]

        for col in numeric_cols:
            # Convertir a str y limpiar formatos locales
            s = df[col].astype(str).str.strip()

            s = (
                s.str.replace('$', '', regex=False)
                .str.replace('(', '', regex=False).str.replace(')', '', regex=False)
                .str.replace(' ', '', regex=False).str.replace('−', '-', regex=False)
            )

            # 1) quitar separador de miles (puntos)
            s = s.str.replace('.', '', regex=False)
            # 2) reemplazar coma decimal por punto
            s = s.str.replace(',', '.', regex=False)

            df[col] = pd.to_numeric(s, errors='coerce').astype(float)

        # Manejo ANO_DE_CORTE si existe
        if 'ANO_DE_CORTE' in df.columns:
            df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].astype(str).str.replace(",", "", regex=False)
            df['ANO_DE_CORTE'] = pd.to_numeric(df['ANO_DE_CORTE'], errors='coerce')
            # rellenar NaN temporalmente con -1, luego filtro
            df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].fillna(-1).astype(int)

        # Filtrar años inválidos (>2000) sólo si columna existe
        if 'ANO_DE_CORTE' in df.columns:
            df = df[df['ANO_DE_CORTE'] > 2000].copy()

        # Llenar numéricos faltantes con 0 (sólo las que existan)
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        return df

    except Exception as e:
        st.error(f"Error cargando o limpiando el archivo CSV: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_assets():
    try:
        models = {
            'cls': joblib.load("model_clasificacion.pkl"),
            'reg_gan': joblib.load("model_reg_ganancia.pkl"),
            'reg_per': joblib.load("model_reg_perdida.pkl"),
        }
        assets = {
            'label_encoders': joblib.load("label_encoders.pkl"),
            'model_features': joblib.load("model_features.pkl"),
            'AGR': joblib.load("growth_rate.pkl"),
            'base_year': joblib.load("base_year.pkl"),
        }
        return models, assets
    except FileNotFoundError as e:
        st.error(f"Error: No se encontró el archivo de activo '{e.filename}'. Asegúrate de que los SIETE archivos .pkl estén en el mismo directorio.")
        return None, None
    except Exception as e:
        st.error(f"Error cargando activos: {e}")
        return None, None

# Ajusta la ruta del CSV a tu entorno
CSV_PATH = "10.000_Empresas_mas_Grandes_del_País_20251115.csv"
df = load_data(CSV_PATH)
models, assets = load_assets()

if df.empty:
    st.warning("El DataFrame quedó vacío después de la carga/limpieza. Revisa el CSV o el proceso de limpieza.")
    st.stop()

if models is None or assets is None:
    st.warning("No se pudieron cargar los activos del modelo (.pkl). Coloca los archivos en el mismo directorio que app.py.")
    st.stop()

# Desempaquetar activos (validar keys)
model_cls = models.get('cls')
model_reg_gan = models.get('reg_gan')
model_reg_per = models.get('reg_per')

label_encoders = assets.get('label_encoders', {})
MODEL_FEATURE_NAMES = assets.get('model_features', [])
AGR = assets.get('AGR', 1.0)
ANO_CORTE_BASE_GLOBAL = assets.get('base_year', df['ANO_DE_CORTE'].max() if 'ANO_DE_CORTE' in df.columns else None)

# --- 2) LÓGICA DE FILTROS Y KPIS ---
st.header("1. Filtros y Datos")

# Asegurarse de que columnas categóricas existan antes de listarlas
macrosectores = ["Todos"]
if 'MACROSECTOR' in df.columns:
    macrosectores += df['MACROSECTOR'].dropna().unique().tolist()

regiones = ["Todos"]
if 'REGION' in df.columns:
    regiones += df['REGION'].dropna().unique().tolist()

col_m, col_r, col_c = st.columns([1, 1, 0.5])

with col_m:
    filtro_macrosector = st.selectbox("Filtrar por Macrosector", macrosectores)

with col_r:
    filtro_region = st.selectbox("Filtrar por Región", regiones)

with col_c:
    ano_corte_mas_reciente_global = df['ANO_DE_CORTE'].max() if 'ANO_DE_CORTE' in df.columns else None
    st.markdown(f"✅ Año de corte máximo global: **{ano_corte_mas_reciente_global}**")

# Aplicar filtros de forma segura
df_filtrado = df.copy()
if filtro_macrosector != "Todos" and 'MACROSECTOR' in df.columns:
    df_filtrado = df_filtrado[df_filtrado['MACROSECTOR'] == filtro_macrosector]
if filtro_region != "Todos" and 'REGION' in df.columns:
    df_filtrado = df_filtrado[df_filtrado['REGION'] == filtro_region]

# KPIs
st.header("2. KPIs Agregados")
if 'ANO_DE_CORTE' in df_filtrado.columns:
    kpis_df = df_filtrado[df_filtrado['ANO_DE_CORTE'] == ano_corte_mas_reciente_global]
else:
    kpis_df = df_filtrado

def format_billones(value):
    # Nota: aquí se divide por 1e9 (miles de millones). Ajusta si necesitas otra escala.
    try:
        return f"${float(value) / 1e9:,.2f}"
    except Exception:
        return "$0"

total_ingresos = kpis_df['INGRESOS_OPERACIONALES'].sum() if 'INGRESOS_OPERACIONALES' in kpis_df.columns else 0.0
promedio_patrimonio = kpis_df['TOTAL_PATRIMONIO'].mean() if 'TOTAL_PATRIMONIO' in kpis_df.columns else 0.0

col_kpi1, col_kpi2 = st.columns(2)
with col_kpi1:
    st.metric(
        label="Ingresos Operacionales Totales (Miles de millones COP)",
        value=format_billones(total_ingresos)
    )
with col_kpi2:
    st.metric(
        label="Patrimonio Promedio (Miles de millones COP)",
        value=format_billones(promedio_patrimonio)
    )

st.markdown("---")

# ----------------------------------------------------
# FUNCIÓN DE PREDICCIÓN RECURSIVA
# ----------------------------------------------------
def predict_recursive(row_base, ano_corte_empresa, ano_prediccion_final,
                      model_cls, model_reg_gan, model_reg_per,
                      label_encoders, MODEL_FEATURE_NAMES, AGR,
                      COLS_TO_PROJECT, LE_COLS, OHE_COLS):

    row_current_base = row_base.copy()
    años_a_predecir = range(int(ano_corte_empresa) + 1, int(ano_prediccion_final) + 1)
    pred_real_final = 0.0

    for ano_actual in años_a_predecir:
        # A. Proyección de Features Numéricos (si existen)
        for col in COLS_TO_PROJECT:
            if col in row_current_base.index:
                try:
                    row_current_base[col] = float(row_current_base[col]) * float(AGR)
                except Exception:
                    row_current_base[col] = row_current_base.get(col, 0.0)

        # B. Preparar fila y codificar
        row_prediccion = row_current_base.to_frame().T.copy()
        row_prediccion["ANO_DE_CORTE"] = ano_actual

        # LE seguro
        for col in LE_COLS:
            if col in row_prediccion.columns:
                encoder = label_encoders.get(col)
                if encoder is not None:
                    row_prediccion[col] = safe_le_transform(encoder, row_prediccion[col].iloc[0])
                else:
                    # si falta encoder, usar -1
                    row_prediccion[col] = -1

        # OHE: convertir a string para get_dummies
        ohe_cols_to_use = [c for c in OHE_COLS if c in row_prediccion.columns]
        for col in ohe_cols_to_use:
            row_prediccion[col] = row_prediccion[col].astype(str)

        row_prediccion_ohe = pd.get_dummies(
            row_prediccion,
            columns=ohe_cols_to_use, prefix=ohe_cols_to_use, drop_first=True, dtype=int
        )

        # C. Alineación con FEATURES del modelo
        X_pred = pd.DataFrame(0, index=[0], columns=MODEL_FEATURE_NAMES)

        common = [c for c in row_prediccion_ohe.columns if c in X_pred.columns]
        if len(common) > 0:
            X_pred.loc[0, common] = row_prediccion_ohe.loc[0, common].values

        cols_to_inject = [c for c in (COLS_TO_PROJECT + LE_COLS + ['ANO_DE_CORTE']) if c in X_pred.columns]
        for col in cols_to_inject:
            if col in row_prediccion.columns:
                try:
                    val = float(row_prediccion[col].iloc[0])
                except Exception:
                    val = 0.0
                X_pred.at[0, col] = val

        X_pred = X_pred.astype(float).fillna(0)

        if X_pred.shape[1] == 0:
            raise ValueError("El DataFrame de predicción (X_pred) está vacío. Revisa model_features.pkl y las columnas del CSV.")

        pred_cls = int(model_cls.predict(X_pred)[0])

        if pred_cls == 1:
            pred_log = float(model_reg_gan.predict(X_pred)[0])
            pred_g_p_actual = np.expm1(pred_log)
        else:
            pred_log = float(model_reg_per.predict(X_pred)[0])
            magnitud_perdida_real = np.expm1(pred_log)
            pred_g_p_actual = -magnitud_perdida_real

        if ano_actual == ano_prediccion_final:
            pred_real_final = pred_g_p_actual

    return pred_real_final

# ----------------------------------------------------
# SECCIÓN: EJECUCIÓN DEL DASHBOARD Y RESULTADOS
# ----------------------------------------------------
st.header("3. Predicción de Ganancia/Pérdida")

col_sel_company, col_sel_year = st.columns(2)
empresas_disponibles = df_filtrado["RAZON_SOCIAL"].dropna().unique().tolist() if 'RAZON_SOCIAL' in df_filtrado.columns else []

if not empresas_disponibles:
    st.warning("No hay empresas disponibles después de aplicar los filtros.")
    st.stop()

with col_sel_company:
    empresa_seleccionada = st.selectbox("Selecciona la Empresa para predecir", empresas_disponibles)

with col_sel_year:
    pred_years = [2026, 2027, 2028, 2029, 2030]
    años_futuros = [y for y in pred_years if (ano_corte_mas_reciente_global is not None and y > ano_corte_mas_reciente_global)]
    if not años_futuros:
        st.warning(f"El año de corte base es {ano_corte_mas_reciente_global}. No hay años futuros disponibles.")
        st.stop()
    ano_prediccion_final = st.selectbox("Selecciona el Año de Predicción (Target)", años_futuros, index=0)

# --- Lógica de Predicción ---
try:
    df_empresa = df_filtrado[df_filtrado["RAZON_SOCIAL"] == empresa_seleccionada]
    if df_empresa.empty:
        st.error("No se encontraron registros para la empresa seleccionada.")
        st.stop()

    ano_corte_empresa = int(df_empresa["ANO_DE_CORTE"].max()) if 'ANO_DE_CORTE' in df_empresa.columns else None

    if ano_corte_empresa is None or ano_corte_empresa <= 2000:
        st.error(f"Error: La empresa '{empresa_seleccionada}' no tiene un año de corte válido.")
        st.stop()

    st.info(f"Predicción recursiva hasta **{ano_prediccion_final}**, iniciando desde la última fecha de corte ({ano_corte_empresa}). Tasa de Crecimiento Asumida (AGR): **{AGR:.4f}**")

    # fila base: toma la fila de la empresa en su último año
    row_data = df_empresa[df_empresa["ANO_DE_CORTE"] == ano_corte_empresa].iloc[[0]].copy()
    ganancia_anterior = float(row_data[TARGET_COL].iloc[0]) if TARGET_COL in row_data.columns else 0.0

    row_base = row_data.drop(columns=[TARGET_COL, 'NIT', 'RAZON_SOCIAL'], errors='ignore').iloc[0]

    pred_real_final = predict_recursive(
        row_base, ano_corte_empresa, ano_prediccion_final,
        model_cls, model_reg_gan, model_reg_per,
        label_encoders, MODEL_FEATURE_NAMES, AGR,
        COLS_TO_PROJECT, LE_COLS, OHE_COLS
    )

    diferencia = pred_real_final - ganancia_anterior
    delta_metric_value = diferencia

    # Delta display robusto
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
            label=f"GANANCIA/PÉRDIDA Predicha ({ano_prediccion_final}) (Unidad original)",
            value=f"${pred_real_final:,.2f}",
            delta=delta_metric_value,
            delta_color="normal"
        )
        st.caption(f"Cambio vs {ano_corte_empresa}: **{delta_display}**")

    with col_res2:
        st.metric(
            label=f"G/P Real (Última fecha de corte registrada) (Unidad original)",
            value=f"${ganancia_anterior:,.2f}",
            delta_color="off"
        )

    st.markdown("---")

    if pred_real_final >= 0.01:
        if ganancia_anterior > 0 and diferencia >= 0:
            st.success(f"📈 El modelo clasifica la operación como **GANANCIA** y predice un **AUMENTO** (Resultado: ${pred_real_final:,.2f}).")
        elif ganancia_anterior < 0:
            st.success(f"🚀 El modelo predice una **RECUPERACIÓN TOTAL** (Resultado: ${pred_real_final:,.2f}).")
        else:
            st.success(f"📈 El modelo predice que la empresa pasa a **GANANCIA** desde equilibrio (Resultado: ${pred_real_final:,.2f}).")
    elif pred_real_final < -0.01:
        st.error(f"📉 El modelo clasifica la operación como **PÉRDIDA** neta (Resultado: **${abs(pred_real_final):,.2f}**).")
    else:
        st.info("ℹ️ El modelo predice que el resultado será **cercano a cero** (equilibrio financiero).")

    st.markdown("---")

except Exception as e:
    st.error(f"❌ ERROR generando la predicción: {e}")
    st.caption("Asegúrate de que la empresa seleccionada tiene datos completos, que todos los SIETE archivos .pkl sean consistentes (especialmente que ANO_DE_CORTE es numérico), y que el formato de números en tu CSV sea el esperado.")
