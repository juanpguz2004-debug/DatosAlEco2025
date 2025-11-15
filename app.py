# ... (imports y normalize_col sin cambios) ...

# ----------------------------------------------------
# 2) CARGAR CSV
# ----------------------------------------------------
@st.cache_data
def load_data():
    csv_file = "10.000_Empresas_mas_Grandes_del_País_20251115.csv"

    if not os.path.exists(csv_file):
        st.error(f"No se encontró el archivo: {csv_file}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"Error al leer el CSV: {e}")
        return pd.DataFrame()

    # Normalizar columnas igual que en el entrenamiento
    df.columns = [normalize_col(c) for c in df.columns]

    # Columnas requeridas para la predicción y vista
    required_cols = [
        'NIT','RAZON_SOCIAL','SUPERVISOR','REGION','DEPARTAMENTO_DOMICILIO',
        'CIUDAD_DOMICILIO','CIIU','MACROSECTOR',
        'INGRESOS_OPERACIONALES','GANANCIA_PERDIDA',
        'TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO','ANO_DE_CORTE'
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas necesarias: {missing}")
        return pd.DataFrame()

    # LIMPIAR columnas numéricas (igual que en Colab)
    numeric_cols = [
        'INGRESOS_OPERACIONALES','GANANCIA_PERDIDA',
        'TOTAL_ACTIVOS','TOTAL_PASIVOS','TOTAL_PATRIMONIO'
    ]

    for col in numeric_cols:
        df[col] = (
            df[col].astype(str)
            .str.replace("$","",regex=False)
            .str.replace(",","",regex=False)
            .str.replace(".","",regex=False)
            .str.replace(" ","",regex=False)
            .str.replace("−","-",regex=False)
            .str.replace("(","-",regex=False)
            .str.replace(")","",regex=False)
            .astype(float)
        )
    
    # REVISIÓN CLAVE: Asegurar que ANO_DE_CORTE es un entero positivo.
    df['ANO_DE_CORTE'] = pd.to_numeric(df['ANO_DE_CORTE'], errors='coerce')
    # Rellenar NaNs con un valor que pueda ser filtrado y convertir a int
    df['ANO_DE_CORTE'] = df['ANO_DE_CORTE'].fillna(-1).astype(int) 
    # Filtrar las filas con años no válidos de manera temprana
    df = df[df['ANO_DE_CORTE'] > 2000] # Asumiendo que los datos son posteriores al año 2000

    return df

# ----------------------------------------------------
# 3) CARGAR MODELO y 4) DASHBOARD PRINCIPAL (sin cambios significativos)
# ----------------------------------------------------

# ... (código de carga de modelo y KPIs sin cambios) ...


# ----------------------------------------------------
# 6) PREDICCIÓN CON COMPARACIÓN (COMPLETA)
# ----------------------------------------------------
st.subheader("🔮 Predicción de Ganancia/Pérdida")

if df_filtrado.empty:
    st.warning("No hay empresas con ese filtro.")
    st.stop()

# Determinar el año base más reciente en el conjunto de datos filtrado
# Esto se usará como la fecha de corte para la predicción
ano_corte_mas_reciente = df_filtrado["ANO_DE_CORTE"].max()

# Si el máximo año sigue siendo <= 0 después del filtro en load_data, hay un problema grave
if ano_corte_mas_reciente <= 0:
    st.warning("No se encontró un año de corte válido (> 0) en los datos filtrados.")
    st.stop()


# --- MEJORA 1: Selectores de Año y Empresa en columnas ---
col_sel_year, col_sel_company = st.columns(2)

with col_sel_year:
    pred_years = [2026, 2027, 2028, 2029, 2030]
    # Filtramos para que solo se pueda predecir años futuros al corte más reciente
    años_futuros = [y for y in pred_years if y > ano_corte_mas_reciente]
    
    if not años_futuros:
        st.warning(f"El año de corte más reciente es {ano_corte_mas_reciente}. No hay años futuros disponibles para predecir en la lista.")
        st.stop()

    ano_prediccion = st.selectbox(
        "Selecciona el Año de Predicción",
        años_futuros,
        index=0 
    )

# ----------------------------------------------------

# Filtrar empresas disponibles para la predicción (basado en el año más reciente)
empresas_disponibles = df_filtrado[df_filtrado["ANO_DE_CORTE"] == ano_corte_mas_reciente]["RAZON_SOCIAL"].unique().tolist()

if not empresas_disponibles:
    st.warning(f"No hay datos de empresas disponibles para el año {ano_corte_mas_reciente} en este filtro.")
    st.stop()

with col_sel_company:
    # --- MEJORA 2: Selector de Empresa por Nombre ---
    empresa_seleccionada = st.selectbox(
        "Selecciona la Empresa para predecir",
        empresas_disponibles
    )

st.info(f"Predicción para **{ano_prediccion}**, comparando contra el año de corte más reciente disponible: **{ano_corte_mas_reciente}**.")

# ... (El resto del código de predicción permanece igual) ...
