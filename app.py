# =========================================================
#   6️⃣ PREDICCIÓN — VERSION DEFINITIVA Y FUNCIONAL
# =========================================================

st.subheader("🔮 Predicción de Ganancia/Pérdida")

# Listado de columnas del modelo (orden exacto)
columnas_modelo = [
    "NIT",
    "RAZON_SOCIAL",
    "SUPERVISOR",
    "REGION",
    "DEPARTAMENTO_DOMICILIO",
    "CIUDAD_DOMICILIO",
    "CIIU",
    "MACROSECTOR",
    "INGRESOS_OPERACIONALES",
    "TOTAL_ACTIVOS",
    "TOTAL_PASIVOS",
    "TOTAL_PATRIMONIO",
    "ANO_DE_CORTE"
]

# Selección de valores categóricos basados en el propio dataset
regiones = df["REGION"].dropna().unique().tolist()
macrosector = df["MACROSECTOR"].dropna().unique().tolist()

with st.form("pred_form"):
    st.write("Ingresa los datos para generar la predicción")

    region_sel = st.selectbox("Región", regiones)
    macro_sel = st.selectbox("Macrosector", macrosector)

    ingresos = st.number_input("Ingresos operacionales", min_value=0.0)
    activos = st.number_input("Total activos", min_value=0.0)
    pasivos = st.number_input("Total pasivos", min_value=0.0)
    patrimonio = st.number_input("Total patrimonio", min_value=0.0)

    enviar = st.form_submit_button("Predecir")

if enviar:

    try:
        # Construimos el DataFrame EXACTAMENTE igual al modelo
        X = pd.DataFrame([{
            "NIT": "0",
            "RAZON_SOCIAL": "DESCONOCIDA",
            "SUPERVISOR": "NO APLICA",
            "REGION": region_sel,
            "DEPARTAMENTO_DOMICILIO": "NO APLICA",
            "CIUDAD_DOMICILIO": "NO APLICA",
            "CIIU": "0000",
            "MACROSECTOR": macro_sel,
            "INGRESOS_OPERACIONALES": ingresos,
            "TOTAL_ACTIVOS": activos,
            "TOTAL_PASIVOS": pasivos,
            "TOTAL_PATRIMONIO": patrimonio,
            "ANO_DE_CORTE": 2025
        }])

        # Reordenar columnas como espera el modelo
        X = X[columnas_modelo]

        # Predecir
        pred = model.predict(X)[0]

        st.success(f"Predicción estimada de ganancia/pérdida: **${pred:,.2f}**")

    except Exception as e:
        st.error(f"Error generando predicción: {e}")
