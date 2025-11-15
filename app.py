# =========================================================
#   PREDICCIÓN — VERSION CORREGIDA
# =========================================================

st.subheader("🔮 Predicción de Ganancia/Pérdida")

if model is not None:

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

    with st.form("pred_form"):
        ingresos = st.number_input("Ingresos operacionales", min_value=0.0)
        activos = st.number_input("Total activos", min_value=0.0)
        pasivos = st.number_input("Total pasivos", min_value=0.0)
        patrimonio = st.number_input("Total patrimonio", min_value=0.0)

        submit = st.form_submit_button("Predecir")

        if submit:
            try:
                # Construir dataframe EXACTAMENTE como el entrenamiento
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

                # 🔥 Reordenar columnas como en el modelo
                X = X[columnas_modelo]

                # 🔥 Sin dummies → XGBoost maneja strings internamente
                pred = model.predict(X)[0]

                st.success(f"Predicción estimada: {pred:,.2f} millones")

            except Exception as e:
                st.error(f"Error generando predicción: {e}")

