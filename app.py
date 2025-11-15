# ==========================================
# 6️⃣ Predicciones (ajustado a columnas reales)
# ==========================================
st.subheader("Predicciones de Ganancias")

# Índice de empresa
max_index = len(df_filtered) - 1
empresa_idx = st.number_input(
    f"Selecciona índice de empresa para predecir ganancia (0 a {max_index})", 
    min_value=0, 
    max_value=max_index, 
    value=0
)

# Columnas EXACTAS que el modelo usa
feature_columns = [
    "TOTAL_PATRIMONIO",
    "TOTAL_ACTIVOS",
    "TOTAL_PASIVOS",
    "INGRESOS_OPERACIONALES",
    "RAZON_SOCIAL",
    "NIT",
    "CIIU",
    "ANO_DE_CORTE",
    "MACROSECTOR",
    "SUPERVISOR",
    "CIUDAD_DOMICILIO",
    "REGION",
    "DEPARTAMENTO_DOMICILIO"
]

# Verificar columnas faltantes
missing = [c for c in feature_columns if c not in df_filtered.columns]
if missing:
    st.error(f"FALTAN COLUMNAS QUE EL MODELO NECESITA: {missing}")
    st.stop()

# Crear X_pred EXACTO como fue entrenado
X_pred = df_filtered[feature_columns]

# Realizar predicción
try:
    pred_value = model.predict(X_pred.iloc[[empresa_idx]])[0]
    st.write(
        f"Predicción de ganancia para la empresa **{df_filtered.iloc[empresa_idx]['RAZON_SOCIAL']}**: "
        f"**${pred_value:,.2f}**"
    )
except Exception as e:
    st.error(f"Error al realizar la predicción: {e}")
