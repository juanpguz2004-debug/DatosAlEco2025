import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# 1. CARGA DEL MODELO
# =========================
@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")

model = load_model()

# =========================
# 2. FUNCIÓN LIMPIEZA COLUMNS
# =========================
def normalizar_columnas(df):
    df.columns = (
        df.columns.str.strip()
        .str.upper()
        .str.normalize('NFKD')
        .str.encode('ascii', errors='ignore')
        .str.decode('utf-8')
        .str.replace(" ", "_")
        .str.replace("Á", "A")
        .str.replace("É", "E")
        .str.replace("Í", "I")
        .str.replace("Ó", "O")
        .str.replace("Ú", "U")
        .str.replace("Ñ", "N")
        .str.replace(r"[^A-Za-z0-9_]", "", regex=True)
    )
    return df

# =========================
# 3. FUNCIÓN DE VALIDACIÓN
# =========================
def validar_columnas(df, requeridas):
    faltantes = [c for c in requeridas if c not in df.columns]
    return faltantes

# =========================
# 4. APP
# =========================
st.title("📊 Dashboard ALECO 2025 - Predicciones Mejoradas")

archivo = st.file_uploader("Sube el dataset", type=["csv"])

if archivo:
    df = pd.read_csv(archivo, encoding="utf-8")
    df = normalizar_columnas(df)

    st.write("### Columnas normalizadas detectadas")
    st.write(df.columns.tolist())
    st.write("Filas:", len(df))

    # Columnas necesarias para predicción
    columnas_requeridas = [
        "NIT","RAZON_SOCIAL","SUPERVISOR","REGION","DEPARTAMENTO_DOMICILIO",
        "CIUDAD_DOMICILIO","CIIU","MACROSECTOR","INGRESOS_OPERACIONALES",
        "GANANCIA_PERDIDA","TOTAL_ACTIVOS","TOTAL_PASIVOS",
        "TOTAL_PATRIMONIO","ANO_DE_CORTE"
    ]

    faltantes = validar_columnas(df, columnas_requeridas)
    if len(faltantes) > 0:
        st.error(f"Faltan columnas necesarias: {faltantes}")
        st.stop()

    # Limpiar numéricos
    for col in ["INGRESOS_OPERACIONALES", "TOTAL_ACTIVOS", "TOTAL_PASIVOS", "TOTAL_PATRIMONIO", "GANANCIA_PERDIDA"]:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace(" ", "", regex=False)
            .astype(float)
        )

    st.success("Dataset cargado correctamente ✔")

    # =========================
    # 5. LISTA DESPLEGABLE DE EMPRESAS
    # =========================
    empresas_lista = df["RAZON_SOCIAL"].unique()
    empresa_seleccionada = st.selectbox("Selecciona una empresa:", empresas_lista)

    df_empresa = df[df["RAZON_SOCIAL"] == empresa_seleccionada]

    # Mostrar filas históricas
    st.write("### Historial de la empresa seleccionada")
    st.dataframe(df_empresa)

    # =========================
    # 6. SELECCIONAR AÑO PARA PREDECIR
    # =========================
    ultimo_anio = int(df["ANO_DE_CORTE"].max())
    anio_pred = st.number_input(
        "Año para predecir (posterior al último año del dataset)",
        min_value=ultimo_anio + 1,
        max_value=ultimo_anio + 10,
        value=ultimo_anio + 1
    )

    # =========================
    # 7. PREPARAR INPUT PARA EL MODELO
    # =========================
    if st.button("🔮 Generar Predicción"):

        try:
            # Tomamos el registro MÁS RECIENTE de esa empresa
            registro_actual = df_empresa.sort_values("ANO_DE_CORTE").iloc[-1].copy()

            # Cambiamos el año al seleccionado
            registro_actual["ANO_DE_CORTE"] = anio_pred

            # Guardamos GANANCIA_PERDIDA actual para comparar
            ganancia_anterior = df_empresa.sort_values("ANO_DE_CORTE").iloc[-1]["GANANCIA_PERDIDA"]

            # Eliminamos GANANCIA para predicción
            registro_actual = registro_actual.drop(labels=["GANANCIA_PERDIDA"])

            # Convertir en DataFrame (1 fila)
            df_pred = pd.DataFrame([registro_actual])

            # =========================
            # 8. APLICAR ONE-HOT *CONSISTENTE*
            # =========================
            df_pred = pd.get_dummies(df_pred)
            df_model_cols = model.get_booster().feature_names

            # Alinear columnas del modelo
            for col in df_model_cols:
                if col not in df_pred.columns:
                    df_pred[col] = 0

            df_pred = df_pred[df_model_cols]

            # =========================
            # 9. PREDICCIÓN
            # =========================
            prediccion = model.predict(df_pred)[0]

            diferencia = prediccion - ganancia_anterior

            # =========================
            # 10. RESULTADOS
            # =========================
            st.success("Predicción generada con éxito ✔")

            st.subheader("📌 Resultados de la predicción")
            st.write(f"**Empresa:** {empresa_seleccionada}")
            st.write(f"**Predicción Año {anio_pred}:** ${prediccion:,.0f}")
            st.write(f"**Año anterior ({int(df_empresa['ANO_DE_CORTE'].max())}):** ${ganancia_anterior:,.0f}")

            if diferencia >= 0:
                st.success(f"▶ Variación: +${diferencia:,.0f} (Mejora)")
            else:
                st.error(f"▼ Variación: {diferencia:,.0f} (Caída)")

        except Exception as e:
            st.error(f"Error generando predicción: {str(e)}")

