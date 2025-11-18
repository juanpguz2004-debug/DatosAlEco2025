import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

st.set_page_config(page_title="Dashboard Inventario de Datos", layout="wide")
st.title("📊 Dashboard Inventario de Activos de Datos Públicos")

# =========================
# CARGA DEL CSV
# =========================
st.sidebar.header("Cargar archivo CSV")
file = st.sidebar.file_uploader("Selecciona el archivo CSV", type=["csv"]) 

if file:
    df = pd.read_csv(file, low_memory=False)

    # Limpiar columnas
    def clean_col(c):
        c = c.lower().strip()
        c = c.replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u")
        c = c.replace(" ", "_").replace(".", "").replace("/", "_").replace("(", "").replace(")", "")
        return c
    df.columns = [clean_col(c) for c in df.columns]

    # COMPLETITUD
    campos_minimos = [
        'titulo', 'descripcion', 'dueño', 'correo_electronico_de_contacto', 'licencia', 'dominio', 'categoria',
        'informacion_de_datos_frecuencia_de_actualizacion', 'common_core_public_access_level',
        'informacion_de_datos_cobertura_geografica'
    ]
    campos_exist = [c for c in campos_minimos if c in df.columns]
    df["campos_diligenciados"] = df[campos_exist].notna().sum(axis=1)
    df["completitud_score"] = (df["campos_diligenciados"] / len(campos_exist)) * 100

    # ANOMALÍAS
    if "antiguedad_datos_dias" in df.columns and "popularidad_score" in df.columns:
        df_model = df[(df["antiguedad_datos_dias"] < 9999) & (df["popularidad_score"] > 0)]
        if not df_model.empty:
            features = df_model[["antiguedad_datos_dias", "popularidad_score", "completitud_score"]]
            iso = IsolationForest(contamination=0.01, random_state=42)
            iso.fit(features)
            df["anomalia_score"] = 0
            df.loc[features.index, "anomalia_score"] = iso.predict(features)
            df.loc[features.index, "anomalia_distancia"] = iso.decision_function(features)
        else:
            df["anomalia_score"] = 0
            df["anomalia_distancia"] = 0

    # RIESGO
    df["riesgo_incumplimiento"] = np.where(df.get("estado_actualizacion") == "🔴 INCUMPLIMIENTO", 3, 0)
    df["riesgo_completitud"] = np.where(df["completitud_score"] < 50, 1.5, 0)
    if "popularidad_score" in df.columns:
        max_pop = df["popularidad_score"].max()
        df["riesgo_demanda"] = (df["popularidad_score"] / max_pop) if max_pop > 0 else 0
    else:
        df["riesgo_demanda"] = 0
    df["riesgo_anomalia"] = np.where(df.get("anomalia_score") == -1, 2, 0)
    df["prioridad_riesgo_score"] = (
        df["riesgo_incumplimiento"] + df["riesgo_completitud"] + df["riesgo_demanda"] + df["riesgo_anomalia"]
    )

    # FILTRADO PUBLICO
    if "publico" in df.columns:
        df_publico = df[df["publico"].astype(str).str.lower().eq("public")]
    else:
        df_publico = df

    st.subheader("📁 Vista general del dataset")
    st.dataframe(df.head(50))

    st.subheader("📈 Análisis visual")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Prioridad vs Antigüedad")
        if "antiguedad_datos_dias" in df_publico.columns:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.scatterplot(
                data=df_publico,
                x="antiguedad_datos_dias",
                y="prioridad_riesgo_score",
                hue="estado_actualizacion",
                alpha=0.6,
                ax=ax
            )
            st.pyplot(fig)
        else:
            st.write("No existe la columna 'antiguedad_datos_dias'")

    with col2:
        st.markdown("### Categorías más frecuentes")
        if "categoria" in df_publico.columns:
            top_cat = df_publico["categoria"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(x=top_cat.values, y=top_cat.index, ax=ax)
            st.pyplot(fig)
        else:
            st.write("No existe la columna 'categoria'")

    st.subheader("🏛 Entidades con mayor incumplimiento")
    if "dueño" in df.columns and "estado_actualizacion" in df.columns:
        res = df_publico.groupby("dueño").agg(
            total=("uid","count") if "uid" in df.columns else ("titulo","count"),
            incumple=("estado_actualizacion", lambda x: (x=="🔴 INCUMPLIMIENTO").sum()),
        ).reset_index()
        res["porcentaje_incumplimiento"] = (res["incumple"] / res["total"]) * 100
        top_ent = res.sort_values("porcentaje_incumplimiento", ascending=False).head(10)
        st.dataframe(top_ent)
    else:
        st.write("Faltan columnas 'dueño' o 'estado_actualizacion'")

else:
    st.info("Sube un archivo CSV para comenzar.")
