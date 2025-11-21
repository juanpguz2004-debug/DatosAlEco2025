import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime, timezone

# ----------------------------------------
# CONFIGURACIÓN STREAMLIT
# ----------------------------------------

st.set_page_config(
    page_title="Evaluación de Calidad – Guía 2025 MinTIC",
    layout="wide"
)


# ----------------------------------------
# API PRINCIPAL: DATOS + METADATOS
# ----------------------------------------

def fetch_api_data(url: str) -> pd.DataFrame:
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame()


def fetch_metadata_from_api(api_endpoint: str):
    """
    Recibe el endpoint oficial:
    Ejemplo: https://www.datos.gov.co/api/v3/views/uzcf-b9dh

    Devuelve un dict con:
    - categoria
    - tags
    - fecha metadatos
    - fecha datos
    - frecuencia actualización
    - licencia
    - publisher (entidad)
    - tema
    - descripción
    """
    try:
        resp = requests.get(api_endpoint)
        resp.raise_for_status()
        meta = resp.json()

        # Extracción segura
        categoria = meta.get("category", "")
        tags = meta.get("tags", [])
        descripcion = meta.get("description", "")

        fecha_meta = meta.get("metadataUpdatedAt", None)
        fecha_datos = meta.get("dataUpdatedAt", None)

        custom_fields = meta.get("metadata", {}).get("custom_fields", {})

        frecuencia = ""
        licencia = ""
        publisher = ""

        if "Información de Datos" in custom_fields:
            frecuencia = custom_fields["Información de Datos"].get(
                "Frecuencia de Actualización", "")

        if "Common Core" in custom_fields:
            licencia = custom_fields["Common Core"].get("License", "")
            publisher = custom_fields["Common Core"].get("Publisher", "")

        return {
            "categoria": categoria,
            "tags": tags,
            "descripcion": descripcion,
            "fecha_metadatos": fecha_meta,
            "fecha_datos": fecha_datos,
            "frecuencia": frecuencia,
            "licencia": licencia,
            "publisher": publisher,
        }
    except Exception as e:
        st.error(f"No se pudieron cargar metadatos: {e}")
        return None


# ----------------------------------------
# AUXILIARES DE ESTADÍSTICAS DEL DF
# ----------------------------------------

def df_stats(df):
    filas = len(df)
    columnas = len(df.columns)
    total_celdas = df.size
    total_nulos = df.isnull().sum().sum()

    col_con_mas_1pct = (df.isnull().sum() / max(filas, 1) > 0.01).sum()

    return {
        "filas": filas,
        "columnas": columnas,
        "total_celdas": total_celdas,
        "total_nulos": total_nulos,
        "col_con_mas_1pct": col_con_mas_1pct
    }


# ----------------------------------------
# CRITERIOS DE CALIDAD – GUÍA 2025
# ----------------------------------------

# 1. Accesibilidad
def criterio_accesibilidad(df, meta):
    # Si existen datos + metadatos → 100
    if df.empty or meta is None:
        return 0
    return 100


# 2. Actualidad – *CORREGIDA CON METADATOS*
def criterio_actualidad(meta):
    fecha_str = meta.get("fecha_datos", None)
    if not fecha_str:
        return 0

    try:
        fecha_update = datetime.fromisoformat(fecha_str.replace("Z", "+00:00"))
        hoy = datetime.now(timezone.utc)
        dias = (hoy - fecha_update).days

        # Si se actualiza hace <= 30 días → 100
        if dias <= 30:
            return 100
        elif dias <= 180:
            return 70
        elif dias <= 365:
            return 40
        else:
            return 0
    except:
        return 0


# 3. Completitud
def criterio_completitud(df, stats):
    if stats["total_celdas"] == 0:
        return 0

    comp_datos = 10 * (1 - (stats["total_nulos"] / stats["total_celdas"]) ** 1.5)

    comp_cols = 10 * (1 - (stats["col_con_mas_1pct"] / max(stats["columnas"], 1)) ** 2)

    cols_no_nulos = (df.isnull().sum() == 0).sum()
    comp_colnovacias = 10 * (cols_no_nulos / max(stats["columnas"], 1))

    return ((comp_datos + comp_cols + comp_colnovacias) / 3) * 10


# 4. Comprensibilidad
def criterio_comprensibilidad(df):
    col_ok = sum(1 for c in df.columns if len(c) < 30 and re.match(r"^[a-z0-9_]+$", c))
    return (col_ok / len(df.columns)) * 100


# 5. Conformidad – con licencia oficial
def criterio_conformidad(meta):
    lic = meta.get("licencia", "").lower()
    if "cc" in lic or "open" in lic:
        return 100
    elif lic != "":
        return 60
    return 0


# 6. Confidencialidad
def criterio_confidencialidad(df):
    sensibles = ["cedula", "identificación", "nombre", "apellido"]
    encontrados = [c for c in df.columns if any(s in c.lower() for s in sensibles)]
    if not encontrados:
        return 100
    return max(0, 100 - 20 * len(encontrados))


# 7. Consistencia
def criterio_consistencia(df):
    col_ok = 0
    for c in df.columns:
        try:
            unicos = df[c].nunique()
            if unicos >= 2:
                col_ok += 1
        except:
            pass
    return (col_ok / len(df.columns)) * 100


# 8. Credibilidad – usa publisher real
def criterio_credibilidad(meta):
    pub = meta.get("publisher", "")
    if pub:
        return 100
    return 40


# 9. Disponibilidad
def criterio_disponibilidad(accesibilidad, actualidad):
    return (accesibilidad + actualidad) / 2


# 10. Eficiencia
def criterio_eficiencia(unicidad, completitud):
    return (unicidad + completitud) / 2


# 11. Exactitud
def criterio_exactitud(df):
    return criterio_consistencia(df)


# 12. Portabilidad
def criterio_portabilidad(completitud, conformidad):
    return (0.50 * 100) + (0.25 * conformidad) + (0.25 * completitud)


# 13. Precisión
def criterio_precision(df):
    return criterio_consistencia(df)


# 14. Recuperabilidad – metadatos reales
def criterio_recuperabilidad(accesibilidad, meta):
    tiene_desc = meta.get("descripcion", "") != ""
    tiene_cat = meta.get("categoria", "") != ""
    puntos = accesibilidad + (50 if tiene_desc else 0) + (50 if tiene_cat else 0)
    return puntos / 3


# 15. Relevancia – *CORREGIDA*
def criterio_relevancia(meta, df):
    medida_categoria = 100 if meta.get("categoria") else 40
    medida_filas = 100 if len(df) >= 50 else 20
    return (medida_categoria + medida_filas) / 2


# 16. Trazabilidad
def criterio_trazabilidad(actualidad, credibilidad):
    return (actualidad + credibilidad) / 2


# 17. Unicidad
def criterio_unicidad(df):
    return (len(df.drop_duplicates()) / len(df)) * 100


# ----------------------------------------
# PROCESO PRINCIPAL
# ----------------------------------------

def evaluar(df, meta):
    stats = df_stats(df)

    acces = criterio_accesibilidad(df, meta)
    actu = criterio_actualidad(meta)
    comp = criterio_completitud(df, stats)
    compr = criterio_comprensibilidad(df)
    conf = criterio_conformidad(meta)
    confid = criterio_confidencialidad(df)
    cons = criterio_consistencia(df)
    cred = criterio_credibilidad(meta)
    disp = criterio_disponibilidad(acces, actu)
    unic = criterio_unicidad(df)
    ef = criterio_eficiencia(unic, comp)
    exact = criterio_exactitud(df)
    port = criterio_portabilidad(comp, conf)
    prec = criterio_precision(df)
    rec = criterio_recuperabilidad(acces, meta)
    rel = criterio_relevancia(meta, df)
    traz = criterio_trazabilidad(actu, cred)

    return {
        "Accesibilidad": acces,
        "Actualidad": actu,
        "Completitud": comp,
        "Comprensibilidad": compr,
        "Conformidad": conf,
        "Confidencialidad": confid,
        "Consistencia": cons,
        "Credibilidad": cred,
        "Disponibilidad": disp,
        "Eficiencia": ef,
        "Exactitud": exact,
        "Portabilidad": port,
        "Precisión": prec,
        "Recuperabilidad": rec,
        "Relevancia": rel,
        "Trazabilidad": traz,
        "Unicidad": unic
    }


# ----------------------------------------
# INTERFAZ STREAMLIT
# ----------------------------------------

def main():

    st.title("🟦 Evaluación de Calidad – Guía MinTIC 2025")
    st.caption("Versión con metadatos oficiales SODA 3.0")

    st.sidebar.header("Ingesta")
    modo = st.sidebar.radio(
        "Origen",
        ["Asset Inventory (API)", "CSV local"]
    )

    df = pd.DataFrame()
    meta = None

    if modo == "Asset Inventory (API)":
        dataset_id = st.sidebar.text_input("Dataset ID", "uzcf-b9dh")

        api_datos = f"https://www.datos.gov.co/resource/{dataset_id}.json?$limit=50000"
        api_meta = f"https://www.datos.gov.co/api/v3/views/{dataset_id}"

        if st.sidebar.button("Cargar"):
            df = fetch_api_data(api_datos)
            meta = fetch_metadata_from_api(api_meta)

    else:
        archivo = st.sidebar.file_uploader("Subir CSV", type=["csv"])
        if archivo:
            df = pd.read_csv(archivo)
            st.info("⚠️ En modo CSV no hay metadatos, varias métricas caerán.")
            meta = {
                "categoria": "",
                "tags": [],
                "descripcion": "",
                "fecha_datos": None,
                "licencia": "",
                "publisher": ""
            }

    if df.empty:
        st.warning("Carga datos para continuar.")
        return

    # EVALUACIÓN
    resultados = evaluar(df, meta)

    st.subheader("🔎 Resultado de los 17 criterios")
    cols = st.columns(4)

    for i, (k, v) in enumerate(resultados.items()):
        cols[i % 4].metric(k, f"{v:.2f}%")

    st.subheader("📌 Score Global")
    st.metric("Calidad Global", f"{np.mean(list(resultados.values())):.2f}%")

    st.subheader("Vista previa")
    st.dataframe(df.head())


main()
