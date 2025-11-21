import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

# ===============================================================
# CONFIGURACIÓN STREAMLIT
# ===============================================================

st.set_page_config(
    page_title="Evaluación de Calidad – Guía MinTIC 2025",
    layout="wide"
)

# ===============================================================
# UTILIDADES
# ===============================================================

def safe_get(url, headers=None, timeout=15):
    """Wrapper seguro para requests.get con manejo de errores."""
    try:
        r = requests.get(url, headers=headers or {}, timeout=timeout)
        if r.status_code == 200:
            return r
        return None
    except:
        return None


# ===============================================================
# DESCARGA DE DATOS
# ===============================================================

def fetch_resource_data(dataset_id: str) -> pd.DataFrame:
    """Descarga datos desde /resource/{id}.json."""
    url = f"https://www.datos.gov.co/resource/{dataset_id}.json?$limit=50000"

    headers = {}
    if "APP_TOKEN" in st.secrets:
        headers["X-App-Token"] = st.secrets["APP_TOKEN"]

    r = safe_get(url, headers)
    if r is None:
        st.error("No se pudo descargar datos del recurso.")
        return pd.DataFrame()

    try:
        return pd.DataFrame(r.json())
    except:
        return pd.DataFrame()

# ===============================================================
# EXTRACCIÓN REAL DE FECHAS DESDE COLUMNAS
# ===============================================================

def extract_real_timestamps(meta_json: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Socrata NO coloca last_metadata_updated_date ni last_data_updated_date
    como campos del recurso.
    
    → Están dentro de meta["columns"] como columnas virtuales.
    → Su valor aparece en meta["columns"][i]["cachedContents"]["top"][0]["item"]
    """

    fecha_creacion = None
    fecha_metadatos = None
    fecha_datos = None

    for col in meta_json.get("columns", []):
        name = col.get("name", "").lower()

        # Intentamos extraer valor ejemplo
        ejemplo = None
        try:
            ejemplo = col["cachedContents"]["top"][0]["item"]
        except:
            ejemplo = None

        if name == "creation_date":
            fecha_creacion = ejemplo

        elif name in ["last_metadata_updated_date", "last_metadata_updated"]:
            fecha_metadatos = ejemplo

        elif name in ["last_data_updated_date", "last_data_updated"]:
            fecha_datos = ejemplo

    return {
        "fecha_creacion_real": fecha_creacion,
        "fecha_metadatos_real": fecha_metadatos,
        "fecha_datos_real": fecha_datos
    }


# ===============================================================
# DESCARGA DIRECTA DE METADATOS
# ===============================================================

def fetch_metadata_from_view(view_id: str) -> Optional[Dict[str, Any]]:
    """
    Descarga metadatos usando /api/v3/views/{id}
    En datos.gov.co el resourceID == viewID.
    """
    url = f"https://www.datos.gov.co/api/v3/views/{view_id}"

    headers = {}
    if "APP_TOKEN" in st.secrets:
        headers["X-App-Token"] = st.secrets["APP_TOKEN"]

    r = safe_get(url, headers)
    if r is None:
        return None

    meta = r.json()

    # Extraer valores desde metadata clásico (si existen)
    categoria = meta.get("category", "")
    tags = meta.get("tags", [])

    # Extraer campos custom
    custom = meta.get("metadata", {}).get("custom_fields", {})
    frecuencia = custom.get("Información de Datos", {}).get("Frecuencia de Actualización", "")
    licencia = custom.get("Common Core", {}).get("License", "")
    publisher = custom.get("Common Core", {}).get("Publisher", "")

    # EXTRAER FECHAS REALES DESDE COLUMNS
    fechas_reales = extract_real_timestamps(meta)

    return {
        "categoria": categoria,
        "tags": tags,
        "frecuencia": frecuencia,
        "licencia": licencia,
        "publisher": publisher,

        # nuevas fechas reales:
        "fecha_creacion": fechas_reales["fecha_creacion_real"],
        "fecha_metadatos": fechas_reales["fecha_metadatos_real"],
        "fecha_datos": fechas_reales["fecha_datos_real"],
    }


# ===============================================================
# FUNCIONES AUXILIARES
# ===============================================================

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

# ===============================================================
# CRITERIOS 0–10
# ===============================================================

def c_accesibilidad(df, meta): return 10 if not df.empty else 0

def parse_fecha(f):
    if not f:
        return None
    try:
        return datetime.fromisoformat(f.replace("Z", "+00:00"))
    except:
        return None

def c_actualidad(meta):
    fecha_str = meta.get("fecha_datos")
    fecha = parse_fecha(fecha_str)
    if not fecha:
        return 0
    dias = (datetime.now(timezone.utc) - fecha).days
    if dias <= 30: return 10
    if dias <= 180: return 7
    if dias <= 365: return 4
    return 1

def c_completitud(df, stats):
    if stats["total_celdas"] == 0: return 0
    comp_datos = 10 * (1 - (stats["total_nulos"]/stats["total_celdas"])**1.5)
    comp_cols = 10 * (1 - (stats["col_con_mas_1pct"]/stats["columnas"])**2)
    cols_no_nulos = (df.isnull().sum() == 0).sum()
    comp_colnovacias = 10 * (cols_no_nulos / stats["columnas"])
    return (comp_datos + comp_cols + comp_colnovacias) / 3

def c_comprensibilidad(df):
    ok = sum(1 for c in df.columns if len(c)<30 and re.match(r"^[a-z0-9_]+$", c))
    return 10*(ok/len(df.columns))

def c_conformidad(meta):
    lic = meta.get("licencia","").lower()
    if "cc" in lic: return 10
    if lic: return 6
    return 0

def c_confidencialidad(df):
    sensibles=["cedula","identificacion","nombre","apellido"]
    encontrados=[c for c in df.columns if any(s in c.lower() for s in sensibles)]
    if not encontrados: return 10
    return max(0,10-2*len(encontrados))

def c_consistencia(df):
    ok=sum(1 for c in df.columns if df[c].nunique()>=2)
    return 10*(ok/len(df.columns))

def c_credibilidad(meta):
    return 10 if meta.get("publisher") else 4

def c_disponibilidad(acc, act):
    return (acc+act)/2

def c_unicidad(df):
    return 10*(len(df.drop_duplicates())/len(df))

def c_eficiencia(unic, comp):
    return (unic+comp)/2

def c_exactitud(df):
    return c_consistencia(df)

def c_portabilidad(comp, conf):
    return 5+0.25*conf+0.25*comp

def c_precision(df):
    return c_consistencia(df)

def c_recuperabilidad(meta, acc):
    tiene = bool(meta.get("categoria"))
    return (acc+(5 if tiene else 0))/2

def c_relevancia(meta, df):
    cat = 10 if meta.get("categoria") else 4
    filas = 10 if len(df)>=50 else 2
    return (cat+filas)/2

def c_trazabilidad(act, cred):
    return (act+cred)/2


# ===============================================================
# EVALUACIÓN PRINCIPAL
# ===============================================================

def evaluar(df, meta):
    stats=df_stats(df)

    acc=c_accesibilidad(df,meta)
    act=c_actualidad(meta)
    comp=c_completitud(df,stats)
    compr=c_comprensibilidad(df)
    conf=c_conformidad(meta)
    confid=c_confidencialidad(df)
    cons=c_consistencia(df)
    cred=c_credibilidad(meta)
    disp=c_disponibilidad(acc,act)
    unic=c_unicidad(df)
    ef=c_eficiencia(unic,comp)
    exact=c_exactitud(df)
    port=c_portabilidad(comp,conf)
    prec=c_precision(df)
    rec=c_recuperabilidad(meta,acc)
    rel=c_relevancia(meta,df)
    traz=c_trazabilidad(act,cred)

    return {
        "Accesibilidad": acc,
        "Actualidad": act,
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


# ===============================================================
# INTERFAZ STREAMLIT
# ===============================================================

def main():
    st.title("Evaluación de Calidad – Guía MinTIC 2025 (0–10 → % visual)")

    dataset_id = st.sidebar.text_input("Dataset ID (resource)", "uzcf-b9dh")

    if st.sidebar.button("Cargar"):

        view_id = dataset_id
        st.success(f"Usando View ID directamente: {view_id}")

        meta = fetch_metadata_from_view(view_id)
        if meta is None:
            st.error("No se pudieron cargar metadatos.")
            meta = {
                "categoria":"",
                "tags":[],
                "fecha_creacion":None,
                "fecha_metadatos":None,
                "fecha_datos":None,
                "frecuencia":"",
                "licencia":"",
                "publisher":"",
            }

        df = fetch_resource_data(dataset_id)
        if df.empty:
            st.error("No se pudo cargar datos del dataset.")
            return

        st.write(f"Filas: {len(df)} — Columnas: {len(df.columns)}")

        resultados = evaluar(df, meta)

        st.subheader("Criterios (en %)")
        cols = st.columns(4)
        for i,(k,v) in enumerate(resultados.items()):
            cols[i%4].metric(k, f"{v*10:.2f}%")

        st.subheader("Score global")
        total=np.mean(list(resultados.values()))
        st.metric("Puntaje total", f"{total*10:.2f}%", f"({total:.2f}/10)")

        st.subheader("Vista previa de datos")
        st.dataframe(df.head())

        st.subheader("Metadatos detectados (incluye fechas reales)")
        st.json(meta)

main()
