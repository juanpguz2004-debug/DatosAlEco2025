# --- 0) FUNCIONES DE UTILIDAD ---

import numpy as np
import unicodedata


# 1. LabelEncoder seguro (corregido)
def safe_le_transform(encoder, val):
    """Transforma un valor usando LabelEncoder, devolviendo -1 si es desconocido."""
    s = str(val)
    try:
        # Intentar transformar directamente
        return int(encoder.transform([s])[0])
    except ValueError:
        # Valor no visto en entrenamiento
        return -1


# 2. Mantener sin cambios (el modelo lo requiere)
def format_ano(year):
    """Convierte el año 2024 a '2,024' para la codificación OHE."""
    year_str = str(year)
    if len(year_str) == 4:
        return f'{year_str[0]},{year_str[1:]}' 
    return year_str


# 3. Normalizador de columnas (corregido)
def normalize_col(col):
    """
    Normaliza nombres de columnas:
    - Quita acentos correctamente
    - Pasa a mayúsculas
    - Reemplaza espacios por _
    - Elimina paréntesis
    - Normaliza Ñ → N
    """
    # Quitar acentos primero
    col = ''.join(
        c for c in unicodedata.normalize('NFD', col)
        if unicodedata.category(c) != 'Mn'
    )

    # Transformaciones estándar
    col = col.strip().upper()
    col = col.replace("Ñ", "N")
    col = col.replace(" ", "_")
    col = col.replace("(", "").replace(")", "")

    return col
