# --- 0) FUNCIONES DE UTILIDAD ---
# Función de Label Encoder segura (Sugerencia 4)
def safe_le_transform(encoder, val):
    """Transforma un valor usando LabelEncoder, devolviendo -1 si es desconocido."""
    s = str(val)
    # Verifica si la clase fue vista en el entrenamiento
    if s in encoder.classes_:
        # Retorna el índice de la clase (el valor codificado)
        return int(np.where(encoder.classes_ == s)[0][0])
    # Retorna -1 si el valor no fue visto en el entrenamiento
    return -1

# Función de formato de año (para OHE, mantenida por si es requerida por el modelo)
def format_ano(year):
    """Convierte el año 2024 a '2,024' para la codificación OHE."""
    year_str = str(year)
    if len(year_str) == 4:
        return f'{year_str[0]},{year_str[1:]}' 
    return year_str

# Función de normalización de columna (mantener tu original si es la que usaste para entrenar)
def normalize_col(col):
    col = col.strip().upper().replace(" ", "_").replace("(", "").replace(")", "").replace("Ñ", "N")
    return ''.join(c for c in unicodedata.normalize('NFD', col) if unicodedata.category(c) != 'Mn')
