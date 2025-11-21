# --- FUNCIONES DE CÁLCULO DE LOS 17 CRITERIOS (FÓRMULAS OFICIALES) ---
# ... (otras funciones)

# Criterio 2. Actualidad (Lógica proxy de 90 días)
def calculate_actuality(df: pd.DataFrame) -> float:
    """Depende de Fecha de última actualización y Frecuencia declarada."""
    # ¡MODIFICA ESTA LÍNEA! Inserta el nombre REAL de la columna de fecha de tu dataset.
    date_column = '[COLUMNA_FECHA_REAL]' 
    
    if df.empty or date_column not in df.columns: return 0.0

    try:
        df_copy = df.copy()
        # Se usa .get para evitar errores si la columna no existe (aunque ya se chequeó)
        df_copy[date_column] = pd.to_datetime(df_copy.get(date_column), errors='coerce', utc=True)
        df_copy.dropna(subset=[date_column], inplace=True)
        
        if df_copy.empty: return 0.0 # Si todas las fechas eran inválidas

        three_months_ago = pd.Timestamp.now(tz='UTC') - pd.DateOffset(months=3)
        recent_count = df_copy[df_copy[date_column] >= three_months_ago].shape[0]
        
        return min(100.0, (recent_count / len(df)) * 100) 
    except Exception:
        # Esto captura cualquier otro error en el proceso de fechas
        return 0.0

# ... (otras funciones)

# Criterio 5. Conformidad (Lógica proxy de cumplimiento de estándares)
def calculate_conformity(df: pd.DataFrame) -> float:
    """Depende del cumplimiento de estándares, formatos y normativas."""
    # ¡MODIFICA ESTA LÍNEA! Inserta el nombre REAL de la columna del tipo de recurso.
    col = '[COLUMNA_TIPO_RECURSO_REAL]'
    
    if df.empty or col not in df.columns: return 0.0
    
    # La Conformidad es el % de filas donde el tipo de recurso está declarado
    conforming_rows = df[col].notna().sum()
    total_rows = len(df)
    return (conforming_rows / total_rows) * 100

# ... (otras funciones)
