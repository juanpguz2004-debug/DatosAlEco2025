EJECUTAR EN UNA CELDA DE COLAB:
print("Iniciando pre-cálculo de datos para Streamlit.")

try:
    # 1. PEDIR AL USUARIO QUE SUBA EL ARCHIVO
    print("Por favor, sube el archivo CSV original (ej. Asset_Inventory_-_Public_20251118.csv)")
    uploaded = files.upload() # Esto abrirá el widget de subida de archivos

    # 2. LEER EL ARCHIVO SUBIDO
    if not uploaded:
        print("❌ ERROR: No se subió ningún archivo. Proceso cancelado.")
    else:
        original_filename = next(iter(uploaded))
        data_io = io.BytesIO(uploaded[original_filename])

        print(f"Archivo subido: {original_filename}")

        # --- DETECCIÓN AUTOMÁTICA DE DELIMITADOR (CORRECCIÓN CLAVE) ---
        try:
            # Intenta primero con la coma (default)
            data_original = pd.read_csv(data_io, low_memory=False)
            if len(data_original.columns) <= 1:
                # Si solo hay 0 o 1 columna, probablemente es el delimitador incorrecto. Reinicia el buffer.
                data_io.seek(0)
                data_original = pd.read_csv(data_io, low_memory=False, sep=';')
                print("  ✅ Aviso: Usando delimitador de punto y coma (;).")
                if len(data_original.columns) <= 1:
                    # Si sigue siendo 0 o 1 columna, vuelve a reiniciar el buffer y prueba con tab.
                    data_io.seek(0)
                    data_original = pd.read_csv(data_io, low_memory=False, sep='\t')
                    print("  ✅ Aviso: Usando delimitador de tabulación (\t).")
                    if len(data_original.columns) <= 1:
                        print("  ❌ ERROR: No se detectaron múltiples columnas con delimitadores comunes.")
                        raise ValueError("El archivo de origen no tiene un formato tabular reconocido (CSV, Semicolon, Tab).")
            else:
                print("  ✅ Aviso: Usando delimitador de coma (,).")

            print(f"Datos cargados. Total de filas: {len(data_original)}")

            # 3. Procesar datos
            data_procesada = process_data(data_original.copy())

            # --- VERIFICACIÓN DE SEGURIDAD ANTES DE GUARDAR ---
            if data_procesada.empty:
                raise Exception("El DataFrame procesado está vacío (0 filas).")

            # 4. Guardar el archivo procesado en el directorio de Colab
            data_procesada.to_csv(ARCHIVO_PROCESADO, index=False)

            print("\n✅ ¡Pre-cálculo completado exitosamente!")
            print(f"El archivo optimizado ({ARCHIVO_PROCESADO}) se ha guardado en el entorno de Colab.")

            # Opcional: Descargar el archivo procesado para subirlo a Streamlit
            print("Descargando el archivo procesado para su uso en Streamlit Cloud...")
            files.download(ARCHIVO_PROCESADO)

        # Este 'except' cierra el bloque 'try' de la lectura con detección automática.
        except Exception as e_lectura:
             print(f"\n❌ ERROR de lectura/detección: {e_lectura}")

    # Este 'except' cierra el bloque 'try' principal del script.
except Exception as e:
    print(f"\n❌ ERROR FATAL durante el pre-cálculo: {e}")
