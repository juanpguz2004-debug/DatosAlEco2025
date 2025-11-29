## 📊 DataSentinel

DataSentinel es una plataforma analítica diseñada para auditar, clasificar y visualizar la calidad de inventarios de datos corporativos. Combina un motor de reglas heurísticas, algoritmos de aprendizaje no supervisado (Machine Learning) y un asistente de IA Generativa (RAG) para identificar activos críticos, obsoletos o en riesgo.

Nota de Implementación: Debido a las restricciones de memoria en el entorno gratuito de Streamlit Community Cloud, los datos son pre-procesados externamente. Se utiliza la totalidad del dataset original, pero se han recortado columnas no prioritarias y pre-calculado métricas pesadas para optimizar el rendimiento en tiempo real.

## Datos y Pre-procesamiento 📋

El sistema ingesta un inventario de activos procesado (Asset_Inventory_PROCESSED.csv) que incluye metadatos clave:

Metadatos Descriptivos: Título, Descripción, Dueño/Entidad, Categoría, Tema.
Métricas de Calidad: Score de Completitud, Antigüedad del dato (días), Nivel de Acceso (Público/Privado).
Optimizaciones: Limpieza de tipos de datos y reducción de dimensionalidad ejecutada previamente para garantizar la estabilidad del despliegue en la nube.

## Métodos de Inteligencia Artificial y Analítica 🧠

El proyecto implementa una arquitectura híbrida de análisis:

### 1. Motor de Riesgo Heurístico (Reglas de Negocio)

Cálculo determinista de un Score de Prioridad de Riesgo (Max 15.0) basado en penalizaciones acumulativas:
Riesgos Universales: Datos incompletos, duplicidad exacta e inconsistencia de tipos.
Riesgos Avanzados: Inconsistencia de metadatos, trazabilidad (falta de dueño), confidencialidad y obsolescencia (>1 año).

### 2. Machine Learning (Scikit-Learn)

Algoritmos de aprendizaje no supervisado para segmentación y detección:
K-Means Clustering: Segmentación de activos en 3 grupos (Riesgo Alto, Medio, Bajo) basado en la relación Riesgo vs. Completitud.
Isolation Forest: Detección de anomalías estadísticas ("outliers") en activos que presentan comportamientos inusuales en sus métricas de calidad.

## 3. IA Generativa con RAG Dinámico (Retrieval-Augmented Generation)
Asistente virtual integrado utilizando Google Gemini 2.0 Flash.

Implementación RAG Dinámico: A diferencia de los sistemas tradicionales estáticos, este sistema utiliza inyección de contexto en tiempo real. Se genera una "memoria a corto plazo" calculando estadísticas y KPIs basados exclusivamente en los filtros activos del usuario (por ejemplo, filtrado por entidad o categoría).

Ventaja Técnica: Esto garantiza que el asistente responda con precisión sobre la "vista actual" de los datos, eliminando alucinaciones y evitando la necesidad de bases de conocimiento estáticas desactualizadas.

## Herramientas y Librerías 🛠️

El proyecto fue construido utilizando el siguiente stack tecnológico en Python:

Interfaz Web: streamlit (Dashboard interactivo).
Manipulación de Datos: pandas, numpy.
Visualización: plotly (Gráficos interactivos), matplotlib, seaborn (Estilos y mapas de color).
Machine Learning: scikit-learn (Modelos KMeans e IsolationForest).
GenAI / LLM: google-genai (Cliente API para modelos Gemini).

## Información Adicional

Estado del Proyecto: Desplegado (Versión Optimizada para Cloud Gratuito).

Autores:
Juan Pablo Guzman Molano,
Karol Astrid Daza Gomez,
Luis Mateo Guagua Cabezas

## Repositorio: 
https://github.com/juanpguz2004-debug/DatosAlEco2025/tree/main
## Producto: 
https://datosaleco2025-49evhfdenjucpqj7appcxpg.streamlit.app/#resultados-del-diagnostico-rapido

Generado para el análisis de calidad de datos e inventarios.
