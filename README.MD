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

### 3. IA Generativa con RAG (Retrieval-Augmented Generation)

Asistente virtual integrado utilizando Google Gemini 2.5 Flash.
Implementación RAG: Se utiliza inyección de contexto mediante una base de conocimiento (knowledge_base.txt) que contiene resúmenes estadísticos pre-calculados.
Limitación Técnica: El alcance del razonamiento complejo se limitó para optimizar el uso de tokens y ajustarse a las cuotas de facturación de Google AI Studio, priorizando respuestas basadas en métricas pre-existentes sobre cálculos en tiempo real.

Herramientas y Librerías 🛠️

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

##Repositorio: 
https://github.com/juanpguz2004-debug/DatosAlEco2025/tree/main
##Producto: 
https://datosaleco2025-49evhfdenjucpqj7appcxpg.streamlit.app/#resultados-del-diagnostico-rapido

Generado para el análisis de calidad de datos e inventarios.
