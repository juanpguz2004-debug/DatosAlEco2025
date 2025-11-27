
Documentación Técnica: Sistema de Análisis de Inventario de Datos

Esta documentación técnica detalla la arquitectura, las herramientas, la lógica matemática de las métricas, la implementación de Machine Learning y la integración de IA Generativa (RAG) utilizada en el Dashboard de Priorización de Activos de Datos.

1. Visión General del Proyecto

Este sistema es una aplicación web interactiva diseñada para auditar, clasificar y visualizar la calidad de un inventario de activos de datos. Combina reglas de negocio heurísticas (cálculo de riesgo), algoritmos de aprendizaje no supervisado (clustering y detección de anomalías) y un asistente de IA para la interpretación de hallazgos.

Limitaciones y Contexto del Despliegue

Debido a las restricciones del entorno de despliegue gratuito (Streamlit Community Cloud) y limitaciones de facturación en la API de Google AI Studio, se aplicaron las siguientes estrategias:

Pre-procesamiento de Datos:
Los datos crudos se limpian y reducen en un script externo (preprocess.py - implícito) para minimizar el consumo de memoria RAM en tiempo de ejecución. Se eliminaron columnas no prioritarias y se pre-calcularon ciertos metadatos.

RAG Optimizado (Context Window):
En lugar de una base de datos vectorial compleja, se utiliza un enfoque de inyección de contexto (Context Injection) mediante un archivo de conocimiento pre-generado (knowledge_base.txt) para mantenerse dentro de los límites de tokens y latencia gratuitos.


---

2. Stack Tecnológico

El proyecto está construido en Python 3.x utilizando las siguientes librerías clave:

Interfaz y Aplicación

Streamlit: Framework principal para la UI. Maneja el estado de la sesión (st.session_state), la caché de datos (st.cache_data) y la interactividad (filtros, tabs).

Procesamiento de Datos

Pandas: Carga de datos (CSV), manipulación de DataFrames, filtrado dinámico y agregaciones (GroupBys para KPIs).
NumPy: Operaciones vectorizadas para el cálculo de scores de riesgo (np.where, np.clip) para asegurar alto rendimiento.

Visualización

Plotly Express (px): Motor principal de gráficos interactivos (Scatter plots, Treemaps, Bar charts). Se priorizó sobre Matplotlib por su capacidad de tooltip y zoom nativo en web.
Matplotlib y Seaborn: Importados para configuraciones de estilo y cálculos auxiliares de color, aunque la renderización final se delega mayoritariamente a Plotly.

Machine Learning (Scikit-Learn)

sklearn.ensemble.IsolationForest: Detección de anomalías.
sklearn.cluster.KMeans: Segmentación de activos.
sklearn.preprocessing.StandardScaler: Normalización de datos para los modelos.

Inteligencia Artificial Generativa

Google GenAI SDK (google-genai): Conector con la API de modelos Gemini (versión gemini-2.5-flash) para el asistente chatbot.


---

3. Motor de Cálculo de Métricas y Riesgo

El núcleo del sistema es un motor de puntuación de riesgo híbrido que combina Riesgos Universales (calidad del dato) con Riesgos Avanzados (negocio y metadatos).

3.1. Fórmula de Riesgo Acumulativo

El Score de Prioridad de Riesgo (S_{riesgo}) se calcula sumando penalizaciones y se limita a un máximo teórico para normalizar la visualización.

MAX_TEORICO = 15.0

A. Riesgos Universales (S_{universal})

Penalizaciones básicas de calidad de datos:

Datos Incompletos: Si la densidad de datos por fila < 70% → +2.0 puntos
Inconsistencia de Tipo: Si una columna mezcla textos y números → +0.5 puntos por columna
Duplicidad: Si el registro es un duplicado exacto → +1.0 punto

B. Riesgos Avanzados (S_{avanzado})

Criterios heurísticos basados en metadatos y contexto:

Inconsistencia Metadatos: Riesgo alto pero antigüedad baja → +1.5
Anomalía Silenciosa: Detectado como anomalía por ML + Baja Popularidad → +1.0
Activos Vacíos Populares: Baja completitud en categorías Top 5 → +2.0
Confidencialidad: Activo Público sin descripción → +1.0
Trazabilidad: Sin dueño asignado → +1.5
Conformidad o Actualidad: Estado INCUMPLIMIENTO o antigüedad > 365 días → +2.0
Relevancia: Baja popularidad + Alto Riesgo → +1.0
Disponibilidad: Riesgo crítico → +1.5
Comprensibilidad: Alto riesgo + baja completitud → +1.0


---

3.2. Score de Calidad

Se presenta al usuario como un porcentaje inverso al riesgo.


---

4. Implementación de Machine Learning (ML)

Se utilizan algoritmos no supervisados para enriquecer el análisis manual.

4.1. Detección de Anomalías (Isolation Forest)

Objetivo: Identificar activos que se comportan de manera extraña comparados con el promedio del inventario.

Features: prioridad_riesgo_score, completitud_score, antiguedad_datos_dias, popularidad_score
Configuración: contamination = auto, n_estimators = 100
Salida: anomalia_score (-1 para anomalía, 1 para normal)
Uso: Las anomalías detectadas suman penalización en Riesgos Avanzados

4.2. Clustering de Calidad (K-Means)

Objetivo: Agrupar los activos en segmentos para facilitar decisiones de priorización.

Features: prioridad_riesgo_score (riesgo) y completitud_score (calidad)
Pre-procesamiento: StandardScaler (normalización)
Clusters (k = 3):

Completo / Riesgo Bajo: activos sanos
Aceptable / Mejora Necesaria: activos intermedios
Incompleto / Riesgo Alto: activos críticos que requieren acción inmediata

Lógica de Asignación:
Los centroides se ordenan dinámicamente restando Riesgo menos Completitud, asignando categorías y colores de manera estable entre ejecuciones.
Aquí tienes tu sección completa, totalmente limpia sin asteriscos ni numerales, integrada en formato Markdown, sin alterar contenido y lista para pegar directamente en tu documentación.


---

Arquitectura RAG (Retrieval-Augmented Generation)

Para el asistente Experto en Datos, se implementó una arquitectura RAG ligera optimizada para ambientes sin infraestructura vectorial ni almacenamiento persistente.

5.1. Fuente de Conocimiento

Se carga un archivo de texto plano llamado knowledge_base.txt.
Este archivo contiene:

Pre-cálculos

Resúmenes estadísticos

Descripciones de metadatos

Métricas consolidadas

Listados categorizados

Estadísticos esenciales generados durante el pre-procesamiento


El propósito es evitar que el modelo LLM tenga que recalcular promedios, percentiles o correlaciones sobre miles de filas en tiempo real, lo cual incrementa el riesgo de errores numéricos o alucinaciones matemáticas.

5.2. Inyección de Contexto (System Prompting)

En lugar de utilizar embeddings vectoriales, índices FAISS o bases de datos externas como ChromaDB o Pinecone, se opta por un enfoque minimalista utilizando System Prompt Injection.

El contenido completo de knowledge_base.txt se incrusta directamente dentro del prompt del sistema al momento de realizar la consulta.

Estructura del Prompt

Rol del Sistema:
Eres un Analista de Inventario de Datos experto, especializado en la calidad, riesgo y priorización de activos de datos corporativos.

Restricción Principal:
Responde única y exclusivamente basándote en la siguiente base de conocimiento. No inventes información adicional y no realices cálculos que no aparezcan explícitamente en ella.

Contexto:
[Contenido íntegro de knowledge_base.txt] disponible en el repositorio 

Instrucciones de Salida:

Citar elementos cuando corresponda

Mantener tono profesional y técnico

Responder únicamente con datos provenientes del archivo

Admitir explícitamente cuando una parte de la información no está contenida en la base de conocimiento


Este enfoque asegura trazabilidad, auditabilidad y evita alucinación del modelo.

5.3. Modelo

Se usa el modelo gemini-2.5-flash del SDK de Google GenAI.

Parámetros relevantes:

Temperatura: 0.1
Una temperatura muy baja prioriza precisión numérica y fidelidad al contexto.

Max Output Tokens: ajustado dinámicamente según carga del usuario

Top-k y Top-p: valores por defecto para minimizar la variabilidad


Este modelo se eligió por su velocidad, bajo costo y excelente rendimiento para tareas de análisis estructurado.



6. Generación de Reportes

El sistema incluye un generador de reportes HTML autocontenido que permite descargar análisis completos sin depender de infraestructura externa.

Características principales:

Genera un HTML autosuficiente con CSS y JavaScript embebidos en línea

No utiliza librerías externas para la conversión server-side, lo cual evita dependencias complejas en Streamlit Cloud

Plotly se carga a través de CDN directamente en el HTML generado

Las figuras de Plotly se exportan como divs HTML usando to_html

Los DataFrames se convierten en tablas HTML utilizando pandas.to_html

Los archivos generados se codifican en Base64 para permitir descarga directa mediante un href sin almacenamiento en disco del servidor

Todo el archivo es portable, reproducible y puede abrirse sin conexión



