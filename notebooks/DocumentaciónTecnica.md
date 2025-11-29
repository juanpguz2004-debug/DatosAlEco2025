
Documentación Técnica: Sistema de Análisis de Inventario de Datos

Esta documentación técnica detalla la arquitectura, las herramientas, la lógica matemática de las métricas, la implementación de Machine Learning y la integración de IA Generativa (RAG) utilizada en el Dashboard de Priorización de Activos de Datos.

1. Visión General del Proyecto

Este sistema es una aplicación web interactiva diseñada para auditar, clasificar y visualizar la calidad de un inventario de activos de datos. Combina reglas de negocio heurísticas (cálculo de riesgo), algoritmos de aprendizaje no supervisado (clustering y detección de anomalías) y un asistente de IA para la interpretación de hallazgos.

Limitaciones y Contexto del Despliegue

Debido a las restricciones del entorno de despliegue gratuito (Streamlit Community Cloud) y limitaciones de facturación en la API de Google AI Studio, se aplicaron las siguientes estrategias:

Pre-procesamiento de Datos:
Los datos crudos se limpian y reducen en un script externo (preprocess.py - implícito) para minimizar el consumo de memoria RAM en tiempo de ejecución. Se eliminaron columnas no prioritarias y se pre-calcularon ciertos metadatos.

RAG Dinámico en Tiempo Real (Dynamic Context Injection): A diferencia de los sistemas RAG tradicionales que dependen de bases de datos vectoriales estáticas o archivos de texto pre-generados, este sistema implementa un motor de contexto dinámico. El sistema genera un contexto de "memoria a corto plazo" basado exclusivamente en los datos que el usuario está visualizando en ese momento (filtrados por entidad, categoría, etc.), garantizando que las respuestas de la IA estén siempre sincronizadas con la vista del dashboard.


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
Google GenAI SDK: Conector con la API de modelos Gemini (versión gemini-2.0-flash) para el asistente chatbot, utilizando una temperatura baja (0.1) para maximizar la precisión analítica.


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


5. Arquitectura de Agente de IA (RAG Dinámico)

Para el asistente "Experto en Datos", se implementó una arquitectura de generación aumentada por recuperación (RAG) dinámica, diseñada para operar sobre datos filtrados en tiempo real.

5.1. Generación de Contexto Dinámico

En lugar de consultar una base de conocimiento estática, el sistema ejecuta la función generate_dynamic_context cada vez que el usuario realiza una pregunta. Este proceso:

Captura el Estado: Toma el DataFrame actual con los filtros activos aplicados por el usuario (por ejemplo, solo activos de una entidad específica).

Calcula KPIs al Vuelo: Genera estadísticas descriptivas instantáneas, incluyendo:

Promedios de riesgo y completitud de la vista actual.

Conteos de activos en incumplimiento y anomalías detectadas.

Top 5 de entidades o categorías con mayor riesgo.

Distribución por grupos de riesgo (Bajo, Medio, Alto, Crítico).

Construye el Prompt: Serializa estos cálculos en un bloque de texto estructurado (Markdown/JSON) que sirve como la "verdad absoluta" para el modelo.

5.2. Inyección de Contexto (System Prompting)

El contexto generado se inyecta en el System Instruction del modelo con instrucciones estrictas:

Rol: Analista de Inventario de Datos experto.

Restricción de Verdad: Responder única y exclusivamente basándose en el "CONTEXTO DE DATOS EN VIVO" suministrado.

Cero Alucinaciones: Si la respuesta no se puede derivar de las estadísticas suministradas, el modelo debe indicarlo honestamente.

5.3. Modelo y Configuración

Modelo: gemini-2.0-flash. Se eligió por su ventana de contexto amplia y su capacidad de razonamiento lógico superior sobre datos estructurados.

Temperatura: 0.1. Configuración casi determinista para asegurar que el modelo cite las cifras exactas calculadas por el motor de Python sin variaciones creativas.
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



