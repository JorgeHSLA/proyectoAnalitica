# Proyecto de Predicción de Precios de Arriendo en Bogotá

## Tabla de Contenidos
- [7. Metodología](#7-metodología)
- [8. Implementación](#8-implementación)
- [9. Resultados y Evaluación](#9-resultados-y-evaluación)
- [10. Interpretación de Resultados y Hallazgos](#10-interpretación-de-resultados-y-hallazgos)
- [11. Conclusiones y Trabajos Futuros](#11-conclusiones-y-trabajos-futuros)
- [12. Apéndices](#12-apéndices)
- [13. Referencias](#13-referencias)

---

## 7. Metodología

### 7.1. Selección del modelo (clasificación, regresión, clúster)

Para este proyecto se implementaron dos enfoques de machine learning:

- **Regresión (Random Forest Regressor)**: Modelo principal para predecir el valor continuo del arriendo en salarios mínimos (Valor_Arriendo_SM). Se seleccionó Random Forest por su capacidad para capturar relaciones no lineales entre variables y manejar múltiples features sin requerir normalización extensiva.

- **Clasificación (Random Forest Classifier)**: Modelo complementario que categoriza los inmuebles en tres segmentos de mercado: "Bajo" (<2 SM), "Medio" (2-4 SM) y "Alto" (>4 SM). Permite identificar el rango de precio esperado basándose en características del inmueble.

### 7.2. Algoritmos y técnicas utilizadas

**Random Forest Regressor**: Modelo de ensamble basado en múltiples árboles de decisión que:
- Reduce el sobreajuste mediante promediación de predicciones
- Captura interacciones complejas entre variables (área, estrato, ubicación)
- Proporciona importancia de características para interpretabilidad

**Random Forest Classifier**: Versión de clasificación del mismo algoritmo para segmentación de mercado.

**Técnicas complementarias**:
- Análisis de correlación de Pearson para identificar variables predictoras
- Visualizaciones exploratorias (heatmaps, scatter plots, boxplots)
- Eliminación de valores atípicos mediante método IQR (Interquartile Range)

### 7.3. Justificación de los hiper-parámetros

**Random Forest (Regresión y Clasificación)**:
- `n_estimators=100`: Balance óptimo entre precisión y costo computacional. 100 árboles proporcionan estabilidad en las predicciones sin incrementar excesivamente el tiempo de entrenamiento.
- `max_depth=10`: Limita la profundidad de cada árbol para evitar sobreajuste. Permite capturar patrones complejos sin memorizar ruido en los datos de entrenamiento.
- `random_state=42`: Garantiza reproducibilidad de los experimentos, permitiendo comparar resultados entre ejecuciones.

**Validación Cruzada (K-Fold)**:
- `n_splits=5`: División estándar en 5 pliegues que balancea varianza del estimador con costo computacional.
- `shuffle=True`: Aleatoriza los datos antes de dividirlos, evitando sesgos por orden de los registros.

**División Train/Test**:
- `test_size=0.2`: 80% entrenamiento, 20% prueba. Proporción estándar que mantiene suficientes datos para entrenamiento mientras reserva un conjunto robusto para evaluación.
- `stratify=y` (clasificación): Mantiene la proporción de clases en train/test, crítico cuando las categorías están desbalanceadas (1592 Bajo, 1311 Medio, 785 Alto).

### 7.4. Validación cruzada y técnicas de re-muestreo

**Validación Cruzada K-Fold (5 pliegues)**:
- Divide el dataset en 5 subconjuntos
- Entrena 5 veces, usando 4 pliegues para entrenar y 1 para validar
- Resultados obtenidos: R² Scores = [0.850, 0.840, 0.850, 0.799, 0.813], R² Promedio = 0.8305 (±0.0414)
- Confirma que el modelo generaliza bien, sin dependencia excesiva de una división particular de datos

**Train/Test Split estratificado**:
- Separación única 80/20 para evaluación final
- En clasificación: estratificación por categoría de precio para mantener distribución representativa
- Evita que el conjunto de prueba sobrerrepresente o subrepresente alguna categoría

**Sin técnicas de re-muestreo adicionales**: No se aplicó SMOTE u oversampling porque las categorías, aunque desbalanceadas, tienen suficientes muestras (785 en la clase minoritaria "Alto") para entrenamiento efectivo.

---

## 8. Implementación

### 8.1. Herramientas y bibliotecas utilizadas

**Lenguaje y entorno**:
- **Python 3.8+**: Lenguaje principal del proyecto

**Manipulación y análisis de datos**:
- **Pandas 2.0+**: Carga, limpieza y transformación de datasets (DataFrames)
- **NumPy**: Operaciones numéricas, manejo de arrays, eliminación de valores infinitos

**Machine Learning**:
- **Scikit-learn 1.3+**: 
  - `RandomForestRegressor`, `RandomForestClassifier`: Modelos principales
  - `train_test_split`, `KFold`, `cross_val_score`: Validación
  - `r2_score`, `mean_squared_error`, `mean_absolute_error`, `accuracy_score`, `classification_report`, `confusion_matrix`: Métricas de evaluación

**Visualización**:
- **Matplotlib**: Gráficos base (scatter plots, histogramas, residuos)
- **Seaborn**: Visualizaciones estadísticas (heatmaps, boxplots, barplots)

### 8.2. Estructura del código y pipeline

**Pipeline de análisis implementado**:

> 📹 **Video tutorial**: [Cómo se extraen los datos - Proceso completo](https://youtu.be/mdHdsDXJUTo)

1. **Carga de datos**: 
   - Archivo: `inmuebles_combinado_limpio.csv` (5567 registros)
   - Columnas: Valor_Arriendo_SM, Area Construida, Estrato, Cuartos, Banos, Garajes, latitud, longitud, distancias a servicios, conteos de servicios cercanos

2. **Análisis Exploratorio de Datos (EDA)**:
   - Matriz de correlación de 18 variables numéricas
   - Análisis por estratos socioeconómicos (0-6)
   - Segmentación geográfica en bins de latitud/longitud
   - Análisis de distancias a servicios vs precio

3. **Preprocesamiento**:
   - Eliminación de valores NaN: `df.dropna()`
   - Eliminación de infinitos: `df.replace([np.inf, -np.inf], np.nan)`
   - Reducción de 5567 a 3688 registros limpios para modelado
   - Sin normalización (Random Forest es invariante a escala)

4. **Modelado**:
   - **Regresión**: Predicción de Valor_Arriendo_SM con 12 features
   - **Clasificación**: Categorización en Bajo/Medio/Alto con 10 features

5. **Evaluación**:
   - Métricas en train y test
   - Validación cruzada 5-fold
   - Generación de visualizaciones (8 gráficos)

6. **Salida**:
   - Resultados en consola con formato estructurado
   - Gráficos guardados en carpeta `eda_plots/`

**Archivos del proyecto**:
```
proyectoAnalitica/
├── analisis_patrones.py              # Script principal de análisis y modelado
├── eda_inmuebles.py                  # Script de exploración visual de datos
├── inmuebles_combinado_limpio.csv    # Dataset final limpio (5567 registros)
├── eda_plots/                        # Carpeta con visualizaciones generadas
└── README.md                         # Este documento
```

### 8.3. Estrategia de experimentación

**Enfoque iterativo**:
1. Comenzar con análisis exploratorio exhaustivo para identificar variables relevantes
2. Seleccionar top features basándose en correlación con precio (threshold |r| > 0.1)
3. Entrenar modelo único (Random Forest) por su balance entre interpretabilidad y performance
4. Validar con train/test split y cross-validation para confirmar generalización
5. Evaluar con múltiples métricas (R², RMSE, MAE) para visión integral del rendimiento

**Decisiones de diseño**:
- **No se compararon múltiples algoritmos** (regresión lineal, XGBoost, SVM) en esta iteración, enfocándose en un modelo robusto
- **Sin optimización de hiperparámetros** (GridSearch): se usaron valores estándar efectivos
- **Validación robusta**: Prioridad en cross-validation sobre ajuste fino de parámetros

---

## 9. Resultados y Evaluación

### 9.1. Métricas de rendimiento

**Modelo de Regresión (Random Forest Regressor)**:

*Conjunto de Entrenamiento*:
- R² Score: **0.9389** (93.89% de varianza explicada)
- RMSE: **0.4510 SM** (error promedio de ~0.45 salarios mínimos)
- MAE: **0.3074 SM** (desviación absoluta promedio)

*Conjunto de Prueba*:
- R² Score: **0.8449** (84.49% de varianza explicada)
- RMSE: **0.7174 SM**
- MAE: **0.4847 SM**

*Validación Cruzada (5-Fold)*:
- R² Promedio: **0.8305** (±0.0414)
- Rango de scores: [0.850, 0.840, 0.850, 0.799, 0.813]

**Interpretación**: Diferencia entre R² train (0.939) y test (0.845) indica leve sobreajuste, pero performance en test sigue siendo sólida. La validación cruzada confirma estabilidad del modelo.

**Modelo de Clasificación (Random Forest Classifier)**:

- **Accuracy Global**: **0.8428** (84.28% de predicciones correctas)

*Métricas por Categoría*:

| Categoría | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| **Alto**  | 0.88      | 0.81   | 0.84     | 157     |
| **Bajo**  | 0.90      | 0.88   | 0.89     | 319     |
| **Medio** | 0.76      | 0.82   | 0.79     | 262     |

- **Macro Avg**: Precision=0.85, Recall=0.84, F1=0.84
- **Weighted Avg**: Precision=0.85, Recall=0.84, F1=0.84

**Interpretación**: 
- La categoría "Bajo" es la mejor predicha (F1=0.89)
- "Medio" tiene menor precision (0.76), probablemente por traslape con límites de "Bajo" y "Alto"
- El modelo es balanceado (weighted avg ≈ macro avg)

**Imagen indicada**: `eda_plots/confusion_matrix_clasificacion.png`

### 9.2. Rendimiento base vs modelo

**Modelo Base (Media por Estrato)**:
- Predicción simplista: asignar precio promedio según estrato del inmueble
- Estratos con precio promedio:
  - Estrato 2: 1.23 SM
  - Estrato 3: 1.33 SM
  - Estrato 4: 2.01 SM
  - Estrato 5: 3.18 SM
  - Estrato 6: 4.35 SM

**Comparación**:
- **Modelo base** (solo estrato): R² estimado ~0.45-0.50 (basado en correlación estrato=0.669)
- **Random Forest** (múltiples features): R² test = **0.845**
- **Mejora**: +39 puntos porcentuales en varianza explicada
- **Valor agregado**: Incorporar área construida, garajes, baños y ubicación geográfica incrementa significativamente la precisión

**Imagen indicada**: `eda_plots/precios_por_estrato.png`

### 9.3. Visualización de resultados

**Gráficos de Regresión**:

1. **Predicciones vs. Valores Reales** (`eda_plots/predicciones_vs_reales.png`):
   - Scatter plot de valores predichos (eje Y) vs. reales (eje X)
   - Línea roja diagonal (y=x) representa predicción perfecta
   - Puntos cercanos a la diagonal indican buenas predicciones
   - R²=0.8449 mostrado en título

2. **Gráfico de Residuos** (`eda_plots/residuos_regresion.png`):
   - Residuos (error = real - predicho) vs. valores predichos
   - Línea horizontal en y=0 indica error cero
   - Distribución aleatoria alrededor de cero sugiere modelo bien calibrado
   - Sin patrones sistemáticos visibles (homocedasticidad)

**Gráficos de Clasificación**:

3. **Matriz de Confusión** (`eda_plots/confusion_matrix_clasificacion.png`):
   - Heatmap 3x3 mostrando predicciones correctas (diagonal) e incorrectas (fuera de diagonal)
   - Categorías: Alto, Bajo, Medio
   - Números en celdas indican cantidad de predicciones
   - Mayor intensidad de color en diagonal confirma buena clasificación

4. **Importancia de Características** (`eda_plots/feature_importance_clasificacion.png`):
   - Gráfico de barras horizontales con top 5 features:
     1. Area Construida: 0.257 (25.7%)
     2. Estrato: 0.179 (17.9%)
     3. Latitud: 0.117 (11.7%)
     4. Garajes: 0.116 (11.6%)
     5. Longitud: 0.104 (10.4%)
   - Confirma que área y estrato son predictores dominantes

**Gráficos Exploratorios**:

5. **Matriz de Correlación** (`eda_plots/correlation_heatmap.png`):
   - Heatmap con 18 variables numéricas
   - Escala de colores: azul (correlación negativa) a rojo (positiva)
   - Diagonal principal = 1.0 (autocorrelación)
   - Revela multicolinealidad (ej: Area Construida-Banos = 0.807)

6. **Mapa Geográfico de Precios** (`eda_plots/mapa_precios_geograficos.png`):
   - Scatter plot con longitud (X) y latitud (Y)
   - Color de puntos indica precio (escala viridis: amarillo=alto, morado=bajo)
   - Identifica zonas caras (cluster en coordenadas 4.663-4.685 lat, -74.059 a -74.041 lon con precio promedio 4.59 SM)

**Nota sobre ROC Curves**: No se generaron curvas ROC en este proyecto. Para clasificación multiclase, se requeriría ROC para cada categoría vs. resto (one-vs-rest), pero se priorizó matriz de confusión por su mayor interpretabilidad en este contexto.

---

## 10. Interpretación de Resultados y Hallazgos

### 10.1. Significado de los resultados obtenidos

**Variables más influyentes en el precio** (correlación con Valor_Arriendo_SM):

1. **Area Construida**: r=0.769 (correlación fuerte positiva)
   - Por cada aumento en metros cuadrados, el precio incrementa proporcionalmente
   - Es el predictor individual más potente

2. **Garajes**: r=0.708
   - Disponibilidad de parqueadero es altamente valorada en Bogotá
   - Refleja necesidad de movilidad privada en la ciudad

3. **Baños**: r=0.679
   - Número de baños indica nivel de confort y tamaño del inmueble
   - Alta correlación con area construida (r=0.807)

4. **Estrato**: r=0.669
   - Nivel socioeconómico de la zona impacta significativamente
   - Refleja calidad de servicios públicos, seguridad y percepción de estatus

5. **Longitud**: r=0.375 (moderada)
   - Ubicación este-oeste en Bogotá influye en precio
   - Zonas más al este (mayor longitud) tienden a ser más costosas

**Patrones identificados**:

- **Estratificación socioeconómica clara**: Precio promedio aumenta linealmente con estrato (Estrato 6: 4.35 SM vs. Estrato 2: 1.23 SM = 3.5x diferencia)

- **Geografía importa**: Zonas en coordenadas (4.663-4.685 lat, -74.059 a -74.041 lon) tienen precios promedio de 4.59 SM, mientras zonas periféricas llegan a 1.58 SM

- **Servicios cercanos con impacto limitado**:
  - Distancia a supermercado: r=0.117 (leve positiva, contraintuitiva)
  - Distancia a TransMilenio: r=0.081 (casi nula)
  - Número de farmacias en 120m: r=-0.004 (irrelevante)
  - **Conclusión**: Proximidad a servicios no es determinante; el estrato y características físicas dominan

- **Universidades cercanas reducen precio**: r=-0.114 (zonas universitarias tienen precios estancados entre 2-4 SM, probablemente inmuebles orientados a estudiantes)

**Imagen indicada**: `eda_plots/correlation_heatmap.png`

### 10.2. Implicaciones en el dominio del negocio

**Para Inmobiliarias**:
- Pueden estimar precios de nuevos listings con precisión (error promedio ±0.48 SM)
- Identificar inmuebles subvalorados o sobrevalorados comparando precio listado vs. predicción del modelo
- Ajustar estrategias de marketing según segmento (Bajo/Medio/Alto) predicho por el clasificador

**Para Inquilinos/Compradores**:
- Validar si un precio solicitado es justo según características del inmueble
- Priorizar búsqueda en zonas geográficas con mejor relación calidad-precio
- Entender qué características (garajes, área) justifican diferencias de precio

**Para Inversores Inmobiliarios**:
- Identificar áreas geográficas con potencial de apreciación (zonas con precios bajos pero buena conectividad)
- Optimizar desarrollo de proyectos: priorizar garajes y área sobre cantidad de servicios cercanos
- Segmentar mercado objetivo (estudiantes en zonas universitarias vs. familias en estratos altos)

**Para Planificación Urbana**:
- Evidencia de segregación espacial: precios varían 3.5x entre estratos
- Conectividad a TransMilenio no reduce significativamente precios (contrario a expectativa de transporte público como igualador)

### 10.3. Consideraciones éticas, justas, o sesgos en los modelos

**Sesgos identificados**:

1. **Sesgo socioeconómico**:
   - El modelo aprende y perpetúa desigualdades existentes (Estrato 6 = 4.35 SM vs. Estrato 2 = 1.23 SM)
   - Usar el modelo para pricing puede reforzar segregación espacial en Bogotá
   - **Riesgo**: Predecir precios bajos en estratos bajos dificulta movilidad social

2. **Sesgo geográfico**:
   - Datos concentrados en zonas centrales (1068 registros en coordenadas 4.663-4.685 lat)
   - Modelo menos preciso en periferia con pocos datos
   - **Riesgo**: Subrepresentación de zonas populares lleva a predicciones injustas

3. **Sesgo en features**:
   - Garajes (correlación 0.708) favorece a población con acceso a vehículo privado
   - Modelo no considera características como accesibilidad para discapacitados, eficiencia energética
   - **Riesgo**: Refuerza valores de mercado que priorizan movilidad privada sobre transporte público

**Consideraciones de fairness**:

- **Discriminación indirecta**: Aunque el modelo no usa variables protegidas (raza, género), el estrato actúa como proxy de nivel socioeconómico y puede correlacionar con estas variables

- **Transparencia**: La importancia de características (`feature_importance_clasificacion.png`) muestra qué factores pesan más, permitiendo auditoría

- **Recomendaciones**:
  1. No usar el modelo para decisiones que afecten acceso a vivienda (ej: aprobar/rechazar aplicaciones de arriendo)
  2. Complementar predicciones con análisis cualitativo de contexto social
  3. Rebalancear datos para incluir más inmuebles en estratos 2-3
  4. Investigar por qué distancia a TransMilenio no reduce precios (¿calidad del servicio?)

**Imagen indicada**: `eda_plots/mapa_precios_geograficos.png`

---

## 11. Conclusiones y Trabajos Futuros

### 11.1. Resumen de los logros

- **Modelo de regresión robusto** con R²=0.845 que predice precios con error promedio de 0.48 SM (equivalente a ~$677,000 COP con SM de $1,400,000)

- **Clasificador efectivo** con 84% de accuracy para segmentar inmuebles en rangos de precio

- **Identificación de variables clave**: Área construida, garajes y baños son los predictores más importantes, seguidos de estrato y ubicación

- **Dataset limpio y estructurado** de 5567 registros con 18 features cuantitativas, resultado de integración de múltiples fuentes (API MetroCuadrado, datos públicos de Bogotá, OSM)

- **Pipeline reproducible** con scripts modulares (`analisis_patrones.py`, `eda_inmuebles.py`) y validación cruzada para garantizar generalización

### 11.2. Desafíos presentados

1. **Pérdida de datos por limpieza**:
   - Dataset original: 9565 registros
   - Después de filtrar outliers: 6863 registros
   - Después de eliminar NaN: 3688 registros utilizables para modelado (38% pérdida)
   - **Causa**: Variables geográficas (distancias, coordenadas) con muchos valores faltantes

2. **Asignación de estratos**:
   - API de MetroCuadrado no incluía estrato en los datos
   - Requirió cruce con dataset público de Bogotá usando latitud/longitud
   - OpenStreetMap no proporciona información de estratos directamente

3. **Sobreajuste leve**:
   - R² train (0.939) vs. R² test (0.845) = 9.4 puntos de diferencia
   - Indica que el modelo memoriza ciertos patrones del entrenamiento
   - Mitigado parcialmente con max_depth=10, pero podría mejorarse

4. **Interpretabilidad de Random Forest**:
   - A diferencia de regresión lineal, no proporciona coeficientes interpretables directos
   - Dificultad para explicar predicciones individuales a stakeholders no técnicos

5. **Desbalance de categorías en clasificación**:
   - Bajo: 1592, Medio: 1311, Alto: 785 (ratio ~2:1.7:1)
   - Afecta recall en categoría "Alto" (81% vs. 88% en "Bajo")

### 11.3. Recomendaciones de mejora

**Mejoras en datos**:
1. **Incluir features temporales**: Edad del inmueble, año de construcción, fecha de última renovación
2. **Variables de calidad**: Estado de conservación, tipo de acabados, presencia de amenidades (piscina, gimnasio, salón social)
3. **Datos externos**: Índices de criminalidad por zona, calidad de colegios cercanos, valorización histórica
4. **Ampliar cobertura geográfica**: Recolectar más datos en estratos 2-3 y zonas periféricas

**Mejoras en modelado**:
1. **Probar algoritmos adicionales**:
   - XGBoost: mejor manejo de valores faltantes y regularización
   - LightGBM: más rápido para datasets grandes
   - Redes neuronales (MLP): para capturar interacciones complejas

2. **Optimización de hiperparámetros**:
   - GridSearchCV o RandomizedSearchCV para `n_estimators`, `max_depth`, `min_samples_split`
   - Reducir sobreajuste aumentando `min_samples_leaf`

3. **Ensamble de modelos**:
   - Combinar Random Forest con regresión lineal (stacking) para balancear interpretabilidad y precisión

4. **Feature engineering avanzado**:
   - Interacciones: `Area_Construida * Estrato`, `Garajes * longitud`
   - Variables polinómicas para capturar relaciones no lineales

**Mejoras en evaluación**:
1. **Validación externa**: Probar modelo con datos de otra ciudad (Medellín, Cali) para verificar transferibilidad
2. **Análisis de errores**: Identificar qué tipos de inmuebles se predicen mal (outliers, zonas específicas)
3. **Métricas adicionales**: MAPE (Mean Absolute Percentage Error) para errores relativos

### 11.4. Ideas para posteriores trabajos o despliegue real

**Despliegue en producción**:

1. **Obtener aun mas datos**
    - hay muy pocas fuentes de datos, y cuando hay varias se repiten demasiado.

2. **Dashboard interactivo**:
   - Herramienta: Streamlit o Dash
   - Funcionalidades:
     - Input manual de características del inmueble
     - Visualización de predicción en mapa de Bogotá
     - Comparación con inmuebles similares
     - Exploración de "qué pasaría si" (ej: agregar un garaje aumenta precio en X%)

3. **Integración con plataformas inmobiliarias**:
   - Plugin para Finca Raíz, MetroCuadrado: validar precios al publicar listing
   - Alertas automáticas: notificar a usuarios si un inmueble está subvalorado según el modelo

**Investigación futura**:

1. **Análisis de series temporales**:
   - Predecir evolución de precios en el tiempo
   - Identificar zonas con tendencia al alza (gentrificación)

2. **Análisis espacial avanzado**:
   - Modelos geográficamente ponderados (GWR) para capturar efectos locales
   - Clustering espacial (DBSCAN) para identificar micro-mercados

3. **Imágenes satelitales y computer vision**:
   - Usar Google Street View para evaluar estado de fachada
   - Análisis de imágenes de satélite para estimar calidad de zona (vegetación, densidad construcción)


**Impacto social**:
- Desarrollar versión del modelo para políticas públicas: identificar zonas con necesidad de vivienda social
- Auditoría algorítmica: evaluar si el modelo discrimina injustamente contra ciertas poblaciones

---

## 12. Apéndices

### 12.1. Diseños de módulos

**Estructura del proyecto**:

```
proyectoAnalitica/
├── analisis_patrones.py                  # Script principal: EDA, modelado, evaluación
├── eda_inmuebles.py                      # Visualizaciones exploratorias adicionales
├── apiMetroCuadrado.py                   # Extracción de datos desde API MetroCuadrado
├── script_metrocuadrado.py               # Procesamiento inicial de datos de MetroCuadrado
├── script_apartamentos_bogota{2-5}.py    # Scripts de scraping/limpieza por fuente
├── conbinarDataLimpia.py                 # Unificación de datasets limpios
├── graficas_distancia_precio.py          # Análisis de relación distancia-precio
│
├── inmuebles_combinado_limpio.csv        # Dataset final (5567 registros)
├── inmuebles_metrocuadrado_limpio.csv    # Datos limpios de MetroCuadrado
├── inmuebles_apartamentos_bogota{2-5}_limpio.csv  # Datasets intermedios
├── Inmuebles_Disponibles_para_Arrendamiento_20251024.csv  # Datos públicos Bogotá
│
├── barriolegalizado.gpkg                 # Archivo geoespacial: barrios Bogotá
├── manzanaestratificacion.gpkg           # Archivo geoespacial: estratos por manzana
├── geocode_cache.sqlite                  # Cache de geocodificación (lat/lon)
│
├── eda_plots/                            # Carpeta con gráficos generados
│   ├── correlation_heatmap.png
│   ├── precios_por_estrato.png
│   ├── mapa_precios_geograficos.png
│   ├── confusion_matrix_clasificacion.png
│   ├── feature_importance_clasificacion.png
│   ├── predicciones_vs_reales.png
│   └── residuos_regresion.png
│
├── cache/                                # Cache de requests HTTP (API)
├── .venv/                                # Entorno virtual Python
├── .git/                                 # Control de versiones Git
├── .gitignore                            # Archivos excluidos de Git
└── README.md                             # Documentación del proyecto
```

**Descripción de módulos clave**:

1. **`analisis_patrones.py`** (Script principal):
   - **Input**: `inmuebles_combinado_limpio.csv`
   - **Funciones**:
     - Carga y preprocesamiento de datos (eliminación de NaN, infinitos)
     - Análisis de correlaciones (matriz triangular inferior)
     - Análisis por estratos, geografía y distancias
     - Entrenamiento de Random Forest Regressor/Classifier
     - Validación cruzada K-Fold
     - Generación de 7 visualizaciones
   - **Output**: 
     - Resultados en consola (métricas, hallazgos)
     - Gráficos en `eda_plots/`
   - **Dependencias**: pandas, numpy, sklearn, matplotlib, seaborn

2. **`eda_inmuebles.py`**:
   - Complementa análisis exploratorio con visualizaciones alternativas
   - Histogramas, distribuciones, boxplots por variables categóricas

3. **`apiMetroCuadrado.py`**:
   - Extracción automatizada de listings desde API de MetroCuadrado
   - Manejo de paginación, rate limiting, caché de requests
   - Output: `datosDeMetroCuadrado.json` / `.csv`

4. **`conbinarDataLimpia.py`**:
   - Unifica múltiples datasets limpiados
   - Maneja duplicados (por dirección, coordenadas)
   - Enriquece con features geográficas (distancias, conteos)
   - Output: `inmuebles_combinado_limpio.csv`

5. **`script_metrocuadrado.py` y `script_apartamentos_bogota{2-5}.py`**:
   - Limpieza inicial de datos crudos:
     - Conversión de precios a salarios mínimos
     - Geocodificación de direcciones
     - Eliminación de outliers (IQR method)
     - Asignación de estratos (cruce con GeoPackage)
   - Output: Archivos `*_limpio.csv`

6. **`graficas_distancia_precio.py`**:
   - Análisis específico de relación entre distancias a servicios y precio
   - Scatter plots, regresiones locales (LOWESS)

**Archivos geoespaciales**:
- **`barriolegalizado.gpkg`**: Polígonos de barrios de Bogotá (formato GeoPackage)
- **`manzanaestratificacion.gpkg`**: Polígonos de manzanas con asignación de estrato
- Usados para spatial join: asignar estrato a coordenadas lat/lon de inmuebles

**Flujo de datos (pipeline)**:

```
[API MetroCuadrado] → [apiMetroCuadrado.py] → datosDeMetroCuadrado.json
                                                       ↓
[Datos públicos Bogotá] → [script_*.py] → inmuebles_*_limpio.csv
                                                       ↓
[Todos los datasets] → [conbinarDataLimpia.py] → inmuebles_combinado_limpio.csv
                                                       ↓
                          [analisis_patrones.py] → Modelos + Gráficos (eda_plots/)
```

### 12.2. Tablas y gráficos generados

**Tabla 1: Estadísticas descriptivas por estrato**

| Estrato | Media (SM) | Mediana (SM) | Desv. Est. | Registros |
|---------|------------|--------------|------------|-----------|
| 0       | 1.94       | 1.94         | 1.35       | 2         |
| 2       | 1.23       | 1.00         | 0.76       | 115       |
| 3       | 1.33       | 1.15         | 0.67       | 844       |
| 4       | 2.01       | 1.77         | 0.96       | 823       |
| 5       | 3.18       | 2.88         | 1.45       | 676       |
| 6       | 4.35       | 3.89         | 1.79       | 1252      |

**Tabla 2: Top 5 zonas geográficas con precios más altos**

| Latitud (rango)    | Longitud (rango)    | Precio Promedio (SM) | Registros |
|--------------------|---------------------|----------------------|-----------|
| (4.663, 4.685]     | (-74.059, -74.041]  | 4.59                 | 1068      |
| (4.642, 4.663]     | (-74.059, -74.041]  | 3.99                 | 331       |
| (4.663, 4.685]     | (-74.041, -74.023]  | 3.81                 | 18        |
| (4.685, 4.706]     | (-74.059, -74.041]  | 3.66                 | 691       |
| (4.685, 4.706]     | (-74.041, -74.023]  | 3.30                 | 417       |

**Tabla 3: Correlaciones de distancias con precio**

| Variable                        | Correlación | Interpretación                          |
|---------------------------------|-------------|-----------------------------------------|
| dist_supermercado_cercano_m     | +0.117      | Mayor distancia → precios ligeramente más altos (contraintuitivo) |
| dist_transmilenio_cercana_m     | +0.081      | Impacto casi nulo                       |
| dist_bus_cercana_m              | +0.012      | Sin relación significativa              |
| dist_farmacia_cercana_m         | -0.004      | Sin relación                            |
| dist_via_principal_m            | -0.091      | Mayor distancia → precios ligeramente más bajos |
| num_universidades_300m          | -0.114      | Más universidades cercanas → precios más bajos |

**Gráficos generados en `eda_plots/`**:

1. **`correlation_heatmap.png`**: Matriz de correlación 18x18 con todas las variables numéricas. Identifica multicolinealidad (Area-Baños: 0.81).

2. **`precios_por_estrato.png`**: Boxplot que muestra distribución de precios en cada estrato. Confirma tendencia lineal de aumento por estrato.

3. **`mapa_precios_geograficos.png`**: Scatter plot geográfico (lat/lon) con color indicando precio. Revela clustering de inmuebles caros en zona norte de Bogotá.

4. **`predicciones_vs_reales.png`**: Validación visual del modelo de regresión. Puntos cercanos a línea diagonal (y=x) indican buenas predicciones.

5. **`residuos_regresion.png`**: Gráfico de residuos vs. valores predichos. Distribución aleatoria confirma homocedasticidad (varianza constante del error).

6. **`confusion_matrix_clasificacion.png`**: Matriz 3x3 para categorías Bajo/Medio/Alto. Diagonal dominante (predicciones correctas).

7. **`feature_importance_clasificacion.png`**: Ranking de importancia de variables en clasificación. Area Construida (25.7%) y Estrato (17.9%) dominan.

### 12.3. Instrumentos de consulta y diccionario de datos

**Diccionario de Variables del Dataset `inmuebles_combinado_limpio.csv`**:

| Variable                        | Tipo     | Descripción                                                                 | Rango/Valores           | Unidad       |
|---------------------------------|----------|-----------------------------------------------------------------------------|-------------------------|--------------|
| **Valor_Arriendo_SM**           | Float    | Precio mensual de arriendo (variable objetivo)                              | 0.1 - 15.0              | Salarios Mínimos (SM) |
| **Area Construida**             | Float    | Área construida del inmueble                                                | 20 - 500+               | Metros cuadrados (m²) |
| **Estrato**                     | Integer  | Nivel socioeconómico del sector (1=bajo, 6=alto)                            | 0, 2, 3, 4, 5, 6        | Categórico   |
| **Cuartos**                     | Integer  | Número de habitaciones/alcobas                                              | 1 - 8                   | Unidades     |
| **Banos**                       | Float    | Número de baños (puede incluir medios baños)                                | 1.0 - 6.0               | Unidades     |
| **Garajes**                     | Integer  | Número de espacios de parqueadero                                           | 0 - 5                   | Unidades     |
| **latitud**                     | Float    | Coordenada latitud (sistema WGS84)                                          | 4.50 - 4.85             | Grados decimales |
| **longitud**                    | Float    | Coordenada longitud (sistema WGS84)                                         | -74.20 - -73.95         | Grados decimales |
| **dist_transmilenio_cercana_m** | Float    | Distancia a estación de TransMilenio más cercana                            | 5 - 2273                | Metros (m)   |
| **dist_bus_cercana_m**          | Float    | Distancia a parada de bus más cercana                                       | 2 - 518                 | Metros (m)   |
| **dist_farmacia_cercana_m**     | Float    | Distancia a farmacia más cercana                                            | 1 - 723                 | Metros (m)   |
| **dist_supermercado_cercano_m** | Float    | Distancia a supermercado más cercano                                        | 4 - 638                 | Metros (m)   |
| **dist_via_principal_m**        | Float    | Distancia a vía principal (avenida) más cercana                             | 0 - 1234                | Metros (m)   |
| **num_farmacias_120m**          | Integer  | Cantidad de farmacias en radio de 120m                                      | 0 - 15                  | Unidades     |
| **num_colegios_120m**           | Integer  | Cantidad de colegios en radio de 120m                                       | 0 - 8                   | Unidades     |
| **num_transmilenio_120m**       | Integer  | Cantidad de estaciones TransMilenio en radio de 120m                        | 0 - 3                   | Unidades     |
| **num_bus_120m**                | Integer  | Cantidad de paradas de bus en radio de 120m                                 | 0 - 20                  | Unidades     |
| **num_universidades_300m**      | Integer  | Cantidad de universidades en radio de 300m                                  | 0 - 5                   | Unidades     |

**Notas sobre el dataset**:
- **Registros totales**: 5567 (después de limpieza y eliminación de outliers)
- **Registros utilizables para modelado**: 3688 (después de eliminar NaN)
- **Fuentes**: API MetroCuadrado, datos abiertos Alcaldía de Bogotá, OpenStreetMap (OSM)
- **Periodo de recolección**: Octubre 2024
- **Salario Mínimo de referencia**: $1,400,000 COP (2024)

**Cálculos derivados**:
- Precio en COP = Valor_Arriendo_SM × $1,400,000
- Precio por m² = (Valor_Arriendo_SM × $1,400,000) / Area Construida

**Categorías de precio (para clasificación)**:
- **Bajo**: Valor_Arriendo_SM < 2.0 (menos de $2,800,000 COP)
- **Medio**: 2.0 ≤ Valor_Arriendo_SM < 4.0 ($2,800,000 - $5,600,000 COP)
- **Alto**: Valor_Arriendo_SM ≥ 4.0 (más de $5,600,000 COP)

---

## 13. Referencias

### 13.1. Artículos académicos

1. Breiman, L. (2001). **"Random Forests"**. *Machine Learning*, 45(1), 5-32. 
   - Paper fundacional del algoritmo Random Forest
   - DOI: 10.1023/A:1010933404324

2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). **"The Elements of Statistical Learning: Data Mining, Inference, and Prediction"** (2nd ed.). Springer.
   - Capítulo 15: Random Forests
   - Disponible en: https://hastie.su.domains/ElemStatLearn/

3. Molnar, C. (2022). **"Interpretable Machine Learning: A Guide for Making Black Box Models Explainable"** (2nd ed.).
   - Capítulo sobre Feature Importance en Random Forest
   - Disponible en: https://christophm.github.io/interpretable-ml-book/

4. Sirmans, S., Macpherson, D., & Zietz, E. (2005). **"The Composition of Hedonic Pricing Models"**. *Journal of Real Estate Literature*, 13(1), 3-43.
   - Fundamentación teórica de modelos hedónicos para valuación inmobiliaria

5. Zurada, J., Levitan, A., & Guan, J. (2011). **"A Comparison of Regression and Artificial Intelligence Methods in a Mass Appraisal Context"**. *Journal of Real Estate Research*, 33(3), 349-388.
   - Comparación de ML vs. métodos tradicionales en valuación de propiedades

### 13.2. Datasets y fuentes de datos

**Fuentes primarias**:

1. **MetroCuadrado API**:
   - Plataforma líder de listings inmobiliarios en Colombia
   - URL: https://www.metrocuadrado.com/
   - Datos extraídos: Precio, área, número de habitaciones/baños, dirección, coordenadas
   - Fecha de extracción: Octubre 2024
   - Total de registros: ~4500 apartamentos en arriendo en Bogotá

2. **Portal de Datos Abiertos de Bogotá**:
   - Dataset: *Inmuebles Disponibles para Arrendamiento* (actualizado 24/10/2024)
   - URL: https://datosabiertos.bogota.gov.co/
   - Variables: Ubicación, estrato, características físicas
   - Total de registros: ~1200 inmuebles públicos

3. **OpenStreetMap (OSM)**:
   - API Overpass para extracción de puntos de interés (POIs)
   - URL: https://www.openstreetmap.org/
   - Datos extraídos: Ubicación de TransMilenio, paraderos de bus, farmacias, supermercados, colegios, universidades, vías principales
   - Librería utilizada: `osmnx` (Python)

4. **Datos Geoespaciales de Bogotá**:
   - Archivo: `manzanaestratificacion.gpkg` (estratos por manzana)
   - Fuente: Secretaría Distrital de Planeación de Bogotá
   - Formato: GeoPackage (polígonos con atributos de estrato)
   - Usado para asignar estrato a coordenadas lat/lon

**Dataset final**:
- Archivo: `inmuebles_combinado_limpio.csv`
- Registros: 5567 (después de limpieza)
- Variables: 18 (numéricas)
- Disponible en repositorio: https://github.com/JorgeHSLA/proyectoAnalitica

### 13.3. Toolkits y bibliotecas

**Lenguaje y entorno**:
- **Python 3.8+**: https://www.python.org/
- **pip**: Gestor de paquetes de Python

**Librerías de análisis de datos**:
- **Pandas 2.0+**: https://pandas.pydata.org/
  - Documentación: https://pandas.pydata.org/docs/
- **NumPy 1.24+**: https://numpy.org/
  - Documentación: https://numpy.org/doc/stable/

**Machine Learning**:
- **Scikit-learn 1.3+**: https://scikit-learn.org/
  - `RandomForestRegressor`: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
  - `RandomForestClassifier`: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
  - Métricas: https://scikit-learn.org/stable/modules/model_evaluation.html

**Visualización**:
- **Matplotlib 3.7+**: https://matplotlib.org/
  - Galería de ejemplos: https://matplotlib.org/stable/gallery/index.html
- **Seaborn 0.12+**: https://seaborn.pydata.org/
  - Galería: https://seaborn.pydata.org/examples/index.html

**Análisis geoespacial**:
- **GeoPandas 0.14+**: https://geopandas.org/
  - Para manejo de archivos GeoPackage (.gpkg)
- **OSMnx 1.6+**: https://osmnx.readthedocs.io/
  - Extracción de datos de OpenStreetMap
- **Geopy 2.4+**: https://geopy.readthedocs.io/
  - Geocodificación (dirección → lat/lon)

**Utilidades**:
- **Requests**: Para llamadas HTTP a API MetroCuadrado
- **SQLite**: Cache de geocodificación (`geocode_cache.sqlite`)

**Entorno virtual**:
```bash
# Crear entorno virtual
python -m venv .venv

# Activar (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install pandas numpy scikit-learn matplotlib seaborn geopandas osmnx geopy requests
```

### 13.4. Otros recursos

**Video tutoriales del proyecto**:

1. **Proceso de Extracción de Datos**:
   - URL: https://youtu.be/mdHdsDXJUTo
   - Duración: [Completar]
   - Descripción: Demostración completa del proceso de extracción de datos desde la API de MetroCuadrado, incluyendo manejo de paginación, limpieza de datos, geocodificación de direcciones y enriquecimiento con datos geoespaciales de Bogotá.
   - Temas cubiertos:
     - Configuración de requests a API MetroCuadrado
     - Parseo de JSON y conversión a DataFrames
     - Geocodificación con Geopy
     - Spatial joins con GeoPackages (estratos)
     - Cálculo de distancias y conteos de POIs usando OSMnx

**Tutoriales y guías**:

1. **Scikit-learn User Guide - Ensemble Methods**:
   - URL: https://scikit-learn.org/stable/modules/ensemble.html
   - Explicación detallada de Random Forest y parámetros

2. **Real Python - Random Forest Classifier**:
   - URL: https://realpython.com/lessons/random-forest-classifier/
   - Tutorial práctico con ejemplos de código

3. **Towards Data Science - Feature Importance**:
   - Artículo: "Explaining Feature Importance by example of a Random Forest"
   - URL: https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e

4. **Documentación de OSMnx**:
   - URL: https://osmnx.readthedocs.io/en/stable/
   - Ejemplos de extracción de POIs: https://osmnx.readthedocs.io/en/stable/user-reference.html#module-osmnx.features

**Contexto de Bogotá**:

1. **Secretaría Distrital de Planeación**:
   - Portal: https://www.sdp.gov.co/
   - Información sobre estratificación socioeconómica en Bogotá

2. **TransMilenio S.A.**:
   - Portal: https://www.transmilenio.gov.co/
   - Mapa de rutas y estaciones

3. **Alcaldía Mayor de Bogotá - Datos Abiertos**:
   - Portal: https://datosabiertos.bogota.gov.co/
   - Catálogo de datasets sobre vivienda, transporte, servicios

**Repositorio del proyecto**:
- **GitHub**: https://github.com/JorgeHSLA/proyectoAnalitica
- Incluye:
  - Código fuente completo
  - Dataset `inmuebles_combinado_limpio.csv`
  - Gráficos generados (`eda_plots/`)
  - README con instrucciones de ejecución

**Contacto**:
- **Autor**: Jorge Hernán Silva López Ardila
- **Email**: [Agregar email]
- **LinkedIn**: [Agregar perfil]
- **Institución**: [Agregar universidad/entidad]

---

**Fin del documento**

*Última actualización: Noviembre 22, 2025*