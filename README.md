# Proyecto de Análisis de Inmuebles en Bogotá

Este proyecto realiza un análisis exhaustivo de datos de inmuebles en arriendo en Bogotá, Colombia, utilizando técnicas de machine learning para modelar precios, clasificar propiedades y segmentar el mercado. El dataset consta de 5567 registros con variables físicas, geográficas y de servicios cercanos.

## 7. Metodología

## 7. Metodología

### 7.1. Selección del modelo (clasificación, regresión, clúster)
- **Regresión**: Utilizada para predecir el precio de arriendo (Valor_Arriendo_SM) basado en características físicas y geográficas.
- **Clasificación**: Categorización de precios en "Bajo" (<2 SM), "Medio" (2-4 SM) y "Alto" (>4 SM).
- **Clúster**: Segmentación de inmuebles en grupos similares utilizando características como área, estrato y ubicación.

### 7.2. Algoritmos y técnicas utilizadas
- **Regresión Lineal (OLS)**: Para modelado estadístico de precios.
- **Random Forest Classifier/Regressor**: Para clasificación y comparación en regresión.
- **K-Means**: Para clustering no supervisado.
- **Técnicas adicionales**: Análisis de correlaciones, normalización (StandardScaler), y visualizaciones (seaborn, matplotlib).

### 7.3. Justificación de los hiper-parámetros
- **Random Forest**: `n_estimators=100` (balance entre precisión y tiempo de cómputo), `max_depth=10` (evita sobreajuste), `random_state=42` (reproducibilidad).
- **K-Means**: `n_init=10` (múltiples inicializaciones para estabilidad), `random_state=42`.
- **KFold**: `n_splits=5` (estándar para validación cruzada), `shuffle=True` (aleatorización).

### 7.4. Validación cruzada y técnicas de re-muestreo
- **Train/Test Split**: 80/20 para evaluación de modelos.
- **Validación Cruzada K-Fold**: 5-fold para estimar rendimiento generalizado.
- **Stratified Split**: En clasificación para mantener distribución de clases.

## 8. Implementación

### 8.1. Herramientas y bibliotecas utilizadas
- **Python**: Lenguaje principal.
- **Pandas**: Manipulación de datos.
- **NumPy**: Cálculos numéricos.
- **Scikit-learn**: Modelos de ML (LinearRegression, RandomForest, KMeans, métricas).
- **Statsmodels**: Regresión OLS con estadísticas detalladas.
- **Matplotlib/Seaborn**: Visualizaciones.
- **Jupyter Notebook** (opcional para exploración).

### 8.2. Estructura del código y pipeline
1. **Carga de datos**: `inmuebles_combinado_limpio.csv`.
2. **EDA**: Análisis descriptivo, correlaciones, visualizaciones.
3. **Preprocesamiento**: Limpieza de NaN, normalización.
4. **Modelado**: Regresión, clasificación, clustering.
5. **Evaluación**: Métricas, validación cruzada, visualizaciones.
6. **Salida**: Gráficos en `eda_plots/`, resultados en consola.

### 8.3. Estrategia de experimentación
- **Iterativa**: Comenzar con EDA, luego modelos simples (regresión lineal), comparar con modelos complejos (Random Forest).
- **Validación**: Separar datos en train/test, usar cross-validation para robustez.
- **Comparación**: Evaluar múltiples modelos (e.g., Regresión Lineal vs. Random Forest).

## 9. Resultados y evaluación

### 9.1. Métricas de rendimiento
- **Regresión Lineal**: R² Test = 0.773, RMSE = 0.869 SM, MAE = 0.628 SM.
- **Random Forest Regressor**: R² Test = 0.845, RMSE = 0.717 SM, MAE = 0.485 SM.
- **Clasificación**: Accuracy = 0.843, Precision/Recall/F1 (Alto: 0.88/0.81/0.84, Bajo: 0.90/0.88/0.89, Medio: 0.76/0.82/0.79).
- **Clustering**: Silhouette Score = 0.307 (k=2), 2 clusters identificados.

### 9.2. Rendimiento base vs modelo
- **Base**: Media de precios por estrato (e.g., Estrato 6: 4.35 SM, Estrato 2: 1.23 SM).
- **Modelo**: Mejora significativa con variables geográficas y físicas (R² +0.07-0.08 vs. base).

### 9.3. Visualización de resultados
- **Matriz de Confusión**: `eda_plots/confusion_matrix_clasificacion.png`.
- **Predicciones vs. Reales**: `eda_plots/predicciones_vs_reales.png`.
- **Residuos**: `eda_plots/residuos_regresion.png`.
- **Curvas ROC**: No implementadas directamente, pero posibles con `sklearn.metrics.roc_curve`.

## 10. Interpretación de resultados y hallazgos

### 10.1. Significado de los resultados obtenidos
- Variables clave: Área construida (corr=0.769), Garajes (0.708), Baños (0.679), Estrato (0.669), Longitud (0.375).
- Patrones: Precios aumentan con estrato (Estrato 6: 4.35 SM promedio vs. Estrato 2: 1.23 SM); ubicación geográfica influye (zonas centrales más caras); proximidad a Transmilenio y supermercados positivos, distancia a vías principales negativa.

### 10.2. Implicaciones en el dominio del negocio
- **Inmobiliarias**: Mejor estimación de precios para listings.
- **Inquilinos**: Identificación de zonas accesibles.
- **Inversores**: Segmentación de mercado para desarrollo.

### 10.3. Consideraciones éticas, justas, o sesgos en los modelos
- **Sesgos geográficos**: Datos pueden reflejar desigualdades socioeconómicas en Bogotá, potencialmente perpetuando segregación.
- **Fairness**: Modelos podrían favorecer áreas de alto estrato; considerar rebalanceo de datos.
- **Transparencia**: Publicar importancia de features para evitar discriminación.

## 11. Conclusiones y trabajos futuros

### 11.1. Resumen de los logros
- Modelos precisos para predicción de precios y clasificación.
- Identificación de patrones clave en mercado inmobiliario bogotano.
- Pipeline completo de análisis de datos.

### 11.2. Desafíos presentados
- Datos faltantes en variables geográficas.
- Sobreajuste en modelos complejos.
- Interpretabilidad de Random Forest.

### 11.3. Recomendaciones de mejora
- Incluir más features (e.g., edad del inmueble).
- Usar técnicas avanzadas (e.g., XGBoost, redes neuronales).
- Validar con datos externos.

### 11.4. Ideas para posteriores trabajos o despliegue real
- API para predicción en tiempo real.
- Dashboard interactivo con Streamlit.
- Integración con datos satelitales para análisis urbano.

## 12. Apéndices

### 12.1. Diseños de módulos
- `analisis_patrones.py`: Script principal de modelado.
- `eda_inmuebles.py`: Exploración de datos y visualizaciones.
- `script_*.py`: Scripts de scraping y limpieza.

### 12.2. Tablas o gráficos
- Ver `eda_plots/` para todos los gráficos generados.
- Ejemplos: `correlation_heatmap.png` (matriz de correlación), `precios_por_estrato.png` (boxplot por estrato), `mapa_precios_geograficos.png` (scatter geográfico), `confusion_matrix_clasificacion.png`, `feature_importance_clasificacion.png`, `elbow_method_clustering.png`, `clusters_geograficos.png`, `precios_por_cluster.png`, `predicciones_vs_reales.png`, `residuos_regresion.png`.

### 12.3. Instrumentos de consulta o diccionarios de datos
- **Diccionario de Datos**:
  - `Valor_Arriendo_SM`: Precio en salarios mínimos.
  - `Area Construida`: Área en m².
  - `Estrato`: Nivel socioeconómico (1-6).
  - `latitud/longitud`: Coordenadas GPS.
  - `dist_*`: Distancias a servicios en metros.
  - `num_*`: Conteo de servicios cercanos.

## 13. Referencias

### 13.1. Artículos académicos
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR. https://jmlr.org/papers/v12/pedregosa11a.html

### 13.2. Datasets
- Datos de MetroCuadrado (inmuebles_metrocuadrado_limpio.csv): 5567 registros de inmuebles en Bogotá.
- Datos combinados (inmuebles_combinado_limpio.csv): Dataset final con features geográficas y físicas.

### 13.3. Toolkits
- Scikit-learn v1.3+: https://scikit-learn.org/
- Pandas v2.0+: https://pandas.pydata.org/
- Seaborn v0.12+: https://seaborn.pydata.org/
- Statsmodels v0.14+: https://www.statsmodels.org/

### 13.4. Otros recursos
- OSMnx para datos geográficos: https://osmnx.readthedocs.io/
- Documentación de Python 3.8+: https://docs.python.org/