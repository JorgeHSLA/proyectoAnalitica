

## 7. Metodología

### 7.1. Selección del modelo (clasificación, regresión)
- **Regresión**: Utilizada para predecir el precio de arriendo (Valor_Arriendo_SM) basado en características físicas y geográficas.
- **Clasificación**: Categorización de precios en "Bajo" (<2 SM), "Medio" (2-4 SM) y "Alto" (>4 SM).

### 7.2. Algoritmos y técnicas utilizadas
- **Random Forest Regressor/Classifier**: Para regresión y clasificación, enfocado en un solo modelo para simplicidad.
- **Técnicas adicionales**: Análisis de correlaciones, visualizaciones (seaborn, matplotlib).

### 7.3. Justificación de los hiper-parámetros
- **Random Forest**: `n_estimators=100` (balance entre precisión y tiempo de cómputo), `max_depth=10` (evita sobreajuste), `random_state=42` (reproducibilidad).
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
- **Scikit-learn**: Modelos de ML (RandomForest, métricas).
- **Matplotlib/Seaborn**: Visualizaciones.

### 8.2. Estructura del código y pipeline
1. **Carga de datos**: `inmuebles_combinado_limpio.csv`.
2. **EDA**: Análisis descriptivo, correlaciones, visualizaciones.
3. **Preprocesamiento**: Limpieza de NaN.
4. **Modelado**: Regresión y clasificación con Random Forest.
5. **Evaluación**: Métricas, validación cruzada, visualizaciones.
6. **Salida**: Gráficos en `eda_plots/`, resultados en consola.

### 8.3. Estrategia de experimentación
- **Enfoque simplificado**: Uso de un solo modelo (Random Forest) para regresión y clasificación.
- **Validación**: Separar datos en train/test, usar cross-validation para robustez.

## 9. Resultados y evaluación

### 9.1. Métricas de rendimiento
- **Random Forest Regressor**: R² Test = 0.8449, RMSE = 0.7174 SM, MAE = 0.4847 SM, R² Promedio CV = 0.8305.
- **Clasificación**: Accuracy = 0.8428, Precision/Recall/F1 (Alto: 0.88/0.81/0.84, Bajo: 0.90/0.88/0.89, Medio: 0.76/0.82/0.79).

### 9.2. Rendimiento base vs modelo
- **Base**: Media de precios por estrato (e.g., Estrato 6: 4.35 SM, Estrato 2: 1.23 SM).
- **Modelo**: Mejora significativa con variables geográficas y físicas (R² 0.84 vs. base simple).

### 9.3. Visualización de resultados
- **Matriz de Confusión**: `eda_plots/confusion_matrix_clasificacion.png`.
- **Predicciones vs. Reales**: `eda_plots/predicciones_vs_reales.png`.
- **Residuos**: `eda_plots/residuos_regresion.png`.
- **Importancia de Características**: `eda_plots/feature_importance_clasificacion.png`.

## 10. Interpretación de resultados y hallazgos

### 10.1. Significado de los resultados obtenidos
- Variables clave: Área construida (corr=0.769), Garajes (0.708), Baños (0.679), Estrato (0.669), Longitud (0.375).
- Patrones: Precios aumentan con estrato (Estrato 6: 4.35 SM promedio vs. Estrato 2: 1.23 SM); ubicación geográfica influye (zonas centrales más caras); proximidad a Transmilenio y supermercados positivos, distancia a vías principales negativa.
- Dataset: 5567 registros, modelo entrenado con 3688 registros limpios.

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
- Ejemplos: `correlation_heatmap.png` (matriz de correlación), `precios_por_estrato.png` (boxplot por estrato), `mapa_precios_geograficos.png` (scatter geográfico), `confusion_matrix_clasificacion.png`, `feature_importance_clasificacion.png`, `predicciones_vs_reales.png`, `residuos_regresion.png`.

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

### 13.2. Datasets
- Datos combinados (inmuebles_combinado_limpio.csv): 5567 registros de inmuebles en Bogotá con features geográficas y físicas.

### 13.3. Toolkits
- Scikit-learn v1.3+: https://scikit-learn.org/
- Pandas v2.0+: https://pandas.pydata.org/
- Seaborn v0.12+: https://seaborn.pydata.org/

### 13.4. Otros recursos
- OSMnx para datos geográficos: https://osmnx.readthedocs.io/
- Documentación de Python 3.8+: https://docs.python.org/