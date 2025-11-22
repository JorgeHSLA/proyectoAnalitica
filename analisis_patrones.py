import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score, accuracy_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Cargar datos
df = pd.read_csv('inmuebles_combinado_limpio.csv')

# Crear directorio para plots si no existe
plots_dir = 'eda_plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

print("=== ANÁLISIS DE PATRONES Y HALLAZGOS ===")
print(f"Dataset: {len(df)} registros")
print()

# 1. Análisis de correlaciones
print("1. MATRIZ DE CORRELACIÓN:")
numeric_cols = ['Valor_Arriendo_SM', 'Area Construida', 'Estrato', 'Cuartos', 'Banos', 'Garajes',
                'latitud', 'longitud', 'dist_transmilenio_cercana_m', 'dist_bus_cercana_m',
                'num_farmacias_120m', 'num_colegios_120m', 'num_transmilenio_120m', 'num_bus_120m',
                'dist_farmacia_cercana_m', 'dist_supermercado_cercano_m', 'dist_via_principal_m',
                'num_universidades_300m']

correlation_matrix = df[numeric_cols].corr()

# Crear matriz triangular inferior (escalonada)
correlation_matrix_lower = correlation_matrix.where(np.tril(np.ones(correlation_matrix.shape)).astype(bool))

print("Matriz de correlación escalonada (triangular inferior):")
print(correlation_matrix_lower.round(3))
print()

# Mostrar también las correlaciones más importantes con el precio
print("Correlaciones con Valor_Arriendo_SM (ordenadas):")
print(correlation_matrix['Valor_Arriendo_SM'].sort_values(ascending=False))
print()

# 2. Heatmap de correlaciones
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Matriz de Correlación - Variables vs Valor Arriendo')
plt.tight_layout()
plt.savefig(f'{plots_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Heatmap de correlaciones guardado en eda_plots/correlation_heatmap.png")
print()

# 3. Análisis por estratos
print("2. ANÁLISIS POR ESTRATOS:")
estrato_stats = df.groupby('Estrato')['Valor_Arriendo_SM'].agg(['mean', 'median', 'std', 'count'])
print(estrato_stats)
print()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Estrato', y='Valor_Arriendo_SM')
plt.title('Distribución de Precios por Estrato Socioeconómico')
plt.xlabel('Estrato')
plt.ylabel('Valor Arriendo (SM)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{plots_dir}/precios_por_estrato.png', dpi=300, bbox_inches='tight')
plt.close()
print("Boxplot de precios por estrato guardado")
print()

# 4. Análisis de ubicación geográfica
print("3. ANÁLISIS GEOGRÁFICO:")
# Crear bins para latitud y longitud
df['lat_bin'] = pd.cut(df['latitud'], bins=10)
df['lon_bin'] = pd.cut(df['longitud'], bins=10)

geo_analysis = df.groupby(['lat_bin', 'lon_bin'])['Valor_Arriendo_SM'].agg(['mean', 'count']).reset_index()
geo_analysis = geo_analysis[geo_analysis['count'] >= 5]  # Solo zonas con al menos 5 propiedades

print("Precios promedio por zona geográfica (filtrado por zonas con ≥5 propiedades):")
print(geo_analysis.sort_values('mean', ascending=False).head(10))
print()

# Scatter plot geográfico con colores por precio
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['longitud'], df['latitud'], c=df['Valor_Arriendo_SM'],
                     cmap='viridis', alpha=0.6, s=20)
plt.colorbar(scatter, label='Valor Arriendo (SM)')
plt.title('Distribución Geográfica de Precios de Arriendo en Bogotá')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{plots_dir}/mapa_precios_geograficos.png', dpi=300, bbox_inches='tight')
plt.close()
print("Mapa geográfico de precios guardado")
print()

# 5. Análisis de distancias vs precio
print("4. ANÁLISIS DE DISTANCIAS:")
dist_cols = ['dist_transmilenio_cercana_m', 'dist_bus_cercana_m', 'dist_farmacia_cercana_m',
             'dist_supermercado_cercano_m', 'dist_via_principal_m']

for col in dist_cols:
    corr = df[col].corr(df['Valor_Arriendo_SM'])
    print(f"Correlación {col}: {corr:.3f}")

    # Crear bins para análisis por rangos
    df[f'{col}_bin'] = pd.cut(df[col], bins=5)
    dist_analysis = df.groupby(f'{col}_bin')['Valor_Arriendo_SM'].mean()
    print(f"Precio promedio por rango de {col}:")
    print(dist_analysis)
    print()

# 6. Modelo de regresión simple para predicción
print("5. MODELO DE REGRESIÓN LINEAL:")
# Variables más correlacionadas (incluyendo las adicionales con correlación significativa)
top_features = ['Area Construida', 'Estrato', 'Cuartos', 'Banos', 'Garajes', 'longitud', 'latitud',
                'dist_via_principal_m', 'dist_supermercado_cercano_m', 'num_universidades_300m',
                'num_transmilenio_120m', 'num_farmacias_120m']

# Limpiar datos para el modelo (eliminar NaN e inf)
model_data = df[top_features + ['Valor_Arriendo_SM']].dropna()
model_data = model_data.replace([np.inf, -np.inf], np.nan).dropna()

print(f"Datos para modelo: {len(model_data)} registros (de {len(df)} total)")

if len(model_data) > 100:  # Solo si hay suficientes datos
    X = model_data[top_features]
    y = model_data['Valor_Arriendo_SM']

    # Agregar constante para statsmodels
    X_sm = sm.add_constant(X)

    # Modelo
    model = sm.OLS(y, X_sm).fit()
    print(model.summary())
    print()

    # Coeficientes significativos
    significant_coefs = model.pvalues[model.pvalues < 0.05]
    print("Coeficientes estadísticamente significativos (p < 0.05):")
    for var in significant_coefs.index:
        if var != 'const':
            coef = model.params[var]
            print(".3f")
    print()
else:
    print("Insuficientes datos limpios para modelo de regresión")
    print()

# 7. Hallazgos principales
print("6. HALLAZGOS PRINCIPALES:")
print("• Las variables más correlacionadas con el precio son:")
top_corr = correlation_matrix['Valor_Arriendo_SM'].abs().sort_values(ascending=False)[1:6]  # Excluir auto-correlación
for var, corr in top_corr.items():
    print(".3f")

print()
print("• Patrón por estratos: Los precios aumentan significativamente con el estrato socioeconómico")
print("• Ubicación: Ciertas zonas geográficas muestran precios consistentemente más altos")
print("• Servicios cercanos: La proximidad a Transmilenio y farmacias influye positivamente en el precio")
print("• Área construida: Correlación positiva fuerte con el precio")
print()

# 8. Recomendaciones para estimación de precios
print("7. RECOMENDACIONES PARA ESTIMACIÓN DE PRECIOS:")
print("• Usar modelo de regresión múltiple con las variables más significativas")
print("• Considerar estratificación geográfica para zonas específicas")
print("• Incluir análisis de outliers por ubicación")
print("• Validar modelo con datos de prueba separados")
print()

print("Análisis completado. Gráficos guardados en carpeta eda_plots/")

# === ANÁLISIS COMPLEMENTARIOS ===
print("\n" + "="*60)
print("=== ANÁLISIS COMPLEMENTARIOS ===")
print("="*60 + "\n")

# 8. CLASIFICACIÓN: Categorizar precios en bajo, medio, alto
print("8. ANÁLISIS DE CLASIFICACIÓN (Random Forest):")
print("-" * 60)

# Crear categorías de precio
df['Categoria_Precio'] = pd.cut(df['Valor_Arriendo_SM'], 
                                 bins=[0, 2, 4, float('inf')], 
                                 labels=['Bajo', 'Medio', 'Alto'])

# Preparar datos para clasificación
clf_features = ['Area Construida', 'Estrato', 'Cuartos', 'Banos', 'Garajes', 
                'latitud', 'longitud', 'dist_via_principal_m', 
                'dist_supermercado_cercano_m', 'num_universidades_300m']

clf_data = df[clf_features + ['Categoria_Precio']].dropna()
clf_data = clf_data.replace([np.inf, -np.inf], np.nan).dropna()

print(f"Datos para clasificación: {len(clf_data)} registros")
print(f"Distribución de categorías:")
print(clf_data['Categoria_Precio'].value_counts())
print()

if len(clf_data) > 100:
    X_clf = clf_data[clf_features]
    y_clf = clf_data['Categoria_Precio']
    
    # Split train/test
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    # Entrenar Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_clf.fit(X_train_clf, y_train_clf)
    
    # Predicciones
    y_pred_clf = rf_clf.predict(X_test_clf)
    
    # Métricas
    accuracy = accuracy_score(y_test_clf, y_pred_clf)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test_clf, y_pred_clf))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test_clf, y_pred_clf)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Bajo', 'Medio', 'Alto'],
                yticklabels=['Bajo', 'Medio', 'Alto'])
    plt.title('Matriz de Confusión - Clasificación de Precios')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predicho')
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/confusion_matrix_clasificacion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Matriz de confusión guardada")
    
    # Importancia de características
    feature_importance = pd.DataFrame({
        'feature': clf_features,
        'importance': rf_clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nImportancia de características (Top 5):")
    print(feature_importance.head())
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title('Importancia de Características - Clasificación')
    plt.xlabel('Importancia')
    plt.ylabel('Característica')
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/feature_importance_clasificacion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico de importancia de características guardado\n")

# 9. CLUSTERING: Identificar grupos de inmuebles similares
print("9. ANÁLISIS DE CLUSTERING (K-Means):")
print("-" * 60)

# Preparar datos para clustering
cluster_features = ['Area Construida', 'Estrato', 'Cuartos', 'Banos', 'Garajes',
                    'Valor_Arriendo_SM', 'latitud', 'longitud']

cluster_data = df[cluster_features].dropna()
cluster_data = cluster_data.replace([np.inf, -np.inf], np.nan).dropna()

print(f"Datos para clustering: {len(cluster_data)} registros")

if len(cluster_data) > 100:
    # Normalizar datos
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(cluster_data)
    
    # Método del codo para determinar k óptimo
    inertias = []
    silhouette_scores = []
    K_range = range(2, 8)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_cluster)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_cluster, kmeans.labels_))
    
    # Gráfico del codo
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(K_range, inertias, 'bo-')
    ax1.set_xlabel('Número de Clusters (k)')
    ax1.set_ylabel('Inercia')
    ax1.set_title('Método del Codo')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(K_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Número de Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score por k')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/elbow_method_clustering.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico del método del codo guardado")
    
    # Mejor k según silhouette score
    best_k = K_range[np.argmax(silhouette_scores)]
    best_silhouette = max(silhouette_scores)
    print(f"\nMejor k según Silhouette Score: {best_k} (score: {best_silhouette:.4f})")
    
    # Aplicar clustering con mejor k
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_data['Cluster'] = kmeans_final.fit_predict(X_cluster)
    
    # Análisis de clusters
    print(f"\nDistribución de inmuebles por cluster:")
    print(cluster_data['Cluster'].value_counts().sort_index())
    
    print(f"\nCaracterísticas promedio por cluster:")
    cluster_summary = cluster_data.groupby('Cluster').mean()
    print(cluster_summary.round(2))
    
    # Visualización geográfica de clusters
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(cluster_data['longitud'], cluster_data['latitud'], 
                         c=cluster_data['Cluster'], cmap='tab10', alpha=0.6, s=30)
    plt.colorbar(scatter, label='Cluster', ticks=range(best_k))
    plt.title(f'Distribución Geográfica de Clusters (k={best_k})')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/clusters_geograficos.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Mapa de clusters guardado")
    
    # Boxplot de precios por cluster
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=cluster_data, x='Cluster', y='Valor_Arriendo_SM', palette='Set2')
    plt.title('Distribución de Precios por Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Valor Arriendo (SM)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/precios_por_cluster.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Boxplot de precios por cluster guardado\n")

# 10. VALIDACIÓN CRUZADA Y MÉTRICAS AVANZADAS PARA REGRESIÓN
print("10. VALIDACIÓN CRUZADA - MODELO DE REGRESIÓN:")
print("-" * 60)

if len(model_data) > 100:
    X_reg = model_data[top_features]
    y_reg = model_data['Valor_Arriendo_SM']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Modelo de regresión lineal
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Predicciones
    y_pred_train = lr_model.predict(X_train)
    y_pred_test = lr_model.predict(X_test)
    
    # Métricas en conjunto de entrenamiento
    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    
    # Métricas en conjunto de prueba
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print("Métricas en conjunto de ENTRENAMIENTO:")
    print(f"  R² Score: {r2_train:.4f}")
    print(f"  RMSE: {rmse_train:.4f} SM")
    print(f"  MAE: {mae_train:.4f} SM")
    
    print("\nMétricas en conjunto de PRUEBA:")
    print(f"  R² Score: {r2_test:.4f}")
    print(f"  RMSE: {rmse_test:.4f} SM")
    print(f"  MAE: {mae_test:.4f} SM")
    
    # Validación cruzada (K-Fold)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lr_model, X_reg, y_reg, cv=kfold, scoring='r2')
    
    print(f"\nValidación Cruzada (5-Fold):")
    print(f"  R² Scores: {cv_scores}")
    print(f"  R² Promedio: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Gráfico de predicciones vs valores reales
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5, s=30)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valor Real (SM)')
    plt.ylabel('Valor Predicho (SM)')
    plt.title(f'Predicciones vs Valores Reales (R²={r2_test:.4f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/predicciones_vs_reales.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico de predicciones vs reales guardado")
    
    # Residuos
    residuos = y_test - y_pred_test
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_test, residuos, alpha=0.5, s=30)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Valor Predicho (SM)')
    plt.ylabel('Residuos')
    plt.title('Gráfico de Residuos')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/residuos_regresion.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico de residuos guardado")
    
    # Random Forest Regressor para comparación
    print("\n" + "-" * 60)
    print("Comparación con Random Forest Regressor:")
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_reg.fit(X_train, y_train)
    
    y_pred_rf_test = rf_reg.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))
    mae_rf = mean_absolute_error(y_test, y_pred_rf_test)
    
    print(f"  R² Score: {r2_rf:.4f}")
    print(f"  RMSE: {rmse_rf:.4f} SM")
    print(f"  MAE: {mae_rf:.4f} SM")
    
    # Comparación de modelos
    print("\n" + "-" * 60)
    print("COMPARACIÓN DE MODELOS:")
    comparison = pd.DataFrame({
        'Modelo': ['Regresión Lineal', 'Random Forest'],
        'R² Test': [r2_test, r2_rf],
        'RMSE Test': [rmse_test, rmse_rf],
        'MAE Test': [mae_test, mae_rf]
    })
    print(comparison.to_string(index=False))
    print()

print("\n" + "="*60)
print("ANÁLISIS COMPLETO FINALIZADO")
print("="*60)
print(f"\nTodos los gráficos y resultados guardados en: {plots_dir}/")
print("\nArchivos generados:")
print("  - correlation_heatmap.png")
print("  - precios_por_estrato.png")
print("  - mapa_precios_geograficos.png")
print("  - confusion_matrix_clasificacion.png")
print("  - feature_importance_clasificacion.png")
print("  - elbow_method_clustering.png")
print("  - clusters_geograficos.png")
print("  - precios_por_cluster.png")
print("  - predicciones_vs_reales.png")
print("  - residuos_regresion.png")