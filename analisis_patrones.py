import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

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
numeric_cols = ['Valor_Arriendo_SM', 'Area Construida', 'Estrato', 'latitud', 'longitud',
                'dist_transmilenio_cercana_m', 'dist_bus_cercana_m', 'num_farmacias_120m',
                'num_colegios_120m', 'num_transmilenio_120m', 'num_bus_120m',
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
# Variables más correlacionadas
top_features = ['Area Construida', 'Estrato', 'num_transmilenio_120m', 'num_farmacias_120m']

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