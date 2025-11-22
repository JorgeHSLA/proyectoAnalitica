import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === Configuración general ===
sns.set(style="whitegrid", context="notebook")
plt.rcParams.update({
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# === Cargar datos ===
df = pd.read_csv('inmuebles_combinado_limpio.csv')

# Filtrar datos donde Estrato es 0 (datos sin estrato válido)
df = df[df['Estrato'] != 0]

# === Crear carpeta de salida ===
output_dir = Path("eda_plots")
output_dir.mkdir(exist_ok=True)

# === Histogramas de variables principales ===
hist_vars = [
    "Valor_Arriendo_SM", "Area Construida", "Estrato", "Cuartos", "Banos", "Garajes",
    "num_farmacias_120m", "num_colegios_120m", "num_transmilenio_120m", 
    "num_bus_120m", "num_universidades_300m"
]

plt.figure(figsize=(18, 12))
for i, var in enumerate(hist_vars, 1):
    plt.subplot(4, 3, i)
    data = df[var].dropna()
    plt.hist(data, bins=30, alpha=0.7, edgecolor='black', density=True)
    plt.title(f'Distribución de {var}')
    plt.xlabel(var)
    plt.ylabel('Densidad')

plt.tight_layout()
plt.savefig(output_dir / "histogramas_principales.png", dpi=300, bbox_inches="tight")
plt.show()

print("Histogramas principales generados en:", output_dir)

# === Comparación de variables respecto al precio de arriendo ===
price_vars = [
    "Area Construida", "Estrato", "Cuartos", "Banos", "Garajes",
    "dist_via_principal_m", "num_farmacias_120m",
    "num_colegios_120m", "num_transmilenio_120m", "num_bus_120m", "num_universidades_300m"
]

for var in price_vars:
    plt.figure(figsize=(8, 6))
    if var in ["Estrato", "Cuartos", "Banos", "Garajes"]:
        # Para variables discretas/categóricas, usar boxplot
        sns.boxplot(data=df, x=var, y='Valor_Arriendo_SM', hue=var, palette='Set2', legend=False)
    else:
        # Para otras variables, scatter plot
        sns.scatterplot(data=df, x=var, y='Valor_Arriendo_SM', alpha=0.6)
    plt.title(f'{var} vs Valor de Arriendo')
    plt.xlabel(var)
    plt.ylabel('Valor de Arriendo (SM)')
    filename = f"comparacion_{var.lower().replace(' ', '_')}_vs_precio.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()  # Cerrar la figura para liberar memoria

print("Gráficos de comparación vs precio generados en:", output_dir)

# === Diagramas de caja adicionales para las nuevas variables ===
# Box plot de Cuartos
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Cuartos', y='Valor_Arriendo_SM', hue='Cuartos', palette='Set3', legend=False)
plt.title('Diagrama de Caja: Valor de Arriendo por Número de Cuartos')
plt.xlabel('Número de Cuartos')
plt.ylabel('Valor de Arriendo (SM)')
plt.savefig(output_dir / "boxplot_arriendo_por_cuartos.png", dpi=300, bbox_inches="tight")
plt.show()

# Box plot de Baños
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Banos', y='Valor_Arriendo_SM', hue='Banos', palette='Set3', legend=False)
plt.title('Diagrama de Caja: Valor de Arriendo por Número de Baños')
plt.xlabel('Número de Baños')
plt.ylabel('Valor de Arriendo (SM)')
plt.savefig(output_dir / "boxplot_arriendo_por_banos.png", dpi=300, bbox_inches="tight")
plt.show()

# Box plot de Garajes
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Garajes', y='Valor_Arriendo_SM', hue='Garajes', palette='Set3', legend=False)
plt.title('Diagrama de Caja: Valor de Arriendo por Número de Garajes')
plt.xlabel('Número de Garajes')
plt.ylabel('Valor de Arriendo (SM)')
plt.savefig(output_dir / "boxplot_arriendo_por_garajes.png", dpi=300, bbox_inches="tight")
plt.show()

# === Diagramas de caja ===
# Box plot de Área Construida por Estrato
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Estrato', y='Area Construida', hue='Estrato', palette='Set2', legend=False)
plt.title('Diagrama de Caja: Área Construida por Estrato')
plt.xlabel('Estrato')
plt.ylabel('Área Construida (m²)')
plt.savefig(output_dir / "boxplot_area_por_estrato.png", dpi=300, bbox_inches="tight")
plt.show()

# Box plot general para Valor de Arriendo
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, y='Valor_Arriendo_SM', color='skyblue')
plt.title('Diagrama de Caja: Valor de Arriendo')
plt.ylabel('Valor del Arriendo (SM)')
plt.savefig(output_dir / "boxplot_arriendo_general.png", dpi=300, bbox_inches="tight")
plt.show()

print("Diagramas de caja generados en:", output_dir)
