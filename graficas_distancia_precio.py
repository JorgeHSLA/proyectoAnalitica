import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Cargar datos
df = pd.read_csv('inmuebles_combinado_limpio.csv')

# Crear directorio para plots si no existe
plots_dir = 'eda_plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Columnas de distancia
dist_columns = [
    'dist_transmilenio_cercana_m',
    'dist_bus_cercana_m',
    'dist_farmacia_cercana_m',
    'dist_supermercado_cercano_m',
    'dist_via_principal_m'
]

# Generar scatter plots para cada columna de distancia vs precio
for col in dist_columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=col, y='Valor_Arriendo_SM', alpha=0.6)
    plt.title(f'Relación entre {col} y Valor Arriendo (SM)')
    plt.xlabel(col.replace('_', ' ').title())
    plt.ylabel('Valor Arriendo (SM)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Guardar plot
    filename = f'{plots_dir}/scatter_{col}_vs_precio.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Gráfico guardado: {filename}')

print('Todos los gráficos de distancia vs precio generados exitosamente.')