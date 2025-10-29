import pandas as pd

# Leer los CSVs limpios
df1 = pd.read_csv('inmuebles_bogota_limpio.csv')
df2 = pd.read_csv('inmuebles_metrocuadrado_limpio.csv')
df3 = pd.read_csv('inmuebles_apartamentos_bogota2_limpio.csv')
df4 = pd.read_csv('inmuebles_apartamentos_bogota3_limpio.csv')
df5 = pd.read_csv('inmuebles_apartamentos_bogota4_limpio.csv')
df6 = pd.read_csv('inmuebles_apartamentos_bogota5_limpio.csv')

print(f"Registros en df1: {len(df1)}")
print(f"Registros en df2: {len(df2)}")
print(f"Registros en df3: {len(df3)}")
print(f"Registros en df4: {len(df4)}")
print(f"Registros en df5: {len(df5)}")
print(f"Registros en df6: {len(df6)}")

# Combinar los dataframes
df_combinado = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)

print(f"Registros antes de filtrar outliers: {len(df_combinado)}")

# Función para remover outliers usando IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Filtrar outliers en columnas clave
df_combinado = remove_outliers_iqr(df_combinado, 'Valor_Arriendo_SM')
print(f"Registros después de filtrar outliers en Valor_Arriendo_SM: {len(df_combinado)}")

df_combinado = remove_outliers_iqr(df_combinado, 'Area Construida')
print(f"Registros después de filtrar outliers en Area Construida: {len(df_combinado)}")

# Filtrar outliers en distancias
df_combinado = remove_outliers_iqr(df_combinado, 'dist_transmilenio_cercana_m')
print(f"Registros después de filtrar outliers en dist_transmilenio_cercana_m: {len(df_combinado)}")

df_combinado = remove_outliers_iqr(df_combinado, 'dist_bus_cercana_m')
print(f"Registros después de filtrar outliers en dist_bus_cercana_m: {len(df_combinado)}")

df_combinado = remove_outliers_iqr(df_combinado, 'dist_farmacia_cercana_m')
print(f"Registros después de filtrar outliers en dist_farmacia_cercana_m: {len(df_combinado)}")

df_combinado = remove_outliers_iqr(df_combinado, 'dist_supermercado_cercano_m')
print(f"Registros después de filtrar outliers en dist_supermercado_cercano_m: {len(df_combinado)}")

df_combinado = remove_outliers_iqr(df_combinado, 'dist_via_principal_m')
print(f"Registros después de filtrar outliers en dist_via_principal_m: {len(df_combinado)}")

# Eliminar duplicados basados en coordenadas y valor de arriendo
df_combinado = df_combinado.drop_duplicates(subset=['latitud', 'longitud', 'Valor Arriendo'], keep='first')
print(f"Registros después de eliminar duplicados: {len(df_combinado)}")

# Guardar el CSV combinado
output_path = 'inmuebles_combinado_limpio.csv'
df_combinado.to_csv(output_path, index=False)

print(f"Archivo combinado guardado en: {output_path}")
print("Columnas finales:", list(df_combinado.columns))
print(f"Registros finales: {len(df_combinado)}")
