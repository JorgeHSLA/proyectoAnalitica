import json
import pandas as pd

# Leer el archivo JSON
json_file = "datosDeMetroCuadrado.json"
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convertir a DataFrame
df = pd.DataFrame(data)

# Guardar como CSV
csv_file = "datosDeMetroCuadrado.csv"
df.to_csv(csv_file, index=False, encoding="utf-8-sig")

print(f"Datos convertidos de {json_file} a {csv_file}")
print(f"Total de registros: {len(df)}")
