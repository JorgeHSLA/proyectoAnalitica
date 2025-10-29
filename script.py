import pandas as pd

# Cambia la ruta si es necesario
csv_path = "c:/Users/Home/Desktop/proyectoAnalitica/Inmuebles_Disponibles_para_Arrendamiento_20251024.csv"

# Leer el archivo CSV y convertirlo en un DataFrame
df = pd.read_csv(csv_path)

# 1. vamos a limpiar los datos para que solo se evalue bogota, ya que es la ciudad
# que mas datos tiene y a las otras ciudades al tener muy pocos datos pueden en vez
# de ayudar hacer ruido, ademas de esto vamos a limitarlo a apartamento, ya que el calculo
# de este es mas facil de predecir que de otros tipos de inmuebles.

# Filtrar el DataFrame para que solo incluya filas donde la ciudad sea "Bogotá D.C."
df = df[df['Ciudad'] == 'Bogotá D.C.']

# Filtrar el DataFrame para que solo incluya filas donde el tipo de inmueble sea "Apartamento"
df = df[df['Tipo Inmueble'] == 'Apartamento']

# Reset index after filtering
df = df.reset_index(drop=True)
print(f"Registros tras filtro (Bogotá & Apartamento): {len(df)}")

# 2. hacemos feature engeneering para crear nuevas columnas que puedan ayudar a predecir mejor
# estas van a ser :
# - latitud (a base de la direccion)
# - longitud (a base de la direccion)
# - metros a la estacion mas cercana
# - n de farmacias en 500 metros
# - n de colegios
# - n de estaciones/paradas mas cercanas
# - metros a la farmacia mas cercana
# - metros al supermercado mas cercano
# - metros a la via principal mas cercana

# Implementación del feature engineering
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import osmnx as ox
import requests_cache
import re

# Función para limpiar direcciones
def clean_address(addr):
    if pd.isna(addr) or addr.strip() == "":
        return ""
    addr = str(addr).upper()
    # Quitar apartamentos, edificios, interiores, conjuntos
    addr = re.sub(r'\bAP \d+', '', addr)
    addr = re.sub(r'\bED \w+', '', addr)
    addr = re.sub(r'\bIN \d+', '', addr)
    addr = re.sub(r'\bCON \w+', '', addr)
    addr = re.sub(r'\bCO \w+', '', addr)
    addr = re.sub(r'\bLT \d+', '', addr)
    addr = re.sub(r'\bCS\b', '', addr)
    # Quitar espacios extra
    addr = ' '.join(addr.split())
    return addr

# Lista de barrios/localidades comunes de Bogotá
barrios_bogota = [
    "CHAPINERO", "USAQUÉN", "TEUSAQUILLO", "SANTA FE", "SAN CRISTÓBAL", "USME", "TUNJUELITO", "BOSA", "KENNEDY", "FONTIBÓN", "ENGATIVÁ", "SUBA", "BARRIOS UNIDOS", "PUENTE ARANDA", "CANDELARIA", "ANTONIO NARIÑO", "MARTIRES", "CIUDAD BOLÍVAR", "SUMAPAZ", "RAFAEL URIBE URIBE",
    "CHAPINERO ALTO", "ROSALES", "POTOSÍ", "BELLA SUIZA", "EL NOGAL", "LA SOLEDAD", "EL REFUGIO", "LA CABRERA", "MAZUREN", "SAN LUIS", "EL CASTILLO", "LA CAROLINA", "QUINTA CAMACHO", "SALITRE", "GALERÍAS", "PARQUE DE LA 93", "EL CHICO", "LA CALLEJA", "BOSQUE IZQUIERDO", "BOSQUE DERECHO"
]

def extract_barrio(addr):
    addr_upper = str(addr).upper()
    for barrio in barrios_bogota:
        if barrio in addr_upper:
            return barrio
    return None

def get_geocode_query(addr):
    barrio = extract_barrio(addr)
    if barrio:
        return f"{barrio}, Bogotá, Colombia"
    else:
        clean = clean_address(addr)
        return f"{clean}, Bogotá, Colombia" if clean else "Bogotá, Colombia"

# Config
PLACE_NAME = "Bogotá, Colombia"
GEOCODE_USER_AGENT = "proyecto_analitica"
MIN_DELAY_SECONDS = 1.0

# Geocoding
requests_cache.install_cache('geocode_cache', expire_after=60*60*24*30)
geolocator = Nominatim(user_agent=GEOCODE_USER_AGENT)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=2.0)

print("Geocodificando direcciones...")
latitudes, longitudes = [], []
for i, addr in enumerate(df['Direccion'].astype(str)):
    query = get_geocode_query(addr)
    try:
        loc = geocode(query, timeout=360)
        if loc:
            latitudes.append(loc.latitude)
            longitudes.append(loc.longitude)
        else:
            latitudes.append(None)
            longitudes.append(None)
    except:
        latitudes.append(None)
        longitudes.append(None)
    if (i+1) % 50 == 0:
        success_count = sum(1 for lat in latitudes if lat is not None)
        print(f"Procesadas {i+1}/{len(df)} direcciones")
        print(f"  Exitosas hasta ahora: {success_count}")

df["latitud"] = latitudes
df["longitud"] = longitudes

# Crear GeoDataFrame solo con filas válidas para cálculos
valid_mask = df["latitud"].notna() & df["longitud"].notna()
df_valid = df[valid_mask].copy()
print(f"Filas con coordenadas válidas: {len(df_valid)}")

if len(df_valid) > 0:
    gdf = gpd.GeoDataFrame(df_valid, geometry=gpd.points_from_xy(df_valid["longitud"], df_valid["latitud"]), crs="EPSG:4326")
    gdf_proj = gdf.to_crs("EPSG:32718")

    # Load POIs
    print("Cargando POIs desde OSM...")
    def load_pois(tags):
        try:
            pois = ox.features_from_place("Bogotá", tags)  # Cambiado a features_from_place para versiones nuevas de osmnx
            if pois.empty:
                print(f"No se encontraron POIs para {tags}")
                return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
            pois["geometry"] = pois.geometry.centroid
            pois = pois[pois.geometry.type == "Point"]
            pois = pois.to_crs("EPSG:32718")
            return pois
        except Exception as e:
            print(f"Error cargando POIs para {tags}: {e}")
            return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    farmacias = load_pois({"amenity": "pharmacy"})
    colegios = load_pois({"amenity": "school"})
    supermercados = load_pois({"shop": "supermarket"})
    estaciones_transmilenio = load_pois({"public_transport": "station"})
    estaciones_bus = load_pois({"highway": "bus_stop"})
    vias = load_pois({"highway": "primary"})
    if vias.empty:
        vias = load_pois({"highway": "trunk"})
    universidades = load_pois({"amenity": "university"})

    # Distances
    def nearest_distance(gdf_proj, pois_proj):
        if pois_proj.empty or len(pois_proj) == 0:
            return pd.Series([float("nan")] * len(gdf_proj))
        try:
            union = unary_union(pois_proj.geometry.values)
            distances = gdf_proj.geometry.distance(union)
            return distances
        except Exception as e:
            print(f"Error en nearest_distance: {e}")
            return pd.Series([float("nan")] * len(gdf_proj))

    print("Calculando distancias...")
    gdf_proj["dist_transmilenio_cercana_m"] = nearest_distance(gdf_proj, estaciones_transmilenio)
    gdf_proj["dist_bus_cercana_m"] = nearest_distance(gdf_proj, estaciones_bus)
    gdf_proj["dist_farmacia_cercana_m"] = nearest_distance(gdf_proj, farmacias)
    gdf_proj["dist_supermercado_cercano_m"] = nearest_distance(gdf_proj, supermercados)
    gdf_proj["dist_via_principal_m"] = nearest_distance(gdf_proj, vias)

    # Counts in 300m
    def count_within(gdf_proj, pois_proj, radius=300):
        if pois_proj.empty or len(pois_proj) == 0:
            return pd.Series([0] * len(gdf_proj))
        try:
            buffers = gdf_proj.geometry.buffer(radius)
            buffers_gdf = gpd.GeoDataFrame(geometry=buffers, crs=gdf_proj.crs)
            joined = gpd.sjoin(buffers_gdf, pois_proj[["geometry"]], how="left", predicate="contains")
            counts = joined.groupby(joined.index).size()
            counts = counts.reindex(gdf_proj.index, fill_value=0)
            return counts
        except Exception as e:
            print(f"Error en count_within: {e}")
            return pd.Series([0] * len(gdf_proj))

    print("Calculando conteos en 120m...")
    gdf_proj["num_farmacias_120m"] = count_within(gdf_proj, farmacias, radius=120)
    gdf_proj["num_colegios_120m"] = count_within(gdf_proj, colegios, radius=120)
    gdf_proj["num_transmilenio_120m"] = count_within(gdf_proj, estaciones_transmilenio, radius=120)
    gdf_proj["num_bus_120m"] = count_within(gdf_proj, estaciones_bus, radius=120)
    gdf_proj["num_universidades_300m"] = count_within(gdf_proj, universidades, radius=300)

    # Asignar las nuevas columnas a df
    df.loc[valid_mask, "dist_transmilenio_cercana_m"] = gdf_proj["dist_transmilenio_cercana_m"]
    df.loc[valid_mask, "dist_bus_cercana_m"] = gdf_proj["dist_bus_cercana_m"]
    df.loc[valid_mask, "dist_farmacia_cercana_m"] = gdf_proj["dist_farmacia_cercana_m"]
    df.loc[valid_mask, "dist_supermercado_cercano_m"] = gdf_proj["dist_supermercado_cercano_m"]
    df.loc[valid_mask, "dist_via_principal_m"] = gdf_proj["dist_via_principal_m"]
    df.loc[valid_mask, "num_farmacias_120m"] = gdf_proj["num_farmacias_120m"]
    df.loc[valid_mask, "num_colegios_120m"] = gdf_proj["num_colegios_120m"]
    df.loc[valid_mask, "num_transmilenio_120m"] = gdf_proj["num_transmilenio_120m"]
    df.loc[valid_mask, "num_bus_120m"] = gdf_proj["num_bus_120m"]
    df.loc[valid_mask, "num_universidades_300m"] = gdf_proj["num_universidades_300m"]
else:
    # Si no hay válidas, asignar NaN
    df["dist_transmilenio_cercana_m"] = float("nan")
    df["dist_bus_cercana_m"] = float("nan")
    df["dist_farmacia_cercana_m"] = float("nan")
    df["dist_supermercado_cercano_m"] = float("nan")
    df["dist_via_principal_m"] = float("nan")
    df["num_farmacias_120m"] = 0
    df["num_colegios_120m"] = 0
    df["num_transmilenio_120m"] = 0
    df["num_bus_120m"] = 0
    df["num_universidades_300m"] = 0



# Ahora df tiene todas las filas, con NaN en distancias para inválidas

# 3. le vamos a quitar las columnas que no aportan
# nada al inmueble, como columnas del propietario,
# codigo sin valor alguno u si esta disponible,
# la gracia es que quede con estas columnas:
# - Valor Arriendo
# - metros cuadrados
# - fecha de publicacion
# - estrato
# - latitud (a base de la direccion)
# - longitud (a base de la direccion)
# - metros a la estacion mas cercana
# - n de farmacias en 500 metros
# - n de colegios
# - n de estaciones/paradas mas cercanas
# - metros a la farmacia mas cercana
# - metros al supermercado mas cercano
# - metros a la via principal mas cercana

# --- Calcular precio en salarios mínimos según fecha de publicación ---
# Detectar columna de fecha (busca cualquier columna que contenga 'fecha')
fecha_col = None
for c in df.columns:
    if 'fecha' in c.lower():
        fecha_col = c
        break

if fecha_col:
    print(f"Usando columna de fecha para cálculo de SM: {fecha_col}")
    df['fecha_publicacion_parsed'] = pd.to_datetime(df[fecha_col], dayfirst=True, errors='coerce')
    df['anio_publicacion'] = df['fecha_publicacion_parsed'].dt.year
else:
    print("No se encontró columna de fecha; no se calculará precio en salarios mínimos.")
    df['anio_publicacion'] = pd.NA

# Mapa de salario mínimo por año (COP) proporcionado por el usuario
salaries = {
    2025: 1423500,
    2024: 1300000,
    2023: 1160000,
    2022: 1000000,
    2021: 908526,
    2020: 877803,
    2019: 828116,
    2018: 781242,
    2017: 737717,
    2016: 689455,
    2015: 644350,
    2014: 616000,
    2013: 589500,
    2012: 566700,
    2011: 535600,
    2010: 515000,
    2009: 496900,
    2008: 461500,
    2007: 433700,
    2006: 408000,
    2005: 381500
}

years_sorted = sorted(salaries.keys())
min_year, max_year = years_sorted[0], years_sorted[-1]

def get_salary_for_year(y):
    if pd.isna(y):
        return pd.NA
    try:
        yy = int(y)
    except:
        return pd.NA
    if yy in salaries:
        return salaries[yy]
    # If year is after max known, use max; if before min, use min; otherwise use nearest lower known
    if yy > max_year:
        return salaries[max_year]
    for candidate in reversed(years_sorted):
        if yy >= candidate:
            return salaries[candidate]
    return salaries[min_year]

df['salario_minimo_cop'] = df['anio_publicacion'].apply(get_salary_for_year)

# Parse Valor Arriendo a número (el original tiene comas, puntos y puede venir como string)
def parse_valor(v):
    if pd.isna(v):
        return pd.NA
    s = str(v)
    # quitar todo lo que no sea dígito o punto
    s = re.sub(r'[^0-9.]', '', s)
    if s == '':
        return pd.NA
    try:
        return float(s)
    except:
        return pd.NA

df['Valor_Arriendo_num'] = df['Valor Arriendo'].apply(parse_valor)

# Precio en salarios mínimos
df['Valor_Arriendo_SM'] = df.apply(lambda r: (r['Valor_Arriendo_num'] / r['salario_minimo_cop']) if pd.notna(r['Valor_Arriendo_num']) and pd.notna(r['salario_minimo_cop']) else pd.NA, axis=1)

# Seleccionar columnas finales (usar la columna de fecha encontrada si existe)
columnas_finales = [
    "Valor Arriendo",
    "Valor_Arriendo_SM",
    "Area Construida",
    "Estrato",
    "latitud",
    "longitud",
    "dist_transmilenio_cercana_m",
    "dist_bus_cercana_m",
    "num_farmacias_120m",
    "num_colegios_120m",
    "num_transmilenio_120m",
    "num_bus_120m",
    "dist_farmacia_cercana_m",
    "dist_supermercado_cercano_m",
    "dist_via_principal_m",
    "num_universidades_300m"
]

# Filtrar columnas existentes
columnas_finales = [c for c in columnas_finales if c and c in df.columns]

# Filtrar filas sin latitud/longitud (no guardar filas sin coordenadas válidas)
df = df.dropna(subset=['latitud', 'longitud'])

df_final = df[columnas_finales]

# Guardar
output_path = "c:/Users/Home/Desktop/proyectoAnalitica/inmuebles_bogota_limpio.csv"
df_final.to_csv(output_path, index=False)
print(f"Archivo guardado en: {output_path}")
print("Columnas finales:", columnas_finales)
print("Registros:", len(df_final))
