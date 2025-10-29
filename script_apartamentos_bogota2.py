import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
import osmnx as ox
import ast
import re

# Ruta del nuevo JSON
json_path = "c:/Users/Home/Desktop/proyectoAnalitica/apartamentos_bogota2.json"

# Leer el archivo JSON
df = pd.read_json(json_path)

# Filtrar para Bogotá D.C. y Apartamento
df['ciudad_str'] = df['mciudad'].apply(lambda x: x.get('nombre') if isinstance(x, dict) else str(x))
df['tipo_str'] = df['mtipoinmueble'].apply(lambda x: x.get('nombre') if isinstance(x, dict) else str(x))
df = df[df['ciudad_str'].str.contains('Bogotá D.C.', na=False)]
df = df[df['tipo_str'].str.contains('Apartamento', na=False)]

# Reset index
df = df.reset_index(drop=True)
print(f"Registros tras filtro (Bogotá & Apartamento): {len(df)}")

# Extraer latitud y longitud de 'localizacion'
def extract_lat_lon(loc_str):
    if pd.isna(loc_str) or not isinstance(loc_str, dict):
        return None, None
    return loc_str.get('lat'), loc_str.get('lon')

df['latitud'], df['longitud'] = zip(*df['localizacion'].apply(extract_lat_lon))

# Crear GeoDataFrame solo con filas válidas
valid_mask = df["latitud"].notna() & df["longitud"].notna()
df_valid = df[valid_mask].copy()
print(f"Filas con coordenadas válidas: {len(df_valid)}")

if len(df_valid) > 0:
    gdf = gpd.GeoDataFrame(df_valid, geometry=gpd.points_from_xy(df_valid["longitud"], df_valid["latitud"]), crs="EPSG:4326")
    gdf_proj = gdf.to_crs("EPSG:4686")  # Usar el mismo CRS que el .gpkg (MAGNA-SIRGAS)

    print("Cargando estratos desde manzanaestratificacion.gpkg...")
    gdf_estratos = gpd.read_file('manzanaestratificacion.gpkg')
    # Ya está en EPSG:4686, no re-proyectar

    # Spatial join para obtener estrato
    joined = gpd.sjoin(gdf_proj, gdf_estratos[['geometry', 'ESTRATO']], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep='first')]  # Evitar duplicados

    df.loc[valid_mask, 'Estrato'] = joined['ESTRATO']

    # Filtrar filas con Estrato inválido (0)
    df = df[df['Estrato'] != 0]

    # Load POIs (mismo código que antes)
    print("Cargando POIs desde OSM...")
    def load_pois(tags):
        try:
            pois = ox.features_from_place("Bogotá", tags)
            if pois.empty:
                print(f"No se encontraron POIs para {tags}")
                return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:3116")
            pois = pois.to_crs("EPSG:4686")  # Re-proyectar primero
            pois["geometry"] = pois.geometry.centroid
            pois = pois[pois.geometry.type == "Point"]
            pois = pois.to_crs("EPSG:3116")  # Reproyectar a proyectado para distancias
            return pois
        except Exception as e:
            print(f"Error cargando POIs para {tags}: {e}")
            return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:3116")

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
            # Reproyectar a CRS proyectado para distancias precisas
            gdf_proj_proj = gdf_proj.to_crs("EPSG:3116") if gdf_proj.crs != "EPSG:3116" else gdf_proj
            pois_proj_proj = pois_proj.to_crs("EPSG:3116") if pois_proj.crs != "EPSG:3116" else pois_proj
            union = unary_union(pois_proj_proj.geometry.values)
            distances = gdf_proj_proj.geometry.distance(union)
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
            # Reproyectar a CRS proyectado para buffers precisos
            gdf_proj_proj = gdf_proj.to_crs("EPSG:3116") if gdf_proj.crs != "EPSG:3116" else gdf_proj
            pois_proj_proj = pois_proj.to_crs("EPSG:3116") if pois_proj.crs != "EPSG:3116" else pois_proj
            buffers = gdf_proj_proj.geometry.buffer(radius)
            buffers_gdf = gpd.GeoDataFrame(geometry=buffers, crs="EPSG:3116")
            joined = gpd.sjoin(buffers_gdf, pois_proj_proj[["geometry"]], how="left", predicate="contains")
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

    # Asignar a df
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

# Calcular precio en SM (salario 2024 para todos, ya que es fecha actual)
salario_2024 = 1300000
df['salario_minimo_cop'] = salario_2024

# Parse Valor Arriendo
df['Valor_Arriendo_num'] = df['mvalorarriendo'].astype(float)

# Precio en SM
df['Valor_Arriendo_SM'] = df['Valor_Arriendo_num'] / df['salario_minimo_cop']

# Mapear columnas
df['Valor Arriendo'] = df['mvalorarriendo']
df['Area Construida'] = df['mareac']

# Columnas finales
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

# Filtrar filas sin lat/lon
df = df.dropna(subset=['latitud', 'longitud'])

df_final = df[columnas_finales]

# Guardar
output_path = "c:/Users/Home/Desktop/proyectoAnalitica/inmuebles_apartamentos_bogota2_limpio.csv"
df_final.to_csv(output_path, index=False)
print(f"Archivo guardado en: {output_path}")
print("Columnas finales:", columnas_finales)
print("Registros:", len(df_final))