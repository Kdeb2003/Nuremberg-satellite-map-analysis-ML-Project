import geopandas as gpd

# load geojson
gdf = gpd.read_file("boundary/nuremberg_boundary.geojson")

# keep only polygon / multipolygon geometries
gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

# save as shapefile
gdf.to_file("nuremberg_boundary.shp")