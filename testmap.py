import geopandas as gpd
import matplotlib.pyplot as plt

gdf = gpd.read_file("boundary/nuremberg_boundary.geojson")

print(gdf)  # just to confirm it loaded

gdf.plot()
plt.show()