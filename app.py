import streamlit as st
import pandas as pd
import geopandas as gpd
import os
import folium
from streamlit_folium import st_folium
import json

st.set_page_config(
    page_title="Nuremberg Land Cover Dashboard",
    layout="wide"
)

st.title("Mapping Urban Change in Nuremberg")
st.write("Interactive dashboard for exploring land cover and urban change using machine learning.")

# -----------------------------
# DATA LOADING
# -----------------------------

@st.cache_data
def load_data():

    # Load yearly datasets
    df_2018 = pd.read_csv("clean_dataset_200m/2018_clean.csv")
    df_2020 = pd.read_csv("clean_dataset_200m/2020_clean.csv")
    df_2022 = pd.read_csv("clean_dataset_200m/2022_clean.csv")
    df_2024 = pd.read_csv("clean_dataset_200m/2024_clean.csv")

    # Load predicted dataset
    df_pred_2024 = pd.read_csv("predicted_landcover_2024_200m.csv")

    return df_2018, df_2020, df_2022, df_2024, df_pred_2024


df_2018, df_2020, df_2022, df_2024, df_pred_2024 = load_data()

# -----------------------------
# LOAD GRID GEOMETRY (2018)
# -----------------------------

grid_2018 = pd.read_excel(
    "data3with200mgridsize/nuremberg_grid_dataset_2018_CORINE_200m.xlsx"
)

# Convert geo column to GeoJSON geometry
grid_2018["geometry"] = grid_2018[".geo"].apply(lambda x: json.loads(x))

st.write("Grid geometry loaded:", len(grid_2018))

# -----------------------------
# LOAD CITY BOUNDARY
# -----------------------------

boundary = gpd.read_file("nuremberg_boundary/nuremberg_boundary.shp")

# -----------------------------
# DISPLAY BASIC INFO
# -----------------------------

st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("2018 Grid Cells", len(df_2018))
col2.metric("2020 Grid Cells", len(df_2020))
col3.metric("2022 Grid Cells", len(df_2022))
col4.metric("2024 Grid Cells", len(df_2024))

st.write("Predicted 2024 rows:", len(df_pred_2024))

# -----------------------------
# MAP SECTION
# -----------------------------

st.subheader("2018 Land Cover Map")

# Create map centered on Nuremberg
m = folium.Map(
    location=[49.4521, 11.0767],
    zoom_start=11
)

# Add boundary
folium.GeoJson(boundary).add_to(m)
# -----------------------------
# ADD GRID CELLS
# -----------------------------

for _, row in grid_2018.iterrows():

    folium.GeoJson(
        row["geometry"],
        style_function=lambda x: {
            "fillColor": "transparent",
            "color": "gray",
            "weight": 0.3
        }
    ).add_to(m)
    
# Display map
st_folium(m, width=900, height=600)

