import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import json
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from chatbot import generate_response, build_context
# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Nuremberg Land Cover Dashboard",
    layout="wide"
)
# -----------------------------
# DARK GRADIENT THEME (PINK / BLUE)
# -----------------------------
st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e6e6f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e, #16213e);
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* Headers */
h1, h2, h3, h4 {
    color: #ff4da6; /* neon pink */
    font-weight: 700;
}

/* Cards / containers */
div[data-testid="stMetric"],
div[data-testid="stPlotlyChart"],
div[data-testid="stDataFrame"],
div[data-testid="stVerticalBlock"] > div {
    background: rgba(20, 20, 40, 0.6);
    border-radius: 14px;
    padding: 12px;
    backdrop-filter: blur(12px);
    box-shadow: 0 0 25px rgba(255, 77, 166, 0.08);
    border: 1px solid rgba(255,255,255,0.05);
}

/* Buttons */
button[kind="primary"] {
    background: linear-gradient(90deg, #ff4da6, #4da6ff);
    border: none;
    border-radius: 10px;
    color: white;
    font-weight: 600;
}

/* Inputs */
input, textarea, .stSelectbox, .stMultiSelect {
    background: rgba(255,255,255,0.08) !important;
    color: white !important;
    border-radius: 8px !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #66b3ff;
    font-weight: bold;
}

/* Chat */
.stChatMessage {
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 8px;
}

/* Text tweaks */
p, label, span {
    color: #dcdcf0 !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(#ff4da6, #4da6ff);
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)
st.title("Mapping Urban Change in Nuremberg")
st.write("Interactive dashboard for exploring land cover and urban change using machine learning.")

st.sidebar.title("Land Cover Dashboard ")
view_mode = st.sidebar.radio(
    "View Mode",
    options=["Single Year", "Multiple Years"],
    index=0
)

selected_year = st.sidebar.selectbox(
    "Select Year",
    options=["2020", "2021", "2024"],
    index=0,
    disabled=(view_mode == "Multiple Years")
)
selected_model = None
compare_both_models = False
first_year = None
second_year = None

if view_mode == "Single Year" and selected_year == "2021":
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=["MLP", "Ridge"],
        index=0
    )
    compare_both_models = st.sidebar.checkbox(
        "Compare Both Models",
        value=False
    )
elif view_mode == "Multiple Years":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Multiple Years")
    year_options = ["2020", "2021", "2024"]
    first_year = st.sidebar.selectbox("1st Year", options=year_options, index=0)
    second_options = [y for y in year_options if y != first_year]
    second_year = st.sidebar.selectbox("2nd Year", options=second_options, index=0)
    if first_year in ["2021", "2024"] or second_year in ["2021", "2024"]:
        selected_model = st.sidebar.selectbox(
            "Select Model",
            options=["MLP", "Ridge"],
            index=0
        )

# -----------------------------
# DATA LOADING
# -----------------------------

@st.cache_data
def load_data():
    # Actual data (2020 only)
    df_2020 = pd.read_csv("clean_dataset_200m/2020_clean.csv")

    # Predictions + actual labels (2021)
    df_2021 = pd.read_csv("predictions_2021.csv")

    # Safety checks for alignment and completeness (2021)
    required_cols_2021 = ["label", "mlp_pred", "ridge_pred"]
    for c in required_cols_2021:
        assert c in df_2021.columns, f"Missing required column in predictions_2021.csv: {c}"
    assert len(df_2021["label"]) == len(df_2021["mlp_pred"]), "Mismatch in prediction length (mlp_pred)"
    assert len(df_2021["label"]) == len(df_2021["ridge_pred"]), "Mismatch in prediction length (ridge_pred)"
    assert df_2021["label"].notna().all(), "Missing labels detected in predictions_2021.csv"
    assert df_2021["mlp_pred"].notna().all(), "Missing MLP predictions detected in predictions_2021.csv"
    assert df_2021["ridge_pred"].notna().all(), "Missing Ridge predictions detected in predictions_2021.csv"

    # Predictions only (2024)
    df_2024 = pd.read_csv("predictions_2024.csv")

    # Fix 2024 geometry for map rendering: predictions_2024 has empty MultiPoint geometries.
    # Reattach polygon grid geometry from clean dataset using system:index.
    df_2024_geo = pd.read_csv("clean_dataset_200m/2024_clean.csv", usecols=["system:index", ".geo"])
    df_2024 = df_2024.merge(df_2024_geo, on="system:index", how="left", suffixes=("", "_clean"))
    df_2024[".geo"] = df_2024[".geo_clean"].fillna(df_2024[".geo"])
    df_2024 = df_2024.drop(columns=[".geo_clean"])

    return df_2020, df_2021, df_2024


df_2020, df_2021, df_2024 = load_data()

year_to_df = {
    "2020": df_2020,
    "2021": df_2021,
    "2024": df_2024
}
selected_df = year_to_df[selected_year]

selected_label_column = "label"
if view_mode == "Single Year" and selected_year == "2021":
    selected_label_column = "mlp_pred" if selected_model == "MLP" else "ridge_pred"
if view_mode == "Single Year" and selected_year == "2024":
    selected_label_column = "mlp_pred"

# -----------------------------
# LOAD GRID GEOJSON (OPTIMIZED)
# -----------------------------

@st.cache_data
def load_grid_geojson(df_labels, label_column="label"):
    # Robust label to color mapping:
    # 1) unified labels (0,1,2,3)
    # 2) WorldCover labels (10,20,...)
    # 3) CORINE labels (111,112,...)
    def normalize_label(raw_label):
        try:
            label = int(raw_label)
        except (TypeError, ValueError):
            return 3  # fallback to "other"

        # Unified labels
        if label in [0, 1, 2, 3]:
            return label

        # WorldCover -> unified
        if label in [10, 30, 40]:
            return 0  # vegetation
        if label == 50:
            return 1  # built-up
        if label == 80:
            return 2  # water

        # CORINE -> unified
        if label in [311, 312, 313, 211, 231]:
            return 0  # vegetation
        if label in [111, 112, 121, 122]:
            return 1  # built-up
        if label == 512:
            return 2  # water

        return 3  # other

    color_map = {
        0: "#2e7d32",  # vegetation
        1: "#d32f2f",  # built-up
        2: "#1976d2",  # water
        3: "#757575",  # other
    }

    # Parse and normalize geometry payload from '.geo'
    def parse_geometry(geo_value):
        if pd.isna(geo_value):
            return None
        try:
            obj = json.loads(geo_value)
        except Exception:
            return None

        # Typical GeoJSON geometry
        if isinstance(obj, dict) and obj.get("type") in ["Polygon", "MultiPolygon"]:
            return {
                "type": obj["type"],
                "coordinates": obj["coordinates"]
            }

        # GeoJSON feature wrapper
        if isinstance(obj, dict) and obj.get("type") == "Feature" and "geometry" in obj:
            geom = obj.get("geometry")
            if isinstance(geom, dict) and geom.get("type") in ["Polygon", "MultiPolygon"]:
                return {
                    "type": geom["type"],
                    "coordinates": geom["coordinates"]
                }

        return None

    # Convert EPSG:3857 meters -> EPSG:4326 lon/lat for Leaflet
    def mercator_to_wgs84(x, y):
        r = 6378137.0
        lon = (x / r) * (180.0 / math.pi)
        lat = (2.0 * math.atan(math.exp(y / r)) - math.pi / 2.0) * (180.0 / math.pi)
        return [lon, lat]

    def maybe_convert_coords(coords):
        if not coords:
            return coords

        # Detect projected meters (not lon/lat)
        sample = coords[0][0] if isinstance(coords[0][0], list) else coords[0]
        sx, sy = sample[0], sample[1]
        is_projected = abs(sx) > 180 or abs(sy) > 90
        if not is_projected:
            return coords

        # Polygon: [ [ [x,y], ... ] , ... ]
        if isinstance(coords[0][0], list):
            return [[mercator_to_wgs84(x, y) for x, y in ring] for ring in coords]

        # Ring fallback: [ [x,y], ... ]
        return [mercator_to_wgs84(x, y) for x, y in coords]

    def normalize_geometry_crs(geometry):
        gtype = geometry.get("type")
        coords = geometry.get("coordinates")
        if gtype == "Polygon":
            return {"type": "Polygon", "coordinates": maybe_convert_coords(coords)}
        if gtype == "MultiPolygon":
            converted = []
            for poly in coords:
                converted.append(maybe_convert_coords(poly))
            return {"type": "MultiPolygon", "coordinates": converted}
        return geometry

    geojson_data = {
        "type": "FeatureCollection",
        "features": []
    }

    for _, row in df_labels.iterrows():
        geometry = parse_geometry(row.get(".geo"))
        if geometry is None:
            continue
        geometry = normalize_geometry_crs(geometry)

        raw_label = row.get(label_column, 3)

        label = normalize_label(raw_label)
        color = color_map[label]

        geojson_data["features"].append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "label": int(label),
                "raw_label": int(raw_label) if pd.notna(raw_label) else None,
                "color": color
            }
        })

    return geojson_data


geojson_data = load_grid_geojson(selected_df, label_column=selected_label_column)

# -----------------------------
# LOAD CITY BOUNDARY
# -----------------------------

@st.cache_data
def load_boundary():
    return gpd.read_file("nuremberg_boundary/nuremberg_boundary.shp")


boundary = load_boundary()


def normalize_label_value(raw_label):
    try:
        label = int(raw_label)
    except (TypeError, ValueError):
        return 3

    if label in [0, 1, 2, 3]:
        return label
    if label in [10, 30, 40]:
        return 0
    if label == 50:
        return 1
    if label == 80:
        return 2
    if label in [311, 312, 313, 211, 231]:
        return 0
    if label in [111, 112, 121, 122]:
        return 1
    if label == 512:
        return 2
    return 3


def render_legend():
    st.markdown(
        """
        <div style="margin-top: 8px; margin-bottom: 8px;">
          <strong>Legend:</strong>
          <span style="margin-left: 10px;"><span style="display:inline-block;width:12px;height:12px;background:#2e7d32;border:1px solid #444;"></span> Vegetation</span>
          <span style="margin-left: 10px;"><span style="display:inline-block;width:12px;height:12px;background:#d32f2f;border:1px solid #444;"></span> Built-up</span>
          <span style="margin-left: 10px;"><span style="display:inline-block;width:12px;height:12px;background:#1976d2;border:1px solid #444;"></span> Water</span>
          <span style="margin-left: 10px;"><span style="display:inline-block;width:12px;height:12px;background:#757575;border:1px solid #444;"></span> Other</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_landcover_pie(df_source, label_column, title):
    label_names = {
        0: "Vegetation",
        1: "Built-up",
        2: "Water",
        3: "Other"
    }
    label_colors = {
        0: "#2e7d32",
        1: "#d32f2f",
        2: "#1976d2",
        3: "#757575"
    }

    normalized = df_source[label_column].apply(normalize_label_value)
    counts = normalized.value_counts().reindex([0, 1, 2, 3], fill_value=0)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    wedges, _ = ax.pie(
        counts.values,
        labels=None,
        colors=[label_colors[i] for i in counts.index],
        startangle=90
    )

    total = counts.sum()
    label_candidates = []
    for i, w in enumerate(wedges):
        if total == 0:
            continue
        pct = (counts.values[i] / total) * 100.0
        if pct <= 0:
            continue

        angle = (w.theta2 + w.theta1) / 2.0
        x = math.cos(math.radians(angle))
        y = math.sin(math.radians(angle))
        side = "right" if x >= 0 else "left"
        label_candidates.append({
            "i": i,
            "x": x,
            "y": y,
            "side": side,
            "y_text": 1.18 * y,
            "text": f"{label_names[counts.index[i]]}: {pct:.1f}%"
        })

    # Simple overlap avoidance: enforce vertical spacing per side.
    min_gap = 0.12
    y_min, y_max = -1.25, 1.25
    for side in ["left", "right"]:
        side_labels = [d for d in label_candidates if d["side"] == side]
        side_labels.sort(key=lambda d: d["y_text"])
        prev_y = None
        for d in side_labels:
            y = max(y_min, min(y_max, d["y_text"]))
            if prev_y is not None and y - prev_y < min_gap:
                y = prev_y + min_gap
            d["y_text"] = min(y, y_max)
            prev_y = d["y_text"]

    for d in label_candidates:
        x_text = 1.36 if d["side"] == "right" else -1.36
        ha = "left" if d["side"] == "right" else "right"
        rad = 0.10 if d["side"] == "right" else -0.10
        ax.annotate(
            d["text"],
            xy=(d["x"], d["y"]),
            xytext=(x_text, d["y_text"]),
            ha=ha,
            va="center",
            fontsize=9,
            arrowprops=dict(
                arrowstyle="-",
                color="#444444",
                lw=0.9,
                connectionstyle=f"arc3,rad={rad}"
            )
        )

    fig.subplots_adjust(bottom=0.16)
    fig.text(0.5, 0.04, title, ha="center", va="center", fontsize=11)
    ax.axis("equal")
    st.pyplot(fig)
    plt.close(fig)


def render_landcover_map(geojson_payload):
    m = folium.Map(
        location=[49.4521, 11.0767],
        zoom_start=11
    )

    folium.GeoJson(
        geojson_payload,
        style_function=lambda feature: {
            "fillColor": feature["properties"]["color"],
            "color": None,
            "weight": 0.3,
            "fillOpacity": 0.75
        }
    ).add_to(m)

    # Keep boundary on top of grid
    folium.GeoJson(
        boundary,
        style_function=lambda _: {
            "color": "#1e40af",
            "weight": 2.5,
            "fillOpacity": 0
        }
    ).add_to(m)

    st_folium(m, width=700, height=520)


def render_accuracy_comparison(y_true, y_pred_mlp, y_pred_ridge):
    acc_mlp = (pd.Series(y_true).values == pd.Series(y_pred_mlp).values).mean()
    acc_ridge = (pd.Series(y_true).values == pd.Series(y_pred_ridge).values).mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    models = ["MLP", "Ridge"]
    scores = [acc_mlp, acc_ridge]
    bars = ax.bar(models, scores, color=["#1f77b4", "#ff7f0e"], width=0.55)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.02, f"{score*100:.1f}%", ha="center", fontsize=10)
    st.pyplot(fig)
    plt.close(fig)


def render_confusion_matrix(y_true, y_pred, title):
    classes = [0, 1, 2, 3]
    cm = pd.crosstab(
        pd.Categorical(y_true, categories=classes),
        pd.Categorical(y_pred, categories=classes),
        dropna=False
    ).values

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)
    plt.close(fig)


def render_distribution_chart(y_true, y_pred_mlp, y_pred_ridge, compare_mode, selected_model_name):
    classes = [0, 1, 2, 3]
    actual_counts = pd.Series(y_true).value_counts().reindex(classes, fill_value=0)
    mlp_counts = pd.Series(y_pred_mlp).value_counts().reindex(classes, fill_value=0)
    ridge_counts = pd.Series(y_pred_ridge).value_counts().reindex(classes, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(classes))
    width = 0.25 if compare_mode else 0.35

    ax.bar(x - width, actual_counts.values, width, label="Actual", color="#2ca02c")
    if compare_mode:
        ax.bar(x, mlp_counts.values, width, label="MLP Predicted", color="#1f77b4")
        ax.bar(x + width, ridge_counts.values, width, label="Ridge Predicted", color="#ff7f0e")
    else:
        if selected_model_name == "MLP":
            ax.bar(x, mlp_counts.values, width, label="MLP Predicted", color="#1f77b4")
        else:
            ax.bar(x, ridge_counts.values, width, label="Ridge Predicted", color="#ff7f0e")

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_xlabel("Class")
    ax.set_ylabel("Grid Cell Count")
    ax.set_title("Actual vs Predicted Distribution")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def render_class_distribution_bar(df_source, label_column, title):
    labels = [0, 1, 2, 3]
    label_names = ["Vegetation", "Built-up", "Water", "Other"]
    color_map = ["#2e7d32", "#d32f2f", "#1976d2", "#757575"]

    normalized = df_source[label_column].apply(normalize_label_value)
    counts = normalized.value_counts().reindex(labels, fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(label_names, counts.values, color=color_map, width=0.6)
    ax.set_title(title)
    ax.set_ylabel("Grid Cell Count")
    for b, c in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{int(c)}", ha="center", va="bottom", fontsize=9)
    st.pyplot(fig)
    plt.close(fig)


def render_feature_histograms(df_source, show_heading=True):
    if show_heading:
        st.markdown("### NDVI / Feature Distribution")
    col1, col2, col3 = st.columns(3)
    feature_cols = ["NDVI", "NDBI", "NDWI"]
    feature_colors = ["#2e7d32", "#d32f2f", "#1976d2"]

    for col, feature, color in zip([col1, col2, col3], feature_cols, feature_colors):
        with col:
            if feature in df_source.columns:
                data = pd.to_numeric(df_source[feature], errors="coerce").dropna()
                fig, ax = plt.subplots(figsize=(4.2, 3.2))
                ax.hist(data, bins=30, color=color, alpha=0.85, edgecolor="white")
                ax.set_title(feature)
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
                plt.close(fig)


def render_ndvi_boxplot_by_class(df_source, label_column, show_heading=True):
    if "NDVI" not in df_source.columns:
        return

    if show_heading:
        st.markdown("### NDVI vs Land Cover Class")
    temp = df_source.copy()
    temp["class_norm"] = temp[label_column].apply(normalize_label_value)
    temp["NDVI"] = pd.to_numeric(temp["NDVI"], errors="coerce")
    temp = temp.dropna(subset=["NDVI"])

    labels = [0, 1, 2, 3]
    label_names = ["Vegetation", "Built-up", "Water", "Other"]
    grouped = [temp.loc[temp["class_norm"] == c, "NDVI"].values for c in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(grouped, patch_artist=True, tick_labels=label_names)
    box_colors = ["#2e7d32", "#d32f2f", "#1976d2", "#757575"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title("Feature vs Class Box Plot (NDVI)")
    ax.set_ylabel("NDVI")
    st.pyplot(fig)
    plt.close(fig)


def render_area_metrics(df_source, label_column, show_heading=True):
    normalized = df_source[label_column].apply(normalize_label_value)
    total = len(normalized)
    vegetation = int((normalized == 0).sum())
    built_up = int((normalized == 1).sum())

    if show_heading:
        st.markdown("### Area Count")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Grid Cells", f"{total}")
    m2.metric("Vegetation Count", f"{vegetation}")
    m3.metric("Built-up Count", f"{built_up}")


def render_correlation_heatmap(df_source):
    st.markdown("### Correlation Heatmap")
    feature_cols = ["B2", "B3", "B4", "B8", "B11", "NDVI", "NDBI", "NDWI"]
    existing = [c for c in feature_cols if c in df_source.columns]
    if len(existing) < 2:
        return

    data = df_source[existing].apply(pd.to_numeric, errors="coerce")
    corr = data.corr()

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(existing)))
    ax.set_yticks(range(len(existing)))
    ax.set_xticklabels(existing, rotation=45, ha="right")
    ax.set_yticklabels(existing)
    ax.set_title("Feature Correlation Matrix")
    for i in range(len(existing)):
        for j in range(len(existing)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)
    plt.close(fig)


def render_full_analytics_block(df_source, label_column, title_prefix):
    render_landcover_pie(df_source, label_column, f"{title_prefix} Land Cover Composition")
    render_class_distribution_bar(df_source, label_column, f"{title_prefix} Class Distribution")
    render_feature_histograms(df_source)
    render_ndvi_boxplot_by_class(df_source, label_column)
    render_area_metrics(df_source, label_column)


def get_class_counts(df_source, label_column="label"):
    labels = [0, 1, 2, 3]
    return df_source[label_column].apply(normalize_label_value).value_counts().reindex(labels, fill_value=0)


def render_class_distribution_comparison(df_source, actual_col, pred_col, model_name):
    classes = [0, 1, 2, 3]
    class_names = ["Vegetation", "Built-up", "Water", "Other"]
    actual_counts = df_source[actual_col].apply(normalize_label_value).value_counts().reindex(classes, fill_value=0)
    pred_counts = df_source[pred_col].apply(normalize_label_value).value_counts().reindex(classes, fill_value=0)

    x = np.arange(len(class_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, actual_counts.values, width, label="Actual", color="#2ca02c")
    ax.bar(x + width / 2, pred_counts.values, width, label=f"Predicted ({model_name})", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Grid Cell Count")
    ax.set_title("Class Distribution Comparison")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def render_feature_distribution_comparison(df_source, model_name):
    st.markdown("### Feature Distribution Comparison")
    features = ["NDVI", "NDBI", "NDWI"]

    st.markdown("**Actual**")
    row1 = st.columns(3)
    for col, feature, color in zip(row1, features, ["#2e7d32", "#d32f2f", "#1976d2"]):
        with col:
            if feature in df_source.columns:
                vals = pd.to_numeric(df_source[feature], errors="coerce").dropna()
                fig, ax = plt.subplots(figsize=(4.0, 3.0))
                ax.hist(vals, bins=30, color=color, alpha=0.85, edgecolor="white")
                ax.set_title(feature)
                st.pyplot(fig)
                plt.close(fig)

    st.markdown(f"**Predicted ({model_name})**")
    row2 = st.columns(3)
    for col, feature, color in zip(row2, features, ["#2e7d32", "#d32f2f", "#1976d2"]):
        with col:
            if feature in df_source.columns:
                vals = pd.to_numeric(df_source[feature], errors="coerce").dropna()
                fig, ax = plt.subplots(figsize=(4.0, 3.0))
                ax.hist(vals, bins=30, color=color, alpha=0.85, edgecolor="white")
                ax.set_title(feature)
                st.pyplot(fig)
                plt.close(fig)


def render_per_class_accuracy(y_true, y_pred):
    classes = [0, 1, 2, 3]
    class_names = ["Vegetation", "Built-up", "Water", "Other"]
    yt = pd.Series(y_true).apply(normalize_label_value)
    yp = pd.Series(y_pred).apply(normalize_label_value)

    accs = []
    for c in classes:
        mask = yt == c
        if mask.sum() == 0:
            accs.append(0.0)
        else:
            accs.append((yp[mask] == c).mean())

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(class_names, accs, color=["#2e7d32", "#d32f2f", "#1976d2", "#757575"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.02, f"{a*100:.1f}%", ha="center", fontsize=9)
    st.pyplot(fig)
    plt.close(fig)


def render_difference_plot(y_true, y_pred_mlp, y_pred_ridge):
    classes = [0, 1, 2, 3]
    class_names = ["Vegetation", "Built-up", "Water", "Other"]
    actual_counts = pd.Series(y_true).apply(normalize_label_value).value_counts().reindex(classes, fill_value=0).values
    mlp_counts = pd.Series(y_pred_mlp).apply(normalize_label_value).value_counts().reindex(classes, fill_value=0).values
    ridge_counts = pd.Series(y_pred_ridge).apply(normalize_label_value).value_counts().reindex(classes, fill_value=0).values

    diff_mlp = mlp_counts - actual_counts
    diff_ridge = ridge_counts - actual_counts

    x = np.arange(len(classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, diff_mlp, width, label="MLP - Actual", color="#1f77b4")
    ax.bar(x + width / 2, diff_ridge, width, label="Ridge - Actual", color="#ff7f0e")
    ax.axhline(0, color="#222222", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Difference in Grid Cell Count")
    ax.set_title("Difference Plot (Prediction - Actual)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def render_per_class_accuracy_comparison(y_true, y_pred_mlp, y_pred_ridge):
    classes = [0, 1, 2, 3]
    class_names = ["Vegetation", "Built-up", "Water", "Other"]
    yt = pd.Series(y_true).apply(normalize_label_value)
    yp_mlp = pd.Series(y_pred_mlp).apply(normalize_label_value)
    yp_ridge = pd.Series(y_pred_ridge).apply(normalize_label_value)

    acc_mlp = []
    acc_ridge = []
    for c in classes:
        mask = yt == c
        if mask.sum() == 0:
            acc_mlp.append(0.0)
            acc_ridge.append(0.0)
        else:
            acc_mlp.append((yp_mlp[mask] == c).mean())
            acc_ridge.append((yp_ridge[mask] == c).mean())

    x = np.arange(len(classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, acc_mlp, width, label="MLP", color="#1f77b4")
    ax.bar(x + width / 2, acc_ridge, width, label="Ridge", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy (MLP vs Ridge)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def render_agreement_map(df_source, col_a, col_b):
    def parse_geometry(geo_value):
        if pd.isna(geo_value):
            return None
        try:
            obj = json.loads(geo_value)
        except Exception:
            return None
        if isinstance(obj, dict) and obj.get("type") in ["Polygon", "MultiPolygon"]:
            return {"type": obj["type"], "coordinates": obj["coordinates"]}
        if isinstance(obj, dict) and obj.get("type") == "Feature" and "geometry" in obj:
            geom = obj.get("geometry")
            if isinstance(geom, dict) and geom.get("type") in ["Polygon", "MultiPolygon"]:
                return {"type": geom["type"], "coordinates": geom["coordinates"]}
        return None

    def mercator_to_wgs84(x, y):
        r = 6378137.0
        lon = (x / r) * (180.0 / math.pi)
        lat = (2.0 * math.atan(math.exp(y / r)) - math.pi / 2.0) * (180.0 / math.pi)
        return [lon, lat]

    def maybe_convert_coords(coords):
        if not coords:
            return coords
        sample = coords[0][0] if isinstance(coords[0][0], list) else coords[0]
        sx, sy = sample[0], sample[1]
        is_projected = abs(sx) > 180 or abs(sy) > 90
        if not is_projected:
            return coords
        if isinstance(coords[0][0], list):
            return [[mercator_to_wgs84(x, y) for x, y in ring] for ring in coords]
        return [mercator_to_wgs84(x, y) for x, y in coords]

    def normalize_geometry_crs(geometry):
        gtype = geometry.get("type")
        coords = geometry.get("coordinates")
        if gtype == "Polygon":
            return {"type": "Polygon", "coordinates": maybe_convert_coords(coords)}
        if gtype == "MultiPolygon":
            return {"type": "MultiPolygon", "coordinates": [maybe_convert_coords(poly) for poly in coords]}
        return geometry

    geojson_data = {"type": "FeatureCollection", "features": []}
    for _, row in df_source.iterrows():
        geometry = parse_geometry(row.get(".geo"))
        if geometry is None:
            continue
        geometry = normalize_geometry_crs(geometry)
        a = normalize_label_value(row.get(col_a))
        b = normalize_label_value(row.get(col_b))
        is_same = (a == b)
        color = "#2e7d32" if is_same else "#fbc02d"  # green/yellow
        geojson_data["features"].append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {"color": color}
        })

    m = folium.Map(location=[49.4521, 11.0767], zoom_start=11)
    folium.GeoJson(
        geojson_data,
        style_function=lambda f: {
            "fillColor": f["properties"]["color"],
            "color": None,
            "weight": 0.3,
            "fillOpacity": 0.75
        }
    ).add_to(m)
    folium.GeoJson(
        boundary,
        style_function=lambda _: {"color": "#1e40af", "weight": 2.5, "fillOpacity": 0}
    ).add_to(m)
    st_folium(m, width=700, height=520)


def render_prediction_difference_map(df_source, col_a, col_b):
    def parse_geometry(geo_value):
        if pd.isna(geo_value):
            return None
        try:
            obj = json.loads(geo_value)
        except Exception:
            return None
        if isinstance(obj, dict) and obj.get("type") in ["Polygon", "MultiPolygon"]:
            return {"type": obj["type"], "coordinates": obj["coordinates"]}
        if isinstance(obj, dict) and obj.get("type") == "Feature" and "geometry" in obj:
            geom = obj.get("geometry")
            if isinstance(geom, dict) and geom.get("type") in ["Polygon", "MultiPolygon"]:
                return {"type": geom["type"], "coordinates": geom["coordinates"]}
        return None

    def mercator_to_wgs84(x, y):
        r = 6378137.0
        lon = (x / r) * (180.0 / math.pi)
        lat = (2.0 * math.atan(math.exp(y / r)) - math.pi / 2.0) * (180.0 / math.pi)
        return [lon, lat]

    def maybe_convert_coords(coords):
        if not coords:
            return coords
        sample = coords[0][0] if isinstance(coords[0][0], list) else coords[0]
        sx, sy = sample[0], sample[1]
        is_projected = abs(sx) > 180 or abs(sy) > 90
        if not is_projected:
            return coords
        if isinstance(coords[0][0], list):
            return [[mercator_to_wgs84(x, y) for x, y in ring] for ring in coords]
        return [mercator_to_wgs84(x, y) for x, y in coords]

    def normalize_geometry_crs(geometry):
        gtype = geometry.get("type")
        coords = geometry.get("coordinates")
        if gtype == "Polygon":
            return {"type": "Polygon", "coordinates": maybe_convert_coords(coords)}
        if gtype == "MultiPolygon":
            return {"type": "MultiPolygon", "coordinates": [maybe_convert_coords(poly) for poly in coords]}
        return geometry

    geojson_data = {"type": "FeatureCollection", "features": []}
    for _, row in df_source.iterrows():
        geometry = parse_geometry(row.get(".geo"))
        if geometry is None:
            continue
        geometry = normalize_geometry_crs(geometry)
        a = normalize_label_value(row.get(col_a))
        b = normalize_label_value(row.get(col_b))
        is_diff = (a != b)
        color = "#ef5350" if is_diff else "#cfd8dc"  # red=diff, light gray=same
        geojson_data["features"].append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {"color": color}
        })

    m = folium.Map(location=[49.4521, 11.0767], zoom_start=11)
    folium.GeoJson(
        geojson_data,
        style_function=lambda f: {
            "fillColor": f["properties"]["color"],
            "color": None,
            "weight": 0.3,
            "fillOpacity": 0.75
        }
    ).add_to(m)
    folium.GeoJson(
        boundary,
        style_function=lambda _: {"color": "#1e40af", "weight": 2.5, "fillOpacity": 0}
    ).add_to(m)
    st_folium(m, width=700, height=520)


def render_model_prediction_comparison_bar(df_source, col_mlp, col_ridge):
    classes = [0, 1, 2, 3]
    class_names = ["Vegetation", "Built-up", "Water", "Other"]
    mlp_counts = df_source[col_mlp].apply(normalize_label_value).value_counts().reindex(classes, fill_value=0).values
    ridge_counts = df_source[col_ridge].apply(normalize_label_value).value_counts().reindex(classes, fill_value=0).values

    x = np.arange(len(classes))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, mlp_counts, width, label="MLP", color="#1f77b4")
    ax.bar(x + width / 2, ridge_counts, width, label="Ridge", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Grid Cell Count")
    ax.set_title("Model Comparison (MLP vs Ridge)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def render_predicted_area_metrics(df_source, pred_col, model_name):
    counts = df_source[pred_col].apply(normalize_label_value).value_counts().reindex([0, 1, 2, 3], fill_value=0)
    st.markdown(f"**{model_name}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vegetation", f"{int(counts.loc[0])}")
    c2.metric("Built-up", f"{int(counts.loc[1])}")
    c3.metric("Water", f"{int(counts.loc[2])}")
    c4.metric("Other", f"{int(counts.loc[3])}")


def render_change_bar_chart(df_year1, df_year2, year1, year2, label_col_year1="label", label_col_year2="label"):
    class_names = ["Vegetation", "Built-up", "Water", "Other"]
    colors = ["#2e7d32", "#d32f2f", "#1976d2", "#757575"]

    c1 = get_class_counts(df_year1, label_col_year1)
    c2 = get_class_counts(df_year2, label_col_year2)
    diff = c2.values - c1.values

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(class_names, diff, color=colors)
    ax.axhline(0, color="#222222", lw=1)
    ax.set_title(f"Change Bar Chart ({year2} - {year1})")
    ax.set_ylabel("Difference in Grid Cell Count")
    for b, d in zip(bars, diff):
        va = "bottom" if d >= 0 else "top"
        ax.text(b.get_x() + b.get_width() / 2, d, f"{int(d):+d}", ha="center", va=va, fontsize=9)
    st.pyplot(fig)
    plt.close(fig)


def render_side_by_side_bar_chart(df_year1, df_year2, year1, year2, label_col_year1="label", label_col_year2="label"):
    class_names = ["Vegetation", "Built-up", "Water", "Other"]
    c1 = get_class_counts(df_year1, label_col_year1)
    c2 = get_class_counts(df_year2, label_col_year2)

    x = np.arange(len(class_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, c1.values, width, label=year1, color="#1f77b4")
    ax.bar(x + width / 2, c2.values, width, label=year2, color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Grid Cell Count")
    ax.set_title("Side-by-Side Class Comparison")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def render_transition_matrix(df_year1, df_year2, year1, year2, label_col_year1="label", label_col_year2="label"):
    classes = [0, 1, 2, 3]

    if "system:index" in df_year1.columns and "system:index" in df_year2.columns:
        y1 = df_year1[["system:index", label_col_year1]].copy()
        y2 = df_year2[["system:index", label_col_year2]].copy()
        y1["system:index"] = y1["system:index"].astype(str)
        y2["system:index"] = y2["system:index"].astype(str)
        y1 = y1.rename(columns={label_col_year1: "class_y1"})
        y2 = y2.rename(columns={label_col_year2: "class_y2"})
        merged = y1.merge(y2, on="system:index", how="inner")
        if len(merged) > 0:
            s1 = merged["class_y1"].apply(normalize_label_value)
            s2 = merged["class_y2"].apply(normalize_label_value)
        else:
            n = min(len(df_year1), len(df_year2))
            s1 = df_year1[label_col_year1].head(n).apply(normalize_label_value)
            s2 = df_year2[label_col_year2].head(n).apply(normalize_label_value)
    else:
        n = min(len(df_year1), len(df_year2))
        s1 = df_year1[label_col_year1].head(n).apply(normalize_label_value)
        s2 = df_year2[label_col_year2].head(n).apply(normalize_label_value)

    tm = pd.crosstab(
        pd.Categorical(s1, categories=classes),
        pd.Categorical(s2, categories=classes),
        dropna=False
    ).values

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(tm, cmap="YlGnBu")
    ax.set_title(f"Transition Matrix ({year1} -> {year2})")
    ax.set_xlabel(f"To ({year2})")
    ax.set_ylabel(f"From ({year1})")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    for i in range(tm.shape[0]):
        for j in range(tm.shape[1]):
            ax.text(j, i, int(tm[i, j]), ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)
    plt.close(fig)


def render_net_change_summary(df_year1, df_year2, year1, year2, label_col_year1="label", label_col_year2="label"):
    class_names = ["Vegetation", "Built-up", "Water", "Other"]
    c1 = get_class_counts(df_year1, label_col_year1)
    c2 = get_class_counts(df_year2, label_col_year2)
    diff = c2.values - c1.values

    st.markdown("### Net Change Summary")
    cols = st.columns(4)
    for i, col in enumerate(cols):
        with col:
            col.metric(
                class_names[i],
                f"{int(c2.values[i])}",
                f"{int(diff[i]):+d} vs {year1}"
            )


def render_percentage_change_chart(df_year1, df_year2, year1, year2, label_col_year1="label", label_col_year2="label"):
    classes = [0, 1, 2, 3]
    class_names = ["Vegetation", "Built-up", "Water", "Other"]
    c1 = get_class_counts(df_year1, label_col_year1).reindex(classes, fill_value=0).values.astype(float)
    c2 = get_class_counts(df_year2, label_col_year2).reindex(classes, fill_value=0).values.astype(float)
    pct = np.where(c1 > 0, ((c2 - c1) / c1) * 100.0, 0.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(class_names, pct, color=["#2e7d32", "#d32f2f", "#1976d2", "#757575"])
    ax.axhline(0, color="#222222", lw=1)
    ax.set_ylabel("% Change")
    ax.set_title(f"Percentage Change ({year1} -> {year2})")
    for b, v in zip(bars, pct):
        va = "bottom" if v >= 0 else "top"
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.1f}%", ha="center", va=va, fontsize=9)
    st.pyplot(fig)
    plt.close(fig)


def render_change_map(df_year1, df_year2, label_col_year1="label", label_col_year2="label"):
    def parse_geometry(geo_value):
        if pd.isna(geo_value):
            return None
        try:
            obj = json.loads(geo_value)
        except Exception:
            return None
        if isinstance(obj, dict) and obj.get("type") in ["Polygon", "MultiPolygon"]:
            return {"type": obj["type"], "coordinates": obj["coordinates"]}
        if isinstance(obj, dict) and obj.get("type") == "Feature" and "geometry" in obj:
            geom = obj.get("geometry")
            if isinstance(geom, dict) and geom.get("type") in ["Polygon", "MultiPolygon"]:
                return {"type": geom["type"], "coordinates": geom["coordinates"]}
        return None

    def mercator_to_wgs84(x, y):
        r = 6378137.0
        lon = (x / r) * (180.0 / math.pi)
        lat = (2.0 * math.atan(math.exp(y / r)) - math.pi / 2.0) * (180.0 / math.pi)
        return [lon, lat]

    def maybe_convert_coords(coords):
        if not coords:
            return coords
        sample = coords[0][0] if isinstance(coords[0][0], list) else coords[0]
        sx, sy = sample[0], sample[1]
        is_projected = abs(sx) > 180 or abs(sy) > 90
        if not is_projected:
            return coords
        if isinstance(coords[0][0], list):
            return [[mercator_to_wgs84(x, y) for x, y in ring] for ring in coords]
        return [mercator_to_wgs84(x, y) for x, y in coords]

    def normalize_geometry_crs(geometry):
        gtype = geometry.get("type")
        coords = geometry.get("coordinates")
        if gtype == "Polygon":
            return {"type": "Polygon", "coordinates": maybe_convert_coords(coords)}
        if gtype == "MultiPolygon":
            return {"type": "MultiPolygon", "coordinates": [maybe_convert_coords(poly) for poly in coords]}
        return geometry

    if "system:index" in df_year1.columns and "system:index" in df_year2.columns:
        y1 = df_year1[["system:index", ".geo", label_col_year1]].copy()
        y2 = df_year2[["system:index", label_col_year2]].copy()
        y1["system:index"] = y1["system:index"].astype(str)
        y2["system:index"] = y2["system:index"].astype(str)
        y1 = y1.rename(columns={label_col_year1: "class_y1"})
        y2 = y2.rename(columns={label_col_year2: "class_y2"})
        merged = y1.merge(y2, on="system:index", how="inner")
        if len(merged) == 0:
            n = min(len(df_year1), len(df_year2))
            merged = pd.DataFrame({
                ".geo": df_year1[".geo"].head(n).values,
                "class_y1": df_year1[label_col_year1].head(n).values,
                "class_y2": df_year2[label_col_year2].head(n).values
            })
    else:
        n = min(len(df_year1), len(df_year2))
        merged = pd.DataFrame({
            ".geo": df_year1[".geo"].head(n).values,
            "class_y1": df_year1[label_col_year1].head(n).values,
            "class_y2": df_year2[label_col_year2].head(n).values
        })

    geojson_data = {"type": "FeatureCollection", "features": []}
    for _, row in merged.iterrows():
        geometry = parse_geometry(row.get(".geo"))
        if geometry is None:
            continue
        geometry = normalize_geometry_crs(geometry)
        c1 = normalize_label_value(row.get("class_y1"))
        c2 = normalize_label_value(row.get("class_y2"))

        # change color rules
        if c1 == 0 and c2 == 1:
            color = "#d32f2f"  # vegetation -> built-up
        elif c1 == 1 and c2 == 0:
            color = "#2e7d32"  # built-up -> vegetation
        elif c1 == 2 or c2 == 2:
            color = "#1976d2"  # water-related change
        elif c1 != c2:
            color = "#fbc02d"  # other change
        else:
            color = "#cfd8dc"  # no change

        geojson_data["features"].append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {"color": color}
        })

    m = folium.Map(location=[49.4521, 11.0767], zoom_start=11)
    folium.GeoJson(
        geojson_data,
        style_function=lambda f: {
            "fillColor": f["properties"]["color"],
            "color": None,
            "weight": 0.3,
            "fillOpacity": 0.75
        }
    ).add_to(m)
    folium.GeoJson(
        boundary,
        style_function=lambda _: {"color": "#1e40af", "weight": 2.5, "fillOpacity": 0}
    ).add_to(m)
    st_folium(m, width=900, height=560)


def render_builtup_growth_metric(df_year1, df_year2, year1, year2, label_col_year1="label", label_col_year2="label"):
    builtup_1 = int(get_class_counts(df_year1, label_col_year1).reindex([0, 1, 2, 3], fill_value=0).loc[1])
    builtup_2 = int(get_class_counts(df_year2, label_col_year2).reindex([0, 1, 2, 3], fill_value=0).loc[1])
    delta = builtup_2 - builtup_1
    pct = (delta / builtup_1 * 100.0) if builtup_1 > 0 else 0.0

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Built-up Count Change", f"{builtup_2}", f"{delta:+d} vs {year1}")
    with c2:
        st.metric("Built-up % Change", f"{pct:+.2f}%")


def render_top_change_insights(df_year1, df_year2, year1, year2, label_col_year1="label", label_col_year2="label"):
    class_names = ["Vegetation", "Built-up", "Water", "Other"]
    c1 = get_class_counts(df_year1, label_col_year1).reindex([0, 1, 2, 3], fill_value=0).values
    c2 = get_class_counts(df_year2, label_col_year2).reindex([0, 1, 2, 3], fill_value=0).values
    diff = c2 - c1
    abs_rank_idx = np.argsort(np.abs(diff))[::-1]

    top_idx = int(abs_rank_idx[0])
    direction = "increase" if diff[top_idx] >= 0 else "decrease"
    st.markdown(
        f"**Top Change:** {class_names[top_idx]} shows the largest {direction} "
        f"({int(diff[top_idx]):+d} cells) from {year1} to {year2}."
    )

    rank_text = ", ".join([f"{class_names[i]} ({int(diff[i]):+d})" for i in abs_rank_idx])
    st.markdown(f"**Change Ranking (by magnitude):** {rank_text}")

    if diff[1] > 0:
        st.markdown("**Insight:** Built-up area is expanding, indicating potential urban growth.")
    elif diff[1] < 0:
        st.markdown("**Insight:** Built-up area is contracting relative to the selected baseline year.")
    else:
        st.markdown("**Insight:** Built-up area is stable across the selected years.")


# -----------------------------
# MAP SECTION
# -----------------------------

if view_mode == "Single Year":
    if selected_year != "2024":
        if selected_year == "2021":
            actual_col = "label"
            mlp_col = "mlp_pred"
            ridge_col = "ridge_pred"
            pred_col = mlp_col if selected_model == "MLP" else ridge_col

            geojson_actual_2021 = load_grid_geojson(selected_df, label_column=actual_col)
            geojson_pred_mlp_2021 = load_grid_geojson(selected_df, label_column=mlp_col)
            geojson_pred_ridge_2021 = load_grid_geojson(selected_df, label_column=ridge_col)

            if compare_both_models:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("Actual Land Cover (2021)")
                    render_landcover_map(geojson_actual_2021)
                    render_legend()
                with col2:
                    st.subheader("Predicted Land Cover (2021 - MLP)")
                    render_landcover_map(geojson_pred_mlp_2021)
                    render_legend()
                with col3:
                    st.subheader("Predicted Land Cover (2021 - Ridge)")
                    render_landcover_map(geojson_pred_ridge_2021)
                    render_legend()

                pie1, pie2, pie3 = st.columns(3)
                with pie1:
                    render_landcover_pie(selected_df, actual_col, "Actual Land Cover Composition (2021)")
                with pie2:
                    render_landcover_pie(selected_df, mlp_col, "Predicted Land Cover Composition (MLP)")
                with pie3:
                    render_landcover_pie(selected_df, ridge_col, "Predicted Land Cover Composition (Ridge)")

                st.markdown("### Model Accuracy Comparison")
                render_accuracy_comparison(selected_df[actual_col], selected_df[mlp_col], selected_df[ridge_col])

                st.markdown("### Confusion Matrix")
                cm_left, cm_right = st.columns(2)
                with cm_left:
                    render_confusion_matrix(selected_df[actual_col], selected_df[mlp_col], "Confusion Matrix - MLP")
                with cm_right:
                    render_confusion_matrix(selected_df[actual_col], selected_df[ridge_col], "Confusion Matrix - Ridge")

                st.markdown("### Actual vs Predicted Distribution")
                render_distribution_chart(
                    selected_df[actual_col],
                    selected_df[mlp_col],
                    selected_df[ridge_col],
                    compare_mode=True,
                    selected_model_name=selected_model
                )

                st.markdown("### Difference Plot")
                render_difference_plot(selected_df[actual_col], selected_df[mlp_col], selected_df[ridge_col])

                st.markdown("### Per-Class Accuracy (MLP vs Ridge)")
                render_per_class_accuracy_comparison(selected_df[actual_col], selected_df[mlp_col], selected_df[ridge_col])

                st.markdown("### Agreement Map")
                render_agreement_map(selected_df, mlp_col, ridge_col)
                st.markdown("**Agreement Legend:** Green = Same Prediction, Yellow = Different Prediction")
            else:
                geojson_pred_2021 = geojson_pred_mlp_2021 if selected_model == "MLP" else geojson_pred_ridge_2021

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Actual Land Cover (2021)")
                    render_landcover_map(geojson_actual_2021)
                    render_legend()
                    render_landcover_pie(selected_df, actual_col, "Actual Land Cover Composition (2021)")
                with col2:
                    st.subheader(f"Predicted Land Cover (2021 - {selected_model})")
                    render_landcover_map(geojson_pred_2021)
                    render_legend()
                    render_landcover_pie(selected_df, pred_col, f"Predicted Land Cover Composition ({selected_model})")

                y_true = selected_df[actual_col].apply(normalize_label_value)
                y_pred = selected_df[pred_col].apply(normalize_label_value)
                accuracy = accuracy_score(y_true, y_pred)

                st.markdown("### Class Distribution Comparison")
                render_class_distribution_comparison(selected_df, actual_col, pred_col, selected_model)

                st.markdown("### Confusion Matrix")
                render_confusion_matrix(y_true, y_pred, f"Confusion Matrix - {selected_model}")

                st.markdown("### Accuracy Score")
                st.metric("Accuracy", f"{accuracy*100:.2f}%")

                render_feature_distribution_comparison(selected_df, selected_model)

                st.markdown("### NDVI vs Class (Actual vs Predicted)")
                b1, b2 = st.columns(2)
                with b1:
                    st.markdown("**Actual**")
                    render_ndvi_boxplot_by_class(selected_df, actual_col, show_heading=False)
                with b2:
                    st.markdown(f"**Predicted ({selected_model})**")
                    render_ndvi_boxplot_by_class(selected_df, pred_col, show_heading=False)

                st.markdown("### Error Map")
                error_df = selected_df.copy()
                error_df["error_flag"] = (y_true != y_pred).astype(int)
                # 0=correct (green), 1=incorrect (red) via existing class color mapping
                geojson_error = load_grid_geojson(error_df, label_column="error_flag")
                render_landcover_map(geojson_error)
                st.markdown("**Error Legend:** Green = Correct, Red = Incorrect")

                st.markdown("### Per-Class Accuracy")
                render_per_class_accuracy(y_true, y_pred)
        else:
            st.subheader(f"{selected_year} Land Cover Map")
            render_landcover_map(geojson_data)
            render_legend()
            # 2020 single-year analytics: only requested plots (no extras/duplicates)
            render_landcover_pie(selected_df, "label", f"Land Cover Composition ({selected_year})")
            st.markdown("### Class Distribution")
            render_class_distribution_bar(selected_df, "label", f"Class Distribution ({selected_year})")
            render_feature_histograms(selected_df, show_heading=True)
            st.markdown("### NDVI vs Class")
            render_ndvi_boxplot_by_class(selected_df, "label", show_heading=False)
            render_area_metrics(selected_df, "label", show_heading=True)
            render_correlation_heatmap(selected_df)
    else:
        geojson_pred_mlp = load_grid_geojson(df_2024, label_column="mlp_pred")
        geojson_pred_ridge = load_grid_geojson(df_2024, label_column="ridge_pred")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Predicted 2024 Map (MLP)")
            render_landcover_map(geojson_pred_mlp)
            render_legend()
        with col2:
            st.subheader("Predicted 2024 Map (Ridge)")
            render_landcover_map(geojson_pred_ridge)
            render_legend()

        pie1, pie2 = st.columns(2)
        with pie1:
            render_landcover_pie(df_2024, "mlp_pred", "Predicted Land Cover Composition (MLP)")
        with pie2:
            render_landcover_pie(df_2024, "ridge_pred", "Predicted Land Cover Composition (Ridge)")

        st.markdown("### Class Distribution Bar Chart")
        b1, b2 = st.columns(2)
        with b1:
            render_class_distribution_bar(df_2024, "mlp_pred", "Class Distribution (MLP)")
        with b2:
            render_class_distribution_bar(df_2024, "ridge_pred", "Class Distribution (Ridge)")

        st.markdown("### Model Comparison (MLP vs Ridge)")
        render_model_prediction_comparison_bar(df_2024, "mlp_pred", "ridge_pred")

        st.markdown("### Prediction Difference Map")
        render_prediction_difference_map(df_2024, "mlp_pred", "ridge_pred")
        st.markdown("**Difference Legend:** Red = Different Prediction, Gray = Same Prediction")

        st.markdown("### Feature Distribution (Predicted Classes)")
        st.markdown("**MLP**")
        render_feature_histograms(df_2024, show_heading=False)
        st.markdown("**Ridge**")
        render_feature_histograms(df_2024, show_heading=False)

        st.markdown("### Area Metrics")
        render_predicted_area_metrics(df_2024, "mlp_pred", "MLP")
        render_predicted_area_metrics(df_2024, "ridge_pred", "Ridge")

        st.markdown("### Uncertainty Map")
        render_agreement_map(df_2024, "mlp_pred", "ridge_pred")
        st.markdown("**Uncertainty Legend:** Green = Models Agree, Yellow = Models Disagree")
else:
    df_year1 = year_to_df[first_year]
    df_year2 = year_to_df[second_year]
    model_col = "mlp_pred" if selected_model == "MLP" else "ridge_pred"

    # Primary comparison pair (year1 -> year2)
    label_col_year1 = "label" if first_year in ["2020", "2021"] else model_col
    label_col_year2 = "label" if second_year in ["2020", "2021"] else model_col

    st.markdown("### Maps")
    years_set = {first_year, second_year}

    if years_set == {"2020", "2021"}:
        geo_2020_actual = load_grid_geojson(df_2020, label_column="label")
        geo_2021_actual = load_grid_geojson(df_2021, label_column="label")
        geo_2021_pred = load_grid_geojson(df_2021, label_column=model_col)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.subheader("2020 Actual")
            render_landcover_map(geo_2020_actual)
            render_legend()
        with m2:
            st.subheader("2021 Actual")
            render_landcover_map(geo_2021_actual)
            render_legend()
        with m3:
            st.subheader(f"2021 Predicted ({selected_model})")
            render_landcover_map(geo_2021_pred)
            render_legend()

    elif years_set == {"2020", "2024"}:
        geo_2020_actual = load_grid_geojson(df_2020, label_column="label")
        geo_2024_mlp = load_grid_geojson(df_2024, label_column="mlp_pred")
        geo_2024_ridge = load_grid_geojson(df_2024, label_column="ridge_pred")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.subheader("2020 Actual")
            render_landcover_map(geo_2020_actual)
            render_legend()
        with m2:
            st.subheader("2024 MLP")
            render_landcover_map(geo_2024_mlp)
            render_legend()
        with m3:
            st.subheader("2024 Ridge")
            render_landcover_map(geo_2024_ridge)
            render_legend()

    elif years_set == {"2021", "2024"}:
        geo_2021_actual = load_grid_geojson(df_2021, label_column="label")
        geo_2021_pred = load_grid_geojson(df_2021, label_column=model_col)
        geo_2024_pred = load_grid_geojson(df_2024, label_column=model_col)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.subheader("2021 Actual")
            render_landcover_map(geo_2021_actual)
            render_legend()
        with m2:
            st.subheader(f"2021 Predicted ({selected_model})")
            render_landcover_map(geo_2021_pred)
            render_legend()
        with m3:
            st.subheader(f"2024 Predicted ({selected_model})")
            render_landcover_map(geo_2024_pred)
            render_legend()

    st.markdown("### Composition")
    if years_set == {"2020", "2021"}:
        p1, p2, p3 = st.columns(3)
        with p1:
            render_landcover_pie(df_2020, "label", "Land Cover Composition (2020 Actual)")
        with p2:
            render_landcover_pie(df_2021, "label", "Land Cover Composition (2021 Actual)")
        with p3:
            render_landcover_pie(df_2021, model_col, f"Land Cover Composition (2021 Predicted - {selected_model})")
    elif years_set == {"2020", "2024"}:
        p1, p2, p3 = st.columns(3)
        with p1:
            render_landcover_pie(df_2020, "label", "Land Cover Composition (2020 Actual)")
        with p2:
            render_landcover_pie(df_2024, "mlp_pred", "Land Cover Composition (2024 MLP)")
        with p3:
            render_landcover_pie(df_2024, "ridge_pred", "Land Cover Composition (2024 Ridge)")
    elif years_set == {"2021", "2024"}:
        p1, p2, p3 = st.columns(3)
        with p1:
            render_landcover_pie(df_2021, "label", "Land Cover Composition (2021 Actual)")
        with p2:
            render_landcover_pie(df_2021, model_col, f"Land Cover Composition (2021 Predicted - {selected_model})")
        with p3:
            render_landcover_pie(df_2024, model_col, f"Land Cover Composition (2024 Predicted - {selected_model})")

    st.markdown("### Change Analysis")
    render_change_bar_chart(df_year1, df_year2, first_year, second_year, label_col_year1, label_col_year2)
    render_side_by_side_bar_chart(df_year1, df_year2, first_year, second_year, label_col_year1, label_col_year2)
    render_percentage_change_chart(df_year1, df_year2, first_year, second_year, label_col_year1, label_col_year2)
    render_transition_matrix(df_year1, df_year2, first_year, second_year, label_col_year1, label_col_year2)
    st.markdown("#### Spatial Change Map")
    render_change_map(df_year1, df_year2, label_col_year1, label_col_year2)
    st.markdown(
        "**Change Map Legend:** Red = Vegetation→Built-up, Green = Built-up→Vegetation, "
        "Blue = Water-related change, Yellow = Other change, Gray = No change"
    )
    st.markdown("#### Actual Trend Summary")
    render_builtup_growth_metric(df_year1, df_year2, first_year, second_year, label_col_year1, label_col_year2)
    render_net_change_summary(df_year1, df_year2, first_year, second_year, label_col_year1, label_col_year2)

    if years_set == {"2020", "2021"}:
        st.markdown("#### Predicted Trend Summary (2020 -> 2021)")
        render_builtup_growth_metric(df_2020, df_2021, "2020", "2021 Predicted", "label", model_col)
        render_net_change_summary(df_2020, df_2021, "2020", "2021 Predicted", "label", model_col)

    st.markdown("### Insights")
    render_top_change_insights(df_year1, df_year2, first_year, second_year, label_col_year1, label_col_year2)

# =====================================================
# CHATBOT SECTION (ALWAYS VISIBLE)
# =====================================================

st.sidebar.markdown("---")
st.sidebar.subheader("💬 Ask the Assistant")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🟣 QUICK ACTION BUTTONS (ADD HERE)
if st.sidebar.button("Explain this page"):
    user_input = "Explain what I am seeing on this page"

elif st.sidebar.button("What changed?"):
    user_input = "What are the main land cover changes?"

elif st.sidebar.button("Is this reliable?"):
    user_input = "Can we trust these predictions?"

else:
    user_input = st.sidebar.chat_input("Ask something...")

# Build context dynamically
accuracy_value = None
if selected_label_column not in selected_df.columns:
    if selected_year == "2024":
        selected_label_column = "mlp_pred" if "mlp_pred" in selected_df.columns else "ridge_pred"
    elif "label" in selected_df.columns:
        selected_label_column = "label"
    else:
        # final fallback to prevent crash
        selected_label_column = selected_df.columns[0]

class_counts = (
    selected_df[selected_label_column]
    .apply(normalize_label_value)
    .value_counts()
    .reindex([0, 1, 2, 3], fill_value=0)
    .astype(int)
    .to_dict()
)

if selected_year == "2021" and not compare_both_models:
    try:
        actual_col = "label"
        pred_col = "mlp_pred" if selected_model == "MLP" else "ridge_pred"
        y_true = selected_df[actual_col].apply(normalize_label_value)
        y_pred = selected_df[pred_col].apply(normalize_label_value)
        accuracy_value = float((y_true == y_pred).mean())
    except:
        accuracy_value = None

context = build_context(
    selected_year,
    selected_model,
    compare_both_models,
    accuracy_value
)
context["class_counts"] = class_counts
context["total_cells"] = len(selected_df)

if selected_year == "2021" and compare_both_models:
    context["mlp_counts"] = (
        selected_df["mlp_pred"]
        .apply(normalize_label_value)
        .value_counts()
        .reindex([0, 1, 2, 3], fill_value=0)
        .astype(int)
        .to_dict()
    )
    context["ridge_counts"] = (
        selected_df["ridge_pred"]
        .apply(normalize_label_value)
        .value_counts()
        .reindex([0, 1, 2, 3], fill_value=0)
        .astype(int)
        .to_dict()
    )

# Handle user input
if user_input:
    response = generate_response(user_input, context)

    # Save history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# Display chat history
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.sidebar.markdown(f"**🧑 You:** {message}")
    else:
        st.sidebar.markdown(f"**🤖 Bot:** {message}")
