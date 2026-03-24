# 🌍 Nuremberg Satellite Map Analysis (ML Project)

Interactive machine learning project for **land-cover mapping** and **urban change analysis** in Nuremberg using Sentinel-2 based features, multiple classification models, and a Streamlit dashboard with map visualizations.

## ✨ What This Project Does

- 🛰️ Builds grid-based land-cover datasets from satellite imagery
- 🧹 Unifies labels (CORINE + WorldCover) into 4 classes
- 🤖 Trains and compares multiple ML models (MLP, Ridge, Logistic, RF, GB, etc.)
- 🗺️ Visualizes predictions and temporal changes (2020, 2021, 2024) on interactive maps
- 📈 Shows confusion matrices, per-class accuracy, transition matrices, and change metrics
- 💬 Includes an in-dashboard assistant (OpenAI-powered with deterministic fallback mode)

## 🧠 Land-Cover Classes

Unified labels used in dashboard/training:

- `0` → Vegetation
- `1` → Built-up
- `2` → Water
- `3` → Other

## 🗂️ Repository Structure

```text
.
|- app.py                                   # Main Streamlit dashboard
|- chatbot.py                               # Chat assistant logic (OpenAI + fallback)
|- requirements.txt
|- README.md
|
|- clean_dataset_200m/                      # Cleaned 200m datasets (CSV)
|- data/ data2/ data3/                      # Raw/intermediate XLSX datasets
|- boundary/                                # GeoJSON boundary
|- nuremberg_boundary/                      # Shapefile boundary used by app
|
|- unify_labels.py                          # Label unification (300m workflow)
|- new_unify_labels_200m.py                 # Label unification (200m workflow)
|
|- model_training_v2.py
|- model_training_v3_for_200m.py
|- model_training_v4_for_200m.py
|- model_training_v5_with_gb.py
|- model_training_v6_for_200m_with_linear_regression.py
|- model_training_v7_ridge_mlp.py
|- model_training_v8_log_rand.py
|- model_training_change.py                 # Learned change model (2020→2021; infer 2024)
|
|- predictions_2021.csv
|- predictions_2024.csv
|- predictions_change_2021.csv
|- predictions_change_2024.csv
|
'- nuremberg_landcover_dataset*.js          # Earth Engine scripts for dataset creation
```

## ⚙️ Installation

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## 🔐 Environment Variables (Chat Assistant)

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT_SECONDS=10
```

Notes:
- ✅ `OPENAI_API_KEY` enables OpenAI responses.
- ✅ If key is missing/invalid, chatbot still works in fallback mode.
- ❌ Never hardcode API keys in source files.

## 🚀 Run the Dashboard

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit (usually `http://localhost:8501`).

## 🧪 Data + Training Workflow

### A) Build/clean datasets

- 300m clean labels:
```bash
python unify_labels.py
```

- 200m clean labels:
```bash
python new_unify_labels_200m.py
```

### B) Train models and generate prediction files

Core files commonly used by the app:

```bash
python model_training_v7_ridge_mlp.py      # creates predictions_2021.csv and predictions_2024.csv
python model_training_change.py            # creates predictions_change_2021.csv and predictions_change_2024.csv
```

Other experimental model scripts are also available (`v2` to `v8`) for comparison.

## 📊 Dashboard Features

### Single Year View

- 2020 actual land-cover map + analytics
- 2021 actual vs predicted (MLP/Ridge), confusion matrix, error map, per-class accuracy
- 2024 predicted maps (MLP vs Ridge), disagreement/uncertainty analysis

### Multiple Years View

- Side-by-side composition comparison
- Net and percentage class change
- Transition matrix
- Spatial change map
- Built-up growth metrics and top change insights
- Optional learned change-model visualization

### Chat Assistant (Sidebar)

- Answers questions about current view, metrics, distributions, transitions, and errors
- Uses exact computed values when available in current dashboard context
- Falls back to deterministic/local logic if OpenAI is unavailable

## 🗺️ Earth Engine Scripts

These scripts generate grid datasets from remote sensing data:

- `nuremberg_landcover_dataset.js`
- `nuremberg_landcover_dataset_2018_with_corine.js`
- `nuremberg_landcover_dataset_for_200m_grid.js`

They include:
- Sentinel-2 feature extraction (`B2, B3, B4, B8, B11`)
- Indices (`NDVI, NDBI, NDWI`)
- Grid reduction and export

## 📌 Important Notes

- The app expects specific generated files (`predictions_2021.csv`, `predictions_2024.csv`) to exist.
- For learned change mode, both change prediction files must exist.
- Large data artifacts are included in repository; cloning may take time.

## 🛠️ Recommended Next Improvements

- Add pinned dependency versions for reproducibility
- Add screenshots/GIFs in README
- Add automated tests for critical data-loading paths
- Add model performance summary table (all versions)

## 👤 Author

Machine Learning Final Project by **Kushagra Deb** (repository owner: `Kdeb2003`).
