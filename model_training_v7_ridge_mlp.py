import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

# =====================================================
# 1. LOAD DATA
# =====================================================

df_2020 = pd.read_excel("nuremberg_grid_dataset_2020_200m.xlsx")
df_2021 = pd.read_excel("nuremberg_grid_dataset_2021_ESA_200m.xlsx")
df_2024 = pd.read_excel("nuremberg_2024_features_clean.xlsx")

df_2020["year"] = 2020
df_2021["year"] = 2021
df_2024["year"] = 2024

print("Loaded shapes:")
print(df_2020.shape, df_2021.shape, df_2024.shape)

# =====================================================
# 2. CLEAN DATA
# =====================================================

def clean_df(df):
    df = df.dropna(subset=["label"])
    df = df.reset_index(drop=True)
    return df

df_2020_clean = clean_df(df_2020.copy())
df_2021_clean = clean_df(df_2021.copy())

# 2024 → no labels
df_2024_clean = df_2024.drop(columns=["system:index"], errors="ignore")

# =====================================================
# 3. FIX LABELS
# =====================================================

def fix_labels(df):
    df["label"] = df["label"].round().astype(int)
    return df

df_2020_clean = fix_labels(df_2020_clean)
df_2021_clean = fix_labels(df_2021_clean)

# =====================================================
# 4. SIMPLIFY LABELS
# =====================================================

def simplify_labels(df):
    def map_class(x):
        if x in [10, 20, 30]: return 0
        elif x in [40, 50]: return 1
        elif x in [60, 70]: return 2
        elif x in [80, 90]: return 3
        else: return 4
    df["label"] = df["label"].apply(map_class)
    return df

df_2020_clean = simplify_labels(df_2020_clean)
df_2021_clean = simplify_labels(df_2021_clean)

# =====================================================
# 5. FEATURE ENGINEERING
# =====================================================

def add_features(df):
    df["SAVI"] = ((df["B8"] - df["B4"]) / (df["B8"] + df["B4"] + 0.5)) * 1.5
    df["BSI"] = ((df["B11"] + df["B4"]) - (df["B8"] + df["B2"])) / (
        (df["B11"] + df["B4"]) + (df["B8"] + df["B2"])
    )
    df["UI"] = (df["B11"] - df["B8"]) / (df["B11"] + df["B8"])
    return df

df_2020_clean = add_features(df_2020_clean)
df_2021_clean = add_features(df_2021_clean)
df_2024_clean = add_features(df_2024_clean)

# =====================================================
# 6. FEATURES
# =====================================================

features = [
    "B11", "B2", "B3", "B4", "B8",
    "NDVI", "NDBI", "NDWI",
    "SAVI", "BSI", "UI"
]

# =====================================================
# 7. TRAIN / VALID / TEST SPLIT (2020)
# =====================================================

X = df_2020_clean[features]
y = df_2020_clean["label"]

# 80% train, 20% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Split temp → 10% val, 10% test
X_val, X_test_internal, y_val, y_test_internal = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# =====================================================
# 8. TEST (2021)
# =====================================================

X_test_2021 = df_2021_clean[features]
y_test_2021 = df_2021_clean["label"]

# =====================================================
# 9. FUTURE (2024)
# =====================================================

X_2024 = df_2024_clean[features]

# =====================================================
# 10. PREPROCESSING
# =====================================================

imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()

X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test_internal = imputer.transform(X_test_internal)
X_test_2021 = imputer.transform(X_test_2021)
X_2024 = imputer.transform(X_2024)

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test_internal = scaler.transform(X_test_internal)
X_test_2021 = scaler.transform(X_test_2021)
X_2024 = scaler.transform(X_2024)

# =====================================================
# 11. SMOTE
# =====================================================

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# =====================================================
# 12. MODELS
# =====================================================

ridge = RidgeClassifier()
ridge.fit(X_train, y_train)

mlp = MLPClassifier(hidden_layer_sizes=(96, 48), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# =====================================================
# 13. EVALUATION
# =====================================================

print("\nValidation:")
print("Ridge:", accuracy_score(y_val, ridge.predict(X_val)))
print("MLP  :", accuracy_score(y_val, mlp.predict(X_val)))

print("\nInternal Test (2020 split):")
print("Ridge:", accuracy_score(y_test_internal, ridge.predict(X_test_internal)))
print("MLP  :", accuracy_score(y_test_internal, mlp.predict(X_test_internal)))

print("\nTest (2021):")
print("Ridge:", accuracy_score(y_test_2021, ridge.predict(X_test_2021)))
print("MLP  :", accuracy_score(y_test_2021, mlp.predict(X_test_2021)))

# =====================================================
# SAVE 2021 PREDICTIONS (ALIGNED OUTPUT)
# =====================================================

df_2021_output = df_2021_clean.copy().reset_index(drop=True)
df_2021_output["mlp_pred"] = mlp.predict(X_test_2021)
df_2021_output["ridge_pred"] = ridge.predict(X_test_2021)

# Safety checks to prevent silent misalignment
assert len(df_2021_output["label"]) == len(df_2021_output["mlp_pred"]), "Mismatch in prediction length"
assert df_2021_output["label"].notna().all(), "Missing labels detected"
assert pd.Series(df_2021_output["mlp_pred"]).notna().all(), "Missing MLP predictions detected"
assert pd.Series(df_2021_output["ridge_pred"]).notna().all(), "Missing Ridge predictions detected"

# Optional debug check
print("2021 output shape:", df_2021_output.shape)
print(df_2021_output[["label", "mlp_pred", "ridge_pred"]].head())

df_2021_output.to_csv("predictions_2021.csv", index=False)

print("Saved predictions_2021.csv")

# =====================================================
# SAVE 2024 PREDICTIONS (FIXED)
# =====================================================

df_2024_output = df_2024.copy()

df_2024_output["mlp_pred"] = mlp.predict(X_2024)
df_2024_output["ridge_pred"] = ridge.predict(X_2024)

df_2024_output.to_csv("predictions_2024.csv", index=False)

print("Saved predictions_2024.csv")

# =====================================================
# 16. OPTIONAL PLOT
# =====================================================

plt.figure(figsize=(8,5))
pd.Series(df_2024_output["mlp_pred"]).value_counts().sort_index().plot(kind="bar")
plt.title("Predicted Land Cover (2024 - MLP)")
plt.show()
