import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
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

# 2024 -> no labels
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
        # Unified 4-class mapping aligned with dashboard:
        # 0=Vegetation, 1=Built-up, 2=Water, 3=Other
        if x in [10, 20, 30, 40]:
            return 0
        elif x == 50:
            return 1
        elif x == 80:
            return 2
        else:
            return 3
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

# Split temp -> 10% val, 10% test
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

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# =====================================================
# 13. EVALUATION
# =====================================================

print("\nValidation:")
print("Logistic Regression:", accuracy_score(y_val, logreg.predict(X_val)))
print("Random Forest      :", accuracy_score(y_val, rf.predict(X_val)))

print("\nInternal Test (2020 split):")
print("Logistic Regression:", accuracy_score(y_test_internal, logreg.predict(X_test_internal)))
print("Random Forest      :", accuracy_score(y_test_internal, rf.predict(X_test_internal)))

print("\nTest (2021):")
print("Logistic Regression:", accuracy_score(y_test_2021, logreg.predict(X_test_2021)))
print("Random Forest      :", accuracy_score(y_test_2021, rf.predict(X_test_2021)))

# =====================================================
# SAVE 2021 PREDICTIONS (ALIGNED OUTPUT)
# =====================================================

df_2021_output = df_2021_clean.copy().reset_index(drop=True)
df_2021_output["logreg_pred"] = logreg.predict(X_test_2021)
df_2021_output["rf_pred"] = rf.predict(X_test_2021)

# Safety checks to prevent silent misalignment
assert len(df_2021_output["label"]) == len(df_2021_output["logreg_pred"]), "Mismatch in prediction length"
assert df_2021_output["label"].notna().all(), "Missing labels detected"
assert pd.Series(df_2021_output["logreg_pred"]).notna().all(), "Missing Logistic Regression predictions detected"
assert pd.Series(df_2021_output["rf_pred"]).notna().all(), "Missing Random Forest predictions detected"

# Optional debug check
print("2021 output shape:", df_2021_output.shape)
print(df_2021_output[["label", "logreg_pred", "rf_pred"]].head())

df_2021_output.to_csv("predictions_2021.csv", index=False)

print("Saved predictions_2021.csv")

# =====================================================
# SAVE 2024 PREDICTIONS
# =====================================================

df_2024_output = df_2024.copy()

df_2024_output["logreg_pred"] = logreg.predict(X_2024)
df_2024_output["rf_pred"] = rf.predict(X_2024)

df_2024_output.to_csv("predictions_2024.csv", index=False)

print("Saved predictions_2024.csv")

# =====================================================
# 16. OPTIONAL PLOT
# =====================================================

plt.figure(figsize=(8, 5))
pd.Series(df_2024_output["rf_pred"]).value_counts().sort_index().plot(kind="bar")
plt.title("Predicted Land Cover (2024 - Random Forest)")
plt.show()