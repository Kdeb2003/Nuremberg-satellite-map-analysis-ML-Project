import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import RandomOverSampler


# =====================================================
# 1. LOAD CLEAN DATASETS
# =====================================================

df_2018 = pd.read_csv("clean_dataset_200m/2018_clean.csv")
df_2020 = pd.read_csv("clean_dataset_200m/2020_clean.csv")
df_2022 = pd.read_csv("clean_dataset_200m/2022_clean.csv")
df_2024 = pd.read_csv("clean_dataset_200m/2024_clean.csv")

df_2018["year"] = 2018
df_2020["year"] = 2020
df_2022["year"] = 2022
df_2024["year"] = 2024

# Combine yearly datasets
df = pd.concat([df_2018, df_2020, df_2022, df_2024], ignore_index=True)

print("Dataset shape:", df.shape)
print(df.head())


# =====================================================
# 2. CLEAN DATA
# =====================================================

# Drop columns not used for modeling
df = df.drop(columns=["system:index", ".geo"], errors="ignore")

# Remove rows without labels
df = df.dropna(subset=["label"])

print("\nOverall label distribution:")
print(df["label"].value_counts())

print("\nClass distribution by year:")
print(df.groupby("year")["label"].value_counts().unstack(fill_value=0))


# =====================================================
# 3. FEATURE ENGINEERING
# =====================================================

# Extra remote sensing indices
df["SAVI"] = ((df["B8"] - df["B4"]) / (df["B8"] + df["B4"] + 0.5)) * 1.5
df["BSI"] = ((df["B11"] + df["B4"]) - (df["B8"] + df["B2"])) / (
    (df["B11"] + df["B4"]) + (df["B8"] + df["B2"])
)
df["UI"] = (df["B11"] - df["B8"]) / (df["B11"] + df["B8"])


# =====================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# =====================================================

train_df = df[df["year"].isin([2018, 2020])].copy()
val_df = df[df["year"] == 2022].copy()
test_df = df[df["year"] == 2024].copy()

print("\nTrain size:", train_df.shape)
print("Validation size:", val_df.shape)
print("Test size:", test_df.shape)


# =====================================================
# 5. FEATURE LIST
# =====================================================

features = [
    "B11",
    "B2",
    "B3",
    "B4",
    "B8",
    "NDBI",
    "NDVI",
    "NDWI",
    "SAVI",
    "BSI",
    "UI",
]

X_train = train_df[features]
y_train = train_df["label"]

X_val = val_df[features]
y_val = val_df["label"]

X_test = test_df[features]
y_test = test_df["label"]


# =====================================================
# 6. HANDLE CLASS IMBALANCE
# =====================================================

ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

print("\nBalanced training distribution:")
print(pd.Series(y_train_resampled).value_counts())


# =====================================================
# 7. IMPUTE MISSING VALUES
# =====================================================

imputer = SimpleImputer(strategy="mean")

X_train_imputed = imputer.fit_transform(X_train_resampled)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)


# =====================================================
# 8. SCALE FEATURES
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)
X_test_scaled = scaler.transform(X_test_imputed)


# =====================================================
# 9. MODEL 1 — RIDGE CLASSIFIER
# =====================================================

print("\nTraining Ridge Classifier...")

ridge = RidgeClassifier(alpha=1.0)
ridge.fit(X_train_scaled, y_train_resampled)

# Validation
y_val_pred_ridge = ridge.predict(X_val_scaled)

print("\nRidge Validation Accuracy:", accuracy_score(y_val, y_val_pred_ridge))
print(confusion_matrix(y_val, y_val_pred_ridge))
print(classification_report(y_val, y_val_pred_ridge, zero_division=0))

# Test
y_test_pred_ridge = ridge.predict(X_test_scaled)

print("\nRidge Test Accuracy:", accuracy_score(y_test, y_test_pred_ridge))
print(confusion_matrix(y_test, y_test_pred_ridge))
print(classification_report(y_test, y_test_pred_ridge, zero_division=0))

# Save Ridge predictions
test_df_ridge = test_df.copy()
test_df_ridge["predicted_label"] = y_test_pred_ridge
test_df_ridge.to_csv("predicted_landcover_ridge.csv", index=False)

print("\nSaved Ridge predictions to predicted_landcover_2024_200m_ridge.csv")


# =====================================================
# 10. MODEL 2 — MLP CLASSIFIER (NONLINEAR MODEL)
# =====================================================

print("\nTraining MLP Classifier...")

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=300,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
)

mlp.fit(X_train_scaled, y_train_resampled)

# Validation
y_val_pred_mlp = mlp.predict(X_val_scaled)

print("\nMLP Validation Accuracy:", accuracy_score(y_val, y_val_pred_mlp))
print(confusion_matrix(y_val, y_val_pred_mlp))
print(classification_report(y_val, y_val_pred_mlp, zero_division=0))

# Test
y_test_pred_mlp = mlp.predict(X_test_scaled)

print("\nMLP Test Accuracy:", accuracy_score(y_test, y_test_pred_mlp))
print(confusion_matrix(y_test, y_test_pred_mlp))
print(classification_report(y_test, y_test_pred_mlp, zero_division=0))

# Save MLP predictions
test_df_mlp = test_df.copy()
test_df_mlp["predicted_label"] = y_test_pred_mlp
test_df_mlp.to_csv("predicted_landcover_mlp.csv", index=False)

print("\nSaved MLP predictions to predicted_landcover_2024_200m_mlp.csv")


# =====================================================
# 11. VISUALIZATIONS
# =====================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# -----------------------------
# PCA plot
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

scatter = axes[0, 0].scatter(
    X_pca[:, 0], X_pca[:, 1], c=y_train_resampled, cmap="tab10", s=5
)
axes[0, 0].set_title("PCA Visualization of Training Data")
axes[0, 0].set_xlabel("PC1")
axes[0, 0].set_ylabel("PC2")

# -----------------------------
# Actual vs Predicted (MLP)
# -----------------------------
actual_counts = pd.Series(y_test).value_counts().sort_index()
pred_counts_mlp = pd.Series(y_test_pred_mlp).value_counts().sort_index()

df_compare = pd.DataFrame(
    {"Actual": actual_counts, "Predicted (MLP)": pred_counts_mlp}
).fillna(0)

df_compare.plot(kind="bar", ax=axes[0, 1])
axes[0, 1].set_title("Actual vs Predicted Class Distribution (MLP)")
axes[0, 1].set_ylabel("Number of Cells")
axes[0, 1].tick_params(axis="x", rotation=0)

# -----------------------------
# Ridge confusion matrix
# -----------------------------
cm_ridge = confusion_matrix(y_test, y_test_pred_ridge)
disp_ridge = ConfusionMatrixDisplay(confusion_matrix=cm_ridge)
disp_ridge.plot(ax=axes[1, 0], colorbar=False)
axes[1, 0].set_title("Confusion Matrix - Ridge")

# -----------------------------
# MLP confusion matrix
# -----------------------------
cm_mlp = confusion_matrix(y_test, y_test_pred_mlp)
disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp)
disp_mlp.plot(ax=axes[1, 1], colorbar=False)
axes[1, 1].set_title("Confusion Matrix - MLP")

plt.tight_layout()
plt.show()

# =====================================================
# 12. FINAL SUMMARY
# =====================================================

print("\n================ FINAL SUMMARY ================")
print("Ridge Validation Accuracy :", accuracy_score(y_val, y_val_pred_ridge))
print("Ridge Test Accuracy       :", accuracy_score(y_test, y_test_pred_ridge))
print("MLP Validation Accuracy   :", accuracy_score(y_val, y_val_pred_mlp))
print("MLP Test Accuracy         :", accuracy_score(y_test, y_test_pred_mlp))
print("==============================================")
