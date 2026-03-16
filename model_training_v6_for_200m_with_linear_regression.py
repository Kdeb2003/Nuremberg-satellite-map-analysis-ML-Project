import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

from imblearn.over_sampling import RandomOverSampler
from imblearn.ensemble import BalancedRandomForestClassifier


# -----------------------------
# LOAD CLEAN DATASETS
# -----------------------------

df_2018 = pd.read_csv("clean_dataset_200m/2018_clean.csv")
df_2020 = pd.read_csv("clean_dataset_200m/2020_clean.csv")
df_2022 = pd.read_csv("clean_dataset_200m/2022_clean.csv")
df_2024 = pd.read_csv("clean_dataset_200m/2024_clean.csv")

df_2018["year"] = 2018
df_2020["year"] = 2020
df_2022["year"] = 2022
df_2024["year"] = 2024


# Combine datasets
df = pd.concat([df_2018, df_2020, df_2022, df_2024], ignore_index=True)

print("Dataset shape:", df.shape)
print(df.head())


# -----------------------------
# CLEAN DATA
# -----------------------------

df = df.drop(columns=["system:index", ".geo"], errors="ignore")
df = df.dropna(subset=["label"])

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nClass distribution per year:")
print(df.groupby("year")["label"].value_counts().unstack(fill_value=0))


# -----------------------------
# ADD NEW FEATURES
# -----------------------------

df["SAVI"] = ((df["B8"] - df["B4"]) / (df["B8"] + df["B4"] + 0.5)) * 1.5
df["BSI"] = ((df["B11"] + df["B4"]) - (df["B8"] + df["B2"])) / ((df["B11"] + df["B4"]) + (df["B8"] + df["B2"]))
df["UI"] = (df["B11"] - df["B8"]) / (df["B11"] + df["B8"])


# -----------------------------
# TRAIN / VAL / TEST SPLIT
# -----------------------------

train_df = df[df["year"].isin([2018, 2020])]
val_df = df[df["year"] == 2022]
test_df = df[df["year"] == 2024]

print("\nTrain size:", train_df.shape)
print("Validation size:", val_df.shape)
print("Test size:", test_df.shape)


# -----------------------------
# FEATURE LIST
# -----------------------------

features = [
    "B11","B2","B3","B4","B8",
    "NDBI","NDVI","NDWI",
    "SAVI","BSI","UI"
]

X_train = train_df[features]
y_train = train_df["label"]

X_val = val_df[features]
y_val = val_df["label"]

X_test = test_df[features]
y_test = test_df["label"]


# -----------------------------
# HANDLE CLASS IMBALANCE
# -----------------------------

ros = RandomOverSampler(random_state=42)

X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

print("\nBalanced training distribution:")
print(pd.Series(y_train_resampled).value_counts())


# -----------------------------
# IMPUTER
# -----------------------------

imputer = SimpleImputer(strategy="mean")

X_train_imputed = imputer.fit_transform(X_train_resampled)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)


# -----------------------------
# SCALING
# -----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_val_scaled = scaler.transform(X_val_imputed)
X_test_scaled = scaler.transform(X_test_imputed)


# -----------------------------
# LINEAR REGRESSION
# -----------------------------

print("\nTraining Linear Regression...")

linreg = LinearRegression()

linreg.fit(X_train_scaled, y_train_resampled)

# validation predictions
y_val_pred = linreg.predict(X_val_scaled)

# convert to classes
y_val_pred = np.clip(np.round(y_val_pred), 0, 3).astype(int)

print("\nLinear Regression Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print(confusion_matrix(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))


# -----------------------------
# TEST PERFORMANCE
# -----------------------------

y_test_pred = linreg.predict(X_test_scaled)
y_test_pred = np.clip(np.round(y_test_pred), 0, 3).astype(int)

print("\nLinear Regression Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))


# -----------------------------
# BALANCED RANDOM FOREST
# -----------------------------

print("\nTraining Balanced Random Forest...")

rf = BalancedRandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

X_train_rf = train_df[features]
y_train_rf = train_df["label"]

X_train_rf_imputed = imputer.transform(X_train_rf)

rf.fit(X_train_rf_imputed, y_train_rf)

y_val_pred_rf = rf.predict(X_val_imputed)

print("\nRandom Forest Validation Accuracy:", accuracy_score(y_val, y_val_pred_rf))
print(confusion_matrix(y_val, y_val_pred_rf))
print(classification_report(y_val, y_val_pred_rf))


# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------

print("\nRandom Forest Feature Importance:")

importance = pd.Series(rf.feature_importances_, index=features)
print(importance.sort_values(ascending=False))


# -----------------------------
# TEST PREDICTIONS (2024)
# -----------------------------

test_predictions = rf.predict(X_test_imputed)

test_df["predicted_label"] = test_predictions

print("\nTest Accuracy:", accuracy_score(y_test, test_predictions))


# -----------------------------
# SAVE PREDICTIONS
# -----------------------------

test_df.to_csv("predicted_landcover_2024_200m_linear.csv", index=False)

print("\nSaved predictions to predicted_landcover_2024_200m_linear.csv")


# -----------------------------
# PCA VISUALIZATION
# -----------------------------

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_train_scaled)

plt.scatter(X_pca[:,0], X_pca[:,1], c=y_train_resampled, cmap="tab10", s=5)
plt.title("PCA class distribution")
plt.show()