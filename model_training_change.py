import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

# =====================================================
# CHANGE LABEL DEFINITION (2020 -> 2021)
# =====================================================
# 0: no change (label_2020 == label_2021)
# 1: vegetation -> built-up (0 -> 1)
# 2: built-up -> vegetation (1 -> 0)
# 3: other change (all remaining transitions)
CHANGE_LABEL_NAMES = {
    0: "no_change",
    1: "veg_to_built",
    2: "built_to_veg",
    3: "other_change",
}


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without labels and reset index."""
    df = df.dropna(subset=["label"])
    df = df.reset_index(drop=True)
    return df


def fix_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Stabilize floating-point class labels from Earth Engine exports."""
    df["label"] = df["label"].round().astype(int)
    return df


def simplify_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw classes to unified 4 classes: 0/1/2/3."""

    def map_class(x: int) -> int:
        if x in [10, 20, 30, 40]:
            return 0
        if x == 50:
            return 1
        if x == 80:
            return 2
        return 3

    df["label"] = df["label"].apply(map_class)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the same engineered features as v7."""
    df["SAVI"] = ((df["B8"] - df["B4"]) / (df["B8"] + df["B4"] + 0.5)) * 1.5
    df["BSI"] = ((df["B11"] + df["B4"]) - (df["B8"] + df["B2"])) / (
        (df["B11"] + df["B4"]) + (df["B8"] + df["B2"])
    )
    df["UI"] = (df["B11"] - df["B8"]) / (df["B11"] + df["B8"])
    return df


def make_change_label(label_2020: int, label_2021: int) -> int:
    """Create assignment-aligned change target."""
    if label_2020 == label_2021:
        return 0
    if label_2020 == 0 and label_2021 == 1:
        return 1
    if label_2020 == 1 and label_2021 == 0:
        return 2
    return 3


def print_per_class_accuracy(cm: np.ndarray, title: str) -> None:
    print(f"\nPer-class accuracy ({title}):")
    for i in range(cm.shape[0]):
        row_total = cm[i].sum()
        acc = (cm[i, i] / row_total) if row_total > 0 else 0.0
        name = CHANGE_LABEL_NAMES.get(i, f"class_{i}")
        print(f"  class {i} ({name}): {acc:.4f} ({cm[i, i]}/{row_total})")


# =====================================================
# 1) LOAD DATA
# =====================================================

df_2020 = pd.read_excel("data3/nuremberg_grid_dataset_2020_200m.xlsx")
df_2021 = pd.read_excel("data3/nuremberg_grid_dataset_2021_ESA_200m.xlsx")
df_2024 = pd.read_excel("data3/nuremberg_2024_features_clean.xlsx")

df_2020["year"] = 2020
df_2021["year"] = 2021
df_2024["year"] = 2024

print("Loaded shapes:")
print("2020:", df_2020.shape, "| 2021:", df_2021.shape, "| 2024:", df_2024.shape)


# =====================================================
# 2) CLEAN + LABEL NORMALIZATION + FEATURE ENGINEERING
# =====================================================

df_2020_clean = simplify_labels(fix_labels(clean_df(df_2020.copy())))
df_2021_clean = simplify_labels(fix_labels(clean_df(df_2021.copy())))
df_2024_clean = df_2024.drop(columns=["label"], errors="ignore").copy()

df_2020_clean = add_features(df_2020_clean)
df_2021_clean = add_features(df_2021_clean)
df_2024_clean = add_features(df_2024_clean)

features = [
    "B11",
    "B2",
    "B3",
    "B4",
    "B8",
    "NDVI",
    "NDBI",
    "NDWI",
    "SAVI",
    "BSI",
    "UI",
]


# =====================================================
# 3) BUILD CHANGE DATASET (2020 -> 2021)
# =====================================================

# We learn transitions from 2020 features and observed 2020->2021 label change.
# Merge on system:index so each row is the same spatial grid cell.
change_df = (
    df_2020_clean[
        ["system:index", ".geo", "year", "label"] + features
    ].rename(columns={"label": "label_2020", "year": "year_2020"})
    .merge(
        df_2021_clean[["system:index", "year", "label"]].rename(
            columns={"label": "label_2021", "year": "year_2021"}
        ),
        on="system:index",
        how="inner",
    )
)

change_df["change_label"] = change_df.apply(
    lambda r: make_change_label(int(r["label_2020"]), int(r["label_2021"])),
    axis=1,
)
change_df["change_binary"] = (change_df["change_label"] != 0).astype(int)

print("\nMerged change dataset shape:", change_df.shape)
print("Change class distribution:")
print(change_df["change_label"].value_counts().sort_index())
print("Binary change distribution:")
print(change_df["change_binary"].value_counts().sort_index())


# =====================================================
# 4) TRAIN / TEST SPLIT + PREPROCESSING
# =====================================================

X = change_df[features]
y = change_df["change_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Optional imbalance handling: apply SMOTE only when each class has >=2 rows.
class_counts_train = y_train.value_counts()
min_count = int(class_counts_train.min())
if min_count >= 2:
    k_neighbors = min(5, min_count - 1)
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"\nSMOTE applied (k_neighbors={k_neighbors}).")
else:
    print("\nSMOTE skipped due to very small class size in training split.")

print("\nTraining class distribution after balancing:")
print(pd.Series(y_train).value_counts().sort_index())


# =====================================================
# 5) TRAIN MODELS
# =====================================================

ridge = RidgeClassifier()
ridge.fit(X_train, y_train)

mlp = MLPClassifier(hidden_layer_sizes=(96, 48), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)


# =====================================================
# 6) EVALUATION ON HOLDOUT CHANGE TEST SET
# =====================================================

y_pred_ridge = ridge.predict(X_test)
y_pred_mlp = mlp.predict(X_test)

print("\n=== HOLDOUT CHANGE EVALUATION ===")
print("Ridge accuracy:", accuracy_score(y_test, y_pred_ridge))
print("MLP accuracy  :", accuracy_score(y_test, y_pred_mlp))

cm_ridge = confusion_matrix(y_test, y_pred_ridge, labels=[0, 1, 2, 3])
cm_mlp = confusion_matrix(y_test, y_pred_mlp, labels=[0, 1, 2, 3])

print("\nRidge confusion matrix (rows=true, cols=pred):")
print(cm_ridge)
print_per_class_accuracy(cm_ridge, "Ridge")

print("\nMLP confusion matrix (rows=true, cols=pred):")
print(cm_mlp)
print_per_class_accuracy(cm_mlp, "MLP")

print("\n=== CHANGE MODEL EVALUATION ===")
print("MLP classification report:")
print(
    classification_report(
        y_test,
        y_pred_mlp,
        labels=[0, 1, 2, 3],
        target_names=[
            CHANGE_LABEL_NAMES[0],
            CHANGE_LABEL_NAMES[1],
            CHANGE_LABEL_NAMES[2],
            CHANGE_LABEL_NAMES[3],
        ],
        zero_division=0,
    )
)

print("\n=== BINARY CHANGE METRICS ===")
y_test_binary = (y_test != 0).astype(int)
y_pred_binary = (y_pred_mlp != 0).astype(int)

binary_acc = accuracy_score(y_test_binary, y_pred_binary)
binary_precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
binary_recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
binary_f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)

print(f"Binary Change Accuracy: {binary_acc:.4f}")
print(f"Binary Change Precision: {binary_precision:.4f}")
print(f"Binary Change Recall: {binary_recall:.4f}")
print(f"Binary Change F1: {binary_f1:.4f}")

print("\n=== ERROR ANALYSIS ===")
no_change_mask = (y_test == 0)
change_mask = (y_test != 0)

if no_change_mask.sum() > 0:
    false_change_rate = ((no_change_mask) & (y_pred_mlp != 0)).sum() / no_change_mask.sum()
else:
    false_change_rate = 0.0

if change_mask.sum() > 0:
    missed_change_rate = ((change_mask) & (y_pred_mlp == 0)).sum() / change_mask.sum()
else:
    missed_change_rate = 0.0

print(f"False Change Rate: {false_change_rate:.4f}")
print(f"Missed Change Rate: {missed_change_rate:.4f}")

print("\n=== STRESS TEST RESULTS ===")
rng = np.random.default_rng(42)
noise = rng.normal(0, 0.1, X_test.shape)
X_test_noisy = X_test + noise
y_pred_noisy = mlp.predict(X_test_noisy)

acc_original = accuracy_score(y_test, y_pred_mlp)
acc_noisy = accuracy_score(y_test, y_pred_noisy)

y_pred_noisy_binary = (y_pred_noisy != 0).astype(int)
binary_acc_original = accuracy_score(y_test_binary, y_pred_binary)
binary_acc_noisy = accuracy_score(y_test_binary, y_pred_noisy_binary)

change_in_predictions = (y_pred_mlp != y_pred_noisy).sum()

print(f"Accuracy (Original): {acc_original:.4f}")
print(f"Accuracy (Noisy): {acc_noisy:.4f}")
print(f"Binary Change Accuracy (Original): {binary_acc_original:.4f}")
print(f"Binary Change Accuracy (Noisy): {binary_acc_noisy:.4f}")
print(f"Predictions changed under noise: {int(change_in_predictions)}")


# =====================================================
# 7) SAVE 2021 CHANGE PREDICTIONS (TRUE + PRED)
# =====================================================

X_all_2020 = scaler.transform(imputer.transform(change_df[features]))
change_df["ridge_change_pred"] = ridge.predict(X_all_2020)
change_df["mlp_change_pred"] = mlp.predict(X_all_2020)
change_df["ridge_change_binary"] = (change_df["ridge_change_pred"] != 0).astype(int)
change_df["mlp_change_binary"] = (change_df["mlp_change_pred"] != 0).astype(int)

pred_change_2021 = change_df[
    [
        "system:index",
        ".geo",
        "year_2020",
        "year_2021",
        "label_2020",
        "label_2021",
        "change_label",
        "change_binary",
        "ridge_change_pred",
        "mlp_change_pred",
        "ridge_change_binary",
        "mlp_change_binary",
    ]
].copy()

pred_change_2021 = pred_change_2021.rename(
    columns={
        "change_label": "change_label_true",
        "change_binary": "change_binary_true",
    }
)
pred_change_2021.to_csv("predictions_change_2021.csv", index=False)
print("\nSaved predictions_change_2021.csv")


# =====================================================
# 8) PREDICT CHANGE FOR 2024
# =====================================================

# We use 2024 features so the model outputs likely change-state categories
# under 2024 observed conditions (assignment-focused forward inference).
X_2024 = scaler.transform(imputer.transform(df_2024_clean[features]))

pred_change_2024 = df_2024.copy()
pred_change_2024["ridge_change_pred"] = ridge.predict(X_2024)
pred_change_2024["mlp_change_pred"] = mlp.predict(X_2024)
pred_change_2024["ridge_change_binary"] = (
    pred_change_2024["ridge_change_pred"] != 0
).astype(int)
pred_change_2024["mlp_change_binary"] = (
    pred_change_2024["mlp_change_pred"] != 0
).astype(int)

pred_change_2024.to_csv("predictions_change_2024.csv", index=False)
print("Saved predictions_change_2024.csv")

print(
    "\nNote: This change model is a simplified transition classifier "
    "using per-cell tabular features; it does not model spatial dependencies."
)
