import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# 1. LOAD DATA
# =====================================================

df2018 = pd.read_csv("clean_dataset_200m/2018_clean.csv")
df2020 = pd.read_csv("clean_dataset_200m/2020_clean.csv")
df2022 = pd.read_csv("clean_dataset_200m/2022_clean.csv")
df2024 = pd.read_csv("clean_dataset_200m/2024_clean.csv")

df2018["year"] = 2018
df2020["year"] = 2020
df2022["year"] = 2022
df2024["year"] = 2024

df = pd.concat([df2018, df2020, df2022, df2024], ignore_index=True)

label_names = {
    0: "Vegetation",
    1: "Built-up",
    2: "Water",
    3: "Other"
}

print("\n================ DATA LOADED ================")
print(df.shape)

# =====================================================
# 2. LABEL DISTRIBUTION CHECK (CRITICAL)
# =====================================================

print("\n================ LABEL DISTRIBUTION ================")

dist = df.groupby("year")["label"].value_counts(normalize=True).unstack(fill_value=0)

print(dist)

# Plot
dist.plot(kind="bar", figsize=(10,6))
plt.title("Label Distribution Shift Across Years")
plt.ylabel("Proportion")
plt.xticks(rotation=0)
plt.legend([label_names[i] for i in range(4)])
plt.tight_layout()
plt.show()

# =====================================================
# 3. ABSOLUTE COUNT COMPARISON
# =====================================================

print("\n================ ABSOLUTE COUNTS ================")

counts = df.groupby("year")["label"].value_counts().unstack(fill_value=0)
print(counts)

counts.plot(kind="bar", figsize=(10,6))
plt.title("Absolute Label Counts Across Years")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.legend([label_names[i] for i in range(4)])
plt.tight_layout()
plt.show()

# =====================================================
# 4. FEATURE vs LABEL CONSISTENCY CHECK
# =====================================================

print("\n================ FEATURE vs LABEL CHECK ================")

features = ["NDVI", "NDBI", "NDWI"]

feature_means = df.groupby("label")[features].mean()
print(feature_means)

# Visualization
feature_means.plot(kind="bar", figsize=(10,6))
plt.title("Feature Means per Class (Should Make Sense Physically)")
plt.ylabel("Value")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# =====================================================
# 5. YEAR-WISE FEATURE DISTRIBUTION
# =====================================================

print("\n================ FEATURE DISTRIBUTION SHIFT ================")

for feature in features:
    plt.figure(figsize=(8,5))
    sns.kdeplot(data=df, x=feature, hue="year", fill=True)
    plt.title(f"{feature} Distribution Across Years")
    plt.show()

# =====================================================
# 6. TRANSITION MATRIX (2018 → 2020)
# =====================================================

print("\n================ TRANSITION MATRIX (2018 → 2020) ================")

# IMPORTANT: matching rows by index (assumes same grid order)
df2018_sorted = df2018.sort_values(by=["system:index"]).reset_index(drop=True)
df2020_sorted = df2020.sort_values(by=["system:index"]).reset_index(drop=True)

transition_matrix = pd.crosstab(df2018_sorted["label"], df2020_sorted["label"])

print("\nTransition Matrix:")
print(transition_matrix)

# Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(transition_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Transition Matrix (2018 → 2020)")
plt.xlabel("2020")
plt.ylabel("2018")
plt.show()

# =====================================================
# 7. IMPOSSIBLE TRANSITIONS DETECTION
# =====================================================

print("\n================ SUSPICIOUS TRANSITIONS ================")

# Built-up (1) → Vegetation (0)
built_to_veg = transition_matrix.loc[1, 0] if (1 in transition_matrix.index and 0 in transition_matrix.columns) else 0

total_built = df2018_sorted["label"].value_counts().get(1, 1)

ratio = built_to_veg / total_built

print(f"Built-up → Vegetation transitions: {built_to_veg}")
print(f"Total Built-up in 2018: {total_built}")
print(f"Ratio: {ratio:.2f}")

if ratio > 0.2:
    print("\n⚠️ WARNING: Extremely high Built-up → Vegetation transition")
    print("Likely DATASET MISMATCH, not real-world change.")

# =====================================================
# 8. SAME FEATURE, DIFFERENT LABEL CHECK
# =====================================================

print("\n================ SAME FEATURE DIFFERENT LABEL CHECK ================")

# Merge same cells
merged = df2018_sorted.copy()
merged["label_2020"] = df2020_sorted["label"]

# Look for conflicting labels
conflicts = merged[merged["label"] != merged["label_2020"]]

print("Total conflicting cells:", len(conflicts))

# Check feature similarity
diff = np.abs(conflicts["NDVI"] - conflicts["NDVI"].mean())

print("\nSample conflicting rows:")
print(conflicts.head())

# =====================================================
# 9. FINAL DIAGNOSIS
# =====================================================

print("\n================ FINAL DIAGNOSIS ================")

print("""
If you observe:

1. Large label distribution shift between 2018 and 2020
2. High Built-up → Vegetation transitions
3. Similar feature values but different labels
4. Feature-label mismatch (e.g., vegetation with low NDVI)

Then:

>>> ROOT CAUSE = LABEL INCONSISTENCY (CORINE vs ESA)

NOT model issue
NOT preprocessing issue
""")