import pandas as pd
import os


# -------------------------
# CREATE OUTPUT FOLDER
# -------------------------

os.makedirs("clean_dataset_200m", exist_ok=True)


# -------------------------
# LOAD DATASETS (200m)
# -------------------------

df2018 = pd.read_excel("data3with200mgridsize/nuremberg_grid_dataset_2018_CORINE_200m.xlsx")
df2020 = pd.read_excel("data3with200mgridsize/nuremberg_grid_dataset_2020_200m.xlsx")
df2022 = pd.read_excel("data3with200mgridsize/nuremberg_grid_dataset_2022_200m.xlsx")
df2024 = pd.read_excel("data3with200mgridsize/nuremberg_grid_dataset_2024_200m.xlsx")


# -------------------------
# CORINE → UNIFIED LABELS
# -------------------------

def corine_to_unified(label):

    if label in [311,312,313,211,231]:
        return 0   # vegetation

    elif label in [111,112,121,122]:
        return 1   # built-up

    elif label in [512]:
        return 2   # water

    else:
        return 3   # other


df2018["label"] = df2018["label"].apply(corine_to_unified)


# -------------------------
# WORLDCOVER → UNIFIED
# -------------------------

def worldcover_to_unified(label):

    if label in [10,30,40]:
        return 0   # vegetation

    elif label == 50:
        return 1   # built-up

    elif label == 80:
        return 2   # water

    else:
        return 3   # other


df2020["label"] = df2020["label"].apply(worldcover_to_unified)
df2022["label"] = df2022["label"].apply(worldcover_to_unified)
df2024["label"] = df2024["label"].apply(worldcover_to_unified)


# -------------------------
# CHECK LABEL DISTRIBUTION
# -------------------------

print("\nLabel distribution after conversion:\n")

print("2018:")
print(df2018["label"].value_counts())

print("\n2020:")
print(df2020["label"].value_counts())

print("\n2022:")
print(df2022["label"].value_counts())

print("\n2024:")
print(df2024["label"].value_counts())


# -------------------------
# SAVE CLEAN DATASETS
# -------------------------

df2018.to_csv("clean_dataset_200m/2018_clean.csv", index=False)
df2020.to_csv("clean_dataset_200m/2020_clean.csv", index=False)
df2022.to_csv("clean_dataset_200m/2022_clean.csv", index=False)
df2024.to_csv("clean_dataset_200m/2024_clean.csv", index=False)


print("\nDatasets cleaned and saved in 'clean_dataset_200m/'")