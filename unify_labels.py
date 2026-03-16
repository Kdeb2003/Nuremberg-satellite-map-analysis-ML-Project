import pandas as pd


# -------------------------
# LOAD DATASETS
# -------------------------

df2018 = pd.read_excel("data2/nuremberg_grid_dataset_2018_CORINE.xlsx")
df2020 = pd.read_excel("data2/nuremberg_grid_dataset_2020.xlsx")
df2022 = pd.read_excel("data2/nuremberg_grid_dataset_2022.xlsx")
df2024 = pd.read_excel("data2/nuremberg_grid_dataset_2024.xlsx")


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
# SAVE CLEAN DATASETS
# -------------------------

df2018.to_csv("2018_clean.csv", index=False)
df2020.to_csv("2020_clean.csv", index=False)
df2022.to_csv("2022_clean.csv", index=False)
df2024.to_csv("2024_clean.csv", index=False)


print("Datasets cleaned and saved.")