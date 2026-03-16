import pandas as pd

df_2018 = pd.read_excel("data/nuremberg_grid_dataset_2018.xlsx")
df_2024 = pd.read_excel("data/nuremberg_grid_dataset_2024.xlsx")

# Compare labels
comparison = (df_2018["label"] == df_2024["label"]).sum()

print("Same labels:", comparison)
print("Total samples:", len(df_2018))
print("Percentage same:", comparison / len(df_2018))