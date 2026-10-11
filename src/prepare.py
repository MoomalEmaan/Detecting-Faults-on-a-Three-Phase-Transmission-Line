# Task 1 - load, clean, split and standardise the data

from pathlib import Path
import numpy as np
import pandas as pd

from load_data import load_raw

Path("results/tables").mkdir(parents=True, exist_ok=True)

# 1.1 Look at the raw file 
raw = load_raw()
print("Raw shape:", raw.shape)
print("Columns:", raw.columns.tolist())
print(raw.head(5).to_string())

# 1.2 Clean 
print("\nInspecting the raw columns:")
info = pd.DataFrame({"missing": raw.isna().sum(), "unique values": raw.nunique()})
print(info)

removed = [c for c in raw.columns if raw[c].isna().all() or raw[c].nunique() <= 1]
df = raw.drop(columns=removed)
print("\nRemoved columns (no information):", removed)

# Rename the label column
df = df.rename(columns={"Output (S)": "fault"})
print("Columns now:", df.columns.tolist())

# Check the six measurement columns are numbers
features = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
print("\nData type of each measurement column:")
print(df[features].dtypes)

print("\nMissing values per column:")
print(df.isna().sum().to_string())

# 1.3 Check the number of rows with fault=1 and fault=0 
n_fault = int((df["fault"] == 1).sum())
n_normal = int((df["fault"] == 0).sum())
print("\nfault = 1:", n_fault, "rows")
print("fault = 0:", n_normal, "rows")

# 1.4 split  
rng = np.random.default_rng(42)
idx = rng.permutation(len(df))
n_train, n_val = int(0.6 * len(df)), int(0.2 * len(df))
train_idx = idx[:n_train]
val_idx = idx[n_train:n_train + n_val]
test_idx = idx[n_train + n_val:]

# Mark every row with its split, so we keep the original row order (needed for the time plot)
df["split"] = ""
df.loc[train_idx, "split"] = "train"
df.loc[val_idx, "split"] = "val"
df.loc[test_idx, "split"] = "test"

print("\nSplit sizes and fault fraction:")
for name in ["train", "val", "test"]:
    part = df[df["split"] == name]
    print(" ", name, ":", len(part), "rows, fault fraction =", round(part["fault"].mean(), 4))

df.to_csv("data/clean_dataset.csv", index=False)
print("\nSaved data/clean_dataset.csv")

# 1.5 standardise
# mean and std come from the training rows only, then the same numbers are used for val and test
is_train = df["split"] == "train"
mu = df.loc[is_train, features].mean()
sigma = df.loc[is_train, features].std(ddof=0)

std_df = (df[features] - mu) / sigma
std_df["fault"] = df["fault"]
std_df["split"] = df["split"]
std_df.to_csv("data/standardised_dataset.csv", index=False)

pd.DataFrame({"mean": mu, "std": sigma}).to_csv("results/scaling_stats.csv")

print("\nTraining mean:", np.round(mu.values, 4))
print("Training std: ", np.round(sigma.values, 4))
# quick check: training part should now have mean 0 and std 1
print("Train mean after scaling:", np.round(std_df.loc[is_train, features].mean().values, 4))
print("Train std after scaling: ", np.round(std_df.loc[is_train, features].std(ddof=0).values, 4))
print("Saved data/standardised_dataset.csv and results/scaling_stats.csv")

# Table 1 
y = df["fault"]

table1 = pd.DataFrame({
    "Item": [
        "Raw rows / raw columns",
        "Columns removed (names)",
        "Missing values after cleaning",
        "Fault rows / no-fault rows",
        "Train / validation / test rows",
        "Fault fraction in train / validation / test",
    ],
    "Value": [
        str(raw.shape[0]) + " / " + str(raw.shape[1]),
        ", ".join(removed),
        df.isna().sum().sum(),
        str(n_fault) + " / " + str(n_normal),
        str(len(train_idx)) + " / " + str(len(val_idx)) + " / " + str(len(test_idx)),
        str(round(y[train_idx].mean(), 3)) + " / " + str(round(y[val_idx].mean(), 3)) + " / " + str(round(y[test_idx].mean(), 3)),
    ],
})

print("\nTable 1 - Data preparation")
print(table1.to_string(index=False))
table1.to_csv("results/tables/table1_data_prep.csv", index=False)
print("Saved results/tables/table1_data_prep.csv")
