# Task 2 - exploratory plots on the RAW (not standardised) data
# run from the project folder:  python src/eda.py   (run prepare.py first)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Path("results/figures").mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/clean_dataset.csv")

normal = df[df["fault"] == 0]
faulty = df[df["fault"] == 1]

# colours used for the two classes in every plot
NO_FAULT_COLOUR = "tab:blue"
FAULT_COLOUR = "tab:orange"


# 2.1 currents over time for the first 2000 rows, fault regions shaded
first = df.iloc[:2000]
rows = np.arange(len(first))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(rows, first["Ia"], label="Ia", linewidth=1)
ax.plot(rows, first["Ib"], label="Ib", linewidth=1)
ax.plot(rows, first["Ic"], label="Ic", linewidth=1)

# shade the background wherever fault = 1 (fills the full height of the plot)
ax.fill_between(rows, 0, 1, where=first["fault"] == 1, transform=ax.get_xaxis_transform(),
                color="grey", alpha=0.25, label="fault = 1")

ax.set_xlabel("Row number (time sample)")
ax.set_ylabel("Current (A)")
ax.set_title("Line currents for the first 2000 rows (shaded = fault)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("results/figures/2_1_currents_vs_time.png", dpi=150)
plt.close(fig)
print("Saved results/figures/2_1_currents_vs_time.png")


# 2.2 scatter of Ia vs Ib coloured by fault
fig, ax = plt.subplots(figsize=(7, 6))
# small dots + some transparency because there are 12000 points
ax.scatter(normal["Ia"], normal["Ib"], s=4, alpha=0.4, color=NO_FAULT_COLOUR, label="no fault (0)")
ax.scatter(faulty["Ia"], faulty["Ib"], s=4, alpha=0.4, color=FAULT_COLOUR, label="fault (1)")
ax.set_xlabel("Ia (A)")
ax.set_ylabel("Ib (A)")
ax.set_title("Ia vs Ib coloured by fault")
ax.legend(markerscale=4)
fig.tight_layout()
fig.savefig("results/figures/2_2_scatter_Ia_Ib.png", dpi=150)
plt.close(fig)
print("Saved results/figures/2_2_scatter_Ia_Ib.png")


# 2.3 histogram of Ia for fault and no-fault rows on the same axes
# use the same bins for both so the bars line up
bins = np.linspace(df["Ia"].min(), df["Ia"].max(), 80)

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(normal["Ia"], bins=bins, alpha=0.6, color=NO_FAULT_COLOUR, label="no fault (0)")
ax.hist(faulty["Ia"], bins=bins, alpha=0.6, color=FAULT_COLOUR, label="fault (1)")
ax.set_xlabel("Ia (A)")
ax.set_ylabel("Number of rows")
ax.set_title("Histogram of Ia for fault and no-fault rows")
ax.legend()
fig.tight_layout()
fig.savefig("results/figures/2_3_hist_Ia.png", dpi=150)
plt.close(fig)
print("Saved results/figures/2_3_hist_Ia.png")


# a few numbers to back up what the plots show
print("\nIa summary for each class:")
print(df.groupby("fault")["Ia"].describe().round(1))
print("\nMean of |Ia| for each class (how big the current is, ignoring sign):")
print(df.groupby("fault")["Ia"].apply(lambda s: s.abs().mean()).round(1))
