# Task 4 - support vector machines (linear and RBF) on Set R

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.svm import SVC

Path("results/tables").mkdir(parents=True, exist_ok=True)
Path("results/figures").mkdir(parents=True, exist_ok=True)

features = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

# Set R = the 6 standardised features from Task 1
df = pd.read_csv("data/standardised_dataset.csv")
train = df[df["split"] == "train"]
val = df[df["split"] == "val"]

X_train, y_train = train[features].to_numpy(), train["fault"].to_numpy()
X_val, y_val = val[features].to_numpy(), val["fault"].to_numpy()

C_values = [0.1, 1, 10, 100]
gamma_values = [0.1, 1]

rows = []      # one row per model for Table 3
models = []    # keep the fitted models so we can save the best ones

# 4.1 linear SVM
for C in C_values:
    model = SVC(kernel="linear", C=C)
    model.fit(X_train, y_train)
    val_acc = np.mean(model.predict(X_val) == y_val)
    n_sv = model.n_support_.sum()
    print("linear  C =", C, "-> val accuracy =", round(val_acc, 4), "| support vectors =", n_sv)
    rows.append({"Kernel": "linear", "C": C, "gamma": "-",
                 "Validation accuracy": round(val_acc, 4), "Support vectors": n_sv})
    models.append(model)

# 4.2 RBF SVM, every combination of C and gamma (8 models)
for C in C_values:
    for gamma in gamma_values:
        model = SVC(kernel="rbf", C=C, gamma=gamma)
        model.fit(X_train, y_train)
        val_acc = np.mean(model.predict(X_val) == y_val)
        n_sv = model.n_support_.sum()
        print("rbf     C =", C, "gamma =", gamma, "-> val accuracy =", round(val_acc, 4),
              "| support vectors =", n_sv)
        rows.append({"Kernel": "rbf", "C": C, "gamma": gamma,
                     "Validation accuracy": round(val_acc, 4), "Support vectors": n_sv})
        models.append(model)

# 4.3 Table 3
table3 = pd.DataFrame(rows)
print("\nTable 3 - SVM validation results")
print(table3.to_string(index=False))
table3.to_csv("results/tables/table3_svm_validation.csv", index=False)

# 4.4 keep the best setting for each kernel and save it
# (if two settings tie, idxmax picks the first one, i.e. the smaller C)
for kernel in ["linear", "rbf"]:
    part = table3[table3["Kernel"] == kernel]
    best_row = part["Validation accuracy"].idxmax()
    best = table3.loc[best_row]
    print("\nBest", kernel, "SVM: C =", best["C"], "gamma =", best["gamma"],
          "val accuracy =", best["Validation accuracy"])
    joblib.dump(models[best_row], "results/svm_" + kernel + "_best.joblib")
    print("Saved results/svm_" + kernel + "_best.joblib")

# figure: validation accuracy against C (log scale)
fig, ax = plt.subplots(figsize=(8, 5))
lin = table3[table3["Kernel"] == "linear"]
ax.plot(lin["C"], lin["Validation accuracy"], marker="o", label="linear")
for gamma in gamma_values:
    part = table3[(table3["Kernel"] == "rbf") & (table3["gamma"] == gamma)]
    ax.plot(part["C"], part["Validation accuracy"], marker="o", label="rbf, gamma = " + str(gamma))
ax.set_xscale("log")
ax.set_xlabel("C (log scale)")
ax.set_ylabel("Validation accuracy")
ax.set_title("SVM validation accuracy against C")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("results/figures/4_svm_val_accuracy_vs_C.png", dpi=150)
plt.close(fig)
print("Saved results/figures/4_svm_val_accuracy_vs_C.png")