# Task 5 - evaluate the four final models on the TEST rows, once
# nothing is trained or tuned here, we only load the saved models and predict

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from logistic import predict
from metrics import accuracy, confusion_matrix

Path("results/tables").mkdir(parents=True, exist_ok=True)

features = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

# test rows for Set R (already standardised with training stats in Task 1)
std_df = pd.read_csv("data/standardised_dataset.csv")
test_R = std_df[std_df["split"] == "test"]
X_test_R = test_R[features].to_numpy()
y_test = test_R["fault"].to_numpy()

# test rows for Set Q: square the raw values, then scale with the TRAINING stats saved in Task 3
clean_df = pd.read_csv("data/clean_dataset.csv")
test_raw = clean_df[clean_df["split"] == "test"]
setQ = test_raw[features].copy()
for c in features:
    setQ[c + "^2"] = test_raw[c] ** 2

stats_Q = pd.read_csv("results/scaling_stats_setQ.csv", index_col=0)
setQ = (setQ - stats_Q["mean"]) / stats_Q["std"]
X_test_Q = setQ.to_numpy()

# logistic regression needs the column of ones in front (x0 = 1)
ones = np.ones((len(y_test), 1))
X_test_R1 = np.hstack([ones, X_test_R])
X_test_Q1 = np.hstack([ones, X_test_Q])

# load the saved models
theta_R = pd.read_csv("results/theta_setR.csv")["theta"].to_numpy()
theta_Q = pd.read_csv("results/theta_setQ.csv")["theta"].to_numpy()
svm_linear = joblib.load("results/svm_linear_best.joblib")
svm_rbf = joblib.load("results/svm_rbf_best.joblib")

linear_name = "Linear SVM (C=" + str(svm_linear.C) + ")"
rbf_name = "RBF SVM (C=" + str(svm_rbf.C) + ", gamma=" + str(svm_rbf.gamma) + ")"

predictions = {
    "Logistic regression, Set R": predict(X_test_R1, theta_R),
    "Logistic regression, Set Q": predict(X_test_Q1, theta_Q),
    linear_name: svm_linear.predict(X_test_R),
    rbf_name: svm_rbf.predict(X_test_R),
}

print("Test rows:", len(y_test), "(faults:", int(y_test.sum()), ")")

rows = []
for name, y_pred in predictions.items():
    acc = accuracy(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]

    print("\n" + name)
    print("  accuracy:", round(acc, 4))
    print("  confusion matrix (rows = actual 0/1, columns = predicted 0/1):")
    print(cm)
    print("  missed faults:", fn, "| false alarms:", fp)
    # sanity check from the assignment: the four boxes must add up to the number of test rows
    print("  entries add up to test rows:", cm.sum() == len(y_test))

    rows.append({"Model": name, "Accuracy": round(acc, 4), "TN": tn,
                 "False alarms": fp, "Missed faults": fn, "TP": tp})

table4 = pd.DataFrame(rows)
print("\nTable 4 - Test results")
print(table4.to_string(index=False))
table4.to_csv("results/tables/table4_test_results.csv", index=False)
print("Saved results/tables/table4_test_results.csv")