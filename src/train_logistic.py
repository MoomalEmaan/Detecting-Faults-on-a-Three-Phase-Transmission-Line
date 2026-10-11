# Task 3 - logistic regression by hand on Set R and Set Q


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic import cost, fit_logistic, predict

Path("results/tables").mkdir(parents=True, exist_ok=True)
Path("results/figures").mkdir(parents=True, exist_ok=True)

features = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

# Build the two feature sets
# Set R: the 6 features, already standardised in Task 1
std_df = pd.read_csv("data/standardised_dataset.csv")
setR = std_df[features]

# Set Q: the 6 raw features + their squares.
# squares are taken from the RAW values first, then all 12 columns are standardised with training stats
clean_df = pd.read_csv("data/clean_dataset.csv")
setQ = clean_df[features].copy()
for c in features:
    setQ[c + "^2"] = clean_df[c] ** 2

is_train = clean_df["split"] == "train"
mu_Q = setQ[is_train].mean()
sigma_Q = setQ[is_train].std(ddof=0)
setQ = (setQ - mu_Q) / sigma_Q

# the app will need these to scale new inputs for the Set Q model
pd.DataFrame({"mean": mu_Q, "std": sigma_Q}).to_csv("results/scaling_stats_setQ.csv")

y_all = clean_df["fault"].to_numpy()
split = clean_df["split"]


def get_split(X_df, name):
    # pick the rows of one split and put a column of ones in front (x0 = 1)
    X = X_df[split == name].to_numpy()
    X = np.hstack([np.ones((len(X), 1)), X])
    y = y_all[split == name]
    return X, y


alphas = [0.01, 0.1, 1, 10]
n_iters = 2000

table2_rows = []
thetas = {}

for set_name, X_df in [("R", setR), ("Q", setQ)]:
    print("\n==========  Set", set_name, "(" + str(X_df.shape[1]) + " features)  ==========")
    X_train, y_train = get_split(X_df, "train")
    X_val, y_val = get_split(X_df, "val")

    # check: with theta = 0, h = 0.5 for every row, so J should be log(2) = 0.693
    J0 = cost(X_train, y_train, np.zeros(X_train.shape[1]))
    print("J at theta = 0:", round(J0, 4), "  (log 2 =", round(np.log(2), 4), ")")

    # 3.4 try every alpha and plot J against iteration
    fig, ax = plt.subplots(figsize=(8, 5))
    good_alphas = []
    for alpha in alphas:
        theta, J_hist = fit_logistic(X_train, y_train, alpha, n_iters)
        J_all = [J0] + J_hist   # include the starting cost so the first step is checked too
        always_down = bool(np.all(np.diff(J_all) < 0))
        if always_down:
            good_alphas.append(alpha)
        print("  alpha =", alpha, "-> final J =", round(J_hist[-1], 4),
              "| J decreased at every iteration:", always_down)
        ax.plot(range(1, n_iters + 1), J_hist, label="alpha = " + str(alpha))

    ax.set_xlabel("Iteration")
    ax.set_ylabel("J(theta)")
    ax.set_title("Cost during gradient descent, Set " + set_name)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/3_4_cost_set" + set_name + ".png", dpi=150)
    plt.close(fig)

    # pick the largest alpha where J went down every single iteration
    best_alpha = max(good_alphas)
    print("Chosen alpha for Set", set_name, ":", best_alpha)

    # 3.5 final model with the chosen alpha
    theta, J_hist = fit_logistic(X_train, y_train, best_alpha, n_iters)
    train_acc = np.mean(predict(X_train, theta) == y_train)
    val_acc = np.mean(predict(X_val, theta) == y_val)
    print("Final J:", round(J_hist[-1], 4))
    print("Training accuracy:  ", round(train_acc, 4))
    print("Validation accuracy:", round(val_acc, 4))

    names = ["theta0 (intercept)"] + list(X_df.columns)
    print("theta:")
    for name, value in zip(names, theta):
        print("  ", name, "=", round(value, 4))

    thetas[set_name] = theta
    # save theta so test_models.py and the app can use it without retraining
    pd.DataFrame({"weight": names, "theta": theta}).to_csv(
        "results/theta_set" + set_name + ".csv", index=False)

    table2_rows.append({
        "Feature set": "Set " + set_name,
        "Chosen alpha": best_alpha,
        "Final J(theta)": round(J_hist[-1], 4),
        "Training accuracy": round(train_acc, 4),
        "Validation accuracy": round(val_acc, 4),
    })

# Table 2
table2 = pd.DataFrame(table2_rows)
print("\nTable 2 - Logistic regression")
print(table2.to_string(index=False))
table2.to_csv("results/tables/table2_logistic.csv", index=False)

# Table 5 - learned theta for Set Q
table5 = pd.DataFrame({
    "Weight": ["theta0 (intercept)"] + list(setQ.columns),
    "Value": np.round(thetas["Q"], 4),
})
print("\nTable 5 - Learned theta (Set Q)")
print(table5.to_string(index=False))
table5.to_csv("results/tables/table5_theta_setQ.csv", index=False)

print("\nSaved tables 2 and 5, theta for both sets, Set Q scaling stats and the two cost plots")