# Task 7 

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# the project folder is one level above app/, so the paths work wherever streamlit is started from
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
from logistic import hypothesis

features = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

# load the saved models and the training mean/std used to scale them
theta_Q = pd.read_csv(ROOT / "results/theta_setQ.csv")["theta"].to_numpy()
stats_Q = pd.read_csv(ROOT / "results/scaling_stats_setQ.csv", index_col=0)
stats_R = pd.read_csv(ROOT / "results/scaling_stats.csv", index_col=0)
svm_rbf = joblib.load(ROOT / "results/svm_rbf_best.joblib")

# a few real rows from the dataset to fill the boxes quickly
examples = {
    "Example: normal operation": [-43.27, 62.71, -23.09, -0.5442, 0.0388, 0.5054],
    "Example: fault (huge current on phase A)": [879.34, 52.81, 1.32, -0.0622, -0.2045, 0.2668],
    "Example: fault (phases A and B)": [636.98, -621.39, -13.48, 0.0226, -0.4202, 0.3975],
}

st.title("Transmission line fault detector")
st.write("Enter the three line currents (A) and three line voltages (per unit) at one instant. "
         "Both saved models will say whether it looks like a fault.")

choice = st.selectbox("Start from an example (you can still change the numbers)", list(examples.keys()))
start = examples[choice]

col_i, col_v = st.columns(2)
values = []
with col_i:
    st.subheader("Currents (A)")
    for i, name in enumerate(["Ia", "Ib", "Ic"]):
        # the key includes the example name so the box resets when a new example is picked
        values.append(st.number_input(name, value=start[i], format="%.2f", key=name + choice))
with col_v:
    st.subheader("Voltages (pu)")
    for i, name in enumerate(["Va", "Vb", "Vc"]):
        values.append(st.number_input(name, value=start[i + 3], format="%.4f", key=name + choice))

x_raw = np.array(values)

# Set Q logistic regression: add squares, scale with Set Q training stats, put 1 in front
x_q = np.concatenate([x_raw, x_raw ** 2])
x_q = (x_q - stats_Q["mean"].to_numpy()) / stats_Q["std"].to_numpy()
x_q = np.concatenate([[1.0], x_q])
p_fault = hypothesis(x_q, theta_Q)
lr_pred = int(p_fault >= 0.5)

# RBF SVM: scale the 6 raw values with the Task 1 training stats
x_r = (x_raw - stats_R["mean"].to_numpy()) / stats_R["std"].to_numpy()
svm_pred = int(svm_rbf.predict(x_r.reshape(1, -1))[0])

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Logistic regression (Set Q)")
    if lr_pred == 1:
        st.error("FAULT")
    else:
        st.success("NO FAULT")
    st.write("Probability of fault:", round(float(p_fault), 4))
with right:
    st.subheader("RBF SVM (C=" + str(svm_rbf.C) + ", gamma=" + str(svm_rbf.gamma) + ")")
    if svm_pred == 1:
        st.error("FAULT")
    else:
        st.success("NO FAULT")
    st.write("Decision value:", round(float(svm_rbf.decision_function(x_r.reshape(1, -1))[0]), 4),
             "(positive = fault)")

if lr_pred != svm_pred:
    st.warning("The two models disagree on this input.")
