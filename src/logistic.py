# Logistic regression 

import numpy as np


def sigmoid(z):
    # g(z) = 1 / (1 + e^-z)
    return 1 / (1 + np.exp(-z))


def hypothesis(X, theta):
    # h(x) = g(theta^T x), done for all rows at once
    return sigmoid(X @ theta)


def cost(X, y, theta):
    # J(theta) = -(1/n) * sum[ y log h + (1 - y) log(1 - h) ]
    n = len(y)
    h = hypothesis(X, theta)
    # clip so we never take log(0)
    h = np.clip(h, 1e-12, 1 - 1e-12)
    return -(1 / n) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))


def fit_logistic(X, y, alpha, n_iters):
    # batch gradient descent starting from theta = 0
    # update: theta_j := theta_j + (alpha / n) * sum[ (y - h(x)) * x_j ]
    # the X.T @ (...) part does this sum for every j at the same time
    n = len(y)
    theta = np.zeros(X.shape[1])
    J_history = []

    for i in range(n_iters):
        error = y - hypothesis(X, theta)
        theta = theta + (alpha / n) * (X.T @ error)
        J_history.append(cost(X, y, theta))   # J after this iteration

    return theta, J_history


def predict(X, theta):
    # fault (1) if h(x) >= 0.5, otherwise no fault (0)
    return (hypothesis(X, theta) >= 0.5).astype(int)
