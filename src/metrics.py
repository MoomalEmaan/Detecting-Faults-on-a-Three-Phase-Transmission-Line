# Task 5.1 - our own accuracy and confusion matrix (no sklearn.metrics allowed)

import numpy as np


def accuracy(y_true, y_pred):
    # fraction of rows where the prediction matches the true label
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sum(y_true == y_pred) / len(y_true)


def confusion_matrix(y_true, y_pred):
    # rows = actual class, columns = predicted class
    #
    #                 predicted 0       predicted 1
    #   actual 0      TN (correct)      FP (false alarm)
    #   actual 1      FN (missed fault) TP (correct)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    return np.array([[tn, fp],
                     [fn, tp]])