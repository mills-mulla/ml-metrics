import numpy as np
from .validation import check_binary_labels

def positive_rate(y):
    y = np.asarray(y)
    return (y == 1).mean()

def accuracy(y_true, y_pred):
    result = check_binary_labels(y_true, y_pred)
    a, b = result
    return (a == b).mean()

def count_between(x, low, high):
    x = np.asarray(x)
    return ((x >= low) & (x <= high)).sum()

def confusion_matrix(y_true, y_pred):
    a, b = check_binary_labels(y_true, y_pred)
    tn = ((a == 0) & (b == 0)).sum()
    fp = ((a == 0) & (b == 1)).sum()
    fn = ((a == 1) & (b == 0)).sum()
    tp = ((a == 1) & (b == 1)).sum()

    return np.array([[tn, fp], [fn, tp]])

