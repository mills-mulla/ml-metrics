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

def _safe_divide_helper(num, denom):
    return 0.0 if denom == 0 else num / denom

def specificity(y_true, y_pred):
    """ specificity = TN / (TN + FP) """
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    denom = (tn + fp)

    return  _safe_divide_helper(tn, denom)

def precision(y_true, y_pred):
    """ precision = TP / (TP + FP)"""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]

    denom = (tp + fp)
    return  _safe_divide_helper(tp, denom)
    

def recall(y_true, y_pred):
    """ recall = TP / (TP +FN)"""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fn = cm[1, 0]
    denom = (tp + fn)
    return  _safe_divide_helper(tp, denom)