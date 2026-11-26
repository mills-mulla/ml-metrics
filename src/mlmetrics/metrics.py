import numpy as np
from .validation import check_binary_labels


def positive_rate(y):
    """
    Return the fraction of samples equal to 1.

    Parameters
    ----------
    y : array-like of shape (n,)
        Binary labels (0 or 1).

    Returns
    -------
    float
        Proportion of positive labels.
    """
    y = np.asarray(y)
    return (y == 1).mean()


def accuracy(y_true, y_pred):
    """
    Return the fraction of predictions that match the true labels.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Ground-truth binary labels.
    y_pred : array-like of shape (n,)
        Predicted binary labels.

    Returns
    -------
    float
        Accuracy score in [0, 1].

    Raises
    ------
    ValueError
        If inputs are invalid (see check_binary_labels).
    """
    a, b = check_binary_labels(y_true, y_pred)
    return (a == b).mean()


def count_between(x, low, high):
    """
    Count how many values in x fall within [low, high] inclusive.

    Parameters
    ----------
    x : array-like
        Input values.
    low : numeric
        Lower bound (inclusive).
    high : numeric
        Upper bound (inclusive).

    Returns
    -------
    int
        Number of values satisfying low <= x <= high.
    """
    x = np.asarray(x)
    return ((x >= low) & (x <= high)).sum()


def confusion_matrix(y_true, y_pred):
    """
    Compute a 2x2 confusion matrix for binary classification.

    The matrix layout is::

        [[TN, FP],
         [FN, TP]]

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Ground-truth binary labels.
    y_pred : array-like of shape (n,)
        Predicted binary labels.

    Returns
    -------
    np.ndarray of shape (2, 2)
        Confusion matrix with integer counts.

    Raises
    ------
    ValueError
        If inputs are invalid (see check_binary_labels).
    """
    a, b = check_binary_labels(y_true, y_pred)
    tn = ((a == 0) & (b == 0)).sum()
    fp = ((a == 0) & (b == 1)).sum()
    fn = ((a == 1) & (b == 0)).sum()
    tp = ((a == 1) & (b == 1)).sum()
    return np.array([[tn, fp], [fn, tp]])


def _safe_divide_helper(num, denom):
    """Return num / denom, or 0.0 when denom is zero."""
    return 0.0 if denom == 0 else num / denom


def specificity(y_true, y_pred):
    """
    Compute specificity (true negative rate): TN / (TN + FP).

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Ground-truth binary labels.
    y_pred : array-like of shape (n,)
        Predicted binary labels.

    Returns
    -------
    float
        Specificity in [0, 1]. Returns 0.0 when TN + FP == 0.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0, 0], cm[0, 1]
    return _safe_divide_helper(tn, tn + fp)


def precision(y_true, y_pred):
    """
    Compute precision: TP / (TP + FP).

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Ground-truth binary labels.
    y_pred : array-like of shape (n,)
        Predicted binary labels.

    Returns
    -------
    float
        Precision in [0, 1]. Returns 0.0 when TP + FP == 0.
    """
    cm = confusion_matrix(y_true, y_pred)
    tp, fp = cm[1, 1], cm[0, 1]
    return _safe_divide_helper(tp, tp + fp)


def recall(y_true, y_pred):
    """
    Compute recall (sensitivity / true positive rate): TP / (TP + FN).

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Ground-truth binary labels.
    y_pred : array-like of shape (n,)
        Predicted binary labels.

    Returns
    -------
    float
        Recall in [0, 1]. Returns 0.0 when TP + FN == 0.
    """
    cm = confusion_matrix(y_true, y_pred)
    tp, fn = cm[1, 1], cm[1, 0]
    return _safe_divide_helper(tp, tp + fn)


def balance_accuracy(y_true, y_pred):
    """
    Compute balanced accuracy: (recall + specificity) / 2.

    Useful for imbalanced datasets where plain accuracy can be misleading.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Ground-truth binary labels.
    y_pred : array-like of shape (n,)
        Predicted binary labels.

    Returns
    -------
    float
        Balanced accuracy in [0, 1].
    """
    return (recall(y_true, y_pred) + specificity(y_true, y_pred)) / 2


def f1(y_true, y_pred):
    """
    Compute the F1 score: 2 * precision * recall / (precision + recall).

    The F1 score is the harmonic mean of precision and recall. It penalises
    large gaps between the two, making it a better single-number summary than
    accuracy on imbalanced data.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Ground-truth binary labels.
    y_pred : array-like of shape (n,)
        Predicted binary labels.

    Returns
    -------
    float
        F1 score in [0, 1]. Returns 0.0 when precision + recall == 0.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return _safe_divide_helper(2 * p * r, p + r)
