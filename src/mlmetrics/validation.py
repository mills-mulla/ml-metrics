import numpy as np


def to_array(values):
    """
    Convert values to a 1-D numpy array.

    Parameters
    ----------
    values : array-like
        Input data to convert.

    Returns
    -------
    np.ndarray of shape (n,)
        1-D array representation of values.

    Raises
    ------
    ValueError
        If the resulting array is not 1-D.
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"expected 1d array, got {arr.ndim} dimensions")
    return arr


def check_binary_labels(y_true, y_pred):
    """
    Validate and return two binary label arrays.

    Checks that both inputs are 1-D, non-empty, contain only 0s and 1s, and
    have the same length.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Ground-truth binary labels.
    y_pred : array-like of shape (n,)
        Predicted binary labels.

    Returns
    -------
    a : np.ndarray of shape (n,)
        Validated ground-truth array.
    b : np.ndarray of shape (n,)
        Validated prediction array.

    Raises
    ------
    ValueError
        If either array is not 1-D, is empty, contains values other than 0/1,
        or the two arrays differ in length.
    """
    a = np.asarray(y_true)
    b = np.asarray(y_pred)

    for arr in (a, b):
        if arr.ndim != 1:
            raise ValueError(f"expected 1-D got {arr.ndim}-D")
        if len(arr) == 0:
            raise ValueError("empty array not allowed")
        if not ((arr == 0) | (arr == 1)).all():
            raise ValueError("only 0 and 1 allowed")

    if len(a) != len(b):
        raise ValueError(f"length mismatch: a has {len(a)} items but b has {len(b)}")

    return a, b
