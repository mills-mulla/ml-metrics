import numpy as np

def to_array(values):
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"expected 1d array, got {arr.ndim} dimensions")
    return arr

def check_binary_labels(y_true, y_pred):
    a = np.asarray(y_true)
    b = np.asarray(y_pred)

    for arr in(a, b):
        if arr.ndim != 1 :
            raise ValueError(f"expected 1-D got {arr.ndim}-D")
        if len(arr) == 0:
            raise ValueError(f"Empty array not allowed")
        if not ((arr == 0 ) | (arr == 1)).all():
            raise ValueError(f"only 0 and 1 allowed")
    if len(a) != len(b):
        raise ValueError(f"length a has {len(a)} items but b has {len(b)}")

    return a, b

