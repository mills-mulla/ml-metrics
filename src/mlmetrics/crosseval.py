import numpy as np


def kfold_indices(n, k, shuffle=False, seed=None):
    """ 
    should:
    1.split arrays into k folds for training
    """
    if not (2 <= k <= n and isinstance(k, int)):
        raise ValueError(f"error procesing request")
    folds = np.array_split(np.arange(n), k)
    for i in range(k):
        test = folds[i]
        others = folds[:i] + folds[i + 1:]
        train = np.concatenate(others)
        yield train, test
  
    
  