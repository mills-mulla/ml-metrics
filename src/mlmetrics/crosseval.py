import numpy as np


def kfold_indices(n, k, shuffle=False, seed=None):
    """
    Generate train/test index splits for k-fold cross-validation.

    Splits the integers 0..n-1 into k non-overlapping folds. Each fold is
    used once as the test set while the remaining k-1 folds form the training
    set.

    Parameters
    ----------
    n : int
        Total number of samples.
    k : int
        Number of folds. Must satisfy 2 <= k <= n.
    shuffle : bool, optional
        Whether to shuffle the indices before splitting. Default is False.
    seed : int or None, optional
        Random seed passed to numpy.random.default_rng when shuffle=True.

    Yields
    ------
    train : np.ndarray
        Indices for the training set.
    test : np.ndarray
        Indices for the test set.

    Raises
    ------
    ValueError
        If k is not an integer or not in the range [2, n].
    """
    if not (2 <= k <= n and isinstance(k, int)):
        raise ValueError(f"k must be an integer with 2 <= k <= n, got k={k}, n={n}")
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(indices)
    folds = np.array_split(indices, k)
    for i in range(k):
        test = folds[i]
        train = np.concatenate(folds[:i] + folds[i + 1:])
        yield train, test
