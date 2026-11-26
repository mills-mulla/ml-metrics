import numpy as np
import pytest
from sklearn.model_selection import KFold

from mlmetrics.crosseval import kfold_indices

def test_fold_size_n10_k3():
    sizes = [len(test) for _, test in kfold_indices(10, 3)]
    assert sizes == [4, 3, 3]

def test_number_of_folds_equal_k():
    k = 3
    n = 10
    assert len(list(kfold_indices(n, k))) == k

def test_every_index_tested_exactly_once():
    n = 10
    k = 3
    folds = list(kfold_indices(n, k))
    all_test = np.concatenate([test for _, test in folds])
    sort = np.sort(all_test)
    assert np.array_equal(sort, np.arange(n)) 

def test_train_test_never_overlap():
    n = 10
    k = 3
    folds = list(kfold_indices(n, k))
    for train, test in folds:
        assert np.intersect1d(train,test).size == 0

def test_train_test_cover_everything():
    n = 10
    k = 3
    folds = list(kfold_indices(n, k))
    for train, test in folds:
        assert len(train) + len(test) == n
        assert np.array_equal(np.sort(np.concatenate([train, test])), np.arange(n))

def test_sizes_n9_k4():
    k = 4
    n = 9
    folds = kfold_indices(n, k)
    size = [len(test) for _, test in folds]
    assert size == [3, 2, 2, 2]

def test_my_function_matches_sklearn():
    n = 10
    k = 3
    mine = list(kfold_indices(n, k))
    sklrn = list(KFold(k).split(np.arange(n)))

    assert len(mine) == len(sklrn)
    for (my_train, my_test), (sk_train, sk_test) in zip(mine, sklrn):
        assert np.array_equal(my_test, sk_test)
        assert np.array_equal(my_train, sk_train)
    
def test_shuffle_still_covers_everything_once():
    n = 10
    folds = list(kfold_indices(n, 3, shuffle=True, seed=0))
    all_tests = np.concatenate([test for _, test in folds])
    assert np.array_equal(np.sort(all_tests), np.arange(n))

    for train, test in folds:
        assert np.intersect1d(train, test).size == 0

def test_same_seed_gives_same_folds():
    a = list(kfold_indices(20, 4, shuffle=True, seed=5))
    b = list(kfold_indices(20, 4, shuffle=True, seed=5))
    c = list(kfold_indices(20, 4, shuffle=True, seed=6))

    for (train_a, test_a), (train_b, test_b) in zip(a,b):
        assert np.array_equal(test_a, test_b)

def test_invalid_k_raises():
    with pytest.raises(ValueError):
        list(kfold_indices(10, 1))
    
    with pytest.raises(ValueError):
        list(kfold_indices(10, 0))
    
    with pytest.raises(ValueError):
        list(kfold_indices(10, 11))
    
    with pytest.raises(ValueError):
        list(kfold_indices(10, 2.5))
