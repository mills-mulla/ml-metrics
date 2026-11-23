import pytest
import numpy as np
from mlmetrics.metrics import positive_rate, accuracy, count_between, confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


Y_TRUE = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

Y_PRED = [1, 1, 0, 0, 0, 0, 0, 0, 0, 1]

WRONG = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]


def test_positive_rate():
    assert positive_rate([1, 0, 1, 1]) == pytest.approx(0.75)

def test_accuracy_shared():
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(0.7)

def test_perfect_predictions_give_1():
    assert accuracy(Y_TRUE, Y_TRUE) == pytest.approx(1.0) #same list twice 

def test_all_wrong_gives_0():
    assert accuracy(Y_TRUE,WRONG) == pytest.approx(0.0)

def test_bad_input_raises():
    with pytest.raises(ValueError):
        accuracy(Y_TRUE, [])

def test_positive_rate_matches_a_loop():
    for seed in range(100):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, size=50)

        count = 0

        for v in y:
            if v == 1:
                count += 1
        expected = count / len(y)

        assert positive_rate(y) == pytest.approx(expected)

def test_accuracy_matches_sklearn():
    for seed in range(1000):
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, 2, size=50)
        y_pred = rng.integers(0, 2, size=50)
        expected = accuracy_score(y_true, y_pred)

        assert accuracy(y_true, y_pred) == pytest.approx(expected), f"failed at seed{seed}"

def test_count_between():
    assert count_between([1, 4, 6, 9, 5], 3, 7) ==3

def test_confusion_matrix_shared():
    cm = confusion_matrix(Y_TRUE, Y_PRED)
    assert cm.tolist() == [[5, 1], [2, 2]]

def test_shape_is_2x2():
    cm = confusion_matrix(Y_TRUE, Y_PRED)
    assert cm.shape == (2, 2)

def test_perfect_predictions():
    cm = confusion_matrix(Y_TRUE, Y_TRUE)
    assert cm.tolist() == [[6, 0], [0, 4]]

def test_all_wrong():
    cm = confusion_matrix(Y_TRUE, WRONG)
    assert cm.tolist() == [[0, 6], [4, 0]]

def test_bad_input_raises():
    with pytest.raises(ValueError):
        confusion_matrix([],[])

def test_matches_accuracy():
    cm = confusion_matrix(Y_TRUE, Y_PRED)
    assert cm.trace() / cm.sum() == pytest.approx(accuracy(Y_TRUE, Y_PRED)) 

def test_matches_sklearn():
    for seed in range(100):
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, 2, size=50)
        y_pred = rng.integers(0, 2, size=50)
        mine = confusion_matrix(y_true, y_pred)

        assert np.array_equal(mine, sk_confusion_matrix(y_true, y_pred, labels=[0, 1]))    
