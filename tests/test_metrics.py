import pytest
import numpy as np
from mlmetrics.metrics import positive_rate, accuracy, count_between, confusion_matrix, specificity, precision, recall, balance_accuracy, f1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

def test_specificity_shared():
    assert specificity(Y_TRUE, Y_PRED) == pytest.approx(5 / 6)

def test_specificity_zero_division():
    assert specificity([1, 1], [1, 1]) == pytest.approx(0.0)


def test_precision_shared():
    assert precision(Y_TRUE, Y_PRED) == pytest.approx(2 / 3)

def test_recall_shared():
    assert recall(Y_TRUE, Y_PRED) == pytest.approx(2 / 4)

def test_precision_perfect_prediction():
    assert precision(Y_TRUE, Y_TRUE) == pytest.approx(1.0)

def test_precision_wrong_prediction():
    assert precision(Y_TRUE, WRONG) == pytest.approx(0.0)

def test_precision_zero_division():
    assert precision([1, 0, 1], [0, 0, 0]) == pytest.approx(0.0)

def test_recall_zero_division():
    assert recall([0, 0], [0, 1])  == pytest.approx(0.0)

def test_bad_input_precision_raises():
    with pytest.raises(ValueError):
        precision([], [])
    
    with pytest.raises(ValueError):
        recall([], [])

def test_precision_recall_matches_sklearn():
    for seed in range(100):
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, 2, size=50)
        y_pred = rng.integers(0, 2, size=50)

        pre = precision(y_true, y_pred)
        rec = recall(y_true, y_pred)

        assert pre == pytest.approx(precision_score(y_true, y_pred, zero_division=0))
        assert rec == pytest.approx(recall_score(y_true, y_pred, zero_division=0)) 

def test_trade_off_precision_and_recall():
    """ predict all 1's and one correct positive only case """
    assert precision(Y_TRUE, [1] * 10) == pytest.approx(0.4)
    assert recall(Y_TRUE,[1] * 10) == pytest.approx(1.0)

    assert precision(Y_TRUE, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) == pytest.approx(1.0)
    assert recall(Y_TRUE,[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) == pytest.approx(0.25)

def test_balance_accuracy_shared():
    assert balance_accuracy(Y_TRUE, Y_PRED) == pytest.approx(2 / 3)

def test_f1_shared():
    assert f1(Y_TRUE, Y_PRED) == pytest.approx(4 / 7)

def test_f1_perfect_prediction():
    assert f1(Y_TRUE, Y_TRUE) == pytest.approx(1.0)
   
def test_f1_all_wrong_prediction():
    assert f1(Y_TRUE, WRONG) == pytest.approx(0.0)

def test_f1_zero_division():
    assert f1([0, 0], [0, 1]) == pytest.approx(0.0)

def test_f1_punsishes_imbalances():
    assert f1([1, 1, 1, 1, 0, 0],[1, 0, 0, 0, 0, 0]) == pytest.approx(0.4)

def test_f1_harmonic_mean_formulas_agree():
    """ formula [1]. F1 = 2.P.R / (P + R) harmonic mean formula [2]. F1 = 2.TP/(2.TP + FP + FN)"""
    pass

def test_bad_input_raises():
    with pytest.raises(ValueError):
        f1([], []) 

def test_f1_matches_sklearn_f1_score():
    for seed in range(100):
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, 2, size=50)
        y_pred = rng.integers(0, 2, size=50)

        assert f1(y_true, y_pred) == pytest.approx(f1_score(y_true, y_pred))

def test_accuracy_misleads_on_imbalance_data():
    y_true = [1] * 5 + [0] *95
    y_pred = [0] * 100
    cm = confusion_matrix(y_true, y_pred)

    assert accuracy(y_true, y_pred) == pytest.approx(0.95)
    assert  cm.tolist() == [[95, 0], [5, 0]]
    assert precision(y_true, y_pred) == pytest.approx(0.0)
    assert recall(y_true, y_pred) == pytest.approx(0.0)
    assert f1(y_true, y_pred) == pytest.approx(0.0)