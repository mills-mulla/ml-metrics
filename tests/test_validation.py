import numpy as np
import pytest
from mlmetrics.validation import to_array,check_binary_labels

def test_to_array_accepts_a_list():
    result = to_array([1,2,3])
    assert result.tolist() == [1,2,3]

def test_to_array_rejects_2d():
    with pytest.raises(ValueError):
        to_array([[1, 0],[1, 1]])

def test_numpy_returns_arrays():
    y_true, y_pred = check_binary_labels([1, 0, 1], [1, 1, 0])
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert y_true.tolist() == [1, 0, 1]

def test_different_lengths_raise():
    with pytest.raises(ValueError):
        check_binary_labels([1, 0, 1], [1, 0])
    
def test_empty_input_raises():
    with pytest.raises(ValueError):
        check_binary_labels([],[])

def test_2d_input_raises():
    with pytest.raises(ValueError):
        check_binary_labels([[1, 0], [1, 1]],[[0, 1], [1, 0]])

    with pytest.raises(ValueError):
        check_binary_labels([1, 1, 1], [[1,1], [1,0]])

def test_values_other_than_0_1_raises():
    with pytest.raises(ValueError):
        check_binary_labels([6, 2, 5], [3, 4, 6])
    with pytest.raises(ValueError):
        check_binary_labels([1, 0, 1], [2, 1, 0])
