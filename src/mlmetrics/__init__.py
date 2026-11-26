"""
mlmetrics — binary classification metrics from scratch.

Public API
----------
Metrics:
    positive_rate, accuracy, count_between, confusion_matrix,
    specificity, precision, recall, balance_accuracy, f1

Validation:
    to_array, check_binary_labels

Cross-validation:
    kfold_indices
"""

from .metrics import (
    positive_rate,
    accuracy,
    count_between,
    confusion_matrix,
    specificity,
    precision,
    recall,
    balance_accuracy,
    f1,
)
from .validation import to_array, check_binary_labels
from .crosseval import kfold_indices

__all__ = [
    "positive_rate",
    "accuracy",
    "count_between",
    "confusion_matrix",
    "specificity",
    "precision",
    "recall",
    "balance_accuracy",
    "f1",
    "to_array",
    "check_binary_labels",
    "kfold_indices",
]
