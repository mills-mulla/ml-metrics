# ml-metrics

A from-scratch implementation of binary classification metrics and k-fold cross-validation using NumPy. Built to understand how these algorithms work under the hood, with results verified against scikit-learn.

---

## What it does

### Metrics (`mlmetrics.metrics`)

| Function | Formula | Description |
|---|---|---|
| `positive_rate(y)` | `sum(y==1) / n` | Fraction of positive labels |
| `accuracy(y_true, y_pred)` | `correct / n` | Overall correctness |
| `count_between(x, low, high)` | — | Count values in a range |
| `confusion_matrix(y_true, y_pred)` | `[[TN, FP], [FN, TP]]` | 2×2 count matrix |
| `specificity(y_true, y_pred)` | `TN / (TN + FP)` | True negative rate |
| `precision(y_true, y_pred)` | `TP / (TP + FP)` | Positive predictive value |
| `recall(y_true, y_pred)` | `TP / (TP + FN)` | Sensitivity / true positive rate |
| `balance_accuracy(y_true, y_pred)` | `(recall + specificity) / 2` | Accuracy adjusted for class imbalance |
| `f1(y_true, y_pred)` | `2·P·R / (P + R)` | Harmonic mean of precision and recall |

### Validation (`mlmetrics.validation`)

| Function | Description |
|---|---|
| `to_array(values)` | Convert any array-like to a 1-D numpy array |
| `check_binary_labels(y_true, y_pred)` | Validate both arrays are 1-D, non-empty, binary, and equal length |

### Cross-validation (`mlmetrics.crosseval`)

| Function | Description |
|---|---|
| `kfold_indices(n, k, shuffle, seed)` | Yield `(train_indices, test_indices)` for each of k folds |

---

## How it works

### Confusion matrix

All metrics are derived from four counts computed by `confusion_matrix`:

```
Predicted →      0       1
Actual 0   [ TN,    FP ]
Actual 1   [ FN,    TP ]
```

- **TN** — correctly predicted negative
- **FP** — negative predicted as positive (false alarm)
- **FN** — positive predicted as negative (missed)
- **TP** — correctly predicted positive

### Why accuracy can mislead

On a dataset with 95 negatives and 5 positives, a model that always predicts 0 scores 95% accuracy while having 0% recall and 0% precision. `balance_accuracy` and `f1` expose this.

### K-fold cross-validation

`kfold_indices(n, k)` splits indices `0..n-1` into k equal (or near-equal) folds using `numpy.array_split`. Each call yields one fold as the test set and the rest concatenated as the training set, so every sample is tested exactly once.

With `shuffle=True` the indices are randomly permuted before splitting, using `numpy.random.default_rng(seed)` for reproducibility.

---

## Installation

```bash
# from the project root
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest
```

All metrics are cross-checked against scikit-learn (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `KFold`) across 100–1000 random seeds.

---

## Project structure

```
ml-metrics/
├── src/mlmetrics/
│   ├── __init__.py      # public API exports
│   ├── crosseval.py     # kfold_indices
│   ├── metrics.py       # all metric functions
│   └── validation.py    # input validation helpers
├── tests/
│   ├── test_crosseval.py
│   ├── test_metrics.py
│   └── test_validation.py
└── pyproject.toml
```
