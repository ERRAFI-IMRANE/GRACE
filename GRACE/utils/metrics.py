"""
Evaluation metrics for multi-label affect recognition.

Implements per-class average precision, mean average precision,
macro-F1, and mean AUC-ROC, using scikit-learn under the hood.
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def per_class_ap(y_true: np.ndarray, y_score: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    """
    Compute average precision for each class independently.

    Args:
        y_true: (N, num_classes) binary ground-truth labels.
        y_score: (N, num_classes) predicted scores/probabilities.
        class_names: names for each of the num_classes columns.

    Returns:
        Dict mapping class name to its average precision.
    """
    aps = {}
    for i, name in enumerate(class_names):
        aps[name] = float(average_precision_score(y_true[:, i], y_score[:, i]))
    return aps


def mean_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean average precision (macro-averaged) across all classes."""
    return float(average_precision_score(y_true, y_score, average="macro"))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Macro-averaged F1 score.

    Args:
        y_true: (N, num_classes) binary ground-truth labels.
        y_pred: (N, num_classes) binary predicted labels (post-threshold).
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def mean_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Mean area under the ROC curve, averaged across classes.

    Classes with only one label value present in `y_true` are skipped,
    since ROC-AUC is undefined in that case.
    """
    aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[:, i], y_score[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")
