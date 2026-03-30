"""
evaluate.py – Shared evaluation helpers for all training scripts.

Every model (baseline or GNN) calls the same functions so that metrics
are computed identically and results JSONs have a consistent schema.

Functions:
    compute_metrics        – accuracy, precision, recall, F1, ROC-AUC
    compute_confusion_matrix – 2x2 confusion matrix (JSON-serialisable)
    metrics_by_node_type   – per-infrastructure-type metric breakdown
    save_results           – write a dict to a JSON file
"""

import json
import pathlib
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

INFRA_TYPES = ["PORT", "PLANT", "WAREHOUSE", "DC"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Return a flat dict of classification metrics.

    Parameters
    ----------
    y_true : array of ground-truth labels (0 or 1).
    y_pred : array of predicted labels (0 or 1).
    y_prob : array of predicted probability for the *positive* class (1).
             If None, ROC-AUC is omitted.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> List[List[int]]:
    """Return the 2x2 confusion matrix as a list-of-lists."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return cm.tolist()


def metrics_by_node_type(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    node_types: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute per-infrastructure-type metrics.

    Skips any type with fewer than 2 distinct classes in that subset
    (ROC-AUC is undefined there, but we still report the other metrics).
    """
    result = {}
    for nt in INFRA_TYPES:
        mask = node_types == nt
        if mask.sum() == 0:
            continue
        nt_prob = y_prob[mask] if y_prob is not None else None
        result[nt] = compute_metrics(y_true[mask], y_pred[mask], nt_prob)
        result[nt]["count"] = int(mask.sum())
    return result


def save_results(results: dict, path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results → {path}")
    return path


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100)
    y_pred = rng.integers(0, 2, size=100)
    y_prob = rng.random(100)
    print("Demo metrics:", compute_metrics(y_true, y_pred, y_prob))
    print("Demo CM:", compute_confusion_matrix(y_true, y_pred))
