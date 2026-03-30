"""
train_baseline.py – Train non-graph baselines for comparison with the GNN.

Models:
  1. Majority-class  – always predicts the most common training label (sanity check)
  2. Logistic Regression – linear model, class-weight balanced
  3. Random Forest – ensemble of decision trees, class-weight balanced

All models use the 26-dim node feature vectors only (no graph edges).
Train/val/test splits come from the masks already stored in graph_data.pt.

Run:  python src/train_baseline.py
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

_SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from evaluate import (
    compute_confusion_matrix,
    compute_metrics,
    metrics_by_node_type,
    save_results,
)

PROCESSED_DIR = _SRC_DIR.parent / "data" / "processed"
OUTPUTS_DIR = _SRC_DIR.parent / "outputs"


def _load_data():
    data = torch.load(PROCESSED_DIR / "graph_data.pt", weights_only=False)
    master = pd.read_csv(PROCESSED_DIR / "master_nodes.csv")

    x = data.x.numpy()
    y = data.y.numpy()
    node_ids = master["node_id"].values
    node_types = master["node_type"].values

    splits = {}
    for name, mask in [("train", data.train_mask),
                       ("val", data.val_mask),
                       ("test", data.test_mask)]:
        idx = mask.numpy().nonzero()[0]
        splits[name] = idx

    return x, y, node_ids, node_types, splits


def _evaluate_model(model_name, y_pred_all, y_prob_all, y, node_types, splits):
    """Evaluate a model on val and test, return a results dict."""
    result = {}
    for split_name in ["val", "test"]:
        idx = splits[split_name]
        yt = y[idx]
        yp = y_pred_all[idx]
        yprob = y_prob_all[idx] if y_prob_all is not None else None

        result[split_name] = compute_metrics(yt, yp, yprob)
        result[split_name]["confusion_matrix"] = compute_confusion_matrix(yt, yp)
        result[f"{split_name}_by_node_type"] = metrics_by_node_type(
            yt, yp, yprob, node_types[idx]
        )
    return result


def train_majority(x, y, splits):
    """Always predict the most frequent class in the training set."""
    train_labels = y[splits["train"]]
    counts = np.bincount(train_labels)
    majority_class = int(counts.argmax())

    all_idx = np.concatenate([splits["train"], splits["val"], splits["test"]])
    y_pred = np.full(len(y), -1, dtype=int)
    y_pred[all_idx] = majority_class

    y_prob = np.full(len(y), 0.0)
    y_prob[all_idx] = 1.0 if majority_class == 1 else 0.0

    print(f"  Majority class = {majority_class}  "
          f"(train counts: 0→{counts[0]}, 1→{counts[1]})")
    return y_pred, y_prob


def train_lr(x, y, splits):
    """Logistic Regression with balanced class weights."""
    tr = splits["train"]
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(x[tr], y[tr])

    all_idx = np.concatenate([splits["train"], splits["val"], splits["test"]])
    y_pred = np.full(len(y), -1, dtype=int)
    y_prob = np.full(len(y), 0.0)
    y_pred[all_idx] = model.predict(x[all_idx])
    y_prob[all_idx] = model.predict_proba(x[all_idx])[:, 1]

    return y_pred, y_prob


def train_rf(x, y, splits):
    """Random Forest with balanced class weights."""
    tr = splits["train"]
    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42,
    )
    model.fit(x[tr], y[tr])

    all_idx = np.concatenate([splits["train"], splits["val"], splits["test"]])
    y_pred = np.full(len(y), -1, dtype=int)
    y_prob = np.full(len(y), 0.0)
    y_pred[all_idx] = model.predict(x[all_idx])
    y_prob[all_idx] = model.predict_proba(x[all_idx])[:, 1]

    return y_pred, y_prob


def main():
    print("Loading data …")
    x, y, node_ids, node_types, splits = _load_data()

    all_results = {}
    predictions = {}  # model_name -> (y_pred, y_prob)

    models = [
        ("majority", train_majority),
        ("logistic_regression", train_lr),
        ("random_forest", train_rf),
    ]

    for model_name, train_fn in models:
        print(f"\nTraining {model_name} …")
        y_pred, y_prob = train_fn(x, y, splits)
        predictions[model_name] = (y_pred, y_prob)

        result = _evaluate_model(model_name, y_pred, y_prob, y, node_types, splits)
        all_results[model_name] = result

    # Build one predictions CSV with a row per labelled node
    all_idx = np.concatenate([splits["train"], splits["val"], splits["test"]])
    split_labels = np.empty(len(y), dtype=object)
    for name in ["train", "val", "test"]:
        split_labels[splits[name]] = name

    merged = pd.DataFrame({
        "node_id": node_ids[all_idx],
        "node_type": node_types[all_idx],
        "split": split_labels[all_idx],
        "y_true": y[all_idx],
    })
    for model_name in predictions:
        yp, yprob = predictions[model_name]
        merged[f"{model_name}_pred"] = yp[all_idx]
        merged[f"{model_name}_prob"] = np.round(yprob[all_idx], 4)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    save_results(all_results, OUTPUTS_DIR / "baseline_results.json")
    merged.to_csv(OUTPUTS_DIR / "baseline_predictions.csv", index=False)
    print(f"Saved predictions → {OUTPUTS_DIR / 'baseline_predictions.csv'}")

    # Print summary
    sep = "=" * 60
    print(f"\n{sep}")
    print("BASELINE RESULTS (test set)")
    print(sep)
    for model_name in all_results:
        m = all_results[model_name]["test"]
        print(f"\n  {model_name}:")
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            if k in m:
                print(f"    {k:12s}: {m[k]:.4f}")


if __name__ == "__main__":
    main()
