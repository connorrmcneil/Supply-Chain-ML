"""
cross_validate.py – Stratified 5-fold cross-validation for LR, RF, and
tuned GraphSAGE on the 400 labelled infrastructure nodes.

The full graph structure (600 nodes, 1593 edges) is kept in every fold.
Only the train/test assignment of labelled infrastructure nodes changes.

GNN early stopping uses a stratified 85/15 inner split of the outer train
fold so that the outer test fold is never seen during training.

Saves:
    outputs/cv_results[_SUFFIX].json   – full fold-level + aggregated results
    outputs/cv_results[_SUFFIX].csv    – one-row-per-fold summary table

Does NOT overwrite any existing outputs.

Run:  python src/cross_validate.py [--suffix v2]
"""

import argparse
import json
import os
import pathlib
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.nn import SAGEConv

_SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from evaluate import compute_metrics, metrics_by_node_type

PROCESSED_DIR = _SRC_DIR.parent / "data" / "processed"
OUTPUTS_DIR = _SRC_DIR.parent / "outputs"
INFRA_TYPES = {"PORT", "PLANT", "WAREHOUSE", "DC"}

SEED = 42
N_FOLDS = 5

GNN_CONFIG = {
    "hidden": 128,
    "dropout": 0.3,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "epochs": 200,
    "patience": 30,
}


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=2, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def _compute_class_weights(y, train_mask):
    train_labels = y[train_mask]
    counts = torch.bincount(train_labels, minlength=2).float()
    return counts.sum() / (2.0 * counts)


def _train_gnn_fold(data, train_global, test_global, device):
    """Train tuned GraphSAGE for one CV fold with proper inner val split.

    1. Split train_global 85/15 stratified -> inner_train / inner_val
    2. Train on inner_train, early-stop on inner_val F1
    3. Evaluate once on test_global (untouched during training)
    """
    seed_everything(SEED)
    cfg = GNN_CONFIG
    n_nodes = data.x.shape[0]
    y_np = data.y.cpu().numpy()

    train_labels = y_np[train_global]
    inner_train_pos, inner_val_pos = train_test_split(
        np.arange(len(train_global)),
        test_size=0.15,
        stratify=train_labels,
        random_state=SEED,
    )
    inner_train_idx = train_global[inner_train_pos]
    inner_val_idx = train_global[inner_val_pos]

    inner_train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    inner_val_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
    inner_train_mask[inner_train_idx] = True
    inner_val_mask[inner_val_idx] = True
    test_mask[test_global] = True

    weights = _compute_class_weights(data.y, inner_train_mask).to(device)
    loss_fn = torch.nn.NLLLoss(weight=weights)

    model = GraphSAGE(
        in_channels=data.num_node_features,
        hidden_channels=cfg["hidden"],
        out_channels=2,
        dropout=cfg["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])

    best_state = None
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[inner_train_mask], data.y[inner_train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(data.x, data.edge_index)
            val_preds = out_eval[inner_val_mask].argmax(dim=1).cpu().numpy()
            val_true = data.y[inner_val_mask].cpu().numpy()
        val_f1 = f1_score(val_true, val_preds, zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= cfg["patience"]:
            break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        all_probs = out.exp()[:, 1].cpu().numpy()
        all_preds = out.argmax(dim=1).cpu().numpy()

    test_idx = test_mask.cpu().numpy().nonzero()[0]
    return all_preds[test_idx], all_probs[test_idx]


def run_cv(suffix: str = ""):
    tag = f"_{suffix}" if suffix else ""
    print("Loading data …")
    data = torch.load(PROCESSED_DIR / "graph_data.pt", weights_only=False)
    master = pd.read_csv(PROCESSED_DIR / "master_nodes.csv")
    print(f"  Feature dimension: {data.num_node_features}")

    x_np = data.x.numpy()
    y_np = data.y.numpy()
    node_types = master["node_type"].values

    infra_mask = np.array([nt in INFRA_TYPES for nt in node_types])
    infra_idx = np.where(infra_mask)[0]
    infra_labels = y_np[infra_idx]

    print(f"Labelled infrastructure nodes: {len(infra_idx)}")
    print(f"  critical: {(infra_labels == 1).sum()}, non-critical: {(infra_labels == 0).sum()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    all_fold_results = defaultdict(list)
    csv_rows = []

    for fold_i, (train_pos, test_pos) in enumerate(skf.split(infra_idx, infra_labels)):
        fold_num = fold_i + 1
        print(f"\n{'='*60}")
        print(f"FOLD {fold_num}/{N_FOLDS}")
        print(f"{'='*60}")

        train_global = infra_idx[train_pos]
        test_global = infra_idx[test_pos]

        y_train = y_np[train_global]
        y_test = y_np[test_global]
        nt_test = node_types[test_global]
        print(f"  train: {len(train_global)} (crit={sum(y_train==1)}, non={sum(y_train==0)})")
        print(f"  test:  {len(test_global)}  (crit={sum(y_test==1)}, non={sum(y_test==0)})")

        # ── Logistic Regression (full outer train fold) ──────────────────
        seed_everything(SEED)
        lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        lr_model.fit(x_np[train_global], y_train)
        lr_pred = lr_model.predict(x_np[test_global])
        lr_prob = lr_model.predict_proba(x_np[test_global])[:, 1]

        lr_metrics = compute_metrics(y_test, lr_pred, lr_prob)
        lr_by_nt = metrics_by_node_type(y_test, lr_pred, lr_prob, nt_test)
        all_fold_results["logistic_regression"].append({
            "fold": fold_num, **lr_metrics, "by_node_type": lr_by_nt,
        })
        print(f"  LR:  F1={lr_metrics['f1']:.4f}  AUC={lr_metrics.get('roc_auc', 0):.4f}")

        # ── Random Forest (full outer train fold) ────────────────────────
        seed_everything(SEED)
        rf_model = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=SEED,
        )
        rf_model.fit(x_np[train_global], y_train)
        rf_pred = rf_model.predict(x_np[test_global])
        rf_prob = rf_model.predict_proba(x_np[test_global])[:, 1]

        rf_metrics = compute_metrics(y_test, rf_pred, rf_prob)
        rf_by_nt = metrics_by_node_type(y_test, rf_pred, rf_prob, nt_test)
        all_fold_results["random_forest"].append({
            "fold": fold_num, **rf_metrics, "by_node_type": rf_by_nt,
        })
        print(f"  RF:  F1={rf_metrics['f1']:.4f}  AUC={rf_metrics.get('roc_auc', 0):.4f}")

        # ── Tuned GraphSAGE (inner 85/15 split for early stopping) ──────
        data_dev = data.to(device)
        gnn_pred, gnn_prob = _train_gnn_fold(data_dev, train_global, test_global, device)

        gnn_metrics = compute_metrics(y_test, gnn_pred, gnn_prob)
        gnn_by_nt = metrics_by_node_type(y_test, gnn_pred, gnn_prob, nt_test)
        all_fold_results["graphsage_tuned"].append({
            "fold": fold_num, **gnn_metrics, "by_node_type": gnn_by_nt,
        })
        print(f"  GNN: F1={gnn_metrics['f1']:.4f}  AUC={gnn_metrics.get('roc_auc', 0):.4f}")

        for model_name, m in [("logistic_regression", lr_metrics),
                               ("random_forest", rf_metrics),
                               ("graphsage_tuned", gnn_metrics)]:
            csv_rows.append({
                "model": model_name, "fold": fold_num,
                **{k: round(v, 4) for k, v in m.items()},
            })

    # ── Aggregate across folds ───────────────────────────────────────────
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    summary = {}

    for model_name, fold_list in all_fold_results.items():
        agg = {}
        for k in metric_keys:
            vals = [f[k] for f in fold_list if k in f]
            if vals:
                agg[f"{k}_mean"] = round(float(np.mean(vals)), 4)
                agg[f"{k}_std"] = round(float(np.std(vals)), 4)

        nt_f1s = defaultdict(list)
        for f in fold_list:
            for nt, nt_m in f.get("by_node_type", {}).items():
                nt_f1s[nt].append(nt_m.get("f1", 0))

        nt_agg = {}
        for nt, vals in nt_f1s.items():
            nt_agg[nt] = {
                "f1_mean": round(float(np.mean(vals)), 4),
                "f1_std": round(float(np.std(vals)), 4),
                "n_folds": len(vals),
            }

        summary[model_name] = {
            "folds": fold_list,
            "aggregate": agg,
            "per_node_type_f1": nt_agg,
        }

        csv_rows.append({
            "model": model_name, "fold": "mean",
            **{k: agg.get(f"{k}_mean") for k in metric_keys},
        })
        csv_rows.append({
            "model": model_name, "fold": "std",
            **{k: agg.get(f"{k}_std") for k in metric_keys},
        })

    # ── Save results ─────────────────────────────────────────────────────
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUTS_DIR / f"cv_results{tag}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {json_path}")

    csv_df = pd.DataFrame(csv_rows)
    csv_path = OUTPUTS_DIR / f"cv_results{tag}.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION SUMMARY ({N_FOLDS}-fold stratified)")
    print(f"{'='*70}")

    for model_name in ["logistic_regression", "random_forest", "graphsage_tuned"]:
        agg = summary[model_name]["aggregate"]
        print(f"\n  {model_name}:")
        for k in metric_keys:
            mean = agg.get(f"{k}_mean", 0)
            std = agg.get(f"{k}_std", 0)
            print(f"    {k:12s}: {mean:.4f} +/- {std:.4f}")

        print(f"    per-node-type F1:")
        for nt in ["PORT", "PLANT", "WAREHOUSE", "DC"]:
            nt_data = summary[model_name]["per_node_type_f1"].get(nt, {})
            if nt_data:
                print(f"      {nt:12s}: {nt_data['f1_mean']:.4f} +/- {nt_data['f1_std']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="", help="e.g. 'v2' → cv_results_v2.json")
    args = parser.parse_args()
    run_cv(suffix=args.suffix)
