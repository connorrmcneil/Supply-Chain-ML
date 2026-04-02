"""
ensemble_cv.py – 5-fold CV comparing Random Forest, GraphSAGE, and an
RF+GraphSAGE probability ensemble across multiple alpha values.

Ensemble rule:
    p_ensemble = alpha * p_gnn + (1 - alpha) * p_rf
    predict critical when p_ensemble >= 0.5

Alpha grid: [0.3, 0.4, 0.5, 0.6, 0.7]

Reports: accuracy, precision, recall, F1, ROC-AUC, per-node-type F1,
         precision@10, precision@20.

Saves:
    outputs/ensemble_cv_results.json
    outputs/ensemble_cv_results.csv

Run:  python src/ensemble_cv.py
"""

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
ALPHAS = [0.3, 0.4, 0.5, 0.6, 0.7]

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
    """Train GraphSAGE for one fold; return (test_preds, test_probs)."""
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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )

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


def precision_at_k(y_true, y_prob, k):
    """Fraction of truly-critical nodes among the top-k by probability."""
    if k <= 0 or len(y_true) == 0:
        return 0.0
    k = min(k, len(y_true))
    top_k_idx = np.argsort(y_prob)[::-1][:k]
    return float(y_true[top_k_idx].sum()) / k


def _evaluate_model(y_true, y_pred, y_prob, nt_test):
    """Standard metrics + precision@10/20 + per-node-type."""
    m = compute_metrics(y_true, y_pred, y_prob)
    m["precision_at_10"] = precision_at_k(y_true, y_prob, 10)
    m["precision_at_20"] = precision_at_k(y_true, y_prob, 20)
    by_nt = metrics_by_node_type(y_true, y_pred, y_prob, nt_test)
    return m, by_nt


def run():
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
    print(f"Labelled infra nodes: {len(infra_idx)}  "
          f"(crit={int((infra_labels == 1).sum())}, non={int((infra_labels == 0).sum())})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    model_names = ["random_forest", "graphsage_tuned"] + [
        f"ensemble_a{a:.1f}" for a in ALPHAS
    ]
    all_fold_results = {m: [] for m in model_names}
    csv_rows = []

    for fold_i, (train_pos, test_pos) in enumerate(skf.split(infra_idx, infra_labels)):
        fold_num = fold_i + 1
        print(f"\n{'=' * 60}\nFOLD {fold_num}/{N_FOLDS}\n{'=' * 60}")

        train_global = infra_idx[train_pos]
        test_global = infra_idx[test_pos]
        y_train = y_np[train_global]
        y_test = y_np[test_global]
        nt_test = node_types[test_global]

        # ── Random Forest ─────────────────────────────────────────────
        seed_everything(SEED)
        rf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=SEED,
        )
        rf.fit(x_np[train_global], y_train)
        rf_pred = rf.predict(x_np[test_global])
        rf_prob = rf.predict_proba(x_np[test_global])[:, 1]

        rf_m, rf_nt = _evaluate_model(y_test, rf_pred, rf_prob, nt_test)
        all_fold_results["random_forest"].append(
            {"fold": fold_num, **rf_m, "by_node_type": rf_nt}
        )
        print(f"  RF:   F1={rf_m['f1']:.4f}  AUC={rf_m.get('roc_auc', 0):.4f}")

        # ── GraphSAGE ─────────────────────────────────────────────────
        data_dev = data.to(device)
        gnn_pred, gnn_prob = _train_gnn_fold(data_dev, train_global, test_global, device)

        gnn_m, gnn_nt = _evaluate_model(y_test, gnn_pred, gnn_prob, nt_test)
        all_fold_results["graphsage_tuned"].append(
            {"fold": fold_num, **gnn_m, "by_node_type": gnn_nt}
        )
        print(f"  GNN:  F1={gnn_m['f1']:.4f}  AUC={gnn_m.get('roc_auc', 0):.4f}")

        # ── Ensembles across alpha grid ───────────────────────────────
        for alpha in ALPHAS:
            ens_prob = alpha * gnn_prob + (1.0 - alpha) * rf_prob
            ens_pred = (ens_prob >= 0.5).astype(int)
            tag = f"ensemble_a{alpha:.1f}"

            ens_m, ens_nt = _evaluate_model(y_test, ens_pred, ens_prob, nt_test)
            all_fold_results[tag].append(
                {"fold": fold_num, **ens_m, "by_node_type": ens_nt}
            )
            print(f"  {tag}: F1={ens_m['f1']:.4f}  AUC={ens_m.get('roc_auc', 0):.4f}")

        for mname in model_names:
            last = all_fold_results[mname][-1]
            csv_rows.append({
                "model": mname, "fold": fold_num,
                **{k: round(v, 4) for k, v in last.items()
                   if k not in ("fold", "by_node_type")},
            })

    # ── Aggregate ─────────────────────────────────────────────────────
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc",
                   "precision_at_10", "precision_at_20"]
    summary = {}

    for mname in model_names:
        fold_list = all_fold_results[mname]
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

        summary[mname] = {
            "folds": fold_list,
            "aggregate": agg,
            "per_node_type_f1": nt_agg,
        }

        csv_rows.append({
            "model": mname, "fold": "mean",
            **{k: agg.get(f"{k}_mean") for k in metric_keys},
        })
        csv_rows.append({
            "model": mname, "fold": "std",
            **{k: agg.get(f"{k}_std") for k in metric_keys},
        })

    # ── Best ensemble alpha ───────────────────────────────────────────
    best_alpha = max(
        ALPHAS,
        key=lambda a: summary[f"ensemble_a{a:.1f}"]["aggregate"].get("f1_mean", 0),
    )
    best_tag = f"ensemble_a{best_alpha:.1f}"
    summary["_best_ensemble"] = {
        "alpha": best_alpha,
        "model_key": best_tag,
    }

    # ── Save ──────────────────────────────────────────────────────────
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUTS_DIR / "ensemble_cv_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {json_path}")

    csv_df = pd.DataFrame(csv_rows)
    csv_path = OUTPUTS_DIR / "ensemble_cv_results.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print(f"ENSEMBLE CV SUMMARY  ({N_FOLDS}-fold, {data.num_node_features} features)")
    print(f"{'=' * 78}")

    header = (f"{'Model':<20s}  {'Acc':>7s}  {'Prec':>7s}  {'Rec':>7s}  "
              f"{'F1':>7s}  {'AUC':>7s}  {'P@10':>7s}  {'P@20':>7s}")
    print(f"\n{header}")
    print("-" * len(header))

    for mname in model_names:
        a = summary[mname]["aggregate"]
        label = mname.replace("_", " ").replace("graphsage tuned", "GraphSAGE")
        best = " *" if mname == best_tag else ""
        print(f"{label:<20s}  {a.get('accuracy_mean', 0):>7.4f}  "
              f"{a.get('precision_mean', 0):>7.4f}  {a.get('recall_mean', 0):>7.4f}  "
              f"{a.get('f1_mean', 0):>7.4f}  {a.get('roc_auc_mean', 0):>7.4f}  "
              f"{a.get('precision_at_10_mean', 0):>7.4f}  "
              f"{a.get('precision_at_20_mean', 0):>7.4f}{best}")

    print(f"\n  Best ensemble alpha: {best_alpha}")
    best_agg = summary[best_tag]["aggregate"]
    print(f"  Best ensemble F1:    {best_agg['f1_mean']:.4f} +/- {best_agg['f1_std']:.4f}")
    print(f"  Best ensemble AUC:   {best_agg['roc_auc_mean']:.4f} +/- {best_agg['roc_auc_std']:.4f}")

    print(f"\n  Per-node-type F1 (best ensemble vs components):")
    print(f"  {'Type':<12s}  {'RF':>8s}  {'GNN':>8s}  {'Ensemble':>8s}")
    print(f"  {'-'*44}")
    for nt in ["PORT", "PLANT", "WAREHOUSE", "DC"]:
        rf_f1 = summary["random_forest"]["per_node_type_f1"].get(nt, {}).get("f1_mean", 0)
        gnn_f1 = summary["graphsage_tuned"]["per_node_type_f1"].get(nt, {}).get("f1_mean", 0)
        ens_f1 = summary[best_tag]["per_node_type_f1"].get(nt, {}).get("f1_mean", 0)
        print(f"  {nt:<12s}  {rf_f1:>8.4f}  {gnn_f1:>8.4f}  {ens_f1:>8.4f}")


if __name__ == "__main__":
    run()
