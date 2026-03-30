"""
tune_gnn.py – Controlled hyperparameter grid search over GraphSAGE.

Grid:
    hidden_dim  : [64, 128]
    dropout     : [0.3, 0.5]
    learning_rate: [0.01, 0.005]

Fixed:
    epochs=200, patience=30, weight_decay=5e-4, seed=42, class_weight=ON

For each of the 8 configs the script:
  1. Resets the random seed
  2. Trains from scratch with early stopping on val F1
  3. Evaluates the best checkpoint on val + test (overall and by node type)

After the sweep it identifies the best config by val F1 and overwrites the
canonical outputs (gnn_results.json, gnn_predictions.csv, etc.) so that the
rest of the pipeline stays consistent.

Run:  python src/tune_gnn.py
"""

import itertools
import json
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

_SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from evaluate import (
    compute_confusion_matrix,
    compute_metrics,
    metrics_by_node_type,
    save_results,
)
from train_gnn import (
    GraphSAGE,
    compute_class_weights,
    evaluate_split,
    seed_everything,
    train_epoch,
)

PROCESSED_DIR = _SRC_DIR.parent / "data" / "processed"
OUTPUTS_DIR = _SRC_DIR.parent / "outputs"

GRID = {
    "hidden": [64, 128],
    "dropout": [0.3, 0.5],
    "lr": [0.01, 0.005],
}
FIXED = {
    "epochs": 200,
    "patience": 30,
    "weight_decay": 5e-4,
    "seed": 42,
}


def run_single_config(data, node_types, config, device, tmp_dir):
    """Train one GraphSAGE config, return metrics dict + best state_dict."""
    seed_everything(FIXED["seed"])

    weights = compute_class_weights(data.y, data.train_mask).to(device)
    loss_fn = torch.nn.NLLLoss(weight=weights)

    model = GraphSAGE(
        in_channels=data.num_node_features,
        hidden_channels=config["hidden"],
        out_channels=2,
        dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=FIXED["weight_decay"],
    )

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    best_state = None
    curves = []

    for epoch in range(1, FIXED["epochs"] + 1):
        train_loss = train_epoch(model, data, optimizer, loss_fn)

        val_loss, val_preds, val_probs = evaluate_split(model, data, data.val_mask)
        val_true = data.y[data.val_mask].cpu().numpy()
        val_m = compute_metrics(val_true, val_preds, val_probs)
        val_f1 = val_m["f1"]

        curves.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_f1": round(val_f1, 5),
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= FIXED["patience"]:
            break

    # Evaluate best checkpoint on val + test
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        full_probs = out.exp()[:, 1].cpu().numpy()
        full_preds = out.argmax(dim=1).cpu().numpy()
    full_true = data.y.cpu().numpy()

    result = {
        "hidden": config["hidden"],
        "dropout": config["dropout"],
        "lr": config["lr"],
        "best_epoch": best_epoch,
        "best_val_f1": round(best_val_f1, 5),
    }

    for split_name, mask in [("val", data.val_mask), ("test", data.test_mask)]:
        idx = mask.cpu().numpy().nonzero()[0]
        yt, yp, yprob = full_true[idx], full_preds[idx], full_probs[idx]
        nt = node_types[idx]
        m = compute_metrics(yt, yp, yprob)
        result[split_name] = m
        result[split_name]["confusion_matrix"] = compute_confusion_matrix(yt, yp)
        result[f"{split_name}_by_node_type"] = metrics_by_node_type(yt, yp, yprob, nt)

    return result, best_state, curves, full_preds, full_probs, full_true


def main():
    print("Loading data …")
    data = torch.load(PROCESSED_DIR / "graph_data.pt", weights_only=False)
    master = pd.read_csv(PROCESSED_DIR / "master_nodes.csv")
    node_ids = master["node_id"].values
    node_types = master["node_type"].values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    configs = [
        dict(zip(GRID.keys(), vals))
        for vals in itertools.product(*GRID.values())
    ]

    print(f"\nRunning {len(configs)} configurations …\n")
    sep = "-" * 70

    all_results = []
    all_curves = {}
    best_overall_val_f1 = 0.0
    best_idx = 0

    for i, cfg in enumerate(configs):
        tag = f"h={cfg['hidden']}  d={cfg['dropout']}  lr={cfg['lr']}"
        print(f"[{i+1}/{len(configs)}] {tag}")

        result, state, curves, preds, probs, true_y = run_single_config(
            data, node_types, cfg, device, None,
        )
        all_results.append(result)
        all_curves[tag] = curves

        vf1 = result["best_val_f1"]
        tf1 = result["test"]["f1"]
        ep = result["best_epoch"]
        print(f"         best_epoch={ep}  val_f1={vf1:.4f}  test_f1={tf1:.4f}")

        if vf1 > best_overall_val_f1:
            best_overall_val_f1 = vf1
            best_idx = i
            best_state = state
            best_preds = preds
            best_probs = probs
            best_true = true_y
            best_curves = curves

        print(sep)

    # ── Summary table ────────────────────────────────────────────────────
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "hidden": r["hidden"],
            "dropout": r["dropout"],
            "lr": r["lr"],
            "best_epoch": r["best_epoch"],
            "val_f1": round(r["best_val_f1"], 4),
            "val_accuracy": round(r["val"]["accuracy"], 4),
            "val_roc_auc": round(r["val"].get("roc_auc", 0), 4),
            "test_f1": round(r["test"]["f1"], 4),
            "test_accuracy": round(r["test"]["accuracy"], 4),
            "test_precision": round(r["test"]["precision"], 4),
            "test_recall": round(r["test"]["recall"], 4),
            "test_roc_auc": round(r["test"].get("roc_auc", 0), 4),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("val_f1", ascending=False).reset_index(drop=True)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUTS_DIR / "tuning_results.csv", index=False)
    print(f"\nSaved tuning summary → {OUTPUTS_DIR / 'tuning_results.csv'}")

    # Full JSON with per-node-type breakdowns
    save_results(all_results, OUTPUTS_DIR / "tuning_results.json")

    # ── Overwrite canonical outputs with best config ─────────────────────
    best_cfg = configs[best_idx]
    best_result = all_results[best_idx]
    print(f"\n{'=' * 70}")
    print(f"BEST CONFIG: hidden={best_cfg['hidden']}, dropout={best_cfg['dropout']}, "
          f"lr={best_cfg['lr']}")
    print(f"  best_epoch = {best_result['best_epoch']}")
    print(f"  val  F1 = {best_result['best_val_f1']:.4f}")
    print(f"  test F1 = {best_result['test']['f1']:.4f}")
    print(f"{'=' * 70}")

    # Save best model checkpoint
    torch.save(best_state, OUTPUTS_DIR / "best_model.pt")
    print(f"Saved best model → {OUTPUTS_DIR / 'best_model.pt'}")

    # Save canonical gnn_results.json
    gnn_results = {"graphsage": {
        "best_epoch": best_result["best_epoch"],
        "best_val_f1": best_result["best_val_f1"],
        "config": best_cfg,
        "val": best_result["val"],
        "val_by_node_type": best_result["val_by_node_type"],
        "test": best_result["test"],
        "test_by_node_type": best_result["test_by_node_type"],
    }}
    save_results(gnn_results, OUTPUTS_DIR / "gnn_results.json")

    # Save canonical gnn_predictions.csv
    all_idx = np.concatenate([
        data.train_mask.cpu().numpy().nonzero()[0],
        data.val_mask.cpu().numpy().nonzero()[0],
        data.test_mask.cpu().numpy().nonzero()[0],
    ])
    split_labels = np.empty(len(best_true), dtype=object)
    for name, mask in [("train", data.train_mask),
                       ("val", data.val_mask),
                       ("test", data.test_mask)]:
        split_labels[mask.cpu().numpy().nonzero()[0]] = name

    pred_df = pd.DataFrame({
        "node_id": node_ids[all_idx],
        "node_type": node_types[all_idx],
        "split": split_labels[all_idx],
        "y_true": best_true[all_idx],
        "gnn_pred": best_preds[all_idx],
        "gnn_prob": np.round(best_probs[all_idx], 4),
    })
    pred_df.to_csv(OUTPUTS_DIR / "gnn_predictions.csv", index=False)
    print(f"Saved predictions → {OUTPUTS_DIR / 'gnn_predictions.csv'}")

    # Save training curves for the best config
    curves_df = pd.DataFrame(best_curves)
    curves_df.to_csv(OUTPUTS_DIR / "gnn_training_curves.csv", index=False)
    print(f"Saved training curves → {OUTPUTS_DIR / 'gnn_training_curves.csv'}")

    # Print full comparison table
    print(f"\n{'=' * 70}")
    print("TUNING RESULTS (sorted by val F1)")
    print(f"{'=' * 70}")
    print(summary_df.to_string(index=False))

    # Print per-node-type for best config
    print(f"\nBest config — test metrics by node type:")
    for nt, m in best_result.get("test_by_node_type", {}).items():
        print(f"  {nt:12s}: F1={m['f1']:.4f}  prec={m['precision']:.4f}  "
              f"rec={m['recall']:.4f}  n={m['count']}")


if __name__ == "__main__":
    main()
