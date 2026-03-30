"""
train_gnn.py – Train a 2-layer GraphSAGE for binary node classification.

Architecture:
    x [600, 26]
      → SAGEConv(26, 64) → ReLU → Dropout(0.5)
      → SAGEConv(64, 2)
      → log_softmax

The forward pass runs on the *full* 600-node graph (all edges), but the
loss is computed only on train_mask nodes.  Early stopping watches val F1
and saves the best checkpoint.

Run:  python src/train_gnn.py
      python src/train_gnn.py --no-class-weight   # disable class weighting
      python src/train_gnn.py --epochs 300
"""

import argparse
import os
import pathlib
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

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

INFRA_TYPES = {"PORT", "PLANT", "WAREHOUSE", "DC"}


# ── Reproducibility ──────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Model ────────────────────────────────────────────────────────────────────

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64,
                 out_channels: int = 2, dropout: float = 0.5):
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


# ── Training helpers ─────────────────────────────────────────────────────────

def compute_class_weights(y, train_mask):
    """Inverse-frequency weights for the two classes in the training set."""
    train_labels = y[train_mask]
    counts = torch.bincount(train_labels, minlength=2).float()
    weights = counts.sum() / (2.0 * counts)
    return weights


@torch.no_grad()
def evaluate_split(model, data, mask):
    """Return (loss, predictions, probabilities) for a given mask."""
    model.eval()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[mask], data.y[mask]).item()
    probs = out[mask].exp()[:, 1].cpu().numpy()
    preds = out[mask].argmax(dim=1).cpu().numpy()
    return loss, preds, probs


def train_epoch(model, data, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-class-weight", action="store_true",
                        help="Disable inverse-frequency class weighting in loss")
    args = parser.parse_args()

    seed_everything(args.seed)

    # ── Load data ────────────────────────────────────────────────────────
    print("Loading data …")
    data = torch.load(PROCESSED_DIR / "graph_data.pt", weights_only=False)
    master = pd.read_csv(PROCESSED_DIR / "master_nodes.csv")
    node_ids = master["node_id"].values
    node_types = master["node_type"].values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    # ── Class weights ────────────────────────────────────────────────────
    if args.no_class_weight:
        loss_fn = torch.nn.NLLLoss()
        print("Class weighting: OFF")
    else:
        weights = compute_class_weights(data.y, data.train_mask).to(device)
        loss_fn = torch.nn.NLLLoss(weight=weights)
        print(f"Class weighting: ON  (weights = {weights.cpu().tolist()})")

    # ── Model ────────────────────────────────────────────────────────────
    in_channels = data.num_node_features
    model = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=args.hidden,
        out_channels=2,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    print(f"Model: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Training loop ────────────────────────────────────────────────────
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    curves = []

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience}) …\n")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, data, optimizer, loss_fn)

        val_loss, val_preds, val_probs = evaluate_split(
            model, data, data.val_mask,
        )
        val_true = data.y[data.val_mask].cpu().numpy()
        val_metrics = compute_metrics(val_true, val_preds, val_probs)
        val_f1 = val_metrics["f1"]

        curves.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_f1": round(val_f1, 5),
        })

        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), OUTPUTS_DIR / "best_model.pt")
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1 or improved:
            tag = " *" if improved else ""
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_f1={val_f1:.4f}{tag}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(best val F1 = {best_val_f1:.4f} at epoch {best_epoch})")
            break

    # ── Load best model and evaluate ─────────────────────────────────────
    model.load_state_dict(torch.load(OUTPUTS_DIR / "best_model.pt", weights_only=True))
    model.eval()

    results = {"graphsage": {"best_epoch": best_epoch, "best_val_f1": round(best_val_f1, 5)}}

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        full_probs = out.exp()[:, 1].cpu().numpy()
        full_preds = out.argmax(dim=1).cpu().numpy()
    full_true = data.y.cpu().numpy()

    for split_name, mask in [("val", data.val_mask), ("test", data.test_mask)]:
        idx = mask.cpu().numpy().nonzero()[0]
        yt = full_true[idx]
        yp = full_preds[idx]
        yprob = full_probs[idx]
        nt = node_types[idx]

        results["graphsage"][split_name] = compute_metrics(yt, yp, yprob)
        results["graphsage"][split_name]["confusion_matrix"] = compute_confusion_matrix(yt, yp)
        results["graphsage"][f"{split_name}_by_node_type"] = metrics_by_node_type(
            yt, yp, yprob, nt,
        )

    # ── Save outputs ─────────────────────────────────────────────────────
    save_results(results, OUTPUTS_DIR / "gnn_results.json")

    # Predictions CSV
    all_idx = np.concatenate([
        data.train_mask.cpu().numpy().nonzero()[0],
        data.val_mask.cpu().numpy().nonzero()[0],
        data.test_mask.cpu().numpy().nonzero()[0],
    ])
    split_labels = np.empty(len(full_true), dtype=object)
    for name, mask in [("train", data.train_mask),
                       ("val", data.val_mask),
                       ("test", data.test_mask)]:
        split_labels[mask.cpu().numpy().nonzero()[0]] = name

    pred_df = pd.DataFrame({
        "node_id": node_ids[all_idx],
        "node_type": node_types[all_idx],
        "split": split_labels[all_idx],
        "y_true": full_true[all_idx],
        "gnn_pred": full_preds[all_idx],
        "gnn_prob": np.round(full_probs[all_idx], 4),
    })
    pred_df.to_csv(OUTPUTS_DIR / "gnn_predictions.csv", index=False)
    print(f"Saved predictions → {OUTPUTS_DIR / 'gnn_predictions.csv'}")

    # Training curves CSV
    curves_df = pd.DataFrame(curves)
    curves_df.to_csv(OUTPUTS_DIR / "gnn_training_curves.csv", index=False)
    print(f"Saved training curves → {OUTPUTS_DIR / 'gnn_training_curves.csv'}")

    # Print summary
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"GRAPHSAGE RESULTS  (best epoch = {best_epoch}, best val F1 = {best_val_f1:.4f})")
    print(sep)
    for split_name in ["val", "test"]:
        m = results["graphsage"][split_name]
        print(f"\n  {split_name}:")
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            if k in m:
                print(f"    {k:12s}: {m[k]:.4f}")


if __name__ == "__main__":
    main()
