"""
Generate illustrative daily revenue figures for each PRODUCT node.

Portfolio interpretation:
    1. Normalize volume_weight across all products so shares sum to 1.
    2. revenue_per_day = portfolio_share × FIRM_DAILY_REVENUE_USD

FIRM_DAILY_REVENUE_USD is a single calibration knob representing
illustrative total firm product revenue per day.

Usage:
    python src/build_product_revenue.py
"""

import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Illustrative total firm product revenue per day (USD)
FIRM_DAILY_REVENUE_USD = 500_000.0

master = pd.read_csv(ROOT / "data" / "processed" / "master_nodes.csv")
products = master[master["node_type"] == "PRODUCT"][["node_id", "volume_weight"]].copy()

products["volume_weight_raw"] = products["volume_weight"].copy()
weights = products["volume_weight"].fillna(0).clip(lower=0).astype(float)

total_weight = weights.sum()
if total_weight == 0:
    products["portfolio_share"] = 1.0 / len(products)
else:
    products["portfolio_share"] = weights / total_weight

products["revenue_per_day"] = products["portfolio_share"] * FIRM_DAILY_REVENUE_USD

out = products[["node_id", "revenue_per_day", "volume_weight_raw", "portfolio_share"]]
out_path = ROOT / "data" / "processed" / "product_revenue.csv"
out.to_csv(out_path, index=False)

print(f"Wrote {len(out)} products to {out_path}")
print(f"  revenue_per_day — min: ${out['revenue_per_day'].min():,.0f}, "
      f"max: ${out['revenue_per_day'].max():,.0f}, "
      f"sum: ${out['revenue_per_day'].sum():,.0f}")
