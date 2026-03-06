# Supply-Chain Redundancy Dataset (Synthetic)

This dataset is a **synthetic supply-chain infrastructure graph** designed for:
- **Redundancy / resilience analysis** (identify single points of failure)
- **Disruption simulation** (remove/disable infrastructure nodes and measure service loss)
- **Graph ML** (e.g., node2vec embeddings + a classifier/regressor to predict infrastructure criticality)

It models a simplified network with **physical infrastructure nodes** (ports, plants, warehouses, distribution centers) plus **context nodes** (products and ingredients) that create realistic dependency structure.

---

## Files

- `nodes_with_notes.csv` — node table including a human-readable `notes` column
- `nodes_no_notes.csv` — same node table, without the `notes` column (cleaner for ML pipelines)
- `edges.csv` — relationship table (directed edges)
- `dataset_summary.json` — quick counts and basic stats

---

## Dataset size (this build)

### Nodes
- Total nodes: **600**
- Infrastructure nodes (prediction targets): **400**
  - PORT: 15
  - PLANT: 90
  - WAREHOUSE: 120
  - DC: 175
- Context nodes: **200**
  - INGREDIENT: 140
  - PRODUCT: 60

### Edges
- Total edges: **1593**
- By relationship type:
- `SHIPS_TO`: 419
- `REQUIRES`: 387
- `REPLENISHES`: 341
- `SUPPLIES`: 266
- `IMPORTS_TO`: 180

### Key realism stats
- Ingredients: **42 single-sourced**, **98 dual-or-more sourced**
- Product recipes (PRODUCT → INGREDIENT):
  - min ingredients per product: 3
  - max ingredients per product: 14
  - avg ingredients per product: 6.45

---

## Node schema

All nodes share:
- `node_id` — unique string id
- `node_type` — one of: `PORT`, `PLANT`, `WAREHOUSE`, `DC`, `INGREDIENT`, `PRODUCT`
- `region` — a coarse region label (infrastructure nodes only)

### Infrastructure node properties (PORT / PLANT / WAREHOUSE / DC)

These nodes are the **ML prediction targets** (they are the ones you will label as critical).

- `capacity_units` — maximum capacity (synthetic units)
- `throughput_units` — current throughput (≤ capacity)
- `recovery_days` — estimated recovery time after a disruption
- `backup_level` — categorical redundancy indicator: `none` / `partial` / `full`
  - heuristic assignment based on structural redundancy (see “Generation logic”)
- `is_backup_capable` — `0/1` flag; `1` means the node has enough spare capacity to help cover others
  - computed by spare ratio: `(capacity_units - throughput_units) / capacity_units >= 0.25`

### INGREDIENT properties

- `criticality_weight` — importance of the ingredient (0–1)
- `substitutable` — `0/1` (0 means difficult/no substitute)

### PRODUCT properties

- `volume_weight` — relative importance / volume (0–1)

### Notes column (only in `nodes_with_notes.csv`)
- `notes` — simple explanation of why the node looks redundant/fragile (helps demos + debugging)

---

## Edge schema

`edges.csv` columns:
- `src_id` — source node id
- `rel_type` — relationship type
- `dst_id` — destination node id
- `lane_capacity` — optional numeric lane capacity (used for shipping-style edges)
- `lead_time_days` — optional numeric lead time for a lane
- `mode` — optional transport mode (`truck`, `rail`, `ship`)
- `is_backup` — `0/1` flag indicating whether the lane is meant as a “backup” path

### Relationship types (directed)

**Dependency / context**
- `PRODUCT -[REQUIRES]-> INGREDIENT`  
  A product recipe/BOM: product requires one or more ingredients.
- `INGREDIENT -[SUPPLIES]-> PLANT`  
  Which plants can provide/source a given ingredient.

**Infrastructure flow**
- `PORT -[IMPORTS_TO]-> PLANT`  
  Ports that can supply/import into plants.
- `PLANT -[SHIPS_TO]-> WAREHOUSE`  
  Plants shipping into warehouses (some lanes marked as backups).
- `WAREHOUSE -[REPLENISHES]-> DC`  
  Warehouses replenishing distribution centers (some lanes marked as backups).

---

## Generation logic (high-level)

### Counts and distributions

**Context node split**
- 140 INGREDIENT
- 60 PRODUCT

**Product recipes (PRODUCT → INGREDIENT)**
- Products require multiple ingredients; average recipe size ≈ 6.45.
- A small set of “common ingredients” are reused across many products.
- Highest-volume products are biased to include several high-criticality ingredients.

**Ingredient sourcing redundancy (INGREDIENT → PLANT)**
- ~30% single-sourced ingredients
- ~50% dual-sourced ingredients
- ~20% multi-sourced ingredients (3 plants)

**Plant port redundancy (PORT → PLANT)**
- 20% of plants have 1 port
- 60% have 2 ports
- 20% have 3 ports

**Plant → warehouse shipping**
- Each plant ships to 3–6 warehouses
- 1–2 lanes are marked primary (`is_backup=0`); the remainder are backup lanes

**Warehouse → DC replenishment**
- 25% of DCs served by 1 warehouse (fragile)
- 55% served by 2 warehouses
- 20% served by 3 warehouses (more resilient)

---

## How this dataset supports your ML + resilience workflow

### 1) Disruption labeling (ground truth)
To label infrastructure node criticality, you typically:
1. Compute **baseline service coverage**.
2. For each infrastructure node (PORT/PLANT/WAREHOUSE/DC), simulate an outage (remove/disable it).
3. Recompute coverage and measure **impact**.

A simple “coverage” definition that matches this dataset:
- A product is **serviceable** if there exists at least one plant that can source **all required ingredients**  
  (via `REQUIRES` + `SUPPLIES`)
- and that plant can reach at least one DC through the infrastructure paths  
  (`PLANT -> WAREHOUSE -> DC`)

Use `volume_weight` to weight products so high-volume products matter more.

### 2) Node2vec embeddings
Run node2vec on the full graph (including context nodes).  
Then train a model **only on infrastructure nodes** using:
- features: node2vec embedding vector (and optionally the raw infra attributes)
- target: disruption-derived label or impact score

---

## Loading into Neo4j (quick outline)

1. Import nodes (example Cypher outline):
   - Create nodes with labels by `node_type` or one label with a `node_type` property.
2. Import edges from `edges.csv` and create relationships using `rel_type`.

If you want, you can keep a single label `:Node` and store `node_type` as a property, or create labels like `:PORT`, `:PLANT`, etc.

---

## Notes / caveats

- This dataset is **synthetic** and intended for learning + demonstration.
- `backup_level` is currently a **heuristic** based on structural redundancy; you can replace it later with a value derived from disruption impacts (quantiles) for more rigor.
- `lane_capacity`, `lead_time_days`, and `mode` are included for realism and possible extensions (e.g., capacity-aware disruption metrics), but they are not required for basic reachability-based coverage metrics.

