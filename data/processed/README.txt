Brewery Supply Chain Synthetic Dataset (150-200 nodes) WITH is_active + backups

Design:
- nodes.csv: is_active (1/0) for node toggling
- relationships.csv: is_active + activation_cost + unit_cost_delta + lead_time_delta for reinforcement toggles

Counts:
- nodes: 165
- relationships: 750
- products: 8
- recipe rows: 77
- disruptions: 420

Backups (inactive):
- 12 backup suppliers (hops/yeast/packaging/fruit/malt/CO2/sugar)
- 1 backup warehouse (WH_3_BK)
- 1 backup DC (DC_4_BK)
Added (inactive by default):
- PLANT_3_BK (backup contract plant)
- PORT_2_BK (backup port)
Also added inactive edges for:
- ROUTES_THROUGH to PORT_2_BK (subset of imported components)
- DELIVERS_TO inbound lanes to PLANT_3_BK
- MAKES edges from PLANT_3_BK to selected products
- SHIPS_TO from PLANT_3_BK to warehouses

New counts:
- nodes: 167
- relationships: 781
- disruptions: 440
